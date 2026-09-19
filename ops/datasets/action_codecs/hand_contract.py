"""Static, asset-bound hand contracts. No learned calibration or asset rebuilding."""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import axis_rotation, urdf_origin

SPEC_SCHEMA = "skynet.unidex-faas/v1"
DEFAULT_SPEC_ROOT = Path(__file__).resolve().parent / "specs"
if not DEFAULT_SPEC_ROOT.is_dir():
    DEFAULT_SPEC_ROOT = Path(__file__).resolve().parents[3] / "config/action_representations/unidex-faas-v1"


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _vector(element, key, default):
    value = np.fromstring(
        element.get(key, default) if element is not None else default, sep=" "
    )
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Invalid URDF vector")
    return value


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    mimic: dict | None


@dataclass
class HandSpec:
    """Verified static asset metadata, independent of any target demonstrations."""

    data: dict
    joints: dict
    root_link: str
    lower: np.ndarray
    upper: np.ndarray
    frame_offset: np.ndarray
    effective_lower: np.ndarray
    effective_upper: np.ndarray

    @property
    def robot(self):
        return self.data["robot"]

    @property
    def spec_hash(self):
        return self.data["spec_hash"]

    @property
    def action_dim(self):
        return len(self.control_names)

    @property
    def state_dim(self):
        return self.action_dim

    @property
    def control_names(self):
        return list(self.data["control_names"])

    @property
    def capture(self):
        return json.loads(json.dumps(self.data["capture"]))

    @property
    def asset(self):
        return json.loads(json.dumps(self.data["asset"]))

    @property
    def mapping_status(self):
        return self.data["mapping_status"]

    @property
    def frame_link(self):
        return self.data["frame_link"]

    @property
    def tip_links(self):
        return tuple(self.data["tip_links"])

    def expand(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape[-1:] != (self.action_dim,) or not np.isfinite(q).all():
            raise ValueError(
                "Native joint coordinates have wrong shape or nonfinite values"
            )
        values = {name: q[..., i] for i, name in enumerate(self.control_names)}
        pending = {name: j for name, j in self.joints.items() if j.mimic}
        while pending:
            ready = [name for name, j in pending.items() if j.mimic["joint"] in values]
            if not ready:
                raise ValueError("Mimic graph has an unresolved or cyclic parent")
            for name in ready:
                j = pending.pop(name)
                m = j.mimic
                values[name] = values[m["joint"]] * m["multiplier"] + m["offset"]
        return values

    def forward_kinematics(self, q):
        """Independent full URDF FK for one native state; no simulator needed."""
        q = np.asarray(q, dtype=float)
        if q.shape != (self.action_dim,):
            raise ValueError("FK expects one native state")
        values = self.expand(q)
        poses = {self.root_link: np.eye(4)}
        pending = dict(self.joints)
        while pending:
            ready = [name for name, j in pending.items() if j.parent in poses]
            if not ready:
                raise ValueError("URDF joint tree is disconnected or cyclic")
            for name in ready:
                j = pending.pop(name)
                motion = np.eye(4)
                if j.kind in ("revolute", "continuous"):
                    motion[:3, :3] = axis_rotation(j.axis, float(values[name]))
                elif j.kind == "prismatic":
                    motion[:3, 3] = j.axis * float(values[name])
                poses[j.child] = poses[j.parent] @ j.origin @ motion
        return poses

    def validate_capture(self, capture):
        """Reject changed native order, controller, source revision, or hand asset."""
        expected = self.capture
        for key in (
            "robot",
            "task",
            "hand",
            "source_revision",
            "action_joint_names",
            "action_semantics",
        ):
            if capture.get(key) != expected.get(key):
                raise ValueError("Capture contract mismatch: " + key)
        for key in ("action_scale", "action_offset"):
            if not np.array_equal(
                np.asarray(capture.get(key)), np.asarray(expected[key])
            ):
                raise ValueError("Capture controller differs: " + key)
        if not np.isclose(
            float(capture.get("step_dt", 0)),
            float(expected["step_dt"]),
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError("Capture timing differs")
        if "hand_asset" in expected:
            if capture.get("hand_asset") != expected["hand_asset"]:
                raise ValueError("Capture hand asset differs")
            if capture.get("kinematics_sha256") != expected["kinematics_sha256"]:
                raise ValueError("Capture kinematics differs")
            if capture.get("wrist_rotation_order") != "intrinsic_XYZ":
                raise ValueError("Capture wrist convention differs")
        elif (
            capture.get("legacy_urdf_sha256", self.asset["urdf_sha256"])
            != self.asset["urdf_sha256"]
        ):
            raise ValueError("Legacy Shadow asset differs")

    def verify_runtime_controller(self, path):
        """Bind the source that applies limits beyond the literal legacy URDF."""
        expected = self.data["runtime_controller"]["source_sha256"]
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Runtime controller differs from pinned codec contract")

    def verify_runtime_asset(self, path):
        """Evaluator must verify actual runtime asset, especially legacy metadata."""
        expected = self.asset.get(
            "simulation_urdf_sha256", self.asset.get("urdf_sha256")
        )
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Runtime URDF differs from pinned codec asset")


def load_spec(robot_id, *, spec_root=None):
    if not isinstance(robot_id, str) or not re.fullmatch(r"[a-z0-9_]+", robot_id):
        raise ValueError("Invalid robot identity")
    root = Path(spec_root or DEFAULT_SPEC_ROOT)
    path = root / (robot_id + ".json")
    if not path.is_file():
        raise ValueError("No exact asset-bound FAAS contract: " + robot_id)
    data = json.loads(path.read_text())
    body = {k: v for k, v in data.items() if k != "spec_hash"}
    if data.get("spec_hash") != canonical_digest(body):
        raise ValueError("Hand spec checksum changed")
    if (
        data.get("schema") != SPEC_SCHEMA
        or data.get("robot") != robot_id
        or data.get("side") != "right"
        or data.get("faas_dim") != 82
    ):
        raise ValueError("Unsupported hand/FAAS contract")
    if (
        data.get("units") != {"length": "m", "angle": "rad"}
        or data.get("reference_frame") != "fixed_robot_base"
    ):
        raise ValueError("Unsupported units or pose reference frame")
    rel = Path(data["kinematics_file"])
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Invalid kinematics path")
    raw = (root / rel).read_bytes()
    if hashlib.sha256(raw).hexdigest() != data["kinematics_sha256"]:
        raise ValueError("Kinematics checksum changed")
    tree = ET.fromstring(raw)
    links = {e.get("name") for e in tree.findall("link")}
    joints = {}
    children = set()
    for e in tree.findall("joint"):
        kind = e.get("type")
        name = e.get("name")
        limit = e.find("limit")
        m = e.find("mimic")
        if kind not in ("fixed", "revolute", "continuous", "prismatic"):
            raise ValueError("Unsupported URDF joint")
        if name in joints:
            raise ValueError("Duplicate joint")
        mimic = (
            None
            if m is None
            else {
                "joint": m.get("joint"),
                "multiplier": float(m.get("multiplier", "1")),
                "offset": float(m.get("offset", "0")),
            }
        )
        axis = _vector(e.find("axis"), "xyz", "1 0 0")
        origin = e.find("origin")
        j = Joint(
            name,
            kind,
            e.find("parent").get("link"),
            e.find("child").get("link"),
            urdf_origin(
                _vector(origin, "xyz", "0 0 0"), _vector(origin, "rpy", "0 0 0")
            ),
            axis,
            float(limit.get("lower", "-inf")) if limit is not None else -np.inf,
            float(limit.get("upper", "inf")) if limit is not None else np.inf,
            mimic,
        )
        if j.child in children or j.parent not in links or j.child not in links:
            raise ValueError("Invalid URDF tree")
        joints[name] = j
        children.add(j.child)
    roots = links - children
    if len(roots) != 1:
        raise ValueError("URDF needs one fixed root")
    names = data["control_names"]
    wrist = data["wrist_names"]
    if len(set(names)) != len(names) or len(wrist) != 6 or not set(wrist) <= set(names):
        raise ValueError("Invalid named controls")
    expected = {n for n, j in joints.items() if j.kind != "fixed" and not j.mimic}
    if set(names) != expected:
        raise ValueError("Controls must exactly cover independent URDF joints")
    for key in ("action_scale", "action_offset"):
        a = np.asarray(data[key], dtype=float)
        if (
            a.shape != (len(names),)
            or not np.isfinite(a).all()
            or (key == "action_scale" and np.any(a == 0))
        ):
            raise ValueError("Invalid controller affine transform")
    frame = np.asarray(data["link_to_frame"], dtype=float)
    if (
        frame.shape != (4, 4)
        or not np.isfinite(frame).all()
        or not np.allclose(frame[3], [0, 0, 0, 1])
        or not np.allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3), atol=1e-8)
        or not np.isclose(np.linalg.det(frame[:3, :3]), 1)
    ):
        raise ValueError("Invalid palm frame transform")
    # Only the exact supported floating chain is inverted analytically. Never
    # silently interpret arbitrary URDF joint coordinates as Euler components.
    parents = {j.child: j for j in joints.values()}
    chain = []
    link = data["frame_link"]
    while link in parents:
        j = parents[link]
        chain.append(j)
        link = j.parent
    chain.reverse()
    moving = [j for j in chain if j.kind != "fixed"]
    if [j.name for j in moving] != wrist:
        raise ValueError("Unsupported wrist chain order")
    fixed = np.eye(4)
    seen = 0
    for j in chain:
        if j.kind == "fixed":
            if seen != 6:
                raise ValueError("Wrist has an unsupported intervening fixed joint")
            fixed = fixed @ j.origin
        else:
            axis = np.eye(3)[seen % 3]
            if (
                j.kind != ("prismatic" if seen < 3 else "revolute")
                or not np.allclose(j.axis, axis)
                or not np.allclose(j.origin, np.eye(4))
            ):
                raise ValueError("Unsupported wrist axis/origin")
            seen += 1
    lower = np.array([joints[n].lower for n in names])
    upper = np.array([joints[n].upper for n in names])
    if (
        not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower >= upper)
    ):
        raise ValueError("Invalid independent joint bounds")
    runtime = data.get("runtime_controller", {})
    expected_overrides = {name: [-1000000.0, 1000000.0] for name in wrist[3:]}
    if (
        runtime.get("function") != "configure_virtual_wrist"
        or runtime.get("source_file") != "ops/xr/wrist.py"
        or not re.fullmatch(r"[a-f0-9]{64}", runtime.get("source_sha256", ""))
        or runtime.get("position_limit_overrides") != expected_overrides
    ):
        raise ValueError("Unsupported runtime controller limit override")
    for name, limits in expected_overrides.items():
        index = names.index(name)
        lower[index], upper[index] = limits

    # Intersect actuator limits with every dependent joint's physical limits.
    # Some upstream mimic coefficients are rounded (e.g. 1.334*.6 > .8), so
    # independent URDF limits alone do not guarantee a realizable target.
    effective_lower, effective_upper = lower.copy(), upper.copy()
    relations = {name: (i, 1.0, 0.0) for i, name in enumerate(names)}
    pending = {name: j for name, j in joints.items() if j.mimic}
    while pending:
        ready = [name for name, j in pending.items() if j.mimic["joint"] in relations]
        if not ready:
            raise ValueError("Cyclic or unresolved mimic graph")
        for name in ready:
            j = pending.pop(name)
            m = j.mimic
            index, gain, offset = relations[m["joint"]]
            gain, offset = (
                gain * m["multiplier"],
                offset * m["multiplier"] + m["offset"],
            )
            relations[name] = (index, gain, offset)
            if gain == 0:
                if not j.lower <= offset <= j.upper:
                    raise ValueError("Impossible constant mimic target")
                continue
            limits = sorted([(j.lower - offset) / gain, (j.upper - offset) / gain])
            effective_lower[index] = max(effective_lower[index], limits[0])
            effective_upper[index] = min(effective_upper[index], limits[1])
    if np.any(effective_lower > effective_upper):
        raise ValueError("Incompatible mimic joint limits")
    spec = HandSpec(
        data,
        joints,
        roots.pop(),
        lower,
        upper,
        fixed @ frame,
        effective_lower,
        effective_upper,
    )
    spec.expand(np.zeros(len(names)))  # Also checks mimic DAG validity.
    if data["mapping_status"] == "VERIFIED":
        mapping = data["mapping"]
        slots = [m["slot"] for m in mapping]
        mapped = [m["joint"] for m in mapping]
        if (
            len(set(slots)) != len(slots)
            or len(set(mapped)) != len(mapped)
            or any(not isinstance(s, int) or not 0 <= s < 32 for s in slots)
        ):
            raise ValueError("Duplicate/invalid FAAS mapping")
        fingers = {n for n, j in joints.items() if j.kind != "fixed"} - set(wrist)
        if set(mapped) != fingers:
            raise ValueError("FAAS map must cover every physical finger joint")
        if any(
            not np.isfinite([m["scale"], m["offset"]]).all() or m["scale"] == 0
            for m in mapping
        ):
            raise ValueError("Invalid FAAS affine transform")
    elif data["mapping_status"] != "BLOCKED":
        raise ValueError("Unknown mapping status")
    return spec
