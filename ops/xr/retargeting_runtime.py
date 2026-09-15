"""Optional upstream solver behind the same named-joint collection boundary."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

REVISION = "3846d3fa207165bb0d498145aac8b885a28ea923"
UTILS_REVISION = "2d8dc1a5abf5899069f9ec73c13de73674f4c897"


def load_upstream(root):
    root = Path(root).resolve()
    receipt = root / ".skynet-retargeting-ready.json"
    if not receipt.is_file():
        raise ValueError(
            "Vector Wrist Joint runtime is not installed on this collection host"
        )
    saved = json.loads(receipt.read_text())
    if (
        saved.get("revision") != REVISION
        or saved.get("utils_revision") != UTILS_REVISION
    ):
        raise ValueError("Vector Wrist Joint runtime revision differs from the session")
    if not saved.get("files"):
        raise ValueError("Missing retargeting runtime checksums")
    for name, digest in saved["files"].items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root)
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Retargeting runtime checksum changed: " + name)
    for path in (
        root / "dependencies",
        root / "source/third_party/utils_python",
        root / "source/src",
    ):
        sys.path.insert(0, str(path))
    from retargeting.core.retargeter import Retargeter

    module_path = Path(sys.modules[Retargeter.__module__].__file__).resolve()
    if not module_path.is_relative_to(root / "source/src"):
        raise ValueError("Another retargeting package shadowed the pinned runtime")
    return Retargeter


def kinematic_hand(xml, hand, destination):
    """Extract one exact hand/wrist branch; visual assets stay in the canonical bundle."""
    tree = ET.fromstring(xml)
    parents = {j.find("child").get("link"): j for j in tree.findall("joint")}
    links = set()
    joints = set()
    for name in [hand["control_frame"], *hand["tips"]]:
        visited = set()
        while name:
            if name in visited:
                raise ValueError("Cycle in recorded hand kinematics")
            visited.add(name)
            links.add(name)
            joint = parents.get(name)
            if joint is None:
                break
            joints.add(joint.get("name"))
            name = joint.find("parent").get("link")
    expected = hand["wrist_joints"] + hand["finger_joints"]
    movable = {
        j.get("name")
        for j in tree.findall("joint")
        if j.get("name") in joints and j.get("type") != "fixed"
    }
    if movable != set(expected):
        raise ValueError("Retargeting branch does not contain every named action joint")
    if any(
        j.find("mimic") is not None
        for j in tree.findall("joint")
        if j.get("name") in joints
    ):
        raise ValueError(
            "Vector Wrist Joint does not support coupled joints; choose DexPilot"
        )
    base_links = links - {
        j.find("child").get("link") for j in parents.values() if j.get("name") in joints
    }
    if len(base_links) != 1:
        raise ValueError("Retargeting hand has no unique root")
    output = ET.Element("robot", name="skynet_retargeting")
    for item in tree:
        if (item.tag == "link" and item.get("name") in links) or (
            item.tag == "joint" and item.get("name") in joints
        ):
            item = copy.deepcopy(item)
            for geometry in list(item.findall("visual")) + list(
                item.findall("collision")
            ):
                item.remove(geometry)
            output.append(item)
    base = next(iter(base_links))
    if "world" not in links:
        ET.SubElement(output, "link", name="world")
        j = ET.SubElement(
            output, "joint", name="skynet_retargeting_world", type="fixed"
        )
        ET.SubElement(j, "parent", link="world")
        ET.SubElement(j, "child", link=base)
    elif base != "world":
        raise ValueError("World frame is not the root of this hand")
    raw = ET.tostring(output, encoding="utf-8", xml_declaration=True)
    Path(destination).write_bytes(raw)
    return expected, list(links | {"world"}), hashlib.sha256(raw).hexdigest()


def link_kinematics(path):
    """Resolve URDF links explicitly, preserving joint names and upstream math."""
    import pinocchio as pin
    from retargeting.core.kinematics.pinocchio_model import RobotPinocchio

    class LinkKinematics(RobotPinocchio):
        def get_frame_index(self, name):
            # URDF permits a joint and a link to share a name. Targets refer to
            # links, so neither JOINT nor FIXED_JOINT frames are valid matches.
            index = self.model.getFrameId(name, pin.FrameType.BODY)
            if index >= self.model.nframes:
                raise ValueError("Unknown hand link: " + name)
            return index

    return LinkKinematics(str(path), "urdf")


def finger_position_weights(manifest, hand, upstream_robot, upstream_profile):
    """Map a known hand profile by source joint name, never by array position.

    A generic hand has no calibrated neutral-pose prior. Joint limits and the
    temporal penalty still apply; borrowing Shadow's weights would constrain
    unrelated joints on other embodiments.
    """
    if manifest.get("hand_key") != "shadow":
        return [0.0] * len(hand["finger_joints"])
    names = upstream_robot["actuated_joints"]
    weights = upstream_profile["retargeting"]["joint_position_weights"]
    if len(names) != len(weights) or len(set(names)) != len(names):
        raise ValueError("Upstream Shadow regularization has an invalid joint map")
    values = np.asarray(weights, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Upstream Shadow regularization has invalid weights")
    upstream = dict(zip(names, values.tolist()))
    source_names = {
        target: source for source, target in manifest["source_names"].items()
    }
    result = []
    for joint in hand["finger_joints"]:
        if joint not in source_names:
            raise ValueError("Missing Shadow source joint: " + joint)
        # Bimanual bundles qualify source names with the side. Canonical names
        # can have additional prefixes, so resolve through the saved source map.
        source = source_names[joint].rsplit(":", 1)[-1]
        source = source.removeprefix("rh_").removeprefix("lh_")
        if source not in upstream:
            raise ValueError("No upstream Shadow regularization for joint: " + source)
        result.append(upstream[source])
    return result


def finger_velocity_weight(config, manifest):
    return (
        config.get("hand_overrides", {})
        .get(manifest.get("hand_key"), {})
        .get("finger_velocity_weight", config["finger_velocity_weight"])
    )


class VectorWristJoint:
    """One upstream solver per hand, with explicit input and action name mapping."""

    def __init__(self, config, manifest, bundle_root, output_root):
        import yaml
        from scipy.spatial.transform import Rotation

        Retargeter = load_upstream(config["runtime_root"])
        from retargeting.config.core import (
            RobotConfig,
            RetargetingConfig,
            RetargetingProfileConfig,
            SolverConfig,
        )
        from retargeting.core.kinematics.adaptor import RobotAdaptor

        if manifest.get("mimic_joints"):
            raise ValueError(
                "Vector Wrist Joint does not support coupled joints; choose DexPilot"
            )
        self.action_names = manifest["wrist_joints"] + manifest["finger_joints"]
        self.entries = {}
        self.metadata = dict(
            config,
            wrist="jointly_optimized",
            hand_digest=manifest["hand_asset"]["digest"],
            hands={},
        )
        xml = (Path(bundle_root) / "simulation.urdf").read_bytes()
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        method = yaml.safe_load(
            (
                Path(config["runtime_root"])
                / "source/configs/retargeting_methods/vector_wrist_joint.yaml"
            ).read_text()
        )
        upstream_profile = yaml.safe_load(
            (
                Path(config["runtime_root"])
                / "source/configs/retargeting_profiles/vector_wrist_joint_panda_shadow.yaml"
            ).read_text()
        )
        upstream_robot = yaml.safe_load(
            (
                Path(config["runtime_root"]) / "source" / upstream_profile["robot"]
            ).read_text()
        )
        for side, hand in manifest["hands"].items():
            path = output_root / ("retargeting-" + side + ".urdf")
            names, links, checksum = kinematic_hand(xml, hand, path)
            model = link_kinematics(path)
            adaptor = RobotAdaptor(model, names)
            neutral = np.array([manifest["neutral"].get(n, 0.0) for n in names])
            model.compute_forward_kinematics(adaptor.forward_qpos(neutral))
            tree = ET.fromstring(xml)
            parents = {
                j.find("child").get("link"): j.find("parent").get("link")
                for j in tree.findall("joint")
            }
            # Nearest ancestor with a distinct origin defines the terminal finger segment.
            bases = []
            for tip in hand["tips"]:
                base = parents[tip]
                tip_pos = model.get_frame_pose(tip)[:3, 3]
                while (
                    np.linalg.norm(tip_pos - model.get_frame_pose(base)[:3, 3]) < 1e-5
                ):
                    if base == hand["control_frame"] or base not in parents:
                        raise ValueError("No terminal finger direction for " + tip)
                    base = parents[base]
                bases.append(base)
            wrist, tips = hand["control_frame"], hand["tips"]
            # Upstream benchmark axes must point along the terminal segment, even
            # when the physical URDF tip axis points backwards (e.g. WUJI -Z).
            augmented = ET.parse(path).getroot()
            benchmark_tips = []
            for i, (base, tip) in enumerate(zip(bases, tips)):
                tip_pose = model.get_frame_pose(tip)
                direction = tip_pose[:3, :3].T @ (
                    tip_pose[:3, 3] - model.get_frame_pose(base)[:3, 3]
                )
                z = direction / np.linalg.norm(direction)
                axis = np.eye(3)[np.argmin(np.abs(z))]
                x = np.cross(axis, z)
                x /= np.linalg.norm(x)
                rotation = np.column_stack((x, np.cross(z, x), z))
                name = f"skynet_direction_{side}_{i}"
                ET.SubElement(augmented, "link", name=name)
                joint = ET.SubElement(
                    augmented, "joint", name=name + "_fixed", type="fixed"
                )
                ET.SubElement(joint, "parent", link=tip)
                ET.SubElement(joint, "child", link=name)
                ET.SubElement(
                    joint,
                    "origin",
                    xyz="0 0 0",
                    rpy=" ".join(
                        map(str, Rotation.from_matrix(rotation).as_euler("xyz"))
                    ),
                )
                benchmark_tips.append(name)
            raw = ET.tostring(augmented, encoding="utf-8", xml_declaration=True)
            path.write_bytes(raw)
            checksum = hashlib.sha256(raw).hexdigest()
            links.extend(benchmark_tips)
            model = link_kinematics(path)
            adaptor = RobotAdaptor(model, names)
            robot = dict(
                name=manifest["robot"] + "_" + side,
                model={"type": "urdf", "path": str(path)},
                actuated_joints=names,
                initial_qpos=neutral.tolist(),
                visual_frame_names=links,
                wrist_frame_name=wrist,
                human_hand_scale=1.0,
                benchmark={
                    "wrist_link_name": wrist,
                    "fingertips": [
                        {
                            "link_name": tip,
                            "human_tip_index": 4 * (i + 1),
                            "human_direction_base_index": 4 * (i + 1) - 1,
                            "robot_direction_axis": "z",
                            "is_thumb": i == 0,
                        }
                        for i, tip in enumerate(benchmark_tips)
                    ],
                },
            )
            profile = dict(
                name=robot["name"],
                robot=robot["name"],
                method=method["name"],
                target={
                    "wrist_link_name": wrist,
                    "link_pairs": [
                        ["world", tips[0]],
                        *[[wrist, t] for t in tips],
                        *[[tips[0], t] for t in tips[1:]],
                        *[list(p) for p in zip(bases, tips)],
                    ],
                },
                objective=upstream_profile["objective"],
                retargeting={
                    "arm_dof": 6,
                    "human_wrist_index": 0,
                    "joint_position_weights": [0.0] * 6
                    + finger_position_weights(
                        manifest, hand, upstream_robot, upstream_profile
                    ),
                    "joint_velocity_weights": [config["wrist_velocity_weight"]] * 6
                    + [finger_velocity_weight(config, manifest)] * (len(names) - 6),
                },
                joint_limit_overrides=[],
            )
            solver_config = {
                "name": "nlopt_slsqp",
                "params": {
                    "ftol_abs": 1e-5,
                    "maxtime": config["solve_budget_seconds"] / len(manifest["hands"]),
                },
            }
            solver = Retargeter(
                adaptor,
                RobotConfig.from_dict(robot),
                RetargetingProfileConfig.from_dict(profile),
                RetargetingConfig.from_dict(method),
                SolverConfig.from_dict(solver_config),
            )
            self.entries[side] = dict(
                solver=solver,
                model=model,
                adaptor=adaptor,
                wrist=wrist,
                neutral=neutral,
                indices=[self.action_names.index(n) for n in names],
            )
            self.metadata["hands"][side] = dict(
                robot=robot,
                profile=profile,
                method=method,
                solver=solver_config,
                kinematics_sha256=checksum,
                position_regularization_source=(
                    upstream_profile["name"]
                    if manifest.get("hand_key") == "shadow"
                    else None
                ),
            )
        self.reset()
        # Pay torch's first-backward initialization cost before collection is ready.
        warm_points = {}
        for side, entry in self.entries.items():
            model = entry["model"]
            model.compute_forward_kinematics(
                entry["adaptor"].forward_qpos(entry["neutral"])
            )
            inverse = np.linalg.inv(model.get_frame_pose(entry["wrist"]))
            pairs = self.metadata["hands"][side]["profile"]["target"]["link_pairs"][
                -len(manifest["hands"][side]["tips"]) :
            ]
            points = np.zeros((21, 3))
            for i, (base, tip) in enumerate(pairs):
                points[4 * (i + 1) - 1] = (inverse @ model.get_frame_pose(base))[:3, 3]
                points[4 * (i + 1)] = (inverse @ model.get_frame_pose(tip))[:3, 3]
            warm_points[side] = points
        self.update(np.zeros(len(self.action_names)), warm_points)
        self.reset()

    def reset(self):
        self.initialized = False
        self.diagnostics = {}
        for entry in self.entries.values():
            entry["solver"].reset()

    def update(self, native_action, points_wrist):
        import torch
        from retargeting.core.types import RetargetingHandObservation

        action = np.asarray(native_action, dtype=float).copy()
        if action.shape != (len(self.action_names),) or not np.isfinite(action).all():
            raise ValueError("Invalid retargeting input command")
        if set(points_wrist) != set(self.entries):
            raise ValueError("Tracking does not match the configured retargeting hands")
        for side, entry in self.entries.items():
            points = np.asarray(points_wrist[side], dtype=float)
            if points.shape != (21, 3) or not np.isfinite(points).all():
                raise ValueError("Retargeting requires 21 finite tracked hand points")
            reference = entry["neutral"].copy()
            reference[:6] = action[entry["indices"][:6]]
            pose = (
                entry["model"]
                .get_frame_pose(
                    entry["wrist"], entry["adaptor"].forward_qpos(reference)
                )
                .copy()
            )
            if not self.initialized:
                entry["solver"].reset(reference)
            # The simulator's outer inference_mode must not disable the optimizer's derivatives.
            with torch.inference_mode(False), torch.enable_grad():
                result = entry["solver"].solve(
                    RetargetingHandObservation(points.copy(), pose, handedness=side)
                )
            status = entry["solver"].optimizer.opt._opt.last_optimize_result()
            if status < 0:
                raise RuntimeError(
                    f"Vector Wrist Joint optimization failed ({status}); no command applied"
                )
            qpos = np.asarray(result.qpos)
            limits = entry["solver"].optimizer.joint_limits
            if qpos.shape != reference.shape or not np.isfinite(qpos).all():
                raise ValueError("Vector Wrist Joint returned invalid joint commands")
            action[entry["indices"]] = np.clip(qpos, limits[:, 0], limits[:, 1])
            self.diagnostics[side] = dict(result.diagnostics, solver_status=status)
        self.initialized = True
        return action


class CollectionRetargeting:
    """Keep headset acquisition and simulator control independent of the solver choice."""

    def __init__(self, cfg, teleop, continuity, output_root):
        self.continuity = continuity
        self.config = cfg.get("retargeting", {"key": "dexpilot", "name": "DexPilot"})
        self.converters = {}
        self.vector = None
        if self.config["key"] == "vector-wrist-joint":
            self.vector = VectorWristJoint(
                self.config,
                cfg["hand_manifest"],
                cfg["hand_bundle"]["root"],
                output_root,
            )
            for retargeter in teleop._retargeters:
                for target in retargeter._tracked_hands:
                    side = target.name.removeprefix("HAND_").lower()
                    self.converters[side] = (retargeter, target)
                # DexVerse still supplies wrist coordinates and tracking visualization.
                # Its finger optimizer is not run when the selected solver owns all joints.
                retargeter._retarget_hand_fingers = lambda *_: np.zeros(
                    0, dtype=np.float32
                )
            if set(self.converters) != set(self.vector.entries):
                raise ValueError(
                    "Headset sides do not match the selected retargeting model"
                )
        elif self.config["key"] != "dexpilot":
            raise ValueError("Unknown collection retargeting method")
        self.metadata = (
            self.vector.metadata
            if self.vector
            else dict(
                self.config,
                provider="dexverse",
                mode="dexpilot",
                wrist="absolute",
                wrist_continuity="equivalent_euler_angles",
            )
        )

    def reset(self):
        self.continuity.reset()
        if self.vector:
            self.vector.reset()

    def update(self, action, raw):
        if self.vector:
            points = {
                side: retargeter._convert_hand_to_canonical_joint_positions(
                    raw.get(target, {}), target
                )
                for side, (retargeter, target) in self.converters.items()
            }
            action = self.vector.update(action, points)
        return self.continuity.update(action)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--check", required=True)
    args = parser.parse_args()
    load_upstream(args.check)
    print(json.dumps({"ready": True, "revision": REVISION}))
