"""Official UniDex FAAS82 slot/wrist convention for exact registered assets.

Dataset absolute wrist poses are transformed to the camera OpenGL frame by
encode_absolute. Chunk-relative targets share one measured observation anchor.
Mappings are static geometry contracts, never inferred from target demos.
"""

import numpy as np

from .geometry import inverse_transform, nearest_xyz, pose9, rotation_xyz, transform9
from .hand_contract import DEFAULT_SPEC_ROOT, HandSpec, load_spec

FAAS_DIM = 82
IDENTITY_POSE = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=float)


class FAASCodec:
    def __init__(self, spec: HandSpec):
        if spec.mapping_status != "VERIFIED":
            raise ValueError("FAAS mapping is blocked: " + spec.data["blocked_reason"])
        self.spec = spec
        self.robot_id = spec.robot
        self.digest = spec.spec_hash
        self.action_names = spec.control_names
        self.action_scale = np.asarray(spec.data["action_scale"], dtype=float)
        self.action_offset = np.asarray(spec.data["action_offset"], dtype=float)
        self.step_dt = float(spec.capture["step_dt"])
        self.wrist_indices = np.array(
            [self.action_names.index(n) for n in spec.data["wrist_names"]]
        )
        self.mapping = tuple(spec.data["mapping"])
        self.independent = tuple(
            m for m in self.mapping if m["joint"] in self.action_names
        )
        self.present_mask = np.zeros(82, dtype=bool)
        self.present_mask[:9] = True
        self.control_mask = np.zeros(82, dtype=bool)
        self.control_mask[:9] = True
        for m in self.mapping:
            self.present_mask[18 + m["slot"]] = True
        for m in self.independent:
            self.control_mask[18 + m["slot"]] = True

    @property
    def action_dim(self):
        return self.spec.action_dim

    @property
    def state_dim(self):
        return self.spec.state_dim

    def _native(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape[-1:] != (self.action_dim,) or not np.isfinite(q).all():
            raise ValueError(
                "Native values must be finite and match exact control order"
            )
        return q

    def targets(self, actions):
        return self._native(actions) * self.action_scale + self.action_offset

    def wrist_pose(self, q):
        """Fixed robot base to canonical palm/control frame (not world).

        The runtime root pose is outside q. A common fixed base transform
        cancels in chunk-relative targets and is not added to native commands.
        """
        q = self._native(q)[..., self.wrist_indices]
        t = np.zeros(q.shape[:-1] + (4, 4), dtype=float)
        t[..., :3, :3] = rotation_xyz(q[..., 3:])
        t[..., :3, 3] = q[..., :3]
        t[..., 3, 3] = 1
        return t @ self.spec.frame_offset

    def _encode(self, q, wrist):
        values = np.zeros(q.shape[:-1] + (82,), dtype=float)
        values[..., 9:18] = IDENTITY_POSE
        values[..., :9] = pose9(wrist)
        physical = self.spec.expand(q)
        for m in self.mapping:
            values[..., 18 + m["slot"]] = (
                physical[m["joint"]] * m["scale"] + m["offset"]
            )
        return values

    def encode_state(self, q_state):
        q = self._native(q_state)
        return self._encode(q, self.wrist_pose(q)), self.present_mask.copy()

    def encode_actions(self, action_chunk, anchor_q):
        """Encode (..., T, A) commands against one (..., A) measured anchor.

        No future measured state is used. Mimic physical slots are populated
        for a self-consistent representation. The returned control mask is
        diagnostic; native UniDex still uses its unchanged all-82 loss.
        """
        targets = self.targets(action_chunk)
        if targets.ndim < 2 or targets.shape[-2] < 1:
            raise ValueError("Actions need a nonempty chunk axis")
        anchor = self._native(anchor_q)
        if anchor.shape != targets.shape[:-2] + (self.action_dim,):
            raise ValueError("One anchor per action chunk is required")
        relative = inverse_transform(self.wrist_pose(anchor))[
            ..., None, :, :
        ] @ self.wrist_pose(targets)
        return self._encode(targets, relative), self.control_mask.copy()

    encode_targets = encode_actions

    def decode_actions(self, faas_chunk, anchor_q, *, clamp=False):
        """Decode (..., T, 82) into native controller commands.

        Inactive/mimic-only output slots never drive actuators. Coupled joint
        predictions are projected onto the independent actuator manifold by
        selecting the actuator's own slot and regenerating its URDF mimics.
        Set clamp=True at the execution boundary to apply exact native bounds;
        encoding never clips or rewrites training labels.
        """
        encoded = np.asarray(faas_chunk, dtype=float)
        if (
            encoded.ndim < 2
            or encoded.shape[-1] != 82
            or encoded.shape[-2] < 1
            or not np.isfinite(encoded).all()
        ):
            raise ValueError("Expected finite nonempty FAAS82 action chunks")
        anchor = self._native(anchor_q)
        if anchor.shape != encoded.shape[:-2] + (self.action_dim,):
            raise ValueError("One anchor per action chunk is required")
        world = self.wrist_pose(anchor)[..., None, :, :] @ transform9(encoded[..., :9])
        control = world @ inverse_transform(self.spec.frame_offset)
        target = np.empty(encoded.shape[:-1] + (self.action_dim,), dtype=float)
        target[..., self.wrist_indices[:3]] = control[..., :3, 3]
        # Select continuous Euler representatives from the one observation
        # anchor, then the preceding decoded command. Physical chunk anchors
        # remain identical even while this angle branch is tracked.
        flat_rot = control[..., :3, :3].reshape((-1, encoded.shape[-2], 3, 3))
        flat_anchor = anchor[..., self.wrist_indices[3:]].reshape((-1, 3))
        angles = np.empty((len(flat_rot), encoded.shape[-2], 3))
        for b, rotations in enumerate(flat_rot):
            previous = flat_anchor[b]
            for k, r in enumerate(rotations):
                previous = nearest_xyz(r, previous)
                angles[b, k] = previous
        target[..., self.wrist_indices[3:]] = angles.reshape(encoded.shape[:-1] + (3,))
        for m in self.independent:
            target[..., self.action_names.index(m["joint"])] = (
                encoded[..., 18 + m["slot"]] - m["offset"]
            ) / m["scale"]
        if clamp:
            target = np.clip(
                target, self.spec.effective_lower, self.spec.effective_upper
            )
        return (target - self.action_offset) / self.action_scale

    decode_targets = decode_actions

    def coverage_from(self, source_codecs):
        """Report genuinely supervised command slots; never invent fallback values."""
        covered = np.zeros(82, dtype=bool)
        for codec in source_codecs:
            covered |= codec.control_mask
        missing = self.control_mask & ~covered
        return {
            "complete": not bool(missing.any()),
            "required_slots": np.flatnonzero(self.control_mask).tolist(),
            "covered_slots": np.flatnonzero(self.control_mask & covered).tolist(),
            "missing_slots": np.flatnonzero(missing).tolist(),
            "missing_finger_slots": np.flatnonzero(missing[18:50]).tolist(),
            "missing_joint_names": [
                m["joint"] for m in self.independent if missing[18 + m["slot"]]
            ],
        }


def load_codec(robot_id, capture_metadata=None, *, spec_root=None):
    spec = load_spec(robot_id, spec_root=spec_root)
    if capture_metadata is not None:
        spec.validate_capture(capture_metadata)
    return FAASCodec(spec)


def supported_robots():
    """Use the same frozen asset-bound declarations as the adapter catalog."""
    result = []
    for path in sorted(DEFAULT_SPEC_ROOT.glob("*.json")):
        spec = load_spec(path.stem)
        if spec.mapping_status == "VERIFIED" and spec.data.get("native_wrist"):
            result.append(spec.robot)
    return result


def preflight(robot_id):
    """Cheap exact-hand check before any GPU observation preparation."""
    codec = load_codec(robot_id)
    return {"robot": robot_id, "schema": codec.spec.data["schema"],
            "codec_sha256": codec.digest, "action_dim": codec.action_dim,
            "upstream_hand": codec.spec.data["native_wrist"]["upstream_hand"]}


def encode_absolute(codec, q, *, camera_from_root):
    """Encode measured joints OR physical command targets in camera coordinates.

    The caller must apply codec.targets() to raw controller actions first.
    camera_from_root is the recorded camera inverse times the recorded robot
    root pose, preceded by the optical-to-OpenGL basis conversion.
    """
    q = codec._native(q)
    transform = np.asarray(camera_from_root, dtype=float)
    expected = q.shape[:-1] + (4, 4)
    if transform.shape != expected or not np.isfinite(transform).all():
        raise ValueError("One finite camera-from-root transform per frame is required")
    r = transform[..., :3, :3]
    if (not np.allclose(transform[..., 3, :], [0, 0, 0, 1])
            or not np.allclose(r @ r.swapaxes(-1, -2), np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(r), 1, atol=1e-5)):
        raise ValueError("Camera-from-root is not a rigid transform")
    return codec._encode(q, transform @ codec.wrist_pose(q))


def anchor_faas_actions(state_absolute, action_absolute):
    """Official UniDex wrist deltas: inv(current measured wrist) @ target.

    All targets in a chunk use the SAME current observation for both wrists.
    Finger targets remain absolute. Inputs preserve inactive identity wrists.
    """
    state = np.asarray(state_absolute, dtype=float)
    actions = np.asarray(action_absolute, dtype=float)
    if (state.shape[-1:] != (82,) or actions.ndim < 2
            or actions.shape[-1] != 82 or not actions.shape[-2]
            or state.shape[:-1] != actions.shape[:-2]
            or not np.isfinite(state).all() or not np.isfinite(actions).all()):
        raise ValueError("Expected one finite FAAS82 state per nonempty action chunk")
    result = actions.copy()
    for start in (0, 9):
        anchor = transform9(state[..., start:start+9])
        target = transform9(actions[..., start:start+9])
        result[..., start:start+9] = pose9(inverse_transform(anchor)[..., None, :, :] @ target)
    return result.astype(np.float32)


def _recording_values(episode, capture):
    """Validate exact controller semantics and extract single-environment poses."""
    codec = load_codec(capture["robot"], capture)
    actions = np.asarray(episode["actions"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != codec.action_dim or len(episode["states"]) != len(actions) + 1:
        raise ValueError("UniDex state/actions must have exact pre-action frame alignment")
    if len(actions) < 1:
        raise ValueError("UniDex requires a nonempty recording")
    source_names = capture["robot_joint_names"]
    if len(set(source_names)) != len(source_names):
        raise ValueError("Duplicate native joint names")
    indices = [source_names.index(name) for name in codec.action_names]
    states = []
    roots = []
    for state in episode["states"][:-1]:
        robot = state["articulation"]["robot"]
        q = np.asarray(robot["joint_position"], dtype=float)
        if "root_pose" not in robot:
            raise ValueError("UniDex requires recorded robot root_pose; identity-root fallback is not supported")
        root = np.asarray(robot["root_pose"], dtype=float)
        if q.shape != (1, len(source_names)) or root.shape != (1, 7):
            raise ValueError("UniDex requires single-environment joints and recorded root poses")
        position, quaternion = root[0, :3], root[0, 3:]
        if not np.isfinite(root).all() or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-4):
            raise ValueError("Invalid recorded robot root quaternion")
        w, x, y, z = quaternion / np.linalg.norm(quaternion)
        transform = np.eye(4)
        transform[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
        transform[:3, 3] = position
        roots.append(transform)
        states.append(q[0, indices])
    states, roots = np.stack(states), np.stack(roots)
    if not np.isfinite(states).all():
        raise ValueError("Nonfinite measured joint state")
    # URDF limits constrain physical joint motion, not recorded position
    # commands. The pinned JointPositionAction sends action*scale+offset
    # directly (no clip configured). Preserve even out-of-range targets; the
    # decoder's optional execution clamp is a separate boundary.
    codec.targets(actions)
    return codec, actions, states, roots


def validate_source(episode, capture):
    """CPU-only preflight before rendering; never invent a missing root or map."""
    codec, actions, _, _ = _recording_values(episode, capture)
    return {"robot": codec.robot_id, "codec_sha256": codec.digest,
            "frames": len(actions), "source_fps": 1.0 / codec.step_dt,
            "action_representation": "skynet.unidex-faas/v1"}


def encode_recording(episode, capture, world_from_camera):
    """Return native absolute FAAS arrays and their exact geometry identity.

    Camera calibration is [T,4,4] world-from-camera in ROS optical convention;
    saved root_pose is [x,y,z,qw,qx,qy,qz]. Both are required, never inferred.
    """
    codec, actions, states, roots = _recording_values(episode, capture)
    camera = np.asarray(world_from_camera, dtype=float)
    if camera.shape != (len(actions), 4, 4):
        raise ValueError("UniDex requires one verified camera calibration per recording frame")
    optical_to_opengl = np.diag([1., -1., -1., 1.])
    camera_from_root = optical_to_opengl @ inverse_transform(camera) @ roots
    arrays = {
        "faas_state_absolute": encode_absolute(codec, states, camera_from_root=camera_from_root).astype(np.float32),
        "faas_action_absolute": encode_absolute(codec, codec.targets(actions), camera_from_root=camera_from_root).astype(np.float32),
    }
    return arrays, {"id": "skynet.unidex-faas/v1", "codec_sha256": codec.digest,
                    "frame": "camera_opengl", "action_semantics": "controller_targets",
                    "robot": codec.robot_id, "state_alignment": "pre_action"}
