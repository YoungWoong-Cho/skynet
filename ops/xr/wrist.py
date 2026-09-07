"""Continuous angles for the simulator's virtual XYZ wrist, in radians."""

import numpy as np


class ContinuousEulerXYZ:
    """Choose an equivalent XYZ solution nearest the previous command.

    Rotation matrices have no turn count. Preserve it across frames instead of
    driving a position-controlled joint from +pi directly back to -pi.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = np.zeros(3)

    def update(self, matrix):
        r = np.asarray(matrix, dtype=float)
        if (
            r.shape != (3, 3)
            or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(r), 1.0, atol=1e-5)
        ):
            raise ValueError("Wrist tracking produced an invalid rotation")
        previous = self.previous
        pitch = np.arctan2(r[0, 2], np.hypot(r[0, 0], r[0, 1]))
        if np.hypot(r[0, 0], r[0, 1]) < 1e-7:
            # At gimbal lock only roll +/- yaw is observable. Keep the
            # unobservable component continuous instead of forcing yaw to zero.
            sign = np.sign(pitch)
            coupled = np.arctan2(sign * r[1, 0], r[1, 1])
            delta = (coupled - previous[0] - sign * previous[2] + np.pi) % (
                2 * np.pi
            ) - np.pi
            angles = np.array(
                [previous[0] + delta / 2, pitch, previous[2] + sign * delta / 2]
            )
            angles[1] += 2 * np.pi * np.round((previous[1] - pitch) / (2 * np.pi))
        else:
            roll = np.arctan2(-r[1, 2], r[2, 2])
            yaw = np.arctan2(-r[0, 1], r[0, 0])
            candidates = np.array(
                [[roll, pitch, yaw], [roll + np.pi, np.pi - pitch, yaw + np.pi]]
            )
            candidates += 2 * np.pi * np.round((previous - candidates) / (2 * np.pi))
            angles = candidates[
                np.argmin(np.linalg.norm(candidates - previous, axis=1))
            ]
        self.previous = angles.copy()
        return angles.copy()


def install_relative_wrist_tracking(retargeter):
    """Keep separate turn histories for each hand, preserving the pinned layout."""
    histories = {}

    def relative(wrist_quat, base_quat, layout):
        wrist = retargeter._get_normalized_wrist_rotation(wrist_quat)
        base = retargeter._get_normalized_wrist_rotation(base_quat)
        rotation = retargeter._express_relative_wrist_rotation_in_action_frame(
            wrist * base.inv(), layout
        )
        key = tuple(layout["wrist_rot_indices"])
        if key not in histories:
            histories[key] = ContinuousEulerXYZ()
        history = histories[key]
        return retargeter._reorder_wrist_euler(
            history.update(rotation.as_matrix()), layout["wrist_rot_order"]
        )

    retargeter._compute_relative_wrist_euler = relative
    return histories.clear


def configure_virtual_wrist(robot, manifest, sides):
    """Remove artificial +/-pi stops only from the floating wrist joints."""
    import torch

    if manifest:
        names = manifest["wrist_joints"][3:]
        if names != ["skynet_roll", "skynet_pitch", "skynet_yaw"]:
            raise ValueError("Unsupported imported wrist rotation layout")
    else:
        prefixes = ["lh_", "rh_"] if len(sides) == 2 else [""]
        names = [
            prefix + axis + "_rotation_joint" for prefix in prefixes for axis in "xyz"
        ]
    missing = set(names) - set(robot.joint_names)
    if missing:
        raise ValueError(
            "Virtual wrist joints are missing: " + ", ".join(sorted(missing))
        )
    indices = [robot.joint_names.index(name) for name in names]
    limits = robot.data.joint_pos_limits[:, indices].clone()
    # Finite near-unbounded values avoid inf-inf in Isaac's soft-limit buffers.
    # These are virtual world-pose axes; physical finger/wrist joints are untouched.
    limits[..., 0], limits[..., 1] = -1_000_000.0, 1_000_000.0
    robot.write_joint_position_limit_to_sim(limits, joint_ids=indices)
    actual = robot.root_physx_view.get_dof_limits()[:, indices].to(limits.device)
    if (
        not torch.allclose(actual, limits)
        or not torch.isfinite(robot.data.soft_joint_pos_limits[:, indices]).all()
    ):
        raise ValueError("Simulator could not configure continuous wrist limits")
