"""NumPy-only rigid transforms. URDF rpy and FAAS row-6D are distinct conventions."""

import numpy as np


def rotation_xyz(angles):
    """Intrinsic XYZ: Rx(roll) @ Ry(pitch) @ Rz(yaw), with arbitrary batch axes."""
    a = np.asarray(angles, dtype=np.float64)
    if a.shape[-1:] != (3,) or not np.isfinite(a).all():
        raise ValueError("Expected finite XYZ angles")
    x, y, z = np.moveaxis(a, -1, 0)
    cx, cy, cz, sx, sy, sz = (
        np.cos(x),
        np.cos(y),
        np.cos(z),
        np.sin(x),
        np.sin(y),
        np.sin(z),
    )
    return np.stack(
        [
            cy * cz,
            -cy * sz,
            sy,
            cx * sz + sx * sy * cz,
            cx * cz - sx * sy * sz,
            -sx * cy,
            sx * sz - cx * sy * cz,
            sx * cz + cx * sy * sz,
            cx * cy,
        ],
        axis=-1,
    ).reshape(a.shape[:-1] + (3, 3))


def urdf_origin(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    """URDF fixed-axis rpy = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    t = np.eye(4)
    x, y, z = rpy
    t[:3, :3] = (
        rotation_xyz([0, 0, z]) @ rotation_xyz([0, y, 0]) @ rotation_xyz([x, 0, 0])
    )
    t[:3, 3] = xyz
    return t


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def inverse_transform(t):
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    out[..., :3, :3] = np.swapaxes(t[..., :3, :3], -2, -1)
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", out[..., :3, :3], t[..., :3, 3])
    out[..., 3, 3] = 1
    return out


def pose9(t):
    t = np.asarray(t, dtype=float)
    return np.concatenate(
        [t[..., :3, 3], t[..., :2, :3].reshape(t.shape[:-2] + (6,))], axis=-1
    )


def transform9(pose):
    """Reconstruct a right-handed rotation from its first two *rows*."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape[-1:] != (9,) or not np.isfinite(pose).all():
        raise ValueError("Expected finite position3 + row-rotation6D")
    a, b = pose[..., 3:6], pose[..., 6:9]
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(n < 1e-8):
        raise ValueError("Degenerate rotation6D first row")
    a = a / n
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    n = np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(n < 1e-8):
        raise ValueError("Degenerate rotation6D collinear rows")
    b = b / n
    out = np.zeros(pose.shape[:-1] + (4, 4), dtype=float)
    out[..., 0, :3], out[..., 1, :3], out[..., 2, :3] = a, b, np.cross(a, b)
    out[..., :3, 3], out[..., 3, 3] = pose[..., :3], 1
    return out


def nearest_xyz(rotation, previous):
    """Inverse intrinsic XYZ with a continuous branch nearest the supplied angle.

    A rotation cannot encode arbitrary turn counts. The anchor and preceding
    decoded command select an equivalent branch; jumps greater than pi require
    a separate turn-count representation and are outside this codec contract.
    """
    r, p = np.asarray(rotation, dtype=float), np.asarray(previous, dtype=float)
    pitch = np.arctan2(r[0, 2], np.hypot(r[0, 0], r[0, 1]))
    if np.hypot(r[0, 0], r[0, 1]) < 1e-8:
        sign = np.sign(pitch)
        coupled = np.arctan2(sign * r[1, 0], r[1, 1])
        delta = (coupled - p[0] - sign * p[2] + np.pi) % (2 * np.pi) - np.pi
        out = np.array([p[0] + delta / 2, pitch, p[2] + sign * delta / 2])
        out[1] += 2 * np.pi * np.round((p[1] - pitch) / (2 * np.pi))
        return out
    roll, yaw = np.arctan2(-r[1, 2], r[2, 2]), np.arctan2(-r[0, 1], r[0, 0])
    choices = np.array([[roll, pitch, yaw], [roll + np.pi, np.pi - pitch, yaw + np.pi]])
    choices += 2 * np.pi * np.round((p - choices) / (2 * np.pi))
    return choices[np.argmin(np.linalg.norm(choices - p, axis=1))]
