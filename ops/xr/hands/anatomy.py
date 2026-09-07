"""Hand frames derived from knuckles, independent of headset wrist-axis conventions."""

import numpy as np


def palm_frame(points, side):
    """Return world-space axes for a 21-point hand (wrist, then four joints/finger).

    +X follows the middle knuckle; +Y points toward the right-hand index side;
    +Z points out of the back of a palm-down hand. Left hands remain mirrored.
    Knuckles define the frame, so opening/closing fingers does not rotate it.
    """
    p = np.asarray(points, dtype=np.float64)
    if side not in {"left", "right"} or p.shape != (21, 3) or not np.isfinite(p).all():
        raise ValueError("Hand tracking is incomplete")
    forward = p[9] - p[0]
    across = (p[5] - p[17]) * (1 if side == "right" else -1)
    if np.linalg.norm(forward) < 0.025 or np.linalg.norm(across) < 0.025:
        raise ValueError("Hand tracking has no valid palm geometry")
    x = forward / np.linalg.norm(forward)
    y = across - x * np.dot(across, x)
    if np.linalg.norm(y) < 0.02:
        raise ValueError("Hand tracking has collapsed knuckle positions")
    y /= np.linalg.norm(y)
    return np.column_stack((x, y, np.cross(x, y)))


def canonical_points(points, side):
    p = np.asarray(points, dtype=np.float64)
    return ((p - p[0]) @ palm_frame(p, side)).astype(np.float32)
