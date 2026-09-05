"""ARKit recording -> the 21 world-space joints consumed by DexVerse.

Only positions and the wrist orientation are consumed by its relative retargeter.
Wrist axes follow XR_EXT_hand_tracking: -Z toward the fingers, +Y dorsal.
The anatomical basis is calibrated once in hand-anchor coordinates; subsequent
wrist rotations come from ARKit, not finger articulation. No mirroring, guessed
missing joints, time stretching, or tracking-gap interpolation is performed.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from pathlib import Path

VERSION = "visionpro-dexverse/1"
JOINTS = {
    "wrist": "wrist",
    "thumb_metacarpal": "thumbKnuckle",
    "thumb_proximal": "thumbIntermediateBase",
    "thumb_distal": "thumbIntermediateTip",
    "thumb_tip": "thumbTip",
}
for _finger, _apple in [
    ("index", "indexFinger"),
    ("middle", "middleFinger"),
    ("ring", "ringFinger"),
    ("little", "littleFinger"),
]:
    for _xr, _ar in [
        ("proximal", "Knuckle"),
        ("intermediate", "IntermediateBase"),
        ("distal", "IntermediateTip"),
        ("tip", "Tip"),
    ]:
        JOINTS[f"{_finger}_{_xr}"] = _apple + _ar
WORLD_FROM_ARKIT = [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def sub(a, b):
    return [x - y for x, y in zip(a, b, strict=True)]


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def unit(a):
    length = math.sqrt(dot(a, a))
    if length < 1e-6:
        raise ValueError(
            "Cannot calibrate a wrist from coincident or collinear knuckles"
        )
    return [x / length for x in a]


def matrix(raw):
    if (
        not isinstance(raw, list)
        or len(raw) != 16
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in raw)
    ):
        raise ValueError("Pose must contain 16 finite column-major numbers")
    m = [[raw[c * 4 + r] for c in range(4)] for r in range(4)]
    if any(abs(a - b) > 1e-4 for a, b in zip(m[3], [0, 0, 0, 1])):
        raise ValueError("Pose is not a homogeneous transform")
    r = [row[:3] for row in m[:3]]
    if (
        any(
            abs(dot(r[i], r[j]) - int(i == j)) > 0.005
            for i in range(3)
            for j in range(3)
        )
        or abs(dot(r[0], cross(r[1], r[2])) - 1) > 0.005
    ):
        raise ValueError("Pose rotation is not a rigid right-handed transform")
    return m


def mv(m, v):
    return [dot(row, v) for row in m]


def mm(a, b):
    return [[dot(row, col) for col in zip(*b)] for row in a]


def position(m):
    return [m[i][3] for i in range(3)]


def quaternion(m):
    # Stable largest-diagonal conversion; output is w,x,y,z (Isaac Lab convention).
    candidates = [
        1 + m[0][0] + m[1][1] + m[2][2],
        1 + m[0][0] - m[1][1] - m[2][2],
        1 - m[0][0] + m[1][1] - m[2][2],
        1 - m[0][0] - m[1][1] + m[2][2],
    ]
    i = max(range(4), key=candidates.__getitem__)
    s = 2 * math.sqrt(max(0, candidates[i]))
    if i == 0:
        q = [
            s / 4,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        ]
    elif i == 1:
        q = [
            (m[2][1] - m[1][2]) / s,
            s / 4,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        ]
    elif i == 2:
        q = [
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            s / 4,
            (m[1][2] + m[2][1]) / s,
        ]
    else:
        q = [
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            s / 4,
        ]
    length = math.sqrt(dot(q, q))
    return [x / length for x in q]


def calibrated_basis(frame):
    joints = checked_joints(frame)
    p = {name: position(value) for name, value in joints.items()}
    z = unit(sub(p["wrist"], p["middle_proximal"]))
    across = sub(p["little_proximal"], p["index_proximal"])
    if frame["hand"] == "left":
        across = [-v for v in across]
    x = unit(sub(across, [dot(across, z) * v for v in z]))
    y = cross(z, x)
    return [list(row) for row in zip(x, y, z)]


def checked_joints(frame):
    if frame.get("tracked") is not True:
        raise ValueError(f"Hand tracking lost at source frame {frame['index']}")
    by_name = {j["name"]: j for j in frame["joints"]}
    result = {}
    for xr, ar in JOINTS.items():
        j = by_name.get(ar)
        if j is None or j.get("tracked") is not True:
            raise ValueError(
                f"Required joint {ar} is missing/untracked at source frame {frame['index']}"
            )
        result[xr] = matrix(j["anchor_from_joint"])
    return result


def convert(path: Path, *, hand="right", fps=60, max_gap_seconds=0.1):
    from skynet_app.local_capture import VisionProTracking

    summary = VisionProTracking().inspect(path)
    header = summary["header"]
    for key, expected in {
        "timestamp_clock": "CACurrentMediaTime",
        "joint_frame": "hand-anchor",
        "coordinate_frame": "ARKit session origin; right-handed; Y up",
    }.items():
        if header.get(key) != expected:
            raise ValueError(
                f"Unsupported {key}; re-record with the current Skynet Capture app"
            )
    if hand not in ("left", "right") or type(fps) is not int or fps not in (30, 60):
        raise ValueError("Choose left/right hand and 30 or 60 Hz")
    if not math.isfinite(max_gap_seconds) or not 0 < max_gap_seconds <= 0.1:
        raise ValueError("Tracking-gap limit must be positive and at most 100 ms")
    frames = []
    with path.open() as source:
        for line in source:
            f = json.loads(line)
            if f.get("type") == "frame" and f["hand"] == hand:
                checked_joints(f)
                matrix(f["origin_from_hand"])
                frames.append(f)
    if len(frames) < 2:
        raise ValueError(f"At least two valid {hand} hand frames are required")
    times = [f["timestamp"] for f in frames]
    gaps = [b - a for a, b in zip(times, times[1:])]
    if max(gaps) > max_gap_seconds:
        raise ValueError(
            f"{hand.title()} hand tracking has a {max(gaps) * 1000:.0f} ms gap; limit is {max_gap_seconds * 1000:.0f} ms. Record again with the hand visible."
        )
    basis = calibrated_basis(frames[0])
    samples = []
    for i in range(math.floor((times[-1] - times[0]) * fps + 1e-7) + 1):
        t = times[0] + i / fps
        f = frames[bisect.bisect_right(times, t + 1e-9) - 1]
        a = matrix(f["origin_from_hand"])
        r = mm(WORLD_FROM_ARKIT, mm([row[:3] for row in a[:3]], basis))
        wristq = quaternion(r)
        joints = {}
        for name, j in checked_joints(f).items():
            worldp = mv(WORLD_FROM_ARKIT, position(mm(a, j)))
            # Finger orientations are intentionally absent: DexPilot consumes positions.
            joints[name] = worldp + wristq
        samples.append(
            {
                "time_seconds": i / fps,
                "source_index": f["index"],
                "receive_timestamp": f["timestamp"],
                "source_timestamp": f["source_timestamp"],
                "joints": joints,
            }
        )
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "schema": "skynet.dexverse-tracking/v1",
        "converter": VERSION,
        "source_sha256": digest,
        "source_session_id": header["session_id"],
        "source_task": header["task"],
        "hand": hand,
        "fps": fps,
        "sampling": "causal hold on receive clock; no interpolation",
        "max_gap_seconds": max(gaps),
        "world_from_arkit_rotation": WORLD_FROM_ARKIT,
        "anchor_from_openxr_wrist_rotation": basis,
        "calibration_source_index": frames[0]["index"],
        "finger_orientations": "unused; wrist orientation repeated for retargeter pose-vector interface",
        "samples": samples,
    }
