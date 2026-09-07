"""CPU kinematics check for a prepared segment retargeter; no recordings are created."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from dex_retargeting.retargeting_config import RetargetingConfig

parser = argparse.ArgumentParser()
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--alignment", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.bundle.resolve()
sys.path.insert(0, str(root))
from anatomy import canonical_points, segment_targets  # noqa: E402

manifest = json.loads((root / "manifest.json").read_text())
assert manifest["retargeting_mode"] == "finger_segments"
cfg = json.loads((root / "retarget.json").read_text())["retargeting"]
cfg["urdf_path"] = str(root / "retarget.urdf")
solver = RetargetingConfig.from_dict(cfg).build()
robot = solver.optimizer.robot
origins = [robot.get_link_index(n) for n in cfg["target_origin_link_names"]]
tasks = [robot.get_link_index(n) for n in cfg["target_task_link_names"]]
indices = np.asarray(cfg["target_link_human_indices"])


def bone_vectors():
    return np.array(
        [
            robot.get_link_pose(b)[:3, 3] - robot.get_link_pose(a)[:3, 3]
            for a, b in zip(origins, tasks)
        ]
    )


neutral = np.array([manifest["neutral"].get(n, 0.0) for n in robot.dof_joint_names])
robot.compute_forward_kinematics(neutral)
lengths = np.linalg.norm(bone_vectors(), axis=1)
p = np.asarray(json.loads(args.alignment.read_text()))
if manifest["side"] == "left":
    p[:, 1] *= -1  # Explicitly synthetic mirror, not a left-headset capture.
p = canonical_points(p, manifest["side"])
results = []


def check(name, points, threshold):
    reference = segment_targets(points, indices, lengths)
    timings = []
    for _ in range(80):
        start = time.perf_counter()
        q = solver.retarget(reference)
        timings.append((time.perf_counter() - start) * 1000)
    robot.compute_forward_kinematics(q)
    actual = bone_vectors()
    cosine = np.sum(actual * reference, axis=1) / (
        np.linalg.norm(actual, axis=1) * lengths
    )
    errors = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    assert np.isfinite(q).all() and errors.max() < threshold, (name, errors)
    results.append(
        dict(
            pose=name,
            max_direction_error_degrees=float(errors.max()),
            segment_errors_degrees=errors.tolist(),
            median_solve_ms=float(np.median(timings[-20:])),
        )
    )


check("recorded_open" if manifest["side"] == "right" else "mirrored_open", p, 8)
for name in ("neutral", "curl", "spread"):
    q = neutral.copy()
    for i, joint in enumerate(robot.dof_joint_names):
        value = 0.0
        if name == "curl":
            value = (
                0.45
                if any(s in joint for s in ("_flex", "_pip", "_dip", "_mcp", "_ip"))
                and "abd" not in joint
                else 0.0
            )
        elif name == "spread" and "abd" in joint:
            value = 0.2 if "index" in joint or "thumb" in joint else -0.15
        low, high = manifest["finger_limits"][joint]
        q[i] = np.clip(value, low, high)
    robot.compute_forward_kinematics(q)
    directions = bone_vectors() / lengths[:, None]
    points = p.copy()
    for i, (start, end) in enumerate(indices.T):
        human_length = lengths[i] * (0.7, 1.3, 1.1)[i % 3]
        points[end] = points[start] + directions[i] * human_length
    check("synthetic_" + name, points, 5)

result = dict(
    robot=manifest["robot"],
    bundle_digest=manifest["digest"],
    method="finger_segments",
    headset_tested=False,
    poses=results,
)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
