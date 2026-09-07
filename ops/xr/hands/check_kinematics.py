"""CPU check of imported hand kinematics and reachable fingertip retargeting targets.

Usage: python check_kinematics.py /path/to/hand/bundle [...]
Requires dex-retargeting, torch, numpy and pinocchio; does not run training or evaluation.
"""

import json
import sys
import time
import numpy as np
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation
from runtime import read_bundle, bounded_fingers


def check(path):
    root, manifest = read_bundle(path)
    floating = RobotWrapper(str(root / "simulation.urdf"))
    wrist_indices = [floating.get_joint_index(n) for n in manifest["wrist_joints"]]
    palm_index = floating.get_link_index("skynet_palm")
    # Pinned SimpleRelativeRetargeter emits intrinsic XYZ Euler angles. Validate
    # combined rotations: single-axis checks cannot detect an incorrect chain order.
    wrist_poses = ([0, 0, 0], [0.3, -0.4, 0.7], [-0.8, 0.4, -0.2], [0.1, 1.0, -0.6])
    for angles in wrist_poses:
        q = floating.q0.copy()
        q[wrist_indices] = [0.5, -0.2, 0.3, *angles]
        floating.compute_forward_kinematics(q)
        pose = floating.get_link_pose(palm_index)
        assert np.allclose(pose[:3, 3], [0.5, -0.2, 0.3], atol=1e-8)
        assert np.allclose(
            pose[:3, :3], Rotation.from_euler("XYZ", angles).as_matrix(), atol=1e-8
        )
    # Independent frame check against the working native Shadow/OpenXR convention:
    # open fingers point +X, thumbs are on +Y (right) / -Y (left), palms face -Z.
    # An optimizer round trip alone cannot catch a shared 90-degree frame error.
    q = floating.q0.copy()
    for name, value in manifest["neutral"].items():
        q[floating.get_joint_index(name)] = value
    floating.compute_forward_kinematics(q)
    open_tips = np.array(
        [
            floating.get_link_pose(floating.get_link_index(n))[:3, 3]
            for n in manifest["tips"]
        ]
    )
    assert np.all(open_tips[1:, 0] > 0.07), "Open fingers must point forward (+X)"
    assert np.all(abs(open_tips[1:, 2]) < 0.04), "Open palm must be horizontal"
    sign = 1 if manifest["side"] == "right" else -1
    assert sign * open_tips[0, 1] > 0.05, "Thumb is on the wrong side"
    assert sign * (open_tips[1, 1] - open_tips[-1, 1]) > 0.03, "Fingers are mirrored"
    cfg = json.loads((root / "retarget.json").read_text())["retargeting"]
    config = RetargetingConfig.from_dict(
        cfg, {"urdf_path": str(root / "retarget.urdf"), "low_pass_alpha": 1.0}
    )
    seq = config.build()
    opt, robot = seq.optimizer, seq.optimizer.robot
    targets = [robot.get_joint_index(n) for n in manifest["finger_joints"]]
    assert len(targets) + 6 == manifest["action_dimension"]
    origins = [robot.get_link_index(n) for n in opt.origin_link_names]
    tasks = [robot.get_link_index(n) for n in opt.task_link_names]
    maximum_error, times = 0.0, []
    for pose_index, fraction in enumerate(
        [*np.linspace(0, 0.4, 21), *np.linspace(0.4, 0, 21)]
    ):
        q = robot.q0.copy()
        q[targets] = np.array(
            [manifest["neutral"][n] for n in manifest["finger_joints"]]
        )
        q[targets] += fraction * (robot.joint_limits[targets, 1] - q[targets])
        if opt.adaptor is not None:
            q = opt.adaptor.forward_qpos(q)
        robot.compute_forward_kinematics(q)
        expected = np.array(
            [
                robot.get_link_pose(t)[:3, 3] - robot.get_link_pose(o)[:3, 3]
                for o, t in zip(origins, tasks)
            ]
        )
        if pose_index == 0:
            seq.set_qpos(q)
        start = time.perf_counter()
        result = seq.retarget(expected)
        times.append((time.perf_counter() - start) * 1000)
        result[targets] = bounded_fingers(
            result[targets],
            [manifest["finger_limits"][n] for n in manifest["finger_joints"]],
        )
        if opt.adaptor is not None:
            result = opt.adaptor.forward_qpos(result)
        assert np.isfinite(result).all(), "Nonfinite retargeting values"
        assert np.all(result[targets] >= robot.joint_limits[targets, 0] - 1e-5), (
            manifest["robot"],
            fraction,
            result[targets] - robot.joint_limits[targets, 0],
        )
        assert np.all(result[targets] <= robot.joint_limits[targets, 1] + 1e-5)
        for joint in manifest["mimic_joints"]:
            parent = joint["mimic"]
            expected_mimic = result[robot.get_joint_index(parent["joint"])] * float(
                parent.get("multiplier", 1)
            ) + float(parent.get("offset", 0))
            assert (
                abs(result[robot.get_joint_index(joint["name"])] - expected_mimic)
                < 1e-5
            )
        robot.compute_forward_kinematics(result)
        actual = np.array(
            [
                robot.get_link_pose(t)[:3, 3] - robot.get_link_pose(o)[:3, 3]
                for o, t in zip(origins, tasks)
            ]
        )
        maximum_error = max(
            maximum_error, float(np.linalg.norm(actual - expected, axis=1).max())
        )
    # DexPilot deliberately projects close thumb/finger targets to a pinch distance.
    assert maximum_error < 0.005, (
        f"Reachable-target error is too large: {maximum_error} m"
    )
    return {
        "robot": manifest["robot"],
        "bundle_digest": manifest["digest"],
        "action_dimension": manifest["action_dimension"],
        "poses_checked": len(times),
        "combined_wrist_poses_checked": len(wrist_poses),
        "palm_frame_checked": True,
        "open_fingertips_m": open_tips.tolist(),
        "max_fingertip_vector_error_m": maximum_error,
        "mean_solve_ms": float(np.mean(times)),
        "mimic_joints_checked": len(manifest["mimic_joints"]),
        "status": "CPU_CHECKED",
        "gpu_tested": False,
        "headset_tested": False,
    }


if __name__ == "__main__":
    for path in sys.argv[1:]:
        result = check(path)
        print(json.dumps(result), flush=True)
