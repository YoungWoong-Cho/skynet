"""Check both production retargeters in PhysX with synthetic tracking, no recordings.

Run under the collection host's nonblocking GPU session lock. Require the JSON
result's GPU_CHECKED status; Isaac shutdown can hide Python failures in its exit code.
"""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback
from types import SimpleNamespace

from retargeting_validation import finger_joint_groups, verify_finger_motion

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", required=True, type=Path)
parser.add_argument("--config", required=True, type=Path)
parser.add_argument("--result", required=True, type=Path)
parser.add_argument(
    "--rapid",
    action="store_true",
    help="Instant close/open steps with per-frame command, pose and timing traces",
)
args = parser.parse_args()
manifest = json.loads((args.bundle / "manifest.json").read_text())
sys.path.insert(0, str(args.bundle))
sys.argv = [
    os.environ["SKYNET_DEXVERSE_RECORDER"],
    "--task",
    "Dexverse-PickCube-v0",
    "--robot_type",
    manifest["robot"],
    "--teleop_device",
    "keyboard",
    "--enable_pinocchio",
    "--headless",
    "--device",
    "cuda:0",
    "--teleop_retargeter",
    "absolute",
    "--retargeting_scheme",
    "dexpilot",
]
args.result.write_text(json.dumps({"status": "STARTING"}))
ns = runpy.run_path(sys.argv[0], run_name="skynet_retargeting_check")
app, env = ns["simulation_app"], None
try:
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation as R
    from dexverse.devices.retargeters.simple_absolute_retargeting import (
        SimpleAbsoluteRetargeter,
        SimpleAbsoluteRetargeterCfg,
    )
    from dexverse.devices.retargeters.simple_relative_retargeting import (
        DEX_RETARGETING_HAND_JOINT_NAMES,
        FINGER_Z_ROTATION_DEG,
    )
    from runtime import install, validate_environment
    from wrist import configure_virtual_wrist, DexVerseWristContinuity
    from retargeting_runtime import CollectionRetargeting

    manifest = install(str(args.bundle))
    cfg, _ = ns["create_environment_config"]()
    cfg = ns["prune_stale_obs_refs"](ns["strip_camera_cfgs"](cfg))
    env = ns["create_environment"](cfg)
    env.reset()
    validate_environment(env, manifest)
    robot = env.scene["robot"]
    sides = list(manifest["hands"])
    configure_virtual_wrist(robot, manifest, sides)
    retargeter = SimpleAbsoluteRetargeter(
        SimpleAbsoluteRetargeterCfg(
            robot_type=manifest["robot"],
            retargeting_scheme="dexpilot",
            sim_device=env.device,
        )
    )
    teleop = SimpleNamespace(_retargeters=[retargeter])
    normalize = R.from_euler("y", 90, degrees=True) * R.from_euler(
        "x", -90, degrees=True
    )
    result = {"status": "RUNNING", "robot": manifest["robot"], "methods": {}}
    for method in ({"key": "dexpilot"}, json.loads(args.config.read_text())):
        with torch.inference_mode():
            env.reset()
        continuity = DexVerseWristContinuity([retargeter], sides)
        commands = CollectionRetargeting(
            {
                "retargeting": method,
                "hand_manifest": manifest,
                "hand_bundle": {"root": str(args.bundle)},
            },
            teleop,
            continuity,
            args.result.parent / "generated-gpu",
        )
        commands.reset()
        times, samples, trace = [], {}, []
        with torch.inference_mode():
            for frame in range(400):
                flex = (
                    min(1.0, max(0.0, (frame - 60) / 100))
                    if frame < 240
                    else max(0.0, (320 - frame) / 80)
                )
                if args.rapid:
                    flex = float(60 <= frame < 240)
                frame_started = time.perf_counter()
                raw = {}
                for target in retargeter._tracked_hands:
                    side = target.name.removeprefix("HAND_").lower()
                    human = np.zeros((21, 3), dtype=np.float32)
                    sign = 1 if side == "right" else -1
                    for finger in range(5):
                        end = (
                            np.array([0.09, 0.075 * sign, -0.01])
                            if finger == 0
                            else np.array(
                                [
                                    0.18 - 0.008 * abs(finger - 2),
                                    (0.055 - 0.022 * finger) * sign,
                                    0,
                                ]
                            )
                        )
                        for joint in range(4):
                            point = end * (0.35 + joint * 0.65 / 3)
                            point[0] *= 1 - 0.45 * flex * joint / 3
                            point[2] -= 0.07 * flex * joint / 3
                            human[1 + 4 * finger + joint] = point
                    # Invert the existing canonical conversion for synthetic input.
                    human = R.from_euler(
                        "z", FINGER_Z_ROTATION_DEG[side], degrees=True
                    ).apply(human)
                    position = np.array(
                        [
                            -0.2 + 0.04 * flex,
                            (-0.22 if side == "right" else 0.22)
                            if len(sides) == 2
                            else 0,
                            1.15,
                        ]
                    )
                    rotation = R.from_euler("XYZ", [0.05 * flex, 0, 0.1 * flex])
                    quat = (rotation * normalize.inv()).as_quat()[[3, 0, 1, 2]]
                    world = rotation.apply(human) + position
                    raw[target] = {
                        name: np.r_[world[i], quat]
                        for i, name in enumerate(DEX_RETARGETING_HAND_JOINT_NAMES)
                    }
                native = retargeter.retarget(raw).detach().cpu().numpy()
                start = time.perf_counter()
                action = commands.update(native, raw)
                times.append(time.perf_counter() - start)
                if method["key"] == "dexpilot":
                    ids = [
                        i
                        for v in retargeter._layout["hands"].values()
                        for i in v["finger_indices"]
                    ]
                    np.testing.assert_array_equal(action[ids], native[ids])
                env.step(
                    torch.as_tensor(
                        action[None], dtype=torch.float32, device=env.device
                    )
                )
                if args.rapid:
                    trace.append(
                        dict(
                            frame=frame,
                            target=action.tolist(),
                            actual=robot.data.joint_pos[0].cpu().numpy().tolist(),
                            wall_ms=(time.perf_counter() - frame_started) * 1000,
                        )
                    )
                assert torch.isfinite(robot.data.body_pos_w).all(), "Nonfinite bodies"
                assert torch.isfinite(robot.data.joint_pos).all(), "Nonfinite joints"
                if frame in (59, 239, 399):
                    label = {59: "open", 239: "closed", 399: "reopened"}[frame]
                    positions = robot.data.body_pos_w[0].cpu().numpy()
                    quaternions = robot.data.body_quat_w[0].cpu().numpy()
                    tips_local = {}
                    for hand in manifest["hands"].values():
                        palm = robot.body_names.index(hand["control_frame"])
                        rotation = R.from_quat(quaternions[palm][[1, 2, 3, 0]])
                        for tip in hand["tips"]:
                            tips_local[tip] = (
                                rotation.inv()
                                .apply(
                                    positions[robot.body_names.index(tip)]
                                    - positions[palm]
                                )
                                .tolist()
                            )
                    samples[label] = dict(
                        target=action.tolist(),
                        actual=robot.data.joint_pos[0].cpu().numpy().tolist(),
                        tips_local=tips_local,
                    )
        groups = {}
        for hand in manifest["hands"].values():
            groups.update(
                finger_joint_groups(
                    (args.bundle / "simulation.urdf").read_bytes(), hand
                )
            )
        action_names = manifest["wrist_joints"] + manifest["finger_joints"]
        measurements = dict(
            frames=400,
            p95_bridge_ms=float(np.percentile(times, 95) * 1000),
            samples=samples,
            action_names=action_names,
            joint_names=robot.joint_names,
            metadata=commands.metadata,
        )
        if args.rapid:
            measurements.update(trace=trace, step_dt=float(env.step_dt))
        result["methods"][method["key"]] = measurements
        # Save measurements before assertions so a failed check remains diagnosable.
        args.result.write_text(json.dumps(result))
        measurements["fingers"] = verify_finger_motion(
            groups, robot.joint_names, action_names, samples
        )
        args.result.write_text(json.dumps(result))
    result["status"] = "GPU_CHECKED"
    args.result.write_text(json.dumps(result))
    print("RETARGETING_GPU_CHECKED", flush=True)
except BaseException as exc:
    args.result.write_text(
        json.dumps(
            {
                "status": "FAILED",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "measurements": locals().get("result", {}),
            }
        )
    )
    raise
finally:
    if env is not None:
        env.close()
    app.close()
