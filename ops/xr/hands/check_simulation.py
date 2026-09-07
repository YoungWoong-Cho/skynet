"""Bounded Isaac/PhysX check of one hand; no headset, recording, training or eval.

Run with SKYNET_DEXVERSE_RECORDER set and --bundle /prepared/bundle.
"""

import argparse
import faulthandler
import json
import os
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--bundle", required=True)
parser.add_argument("--result", required=True)
args = parser.parse_args()
root = Path(args.bundle)
manifest = json.loads((root / "manifest.json").read_text())
sys.path.insert(0, str(root))
from runtime import install, validate_environment  # noqa: E402

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
    "--retargeting_scheme",
    manifest["retargeting_scheme"],
]
namespace = runpy.run_path(sys.argv[0], run_name="skynet_validation")
app = namespace["simulation_app"]
try:
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from dexverse.devices.retargeters.simple_relative_retargeting import (
        SimpleRelativeRetargeter,
        SimpleRelativeRetargeterCfg,
        DEX_RETARGETING_HAND_JOINT_NAMES,
    )
    from isaaclab.devices.device_base import DeviceBase

    install(root)
    cfg, _ = namespace["create_environment_config"]()
    cfg = namespace["strip_camera_cfgs"](cfg)
    cfg = namespace["prune_stale_obs_refs"](cfg)
    faulthandler.dump_traceback_later(35, repeat=True)
    env = namespace["create_environment"](cfg)
    print("CHECK: resetting", flush=True)
    env.reset()
    print("CHECK: reset complete", flush=True)
    validate_environment(env, manifest)
    robot = env.scene["robot"]
    tip_ids = [robot.body_names.index(n) for n in manifest["tips"]]
    palm_id = robot.body_names.index(manifest["palm"])
    initial_tips = (
        (robot.data.body_pos_w[0, tip_ids] - robot.data.body_pos_w[0, palm_id])
        .cpu()
        .numpy()
    )
    assert np.all(initial_tips[1:, 0] > 0.07), (
        "Fingers do not point forward",
        initial_tips,
    )
    assert np.all(abs(initial_tips[1:, 2]) < 0.04), (
        "Palm is not horizontal",
        initial_tips,
    )
    hand = (
        DeviceBase.TrackingTarget.HAND_RIGHT
        if manifest["side"] == "right"
        else DeviceBase.TrackingTarget.HAND_LEFT
    )
    retargeter = SimpleRelativeRetargeter(
        SimpleRelativeRetargeterCfg(
            robot_type=manifest["robot"],
            bound_hand=hand,
            retargeting_scheme=manifest["retargeting_scheme"],
        )
    )
    # Independent human points in a palm-down frame. Exercise the actual OpenXR
    # conversion, including nontrivial wrist rotation, not just robot-FK targets.
    human = np.zeros((21, 3), dtype=np.float32)
    side_sign = 1 if manifest["side"] == "right" else -1
    for f in range(5):
        end = (
            np.array([0.09, 0.075 * side_sign, -0.01])
            if f == 0
            else np.array(
                [0.18 - 0.008 * abs(f - 2), (0.055 - 0.022 * f) * side_sign, 0.0]
            )
        )
        for j in range(4):
            human[1 + f * 4 + j] = end * (0.35 + j * 0.65 / 3)
    wrist = Rotation.from_euler("XYZ", [0.3, -0.4, 0.7])
    normalization = Rotation.from_euler("y", 90, degrees=True) * Rotation.from_euler(
        "x", -90, degrees=True
    )
    raw = wrist * normalization.inv()
    quat = raw.as_quat()[[3, 0, 1, 2]]
    position = np.array([0.2, -0.1, 0.7])
    data = {
        name: np.r_[wrist.apply(human[i]) + position, quat]
        for i, name in enumerate(DEX_RETARGETING_HAND_JOINT_NAMES)
    }
    canonical = retargeter._convert_hand_to_canonical_joint_positions(data, hand)
    assert np.allclose(canonical, human, atol=1e-6), (
        "Human points rotated relative to the model"
    )
    # Exercise actual per-hand optimizer/name mapping with gradual open/close input.
    for fraction in np.r_[np.linspace(0, 1, 30), np.linspace(1, 0, 30)]:
        points = human.copy()
        for f in range(5):
            for j in range(4):
                points[1 + 4 * f + j, 0] *= 1 - 0.45 * fraction * (j / 3)
                points[1 + 4 * f + j, 2] -= 0.07 * fraction * (j / 3)
        dex = retargeter._dex_retgt[hand]
        fingers = dex.retarget(retargeter._compute_dex_ref_value(dex, points))
        action = np.zeros(manifest["action_dimension"], dtype=np.float32)
        retargeter._assign_hand_fingers(action, hand, fingers)
        action[2] = 0.2  # Keep the check clear of table/object contacts.
        action[3:6] = [0.15 * fraction, -0.2 * fraction, 0.25 * fraction]
        for _ in range(3):
            env.step(torch.tensor(action[None], device=env.device))
        assert torch.isfinite(robot.data.joint_pos).all(), (
            "Physics produced nonfinite joints"
        )
    # Hold the final pose and check tracking, rather than accepting a frozen but finite hand.
    for _ in range(60):
        env.step(torch.tensor(action[None], device=env.device))
    ids = [
        robot.joint_names.index(n)
        for n in manifest["wrist_joints"] + manifest["finger_joints"]
    ]
    actual = robot.data.joint_pos[0, ids].cpu().numpy()
    target = action.copy()
    target[:6] += robot.data.default_joint_pos[0, ids[:6]].cpu().numpy()
    residual = np.abs(actual - target)
    print("CHECK: target", target.tolist(), flush=True)
    print("CHECK: actual", actual.tolist(), flush=True)
    print(
        "CHECK: limits",
        robot.data.joint_pos_limits[0, ids].cpu().numpy().tolist(),
        flush=True,
    )
    print("CHECK: final joint errors", residual.tolist(), flush=True)
    assert residual[:3].max() < 0.03, "Wrist translation did not follow the command"
    assert residual[3:6].max() < 0.10, "Wrist rotation did not follow the command"
    human_contact_residual = residual[6:].tolist()
    # Human fingertip goals can be blocked by real self-contact. Validate the
    # joint-name/action map separately with the model's own neutral pose.
    action[6:] = [manifest["neutral"][name] for name in manifest["finger_joints"]]
    for _ in range(90):
        env.step(torch.tensor(action[None], device=env.device))
    neutral_error = np.abs(robot.data.joint_pos[0, ids[6:]].cpu().numpy() - action[6:])
    print("CHECK: neutral finger errors", neutral_error.tolist(), flush=True)
    assert neutral_error.max() < 0.10, "Hand cannot return to its neutral pose"
    result = {
        "robot": manifest["robot"],
        "bundle_digest": manifest["digest"],
        "status": "GPU_CHECKED",
        "physics_steps": 330,
        "wrist_tracking_error": residual[:6].tolist(),
        "human_goal_finger_residual_rad": human_contact_residual,
        "neutral_finger_error_rad": neutral_error.tolist(),
        "palm_frame_checked": True,
        "openxr_conversion_checked": True,
        "human_open_close_checked": True,
        "initial_fingertips_relative_m": initial_tips.tolist(),
        "headset_tested": False,
    }
    Path(args.result).write_text(json.dumps(result, indent=2) + "\n")
    print("SKYNET_CHECK " + json.dumps(result), flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    faulthandler.cancel_dump_traceback_later()
    if "env" in locals():
        env.close()
    app.close()
