"""Exercise native DexVerse retargeting -> action mapping -> actual PhysX bodies.

Synthetic hand input; no user recordings, training, or evaluation are created.
Run separately for each native robot or imported bundle on an idle GPU.
"""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

p = argparse.ArgumentParser()
p.add_argument("--robot", required=True)
p.add_argument("--bundle")
p.add_argument("--result", required=True)
p.add_argument("--images", action="store_true")
args = p.parse_args()
if args.bundle:
    sys.path.insert(0, args.bundle)
sys.argv = [
    os.environ["SKYNET_DEXVERSE_RECORDER"],
    "--task",
    "Dexverse-PickCube-v0",
    "--robot_type",
    args.robot,
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
if args.images:
    sys.argv.append("--enable_cameras")
ns = runpy.run_path(sys.argv[0], run_name="skynet_following_check")
app, env, camera = ns["simulation_app"], None, None
try:
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation as R
    from dexverse.devices.retargeters.simple_absolute_retargeting import (
        SimpleAbsoluteRetargeter,
        SimpleAbsoluteRetargeterCfg,
    )
    from dexverse.devices.retargeters.simple_relative_retargeting import (
        DEX_RETARGETING_HAND_JOINT_NAMES as names,
    )
    from wrist import configure_virtual_wrist, DexVerseWristContinuity

    manifest = None
    if args.bundle:
        from runtime import install, validate_environment

        manifest = install(args.bundle)
    cfg, _ = ns["create_environment_config"]()
    cfg = ns["prune_stale_obs_refs"](ns["strip_camera_cfgs"](cfg))
    env = ns["create_environment"](cfg)
    env.reset()
    if manifest:
        validate_environment(env, manifest)
    robot = env.scene["robot"]
    sides = (
        ["right", "left"]
        if args.robot.endswith("bimanual")
        else [args.robot.rsplit("_", 1)[1]]
    )
    configure_virtual_wrist(robot, manifest, sides)
    retargeter = SimpleAbsoluteRetargeter(
        SimpleAbsoluteRetargeterCfg(
            robot_type=args.robot,
            retargeting_scheme="dexpilot",
            sim_device=env.device,
        )
    )
    continuity = DexVerseWristContinuity([retargeter], sides)
    normalize = R.from_euler("y", 90, degrees=True) * R.from_euler(
        "x", -90, degrees=True
    )
    layouts = retargeter._layout["hands"]
    positions, rotations, points, palms, neutral_frames = {}, {}, {}, {}, {}
    for hand in layouts:
        side = hand.name.removeprefix("HAND_").lower()
        positions[hand] = np.array(
            [-0.2, (-0.22 if side == "right" else 0.22) if len(sides) == 2 else 0, 1.15]
        )
        rotations[hand] = R.identity()
        human = np.zeros((21, 3), dtype=np.float32)
        sign = 1 if side == "right" else -1
        for f in range(5):
            end = (
                np.array([0.09, 0.075 * sign, -0.01])
                if f == 0
                else np.array(
                    [
                        0.18 - 0.008 * abs(f - 2),
                        (0.055 - 0.022 * f) * sign,
                        0,
                    ]
                )
            )
            for j in range(4):
                human[1 + 4 * f + j] = end * (0.35 + j * 0.65 / 3)
        points[hand] = human
        palm = (
            manifest.get("hands", {side: {"control_frame": "skynet_palm"}})[side][
                "control_frame"
            ]
            if manifest
            else (
                ("rh_" if side == "right" else "lh_") + "palm"
                if len(sides) == 2
                else "palm"
            )
        )
        palms[hand] = robot.body_names.index(palm)
        neutral_quat = robot.data.body_quat_w[0, palms[hand]].cpu().numpy()
        neutral_frames[hand] = R.from_quat(neutral_quat[[1, 2, 3, 0]])
    if args.images:
        from video import SceneCamera

        camera = SceneCamera(env)
        import omni.replicator.core as rep

        with camera.camera:
            rep.modify.pose(position=(0.35, -0.5, 1.5), look_at=(-0.15, 0, 1.1))

    def snapshot(label):
        if camera:
            from PIL import Image

            Image.fromarray(camera.frame()).save(
                str(Path(args.result).with_suffix("")) + "-" + label + ".png"
            )

    max_position, max_rotation, max_mimic = 0.0, 0.0, 0.0
    steps = 0
    finger_commands = []

    def advance(flex=0.0, ticks=1):
        global steps
        for _ in range(ticks):
            raw = {}
            for hand in layouts:
                human = points[hand].copy()
                for f in range(5):
                    for j in range(4):
                        human[1 + 4 * f + j, 0] *= 1 - 0.45 * flex * j / 3
                        human[1 + 4 * f + j, 2] -= 0.07 * flex * j / 3
                quat = (rotations[hand] * normalize.inv()).as_quat()[[3, 0, 1, 2]]
                world = rotations[hand].apply(human) + positions[hand]
                raw[hand] = {n: np.r_[world[i], quat] for i, n in enumerate(names)}
            native = retargeter.retarget(raw).detach().cpu().numpy()
            action = continuity.update(native)
            # Continuity must leave native solver results and wrist poses intact.
            for hand, layout in layouts.items():
                fixed = list(layout["wrist_trans_indices"]) + list(
                    layout["finger_indices"]
                )
                np.testing.assert_array_equal(action[fixed], native[fixed])
                ids = list(layout["wrist_rot_indices"])
                order = layout["wrist_rot_order"]
                a, b = action[ids], native[ids]
                if order == "yaw_pitch_roll":
                    a, b = a[::-1], b[::-1]
                np.testing.assert_allclose(
                    R.from_euler("XYZ", a).as_matrix(),
                    R.from_euler("XYZ", b).as_matrix(),
                    atol=1e-6,
                )
            env.step(torch.as_tensor(action[None], device=env.device))
            assert torch.isfinite(robot.data.body_pos_w).all(), (
                "Nonfinite hand body positions"
            )
            assert torch.isfinite(robot.data.joint_pos).all(), "Nonfinite hand joints"
            steps += 1
        finger_commands.append(
            np.concatenate(
                [action[list(v["finger_indices"])] for v in layouts.values()]
            )
        )
        return action

    def verify_pose(label):
        global max_position, max_rotation, max_mimic
        for hand, body in palms.items():
            pos = robot.data.body_pos_w[0, body].cpu().numpy()
            quat = robot.data.body_quat_w[0, body].cpu().numpy()
            position_error = float(np.linalg.norm(pos - positions[hand]))
            rotation_error = float(
                (
                    R.from_quat(quat[[1, 2, 3, 0]]).inv()
                    * rotations[hand]
                    * neutral_frames[hand]
                ).magnitude()
            )
            max_position = max(max_position, position_error)
            max_rotation = max(max_rotation, rotation_error)
            assert position_error < 0.025, (
                label,
                hand,
                "position",
                pos,
                positions[hand],
            )
            assert rotation_error < 0.08, (label, hand, "rotation", rotation_error)
        if manifest:
            q = robot.data.joint_pos[0].cpu().numpy()
            for j in manifest["mimic_joints"]:
                parent = j["mimic"]
                expected = q[robot.joint_names.index(parent["joint"])] * float(
                    parent.get("multiplier", 1)
                ) + float(parent.get("offset", 0))
                error = abs(float(q[robot.joint_names.index(j["name"])] - expected))
                max_mimic = max(max_mimic, error)
                assert error < 0.10, (label, j["name"], "mimic", error)
        print("POSE", label, "passed", flush=True)

    advance(ticks=120)
    snapshot("open")
    verify_pose("open")
    snapshot("open")
    home = {h: x.copy() for h, x in positions.items()}
    # Independent hand movement catches swapped left/right action indices.
    for hand in layouts:
        for axis in range(3):
            positions[hand] = home[hand].copy()
            positions[hand][axis] += 0.12
            advance(ticks=80)
            verify_pose(f"{hand.name}-translation-{axis}")
        positions[hand] = home[hand].copy()
    # Cross both signs of the Euler branch cut while rotating the real physics joints.
    for axis in range(3):
        continuity.reset()
        for angle in np.r_[
            np.linspace(0, 3.7, 100),
            np.linspace(3.7, -3.7, 200),
            np.linspace(-3.7, 0, 100),
        ]:
            for index, hand in enumerate(layouts):
                angles = np.array([0.1, -0.2, 0.15])
                angles[axis] = angle * (-1 if index else 1)
                rotations[hand] = R.from_euler("XYZ", angles)
            advance()
        advance(ticks=80)
        verify_pose(f"rotation-{axis}")
    for hand in layouts:
        rotations[hand] = R.identity()
    for fraction in np.linspace(0, 1, 40):
        advance(float(fraction), ticks=3)
    advance(1, ticks=90)
    snapshot("closed")
    verify_pose("closed")
    for fraction in np.linspace(1, 0, 40):
        advance(float(fraction), ticks=3)
    advance(ticks=90)
    verify_pose("reopened")
    for hand, layout in layouts.items():
        # Require native finger commands to respond, not merely remain finite.
        assert np.ptp(np.array(finger_commands), axis=0).max() > 0.2, (
            "Frozen finger retargeting"
        )
    env.reset()
    continuity.reset()
    advance(ticks=120)
    verify_pose("restart")
    result = dict(
        robot=args.robot,
        bundle_digest=manifest["digest"] if manifest else None,
        retargeter="DexVerse SimpleAbsoluteRetargeter",
        finger_solver="DexPilot",
        max_wrist_position_error_m=max_position,
        max_wrist_rotation_error_rad=max_rotation,
        max_mimic_error_rad=max_mimic,
        physics_steps=steps,
        hands_checked=len(sides),
        status="GPU_CHECKED",
        headset_tested=False,
    )
    Path(args.result).write_text(json.dumps(result, indent=2) + "\n")
    print("SKYNET_CHECK", json.dumps(result), flush=True)
except BaseException as exc:
    import traceback

    traceback.print_exc()
    Path(args.result).write_text(
        json.dumps(
            {"robot": args.robot, "status": "FAILED", "error": str(exc)}, indent=2
        )
    )
    raise
finally:
    if camera:
        camera.close()
    if env:
        env.close()
    app.close()
