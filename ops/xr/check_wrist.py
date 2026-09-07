"""GPU regression: actual virtual wrist joints must not reverse at +/-180 degrees."""

import json
import os
from pathlib import Path
import runpy
import sys

native = sys.argv[1] in {"floating_shadow_right", "floating_shadow_left"}
bundle = Path(sys.argv[1])
manifest = (
    dict(
        robot=sys.argv[1],
        side=sys.argv[1].rsplit("_", 1)[1],
        retargeting_scheme="dexpilot",
        wrist_joints=[
            "x_translation_joint",
            "y_translation_joint",
            "z_translation_joint",
            "x_rotation_joint",
            "y_rotation_joint",
            "z_rotation_joint",
        ],
        digest=None,
    )
    if native
    else json.loads((bundle / "manifest.json").read_text())
)
if not native:
    sys.path.insert(0, str(bundle))
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
]
ns = runpy.run_path(sys.argv[0], run_name="skynet_wrist_check")
app = ns["simulation_app"]
try:
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from wrist import (
        ContinuousEulerXYZ,
        configure_virtual_wrist,
        install_relative_wrist_tracking,
    )
    from anatomy import palm_frame
    from dexverse.devices.retargeters.simple_relative_retargeting import (
        SimpleRelativeRetargeter,
        SimpleRelativeRetargeterCfg,
    )
    from isaaclab.devices.device_base import DeviceBase

    if not native:
        from runtime import install, validate_environment

        install(bundle)
    cfg, _ = ns["create_environment_config"]()
    cfg = ns["prune_stale_obs_refs"](ns["strip_camera_cfgs"](cfg))
    env = ns["create_environment"](cfg)
    env.reset()
    if not native:
        validate_environment(env, manifest)
    robot = env.scene["robot"]
    before_limits = robot.data.joint_pos_limits.clone()
    configure_virtual_wrist(robot, None if native else manifest, [manifest["side"]])
    axes = [robot.joint_names.index(name) for name in manifest["wrist_joints"][3:]]
    fingers = [i for i in range(len(robot.joint_names)) if i not in axes]
    torch.testing.assert_close(
        before_limits[:, fingers], robot.data.joint_pos_limits[:, fingers]
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
    reset_relative = install_relative_wrist_tracking(retargeter)
    layout = retargeter._layout["hands"][hand]
    normalize = Rotation.from_euler("y", 90, degrees=True) * Rotation.from_euler(
        "x", -90, degrees=True
    )
    base_quat = normalize.inv().as_quat()[[3, 0, 1, 2]]
    human = np.zeros((21, 3))
    for f, y in enumerate([0.075, 0.03, 0, -0.025, -0.05]):
        for j in range(4):
            human[1 + f * 4 + j] = [0.065 + j * 0.03, y, 0]
    if manifest["side"] == "left":
        human[:, 1] *= -1
    reports = []
    for name, axis, endpoint in [
        ("palm_roll_forward", 0, 2.5 * np.pi),
        ("palm_roll_backward", 0, -2.5 * np.pi),
        ("pitch_forward", 1, np.pi),
        ("pitch_backward", 1, -np.pi),
        ("yaw_forward", 2, 2.5 * np.pi),
        ("yaw_backward", 2, -2.5 * np.pi),
    ]:
        env.reset()
        tracker = ContinuousEulerXYZ()
        reset_relative()
        actual, commanded, errors = [], [], []
        angles = np.zeros(3)
        initial = robot.data.joint_pos.clone()
        ticks = int(abs(endpoint) / 0.035) + 1
        for t in np.r_[np.linspace(0, 1, ticks), np.ones(80)]:
            angles[axis] = endpoint * t
            # Use the same anatomical frame and continuous conversion as collection.
            expected_rotation = Rotation.from_euler("XYZ", angles)
            command = tracker.update(
                palm_frame(expected_rotation.apply(human), manifest["side"])
            )
            np.testing.assert_allclose(command, angles, atol=1e-5)
            quat = (expected_rotation * normalize.inv()).as_quat()[[3, 0, 1, 2]]
            relative = retargeter._compute_relative_wrist_euler(quat, base_quat, layout)
            expected_relative = (
                angles if layout["wrist_rot_order"] == "xyz" else angles[[2, 1, 0]]
            )
            np.testing.assert_allclose(relative, expected_relative, atol=1e-5)
            target = initial.clone()
            target[:, axes] = torch.as_tensor(
                command, device=env.device, dtype=target.dtype
            )
            robot.set_joint_position_target(target)
            for _ in range(env.cfg.decimation):
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
            measured = robot.data.joint_pos[0, axes].cpu().numpy().copy()
            assert np.isfinite(measured).all()
            actual.append(measured)
            commanded.append(command)
            errors.append(float(np.linalg.norm(measured - command)))
        actual, commanded = np.asarray(actual), np.asarray(commanded)
        delta = np.diff(actual[:, axis])
        reverse = float(np.max(np.maximum(-np.sign(endpoint) * delta, 0)))
        assert reverse < 0.015, (name, "Wrist reversed", reverse)
        assert np.max(np.abs(np.diff(commanded, axis=0))) < 0.04
        assert errors[-1] < 0.03, (
            name,
            "Wrist did not reach target beyond its former limit",
            errors[-1],
        )
        assert np.max(errors) < 0.65, (
            name,
            "Wrist jumped away from its target",
            np.max(errors),
        )
        reports.append(
            dict(
                trajectory=name,
                steps=len(actual),
                final_angle=float(actual[-1, axis]),
                target_angle=endpoint,
                max_reverse_step=reverse,
                final_error=errors[-1],
                max_tracking_error=max(errors),
            )
        )
    print(
        "SKYNET_WRIST_CHECK "
        + json.dumps(
            dict(
                synthetic=True,
                robot=manifest["robot"],
                bundle_digest=manifest["digest"],
                physical_finger_limits_unchanged=True,
                trajectories=reports,
            )
        ),
        flush=True,
    )
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    if "env" in locals():
        env.close()
    app.close()
