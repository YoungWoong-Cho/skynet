"""Isolated two-episode lifecycle check with synthetic input, never user training data."""

import json
import os
from pathlib import Path
import runpy
import sys
import time
from types import SimpleNamespace

bundle, output = map(Path, sys.argv[1:3])
manifest = json.loads((bundle / "manifest.json").read_text())
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
ns = runpy.run_path(sys.argv[0], run_name="skynet_check_upstream")
app = ns["simulation_app"]
try:
    import numpy as np
    from scipy.spatial.transform import Rotation
    from runtime import install, validate_environment
    import collection as collection_runtime
    from collection import run_loop
    from dexverse.devices.retargeters.simple_relative_retargeting import (
        SimpleRelativeRetargeter,
        SimpleRelativeRetargeterCfg,
        DEX_RETARGETING_HAND_JOINT_NAMES,
    )
    from isaaclab.devices.device_base import DeviceBase

    install(bundle)
    cfg, success_term = ns["create_environment_config"]()
    cfg = ns["prune_stale_obs_refs"](ns["strip_camera_cfgs"](cfg))
    env = ns["create_environment"](cfg)
    env.reset()
    validate_environment(env, manifest)
    hand = DeviceBase.TrackingTarget.HAND_RIGHT
    retargeter = SimpleRelativeRetargeter(
        SimpleRelativeRetargeterCfg(
            robot_type=manifest["robot"],
            bound_hand=hand,
            retargeting_scheme=manifest["retargeting_scheme"],
        )
    )
    events, phase_history, callbacks = [], [], {}
    write_json = collection_runtime.atomic_json

    def observe_status(path, value):
        if path.name == "collection-status.json" and (
            not phase_history or phase_history[-1] != value["phase"]
        ):
            phase_history.append(value["phase"])
        write_json(path, value)

    collection_runtime.atomic_json = observe_status
    output.mkdir(parents=True, exist_ok=True)
    robot = env.scene["robot"]
    human = np.zeros((21, 3), dtype=np.float32)
    for f, y in enumerate([0.075, 0.03, 0, -0.025, -0.05]):
        for j in range(4):
            human[1 + 4 * f + j] = [0.065 + j * 0.03, y, 0]
    normalization = Rotation.from_euler("y", 90, degrees=True) * Rotation.from_euler(
        "x", -90, degrees=True
    )
    # Deliberate 25-degree mismatch between wrist axes and actual finger heading.
    quat = (Rotation.from_euler("z", 25, degrees=True) * normalization.inv()).as_quat()[
        [3, 0, 1, 2]
    ]
    neutral_wrist = (
        robot.data.body_pos_w[0, robot.body_names.index(manifest["palm"])]
        .cpu()
        .numpy()
        .copy()
    )
    raw = {
        name: np.r_[human[i] + neutral_wrist, quat]
        for i, name in enumerate(DEX_RETARGETING_HAND_JOINT_NAMES)
    }
    np.testing.assert_allclose(
        retargeter._convert_hand_to_canonical_joint_positions(raw, hand),
        human,
        atol=1e-6,
    )
    drawn_points = None
    draw = retargeter._canonical_markers.visualize

    def observe_points(**kwargs):
        global drawn_points
        drawn_points = kwargs["translations"].copy()
        draw(**kwargs)

    retargeter._canonical_markers.visualize = observe_points
    wrist_motion = Rotation.from_euler("XYZ", [0.2, -0.15, 0.25])
    displacement = np.array([0.03, -0.02, 0.05])
    rotated_overlay_checked = False

    class Bus:
        def create_subscription_to_pop_by_type(self, event, callback):
            events.append(callback)
            return callback

    class XR:
        def get_message_bus(self):
            return Bus()

        def get_input_device(self, path):
            return SimpleNamespace(
                get_all_virtual_world_poses=lambda: {
                    n: SimpleNamespace(validity_flags=0 if dropout else 15) for n in raw
                }
            )

    started = time.monotonic()
    dropout, dropped_once, dropout_since = False, False, 0.0
    brief_once, dropout_duration = False, 1.0

    class Teleop:
        _xr_core = XR()
        _retargeters = [retargeter]

        def _get_raw_data(self):
            return {hand: raw}

        def advance(self):
            global dropout, dropped_once, dropout_since, brief_once, dropout_duration
            global rotated_overlay_checked
            status = json.loads((output / "collection-status.json").read_text())
            if steps == 5 and not dropped_once:
                dropout, dropped_once, dropout_since = True, True, time.monotonic()
            if steps == 30 and not brief_once:
                dropout, brief_once, dropout_since, dropout_duration = (
                    True,
                    True,
                    time.monotonic(),
                    0.2,
                )
            if dropout and time.monotonic() - dropout_since > dropout_duration:
                dropout = False
            command = "end" if status["saved"] >= 2 else "heartbeat"
            for callback in events:
                callback(
                    SimpleNamespace(
                        payload={"message": json.dumps({"command": command})}
                    )
                )
            if time.monotonic() - started > 45:
                raise TimeoutError(
                    "Automatic collection failed to complete two synthetic episodes"
                )
            moving = status["phase"] == "recording"
            rotation = wrist_motion if moving else Rotation.identity()
            position = neutral_wrist + (displacement if moving else 0)
            current_quat = (
                rotation * Rotation.from_quat(quat[[1, 2, 3, 0]])
            ).as_quat()[[3, 0, 1, 2]]
            for i, name in enumerate(DEX_RETARGETING_HAND_JOINT_NAMES):
                raw[name] = np.r_[rotation.apply(human[i]) + position, current_quat]
            action = retargeter.retarget(self._get_raw_data())
            if moving:
                np.testing.assert_allclose(
                    drawn_points, wrist_motion.apply(human) + position, atol=1e-5
                )
                np.testing.assert_allclose(
                    action[:3].cpu().numpy(), displacement, atol=1e-5
                )
                np.testing.assert_allclose(
                    action[3:6].cpu().numpy(), [0.2, -0.15, 0.25], atol=1e-5
                )
                rotated_overlay_checked = True
            return action

        def reset(self):
            pass

        def add_callback(self, key, callback):
            callbacks[key] = callback

    steps = 0

    def synthetic_success(env, term, count):
        global steps
        steps += 1
        return count + 1, count >= 19

    ns["check_success"] = synthetic_success
    recorder = ns["TrajectoryPickleRecorder"](
        str(output / "unused.pkl"),
        task_name="Dexverse-PickCube-v0",
        env_name="Dexverse-PickCube-v0",
        robot_type=manifest["robot"],
    )
    result = run_loop(
        ns,
        dict(
            hand="right",
            instructions="Synthetic lifecycle check",
            hand_bundle=True,
            hand_manifest=manifest,
        ),
        output,
        env,
        Teleop(),
        success_term,
        recorder,
        pose_validity=SimpleNamespace(POSITION_VALID=2, POSITION_TRACKED=8),
    )
    assert result == 2
    assert rotated_overlay_checked, (
        "Wrist rotation and displayed control points were not exercised"
    )
    assert brief_once, "Brief tracking dropout was not exercised"
    receipts = json.loads((output / "episodes.json").read_text())
    assert len(receipts) == 2 and all(r["steps"] == 20 for r in receipts)
    assert (
        json.loads((output / "collection-status.json").read_text())["phase"] == "ended"
    )
    assert phase_history.count("recording") == 3, phase_history
    assert phase_history.count("saving") == 2 and "interrupted" in phase_history, (
        phase_history
    )
    print(
        "SKYNET_COLLECTION_CHECK "
        + json.dumps(
            dict(
                synthetic=True,
                episodes=result,
                steps=steps,
                phases=phase_history,
                wrist_axis_bias_degrees=25,
                rotated_control_points_checked=rotated_overlay_checked,
                bundle_digest=manifest["digest"],
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
