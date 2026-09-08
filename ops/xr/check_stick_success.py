"""GPU regression with synthetic stick poses; writes only to a validation directory."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument("robot", choices=["floating_shadow_left", "floating_shadow_right"])
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
task = "Dexverse-PickUpStick-v0"
sys.argv = [
    os.environ["SKYNET_DEXVERSE_RECORDER"],
    "--task",
    task,
    "--robot_type",
    args.robot,
    "--teleop_device",
    "keyboard",
    "--headless",
    "--device",
    "cuda:0",
]
ns = runpy.run_path(sys.argv[0], run_name="skynet_stick_check")
app = ns["simulation_app"]
env = None
try:
    import torch
    from scipy.spatial.transform import Rotation
    from collection import EpisodeStore, collection_success_term

    cfg, upstream = ns["create_environment_config"]()
    cfg = ns["prune_stale_obs_refs"](ns["strip_camera_cfgs"](cfg))
    env = ns["create_environment"](cfg)
    env.reset()
    corrected = collection_success_term(task, upstream)
    obj = env.scene["object"]

    def place(angle, height, sign=1):
        state = obj.data.default_root_state.clone()
        state[:, :3] += env.scene.env_origins
        state[:, 2] += height
        quat = (
            Rotation.from_euler("x", angle, degrees=True).as_quat()[[3, 0, 1, 2]] * sign
        )
        state[:, 3:7] = torch.as_tensor(quat, device=env.device, dtype=state.dtype)
        obj.write_root_state_to_sim(state)

    cases = []
    for angle, height, expected in [
        (0, 0.25, True),
        (180, 0.25, True),
        (29, 0.25, True),
        (151, 0.25, True),
        (31, 0.25, False),
        (149, 0.25, False),
        (90, 0.25, False),
        (0, 0.19, False),
        (180, 0.19, False),
    ]:
        for sign in [1, -1]:
            place(angle, height, sign)
            before = bool(upstream.func(env, **upstream.params)[0])
            after = bool(corrected.func(env, **corrected.params)[0])
            assert after is expected, (angle, height, sign, after)
            if angle == 180 and height == 0.25:
                assert before is False, "Expected to reproduce the directed-axis bug"
            cases.append(
                dict(
                    angle_degrees=angle,
                    height_m=height,
                    quaternion_sign=sign,
                    before=before,
                    after=after,
                )
            )

    recorder = ns["TrajectoryPickleRecorder"](
        str(args.output / "synthetic.pkl"),
        task_name=task,
        env_name=task,
        robot_type=args.robot,
    )
    recorder._metadata["synthetic_validation"] = True
    store = EpisodeStore(args.output, recorder._metadata)
    resets = []
    for angle in [0, 180]:
        ns["handle_reset"](env)
        count, success = ns["check_success"](env, corrected, 0)
        assert count == 0 and not success
        resets.append(True)
        recorder.start_episode(initial_state=env.scene.get_state(is_relative=True))
        action = torch.zeros(env.action_space.shape, device=env.device)
        for step in range(10):
            env.step(action)
            place(angle, 0.25)
            recorder.record_action(action[0])
            recorder.record_state(env.scene.get_state(is_relative=True))
            count, success = ns["check_success"](env, corrected, count)
            assert success is (step == 9), (step, count, success)
        recorder.finalize_episode(True)
        store.save(recorder._episodes[-1])
    assert len(store.receipts) == 2
    result = dict(
        synthetic=True,
        robot=args.robot,
        cases=cases,
        saved_episodes=len(store.receipts),
        success_hold_steps=10,
        reset_clears_success=all(resets),
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print("SKYNET_STICK_CHECK " + json.dumps(result), flush=True)
finally:
    if env is not None:
        env.close()
    app.close()
