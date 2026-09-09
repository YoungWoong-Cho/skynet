"""Isolated camera smoke check. Synthetic data stays outside collection sessions."""
import json
import os
from pathlib import Path
import runpy
import sys

output = Path(sys.argv[1])
robot_type = sys.argv[2] if len(sys.argv) > 2 else "floating_shadow_right"
recorder_path = os.environ["SKYNET_DEXVERSE_RECORDER"]
sys.argv = [recorder_path, "--task", "Dexverse-PickCube-v0", "--robot_type", robot_type, "--teleop_device", "keyboard", "--headless", "--device", "cuda:0", "--enable_cameras", "--teleop_retargeter", "absolute"]
ns = runpy.run_path(recorder_path, run_name="skynet_image_check")
app, env = ns["simulation_app"], None
try:
    import numpy as np
    import torch
    import h5py
    from images import configure_cameras, ImageRecorder
    cfg, _ = ns["create_environment_config"]()
    configure_cameras(cfg)
    env = ns["create_environment"](cfg)
    env.reset()
    profile = dict(robot=robot_type, task="Dexverse-PickCube-v0", hand="both" if "bimanual" in robot_type else robot_type.rsplit("_", 1)[1], source_revision="30cc673e27684b9f10186fa6bea731aed246bc9f")
    images = ImageRecorder(output, env, profile, [])
    images.begin()
    states, actions = [], []
    def numeric(value):
        if isinstance(value, dict): return {k: numeric(v) for k, v in value.items()}
        if hasattr(value, "detach"): return value.detach().cpu().numpy().copy()
        return value
    states.append(numeric(env.scene.get_state(is_relative=True)))
    for i in range(6):
        action = torch.zeros(env.action_manager.total_action_dim, device=env.device)
        images.append(action)
        actions.append(action.cpu().numpy().copy())
        env.step(action[None])
        states.append(numeric(env.scene.get_state(is_relative=True)))
    receipt = images.finish()
    with h5py.File(output / receipt["path"]) as h5:
        checks = {}
        for name in ["scene_front", "scene_left", "scene_right"]:
            frames = h5["images/" + name][:]
            assert frames.shape == (6, 256, 256, 3)
            assert frames.std() > 5, (name, "empty or unrendered frames")
            checks[name] = {"shape": list(frames.shape), "min": int(frames.min()), "max": int(frames.max()), "std": float(frames.std())}
        meta = json.loads(h5.attrs["metadata"])
        ids = [meta["robot_joint_names"].index(n) for n in meta["action_joint_names"]]
        expected = np.stack([s["articulation"]["robot"]["joint_position"][0, ids] for s in states[:-1]])
        np.testing.assert_array_equal(expected, h5["state"][:])
        np.testing.assert_array_equal(actions, h5["action"][:])
    result = {"robot": robot_type, "steps": 6, "cameras": checks, "alignment": "passed", "receipt": receipt}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print("SKYNET_IMAGE_CHECK " + json.dumps(result), flush=True)
finally:
    if env is not None: env.close()
    app.close()
