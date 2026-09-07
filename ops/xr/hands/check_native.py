"""Check the pinned native Shadow variants without a headset or recording."""

import argparse
import os
import runpy
import sys
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--robot", required=True)
p.add_argument("--result", required=True)
args = p.parse_args()
sys.argv = [
    os.environ["SKYNET_DEXVERSE_RECORDER"],
    "--task",
    "Dexverse-PickCube-v0",
    "--robot_type",
    args.robot,
    "--teleop_device",
    "keyboard",
    "--headless",
    "--device",
    "cuda:0",
]
n = runpy.run_path(sys.argv[0], run_name="skynet_native_check")
app = n["simulation_app"]
try:
    import numpy as np
    import torch

    cfg, _ = n["create_environment_config"]()
    cfg = n["prune_stale_obs_refs"](n["strip_camera_cfgs"](cfg))
    env = n["create_environment"](cfg)
    env.reset()
    r = env.scene["robot"]
    palms = [name for name in r.body_names if name == "palm" or name.endswith("_palm")]
    assert len(palms) == (2 if "bimanual" in args.robot else 1)
    for palm in palms:
        prefix = palm[:-4]
        names = [prefix + x for x in ["thtip", "fftip", "mftip", "rftip", "lftip"]]
        tips = (
            (
                r.data.body_pos_w[0, [r.body_names.index(x) for x in names]]
                - r.data.body_pos_w[0, r.body_names.index(palm)]
            )
            .cpu()
            .numpy()
        )
        assert np.all(tips[1:, 0] > 0.12) and np.all(abs(tips[1:, 2]) < 0.04), tips
        sign = -1 if ("left" in args.robot or prefix == "lh_") else 1
        assert sign * tips[0, 1] > 0.05, tips
    action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
    for _ in range(120):
        env.step(action)
        assert torch.isfinite(r.data.joint_pos).all()
    v = {
        "robot": args.robot,
        "status": "GPU_CHECKED",
        "physics_steps": 120,
        "palm_frame_checked": True,
        "hands_checked": len(palms),
        "headset_tested": False,
    }
    Path(args.result).write_text(json.dumps(v, indent=2) + "\n")
    print("SKYNET_CHECK", json.dumps(v), flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    if "env" in locals():
        env.close()
    app.close()
