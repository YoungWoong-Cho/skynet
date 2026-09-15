"""Load self-contained Native ACT checkpoints with their original model and stats."""

import os

from act_native_checkpoint import SCHEMA
from act_native_data import CAMERAS
from xpolicy_runtime import repository


def load_native_act(context, source_dir, manifest):
    import numpy as np
    import torch

    checkpoint = context["checkpoint"]["path"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = context["policy"]["native_config"]
    revision = context["policy"]["source"]["revision"]
    identity = payload.get("identity", {})
    if (payload.get("schema") != SCHEMA or identity.get("source_revision") != revision
            or identity.get("dataset_manifest_sha256") != config["dataset_manifest_sha256"]):
        raise ValueError("Evaluation requires a Native ACT checkpoint saved with model, configuration and normalization; weights-only legacy .ckpt files cannot be used automatically")
    if manifest.get("contract") != "skynet.act-rgb-joints/v1":
        raise ValueError("Native ACT simulator evaluation requires recorded RGB/joint data")
    dimension = len(manifest["policy_to_source_indices"])
    if identity.get("state_dim") != dimension or identity["policy_config"]["camera_names"] != CAMERAS:
        raise ValueError("Native ACT checkpoint camera/joint layout differs from the recording")
    repository(source_dir, revision, "ACT")
    os.environ["ACT_ACTION_DIM"] = str(dimension)
    from detr.main import get_args_parser
    from detr.act_policy import ACTPolicy

    native = identity["native_args"]
    parser = get_args_parser()
    args = parser.parse_args([
        "--ckpt_dir", ".", "--policy_class", "ACT", "--bench_name", native["bench_name"],
        "--task_name", native["task_name"], "--seed", str(native["seed"]),
        "--num_epochs", str(payload["config"]["num_epochs"]),
    ])
    # Passing model_cfg bypasses upstream args_override handling. Apply both
    # the saved CLI arguments and resolved policy configuration explicitly.
    for key, value in {**payload["config"]["native_args"], **identity["policy_config"]}.items():
        setattr(args, key, value)
    model = ACTPolicy(identity["policy_config"], args)
    model.load_state_dict(payload["model"])
    stats = {}
    for key in ("qpos_mean", "qpos_std", "action_mean", "action_std"):
        value = np.asarray(payload["normalization"][key], dtype=np.float32)
        if value.shape != (dimension,) or not np.isfinite(value).all() or (key.endswith("std") and (value <= 0).any()):
            raise ValueError("Invalid Native ACT checkpoint normalization")
        stats[key.replace("qpos", "state")] = value
    scenes = [manifest["camera_slots"][camera] for camera in CAMERAS]
    return model, stats, scenes, int(identity["policy_config"]["num_queries"])
