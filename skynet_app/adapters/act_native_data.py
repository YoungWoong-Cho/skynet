"""Connect verified recorded ACT inputs to the unmodified XPolicyLab loader."""

import json
import sys

FORMAT = "xpolicylab-act-hdf5/v1"
CONTRACT = "skynet.act-rgb-joints/v1"
CAMERAS = ["cam_head", "cam_right_wrist", "cam_left_wrist"]
LAUNCH = dict(bench_name="Skynet", task_name="recordings", env_cfg_type="skynet", action_type="joint")


def validate_recorded_act(dataset, manifest):
    import h5py
    import numpy as np

    if manifest.get("format") != FORMAT or manifest.get("contract") != CONTRACT:
        raise ValueError("ACT Native requires a verified ACT RGB/joint HDF5 dataset")
    if manifest.get("validation", {}).get("status") != "PASSED":
        raise ValueError("ACT dataset conversion has not passed validation")
    episodes = manifest.get("episodes", [])
    if len(episodes) < 2:
        raise ValueError("The original ACT loader requires at least two episodes for its 80/20 split. Use ACT · Skynet recordings for single-episode training without validation.")
    capture = manifest["capture"]
    names = capture["action_joint_names"]
    order = manifest["policy_to_source_indices"]
    if not names or len(set(names)) != len(names) or sorted(order) != list(range(len(names))):
        raise ValueError("ACT joint mapping must cover each recorded command exactly once")
    if capture.get("action_semantics") != "raw_joint_position_command; target = action * scale + offset":
        raise ValueError("Unsupported recorded ACT command semantics")
    groups = capture["groups"]
    expected_order = [i for group in groups for i in group["wrist_indices"] + group["finger_indices"]]
    if order != expected_order or any(len(g["wrist_indices"]) != 6 or not g["finger_indices"] for g in groups):
        raise ValueError("ACT wrist/finger layout differs from its recorded joint order")
    expected = {"arm_dim": [len(g["wrist_indices"]) for g in groups],
                "ee_dim": [len(g["finger_indices"]) for g in groups]}
    config = json.loads((dataset / "robot_config.json").read_text())
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("ACT robot configuration differs from the recorded dimensions")
    slots = manifest.get("camera_slots", {})
    if set(slots) != set(CAMERAS) or len(set(slots.values())) != len(CAMERAS):
        raise ValueError("ACT requires three distinct recorded camera views")
    if not set(slots.values()).issubset(capture.get("cameras", {})):
        raise ValueError("ACT camera calibration is missing")
    required = {"robot_config.json"}
    for index, episode in enumerate(episodes):
        name = f"dataset/episode_{index}.hdf5"
        required.add(name)
        if episode.get("index") != index or name not in manifest["files"]:
            raise ValueError("ACT episodes must have consecutive, verified HDF5 files")
        steps = episode["steps"]
        with h5py.File(dataset / name, "r") as file:
            for key in ("observations/qpos", "action"):
                array = file[key]
                if array.shape != (steps, len(names)) or steps < 1 or array.dtype.kind != "f" or not np.isfinite(array[:]).all():
                    raise ValueError(f"Invalid ACT state/action array in {name}: {key}")
            for camera in CAMERAS:
                array = file["observations/images/" + camera]
                if array.shape != (steps, 480, 640, 3) or array.dtype != np.uint8:
                    raise ValueError(f"Invalid ACT RGB camera in {name}: {camera}")
    if not required.issubset(manifest["files"]):
        raise ValueError("ACT configuration is missing from the verified inventory")
    return expected


def merge_json(path, key, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    original = json.loads(path.read_text()) if path.exists() else {}
    original[key] = value
    path.write_text(json.dumps(original, indent=2, allow_nan=False))


def prepare_recorded_act(dataset, manifest, workspace, dimensions):
    """Only write private configuration. Upstream reads the pinned HDF5 in place."""
    directory = workspace / "XPolicyLab/policy/ACT"
    key = "-".join(LAUNCH.values())
    merge_json(directory / "TASK_CONFIGS.json", key, {
        "dataset_dir": str(dataset / "dataset"), "num_episodes": len(manifest["episodes"]),
        "episode_len": max(e["steps"] for e in manifest["episodes"]), "camera_names": CAMERAS,
    })
    merge_json(workspace / "XPolicyLab/utils/robot/_robot_info.json", "skynet", dimensions)
    mapping = {
        "joint_names": [manifest["capture"]["action_joint_names"][i] for i in manifest["policy_to_source_indices"]],
        "policy_to_source_indices": manifest["policy_to_source_indices"],
        "capture": manifest["capture"], "camera_slots": manifest["camera_slots"],
        "dataset_path": str(dataset),
        "training_split": "upstream random 80/20; dataset split.json is not used",
        "normalization": "upstream statistics over all episodes; normalization.json is not used",
        "action_alignment": "upstream loader starts actions at max(0, observation step - 1)",
    }
    (directory / "skynet-recording-mapping.json").write_text(json.dumps(mapping, indent=2, allow_nan=False))
    return {**LAUNCH, "required_paths": ["XPolicyLab/policy/ACT/TASK_CONFIGS.json", "XPolicyLab/utils/robot/_robot_info.json"]}


def verify_loader(directory, manifest_sha, seed):
    """Run the actual upstream loader, including its split and normalization."""
    import numpy as np
    import torch

    sys.path.insert(0, str(directory))
    from utils import load_data
    from constants import TASK_CONFIGS

    config = TASK_CONFIGS["-".join(LAUNCH.values())]
    np.random.seed(seed)
    torch.manual_seed(seed)
    train, validation, stats, _ = load_data(config["dataset_dir"], config["num_episodes"], config["camera_names"], 1, 1)
    shapes = {}
    for label, loader in (("train", train), ("validation", validation)):
        images, state, actions, mask = next(iter(loader))
        if not all(torch.isfinite(value).all() for value in (images, state, actions)) or mask.all():
            raise ValueError("Original ACT loader returned invalid training values")
        shapes[label] = {"images": list(images.shape), "state": list(state.shape), "actions": list(actions.shape)}
    if not all(np.isfinite(stats[k]).all() for k in ("qpos_mean", "qpos_std", "action_mean", "action_std")):
        raise ValueError("Original ACT normalization contains nonfinite values")
    return {"schema": "skynet.act-native-loader-validation/v1", "manifest_sha256": manifest_sha,
            "observation_mode": "rgb", "episodes": config["num_episodes"],
            "train_episodes": len(train.dataset), "validation_episodes": len(validation.dataset), "batch_shapes": shapes}
