"""Reader bridge from shared recording streams into the native ACT trainer."""
import json
import sys

from recording_dataset import FORMAT, RecordingDataset, verify_dataset

CONTRACT = "skynet.act-rgb-joints/v1"
CAMERAS = ["cam_head", "cam_right_wrist", "cam_left_wrist"]
STREAMS = {"cam_head": "scene_front", "cam_right_wrist": "scene_right", "cam_left_wrist": "scene_left"}
LAUNCH = dict(bench_name="Skynet", task_name="recordings", env_cfg_type="skynet", action_type="joint")


def validate_recorded_act(dataset, manifest):
    if manifest.get("format") != FORMAT or manifest.get("contract") != CONTRACT:
        raise ValueError("ACT Native requires a shared RGB/joint recording dataset")
    if len(manifest["episodes"]) < 2 or not manifest["split"]["validation"]:
        raise ValueError("ACT Native requires nonempty, separate train and validation episodes")
    capture = manifest["capture"]
    names, order = capture["action_joint_names"], manifest["policy_to_source_indices"]
    if not names or len(set(names)) != len(names) or sorted(order) != list(range(len(names))):
        raise ValueError("ACT joint mapping must cover each recorded command exactly once")
    if capture.get("action_semantics") != "raw_joint_position_command; target = action * scale + offset":
        raise ValueError("Unsupported recorded ACT command semantics")
    groups = capture["groups"]
    expected_order = [i for group in groups for i in group["wrist_indices"] + group["finger_indices"]]
    if order != expected_order or any(len(g["wrist_indices"]) != 6 or not g["finger_indices"] for g in groups):
        raise ValueError("ACT wrist/finger layout differs from its recorded joint order")
    for episode in manifest["episodes"]:
        for key in ("state", "action"):
            if episode["streams"][key]["shape"] != [episode["steps"], len(names)]:
                raise ValueError("ACT joint dimensions differ across recordings")
        for key in STREAMS.values():
            reference = episode["streams"][key]
            if reference["dtype"] != "uint8" or len(reference["shape"]) != 4 or reference["shape"][-1] != 3:
                raise ValueError("ACT requires three uint8 RGB camera streams")
    return {"arm_dim": [len(g["wrist_indices"]) for g in groups],
            "ee_dim": [len(g["finger_indices"]) for g in groups]}


def merge_json(path, key, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    original = json.loads(path.read_text()) if path.exists() else {}
    original[key] = value
    path.write_text(json.dumps(original, indent=2, allow_nan=False))


class NativeACTDataset:
    """One randomly sampled timestep per episode, with native ACT tensor order."""
    def __init__(self, recordings, episode_ids, camera_names, stats):
        self.recordings = recordings
        self.episode_ids = list(episode_ids)
        self.camera_names = list(camera_names)
        self.stats = stats
        self.is_sim = False
        self.max_steps = max(e["steps"] for e in recordings.manifest["episodes"])

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        import numpy as np
        import torch
        episode = self.recordings.episode(self.episode_ids[index])
        step = int(np.random.randint(episode.steps))
        qpos = episode.joint("state", step).astype(np.float32)
        images = np.stack([episode.rgb(STREAMS[name], step, size=(640, 480)) for name in self.camera_names])
        # The native ACT loader's real-recording alignment is retained explicitly.
        actions = episode.joint("action", slice(max(0, step - 1), None)).astype(np.float32)
        padded = np.zeros((self.max_steps, len(qpos)), dtype=np.float32)
        padded[:len(actions)] = actions
        mask = np.arange(self.max_steps) >= len(actions)
        qpos = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        padded = (padded - self.stats["action_mean"]) / self.stats["action_std"]
        return (torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float() / 255,
                torch.from_numpy(qpos), torch.from_numpy(padded), torch.from_numpy(mask))


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, *, num_workers=1):
    """Native ACT load_data signature, backed solely by shared streams."""
    from torch.utils.data import DataLoader
    recordings = RecordingDataset(dataset_dir)
    manifest = recordings.manifest
    validate_recorded_act(dataset_dir, manifest)
    if num_episodes != len(recordings) or any(name not in STREAMS for name in camera_names):
        raise ValueError("ACT task config differs from its registered recording dataset")
    stats = recordings.normalization()
    stats["qpos_mean"] = stats.pop("state_mean")
    stats["qpos_std"] = stats.pop("state_std")
    stats["example_qpos"] = recordings.episode(manifest["split"]["train"][0]).joint("state")
    datasets = {split: NativeACTDataset(recordings, manifest["split"][split], camera_names, stats)
                for split in ("train", "validation")}
    worker_options = dict(num_workers=num_workers, **({"prefetch_factor": 1} if num_workers else {}))
    loaders = [DataLoader(datasets[split], batch_size=batch, shuffle=True, pin_memory=True, **worker_options)
               for split, batch in (("train", batch_size_train), ("validation", batch_size_val))]
    return *loaders, stats, False


def prepare_recorded_act(dataset, manifest, workspace, dimensions):
    """Configure private native source and replace only its data-reader binding."""
    directory = workspace / "XPolicyLab/policy/ACT"
    key = "-".join(LAUNCH.values())
    merge_json(directory / "TASK_CONFIGS.json", key, {
        "dataset_dir": str(dataset), "num_episodes": len(manifest["episodes"]),
        "episode_len": max(e["steps"] for e in manifest["episodes"]), "camera_names": CAMERAS,
    })
    merge_json(workspace / "XPolicyLab/utils/robot/_robot_info.json", "skynet", dimensions)
    path = directory / "utils.py"
    source = path.read_text()
    import ast
    if not any(isinstance(node, ast.FunctionDef) and node.name == "load_data" for node in ast.parse(source).body):
        raise ValueError("Pinned ACT source has no expected load_data reader entrypoint")
    path.write_text(source + "\n# Skynet shared recording store: data reader only.\nfrom act_native_data import load_data\n")
    mapping = {
        "joint_names": [manifest["capture"]["action_joint_names"][i] for i in manifest["policy_to_source_indices"]],
        "policy_to_source_indices": manifest["policy_to_source_indices"],
        "capture": manifest["capture"], "camera_slots": manifest["camera_slots"], "dataset_path": str(dataset),
        "training_split": "registered immutable manifest split",
        "normalization": "registered training episodes only",
        "action_alignment": "native ACT starts actions at max(0, observation step - 1)",
    }
    (directory / "skynet-recording-mapping.json").write_text(json.dumps(mapping, indent=2, allow_nan=False))
    return {**LAUNCH, "required_paths": ["XPolicyLab/policy/ACT/TASK_CONFIGS.json", "XPolicyLab/utils/robot/_robot_info.json"]}


def verify_loader(directory, manifest_sha, seed):
    import numpy as np
    import torch
    config = json.loads((directory / "TASK_CONFIGS.json").read_text())["-".join(LAUNCH.values())]
    verify_dataset(config["dataset_dir"], manifest_sha)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Validation reads one batch per split. It does not need prefetch worker
    # processes or shared-memory daemons; native training keeps its worker path.
    train, validation, stats, _ = load_data(config["dataset_dir"], config["num_episodes"], config["camera_names"], 1, 1, num_workers=0)
    shapes = {}
    for label, loader in (("train", train), ("validation", validation)):
        images, state, actions, mask = next(iter(loader))
        if not all(torch.isfinite(value).all() for value in (images, state, actions)) or mask.all():
            raise ValueError("ACT recording reader returned invalid training values")
        shapes[label] = {"images": list(images.shape), "state": list(state.shape), "actions": list(actions.shape)}
    return {"schema": "skynet.act-native-loader-validation/v2", "manifest_sha256": manifest_sha,
            "observation_mode": "rgb", "episodes": config["num_episodes"],
            "split": {"train": train.dataset.episode_ids, "validation": validation.dataset.episode_ids},
            "normalization": "training_episodes_only", "batch_shapes": shapes}
