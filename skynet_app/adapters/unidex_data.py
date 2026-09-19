"""Read-only episode windows for the pinned native UniDex model.

No upstream dataset cache, window-level split, random retry, fitted statistics or
future measured-state labels are used here. The converter freezes whole-episode
splits and controller targets; every action wrist uses the current state's anchor.
"""
import math

import numpy as np

FORMAT = "skynet.recording-dataset/v1"
CONTRACT = "skynet.unidex-pointcloud-faas/v1"


def _portable_imports():
    try:
        from .recording_dataset import verify_dataset, RecordingDataset
    except ImportError:
        from recording_dataset import verify_dataset, RecordingDataset
    try:
        from ops.datasets.action_codecs.unidex import anchor_faas_actions
    except ImportError:
        from action_codecs.unidex import anchor_faas_actions
    return verify_dataset, RecordingDataset, anchor_faas_actions


class NativeNormalizer:
    """The official fixed min/max formula, including its 1e-6 denominator."""

    def __init__(self, config):
        self.config = config
        self.stats = {}
        for key in ("state", "action"):
            if config.get("norm_type", {}).get(key) != "minmax":
                raise ValueError("UniDex requires the pinned fixed min/max normalizer")
            bounds = config.get("norm_stats", {}).get(key, {})
            low, high = (np.asarray(bounds.get(k), dtype=np.float32) for k in ("min", "max"))
            if low.shape != (82,) or high.shape != (82,) or not np.isfinite([low, high]).all() or np.any(high < low):
                raise ValueError("Invalid UniDex FAAS normalization bounds")
            self.stats[key] = (low, high)
        if config.get("norm_type", {}).get("pointcloud") != "identity":
            raise ValueError("UniDex point clouds must retain metric coordinates")

    def normalize(self, key, value):
        low, high = self.stats[key]
        return ((np.asarray(value, dtype=np.float32) - (high + low) / 2) / (high - low + 1e-6) * 2).astype(np.float32)

    def unnormalize(self, key, value):
        low, high = self.stats[key]
        return (np.asarray(value, dtype=np.float32) / 2 * (high - low + 1e-6) + (high + low) / 2).astype(np.float32)


def validate_manifest(root, sha):
    verify, _, _ = _portable_imports()
    manifest = verify(root, sha)
    if manifest.get("format") != FORMAT or manifest.get("contract") != CONTRACT:
        raise ValueError("Convert recordings with the current native UniDex adapter; old dataset formats are not supported")
    episodes = manifest.get("episodes", [])
    identities = [episode["id"] for episode in episodes]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate recording episode identity")
    try:
        from ops.datasets.action_codecs.unidex import supported_robots, load_codec
    except ImportError:
        from action_codecs.unidex import supported_robots, load_codec
    supported = set(supported_robots())
    for episode in episodes:
        if not isinstance(episode.get("prompt"), str) or not episode["prompt"].strip():
            raise ValueError("Each UniDex episode needs an explicit task prompt")
        if episode.get("hand_id") not in supported:
            raise ValueError("UniDex FAAS mapping is not verified for this exact hand asset")
        codec = load_codec(episode["hand_id"])
        representation = episode.get("action_representation", {})
        if representation.get("codec_sha256") != codec.digest or representation.get("frame") != "camera_opengl":
            raise ValueError("UniDex episode must pin the exact verified asset-bound FAAS codec")
        count = int(episode["steps"])
        streams = episode["streams"]
        for name in ("faas_state_absolute", "faas_action_absolute"):
            if name not in streams or streams[name]["shape"] != [count, 82] or streams[name]["dtype"] != "float32":
                raise ValueError("UniDex requires native FAAS82 absolute state and controller-target streams")
        cloud = streams.get("scene_front_pointcloud", {})
        if cloud.get("shape") != [count, 1024, 6] or cloud.get("dtype") != "float32":
            raise ValueError("UniDex requires 1024 aligned scene-front XYZRGB points per frame")
    temporal = manifest.get("temporal", {})
    stride, horizon, execute = (temporal.get(k) for k in ("frame_stride", "action_horizon", "execution_horizon"))
    if any(type(v) is not int or v < 1 for v in (stride, horizon, execute)) or execute > horizon:
        raise ValueError("Explicit positive frame stride, action horizon and execution horizon are required")
    source_hz, control_hz = (float(temporal.get(k, 0)) for k in ("source_fps", "control_hz"))
    if not all(math.isfinite(v) and v > 0 for v in (source_hz, control_hz)) or not math.isclose(source_hz / stride, control_hz, rel_tol=1e-6):
        raise ValueError("UniDex control frequency must equal source frequency / stride")
    if any(not math.isclose(float(ep.get("capture", {}).get("step_dt", 0)) * source_hz, 1.0, rel_tol=1e-6) for ep in episodes):
        raise ValueError("UniDex temporal settings differ from recorded control timing")
    split = manifest["split"]
    train, valid = split["train"], split["validation"]
    if not train or not valid or sorted(train + valid) != list(range(len(episodes))):
        raise ValueError("UniDex requires disjoint complete source-episode splits before creating windows")
    preprocessing = manifest.get("preprocessing", {})
    if preprocessing.get("pointcloud_frame") != "camera_ros_optical" or preprocessing.get("pointcloud_native_frame") != "camera_opengl":
        raise ValueError("UniDex requires an explicit ROS-optical to OpenGL pointcloud view")
    representation = manifest.get("action_representation", {})
    if representation.get("id") != "skynet.unidex-faas/v1" or representation.get("frame") != "camera_opengl" or representation.get("action_semantics") != "controller_targets":
        raise ValueError("UniDex requires camera-OpenGL FAAS and explicit controller-target action semantics")
    return manifest


class UniDexDataset:
    """Pure reading of the immutable shared recording store, no random fallback."""

    def __init__(self, root, manifest, split, normalizer):
        if split not in {"train", "validation"}:
            raise ValueError("Unknown UniDex split")
        _, RecordingDataset, _ = _portable_imports()
        self.dataset = RecordingDataset(root)
        if self.dataset.manifest != manifest:
            raise ValueError("Dataset changed after verification")
        self.normalizer = normalizer
        self.temporal = manifest["temporal"]
        self.stride = self.temporal["frame_stride"]
        self.horizon = self.temporal["action_horizon"]
        self.windows = []
        for index in manifest["split"][split]:
            count = int(manifest["episodes"][index]["steps"])
            # No padded labels or masked-loss variant: retain complete chunks.
            self.windows.extend((index, start) for start in range(0, count - (self.horizon - 1) * self.stride, self.stride))
        if not self.windows:
            raise ValueError(f"No complete {self.horizon}-step UniDex windows in {split} episodes")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode_index, start = self.windows[index]
        reader = self.dataset.episode(episode_index)
        _, _, anchor = _portable_imports()
        state = np.asarray(reader.read("faas_state_absolute", start), dtype=np.float32)
        actions = np.asarray(reader.read("faas_action_absolute", slice(start, start + self.horizon * self.stride, self.stride)), dtype=np.float32)
        cloud = np.asarray(reader.read("scene_front_pointcloud", slice(start, start + 1)), dtype=np.float32)
        if not all(np.isfinite(x).all() for x in (state, actions, cloud)):
            raise ValueError("Non-finite UniDex data; refusing random replacement samples")
        if np.any(cloud[..., 3:] < 0) or np.any(cloud[..., 3:] > 1):
            raise ValueError("UniDex pointcloud colors must be in [0, 1]")
        cloud = cloud.copy()
        cloud[..., 1:3] *= -1  # Shared ROS optical XYZ → native OpenGL XYZ.
        actions = anchor(state, actions)
        return {"pointcloud": cloud, "state": self.normalizer.normalize("state", state[None]),
                "action": self.normalizer.normalize("action", actions),
                "prompt": self.dataset.manifest["episodes"][episode_index]["prompt"]}
