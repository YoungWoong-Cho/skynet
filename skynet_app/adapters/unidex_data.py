"""Read-only episode windows for the pinned native UniDex model.

No upstream dataset cache, window-level split, random retry, fitted statistics or
future measured-state labels are used here. The converter freezes whole-episode
splits and controller targets; every action wrist uses the current state's anchor.
"""
from bisect import bisect_right
from collections import defaultdict
import math
import random

import numpy as np

try:
    from .recording_time import resolve_sampling
    from .unidex_subset import subset_windows
    from .unidex_input import dataset_pointcloud_recipe, collection_pointcloud_recipe
except ImportError:
    from recording_time import resolve_sampling
    from unidex_subset import subset_windows
    from unidex_input import dataset_pointcloud_recipe, collection_pointcloud_recipe

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
    pointcloud_recipe = dataset_pointcloud_recipe(manifest)
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
        step_dt = float(episode.get("capture", {}).get("step_dt", 0))
        if count < 1 or not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("UniDex requires nonempty recordings with explicit positive source timing")
        streams = episode["streams"]
        for name in ("faas_state_absolute", "faas_action_absolute"):
            if name not in streams or streams[name]["shape"] != [count, 82] or streams[name]["dtype"] != "float32":
                raise ValueError("UniDex requires native FAAS82 absolute state and controller-target streams")
        cloud = streams.get("scene_front_pointcloud", {})
        if cloud.get("shape") != [count, pointcloud_recipe["num_points"], 6] or cloud.get("dtype") != "float32":
            raise ValueError("UniDex requires aligned scene-front XYZRGB points matching the frozen recipe")
    split = manifest["split"]
    train, valid = split["train"], split["validation"]
    if (not episodes or any(type(i) is not int for i in train + valid)
            or sorted(train + valid) != list(range(len(episodes)))):
        raise ValueError("UniDex requires disjoint complete source-episode splits before creating windows")
    preprocessing = manifest.get("preprocessing", {})
    if preprocessing.get("pointcloud_frame") != "camera_ros_optical" or preprocessing.get("pointcloud_native_frame") != "camera_opengl":
        raise ValueError("UniDex requires an explicit ROS-optical to OpenGL pointcloud view")
    representation = manifest.get("action_representation", {})
    if representation.get("id") != "skynet.unidex-faas/v1" or representation.get("frame") != "camera_opengl" or representation.get("action_semantics") != "controller_targets":
        raise ValueError("UniDex requires camera-OpenGL FAAS and explicit controller-target action semantics")
    return manifest


def validate_recorded_values(root, manifest):
    """Inspect native frame values without imposing any experiment chunk length."""
    _, RecordingDataset, _ = _portable_imports()
    dataset = RecordingDataset(root)
    if dataset.manifest != manifest:
        raise ValueError("Dataset changed after verification")
    for index in range(len(manifest["episodes"])):
        reader = dataset.episode(index)
        values = [np.asarray(reader.read(name, 0)) for name in (
            "faas_state_absolute", "faas_action_absolute", "scene_front_pointcloud")]
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError("Non-finite UniDex recorded values")
        if np.any(values[-1][..., 3:] < 0) or np.any(values[-1][..., 3:] > 1):
            raise ValueError("UniDex pointcloud colors must be in [0, 1]")


class UniDexDataset:
    """Pure reading of the immutable shared recording store, no random fallback."""

    def __init__(self, root, manifest, split, normalizer, *, sampling=None, control_hz=None, action_steps=30,
                 allow_empty=False):
        if split not in {"train", "validation"}:
            raise ValueError("Unknown UniDex split")
        _, RecordingDataset, _ = _portable_imports()
        self.dataset = RecordingDataset(root)
        if self.dataset.manifest != manifest:
            raise ValueError("Dataset changed after verification")
        self.pointcloud_recipe = dataset_pointcloud_recipe(manifest)
        self.normalizer = normalizer
        self.sampling = sampling if sampling is not None else resolve_sampling(
            manifest, control_hz=control_hz, action_steps=action_steps,
            window_policy="complete", require_validation=True)
        self.horizon = self.sampling["action_steps"]
        self.strides = {episode["index"]: episode["stride"] for episode in self.sampling["episodes"]}
        self.windows = []
        for index in manifest["split"][split]:
            count = int(manifest["episodes"][index]["steps"])
            stride = self.strides[index]
            # No padded labels or masked-loss variant: retain complete chunks.
            self.windows.extend((index, start) for start in range(0, count - (self.horizon - 1) * stride, stride))
        if split == "train":
            selected = subset_windows(manifest, self.sampling)
            if selected is not None:
                self.windows = selected
        if not self.windows and not allow_empty:
            raise ValueError(f"No complete {self.horizon}-step UniDex windows in {split} episodes")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode_index, start = self.windows[index]
        stride = self.strides[episode_index]
        reader = self.dataset.episode(episode_index)
        _, _, anchor = _portable_imports()
        state = np.asarray(reader.read("faas_state_absolute", start), dtype=np.float32)
        actions = np.asarray(reader.read("faas_action_absolute", slice(start, start + self.horizon * stride, stride)), dtype=np.float32)
        cloud = np.asarray(reader.read("scene_front_pointcloud", slice(start, start + 1)), dtype=np.float32)
        if cloud.shape != (1, self.pointcloud_recipe["num_points"], 6):
            raise ValueError("UniDex point-cloud data differs from its frozen recipe")
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


class UniDexCollectionDataset:
    """Concatenate immutable episode splits without copying their physical streams."""

    def __init__(self, selections, split, normalizer, sampling):
        self.pointcloud_recipe = collection_pointcloud_recipe(selections)
        self.datasets, self.cumulative_sizes, self.hands = [], [], []
        rows = sampling.get("datasets", [])
        if len(rows) != len(selections):
            raise ValueError("Collection sampling does not match the selected datasets")
        for selection, row in zip(selections, rows):
            if any(selection.get(key) != row.get(key) for key in ("position", "version_id", "manifest_sha256")):
                raise ValueError("Collection sampling identity differs from the selected datasets")
            dataset = UniDexDataset(selection["path"], selection["metadata"], split, normalizer,
                                    sampling=row["sampling"], allow_empty=True)
            if not len(dataset):
                continue
            self.datasets.append(dataset)
            self.cumulative_sizes.append(len(dataset) + (self.cumulative_sizes[-1] if self.cumulative_sizes else 0))
            self.hands.extend(selection["metadata"]["episodes"][index]["hand_id"] for index, _ in dataset.windows)
        if not self.datasets:
            raise ValueError(f"No complete UniDex windows in the collection's {split} episodes")

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("UniDex collection window index out of range")
        dataset_index = bisect_right(self.cumulative_sizes, index)
        start = self.cumulative_sizes[dataset_index - 1] if dataset_index else 0
        return self.datasets[dataset_index][index - start]


class UniDexMixtureSampler:
    """Skynet's deterministic sampling around the unchanged native model.

    window_proportional visits every window once, matching upstream concatenation.
    hand_balanced assigns equal hand quotas (within one sample), then draws each
    hand's windows with replacement. DDP shards one shared epoch sequence and pads
    its tail like DistributedSampler so every rank performs the same batch count.
    """

    SCHEMA = "skynet.unidex-mixture-sampling/v1"
    POLICIES = ("window_proportional", "hand_balanced")

    def __init__(self, hands, policy="window_proportional", *, seed=42, shuffle=True, rank=None, replicas=None,
                 batch_size=1, gradient_accumulation_steps=1):
        if policy not in self.POLICIES or not hands or any(not isinstance(hand, str) or not hand for hand in hands):
            raise ValueError("UniDex sampling requires a known policy and a hand identity for every window")
        if type(seed) is not int or seed < 0:
            raise ValueError("UniDex sampling seed must be a nonnegative integer")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Sampler batch size must be a positive integer")
        if type(gradient_accumulation_steps) is not int or gradient_accumulation_steps < 1:
            raise ValueError("Sampler gradient accumulation must be a positive integer")
        if (rank is None) != (replicas is None) or (rank is not None and (
                type(rank) is not int or type(replicas) is not int or replicas < 1 or not 0 <= rank < replicas)):
            raise ValueError("Invalid distributed sampler rank or replica count")
        self.hands, self.policy, self.seed, self.shuffle = tuple(hands), policy, seed, shuffle
        self.rank, self.replicas, self.epoch = rank, replicas, 0
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.consumed_samples = 0
        self.groups = defaultdict(list)
        for index, hand in enumerate(self.hands):
            self.groups[hand].append(index)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Sampler epoch must be a nonnegative integer")
        if self.epoch != epoch:
            if 0 < self.consumed_samples < len(self):
                raise ValueError("Checkpoint ended an incomplete data epoch; resume a batch-boundary checkpoint instead")
            self.consumed_samples = 0
        self.epoch = epoch

    def mark_consumed(self, count):
        """Acknowledge completed batches, never DataLoader's prefetched indices."""
        if type(count) is not int or count < 1 or self.consumed_samples + count > len(self):
            raise ValueError("Invalid completed training sample count")
        self.consumed_samples += count

    def state_dict(self):
        return {"schema": "skynet.unidex-sampler-state/v1", "identity": self.identity(),
                "shuffle": self.shuffle, "replicas": self._distribution()[1],
                "epoch": self.epoch, "consumed_samples": self.consumed_samples}

    def load_state_dict(self, state):
        if (state.get("schema") != "skynet.unidex-sampler-state/v1"
                or state.get("identity") != self.identity() or state.get("shuffle") != self.shuffle
                or state.get("replicas") != self._distribution()[1]):
            raise ValueError("Resume sampler identity or distributed world size changed")
        epoch, consumed = state.get("epoch"), state.get("consumed_samples")
        if type(epoch) is not int or epoch < 0 or type(consumed) is not int or not 0 <= consumed <= len(self):
            raise ValueError("Invalid saved training data position")
        self.epoch, self.consumed_samples = epoch, consumed

    def _distribution(self):
        if self.rank is not None:
            return self.rank, self.replicas
        try:
            import torch.distributed as distributed
            if distributed.is_available() and distributed.is_initialized():
                return distributed.get_rank(), distributed.get_world_size()
        except ImportError:
            pass
        return 0, 1

    def global_indices(self):
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        if self.policy == "window_proportional":
            indices = list(range(len(self.hands)))
        else:
            hands = sorted(self.groups)
            rng.shuffle(hands)
            indices = [rng.choice(self.groups[hands[index % len(hands)]]) for index in range(len(self.hands))]
        if self.shuffle:
            rng.shuffle(indices)
        return indices

    def __iter__(self):
        rank, replicas = self._distribution()
        indices = self.global_indices()
        total = len(self) * replicas
        indices += (indices * math.ceil((total - len(indices)) / len(indices)))[:total - len(indices)]
        return iter(indices[rank:total:replicas][self.consumed_samples:])

    def __len__(self):
        _, replicas = self._distribution()
        return self._rank_samples(replicas)

    def _rank_samples(self, replicas):
        # Pad the distributed epoch to complete optimizer updates, including its
        # final accumulated microbatch. Otherwise Lightning flushes a smaller
        # final update, changing the requested global batch at every epoch end.
        alignment = self.batch_size * self.gradient_accumulation_steps
        return math.ceil(len(self.hands) / (replicas * alignment)) * alignment

    def epoch_plan(self, replicas):
        """Record padding using the planned world size before workers launch."""
        if type(replicas) is not int or replicas < 1:
            raise ValueError("Epoch sampling plan requires a positive replica count")
        rank_samples = self._rank_samples(replicas)
        total = rank_samples * replicas
        alignment = self.batch_size * self.gradient_accumulation_steps
        return {"replicas": replicas, "draws_before_padding": len(self.hands),
                "draws_after_padding": total, "padding_draws": total - len(self.hands),
                "samples_per_rank": rank_samples, "microbatches_per_rank": rank_samples // self.batch_size,
                "optimizer_updates": rank_samples // alignment, "global_batch_size": replicas * alignment}

    def identity(self):
        result = {"schema": self.SCHEMA, "policy": self.policy, "seed": self.seed,
                "epoch_windows": len(self.hands), "hand_windows": {hand: len(self.groups[hand]) for hand in sorted(self.groups)},
                "distributed": "shared_epoch_sequence_strided_by_rank_with_tail_padding"}
        if self.batch_size > 1:
            result["complete_batch_size"] = self.batch_size
        if self.gradient_accumulation_steps > 1:
            result["gradient_accumulation_steps"] = self.gradient_accumulation_steps
            result["optimizer_step_sample_alignment_per_rank"] = self.batch_size * self.gradient_accumulation_steps
        return result


def make_training_dataloader(dataset, sampler, *, batch_size, num_workers, seed):
    """Lightning's public stateful-loader protocol, backed by completed batches.

    The full epoch length stays unchanged because Lightning restores its own
    completed-batch counters. Only iteration skips the acknowledged rank-local
    prefix. Worker prefetch therefore cannot advance the checkpoint position.
    Recorded dataset reads are deterministic and perform no random augmentation.
    """
    from torch import Generator
    from torch.utils.data import DataLoader

    class TrainingDataLoader(DataLoader):
        def state_dict(self):
            return {"schema": "skynet.unidex-loader-state/v1", "batch_size": self.batch_size,
                    "sampler": self.sampler.state_dict()}

        def load_state_dict(self, state):
            if state.get("schema") != "skynet.unidex-loader-state/v1" or state.get("batch_size") != self.batch_size:
                raise ValueError("Resume training loader configuration changed")
            self.sampler.load_state_dict(state["sampler"])
            consumed = self.sampler.consumed_samples
            if consumed % self.batch_size and consumed != len(self.sampler):
                raise ValueError("Saved training position is not a completed batch")

    return TrainingDataLoader(dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, persistent_workers=num_workers > 0,
                              generator=Generator().manual_seed(seed))
