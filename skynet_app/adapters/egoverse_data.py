"""Reference-only recording leaves for the pinned EgoVerse MultiDataset.

EgoVerse keeps responsibility for normalization, model code and optimization.
Only its storage leaf is replaced; no Zarr/JPEG dataset is materialized.
"""

from pathlib import Path

import numpy as np
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

try:
    from .recording_dataset import RecordingDataset
except ImportError:  # Frozen capsule.
    from recording_dataset import RecordingDataset


class RecordingEpisode:
    """The small leaf interface consumed by native MultiDataset statistics."""

    embodiment = 100

    def __init__(self, episode, reader, key_map, root):
        self.episode_id = episode["id"]
        self.episode_path = Path(root) / "episodes" / self.episode_id
        self.total_frames = int(reader.steps)
        self.metadata = {"total_frames": self.total_frames, "embodiment": 100}
        self.reader = reader
        self.key_map = key_map
        self.transform = []

    def __len__(self):
        return self.total_frames

    def __getitem__(self, index):
        import torch

        if index < 0 or index >= len(self):
            raise IndexError(index)
        result = {}
        for key, spec in self.key_map.items():
            stream = {"joint_positions": "state", "actions_joints": "action"}.get(key, key)
            horizon = spec.get("horizon")
            selection = index if horizon is None else slice(index, min(index + horizon, len(self)))
            read = self.reader.joint if stream in {"state", "action"} else self.reader.read
            values = np.asarray(read(stream, selection)).copy()
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite recorded values in {key}: {self.episode_id}[{index}]")
            if horizon is not None and len(values) < horizon:
                values = np.concatenate([values, np.repeat(values[-1:], horizon - len(values), axis=0)])
            if spec.get("key_type") == "camera_keys":
                if values.dtype != np.uint8 or values.ndim != 3 or values.shape[-1] != 3:
                    raise ValueError("EgoVerse requires frame-aligned uint8 RGB streams")
                values = np.moveaxis(values, -1, 0).astype(np.float32) / 255.0
            result[key] = torch.from_numpy(np.ascontiguousarray(values)).float()
        result.update(embodiment=100, episode_hash=self.episode_id,
                      intrinsics=torch.full((3, 4), float("nan")))
        return result


class RecordingResolver:
    def __init__(self, root, split, key_map, manifest_sha=None, control_hz=None):
        self.root, self.split, self.key_map = root, split, key_map
        self.manifest_sha = manifest_sha
        self.control_hz = control_hz

    def resolve(self, filters=None):
        if filters is not None:
            raise ValueError("Recording membership is frozen in the dataset manifest")
        dataset = RecordingDataset(self.root, self.manifest_sha, control_hz=self.control_hz)
        return {
            dataset.manifest["episodes"][i]["id"]: RecordingEpisode(
                dataset.manifest["episodes"][i], dataset.episode(i), self.key_map, self.root)
            for i in dataset.manifest["split"][self.split]
        }


class JointDataset(MultiDataset):
    def __init__(self, *args, reject_outliers=True, **kwargs):
        self.reject_outliers = reject_outliers
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx, _attempts=None):
        # Retain the explicit native outlier filter, but never replace a corrupt
        # recording or failed read with a random example.
        attempts = _attempts
        while True:
            name, local_idx = self.index_map[idx]
            dataset = self.datasets[name]
            data = dataset[local_idx]
            if isinstance(dataset, MultiDataset):
                return data
            violation = self._check_bounds(data, dataset, local_idx, name)
            if violation is not None:
                idx, attempts = self._next_after_failure(idx, name, attempts, reason=violation)
                continue
            return self.normalize(data, data["embodiment"]) if self.norm_stats and data.get("embodiment") in self.norm_stats else data

    def _check_bounds(self, data, dataset, idx, dataset_name):
        for key in self.zarr_keys.get(data.get("embodiment"), {}).values():
            if key not in data:
                continue
            values = data[key]
            if not hasattr(values, "dtype") or str(values.dtype) in {"object", "str"}:
                continue
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite recorded values in {key}")
        if self.reject_outliers:
            return super()._check_bounds(data, dataset, idx, dataset_name)
        return None
