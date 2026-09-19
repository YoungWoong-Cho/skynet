"""One immutable recording store, referenced by small adapter dataset manifests.

HDF5 contains physical streams; a dataset contains references and split metadata
only. Handles are opened lazily per process, bounded, and never pickled to workers.
"""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

try:
    from .recording_time import frequency_stride, source_frequency
except ImportError:
    from recording_time import frequency_stride, source_frequency

FORMAT = "skynet.recording-dataset/v1"
_HANDLES = OrderedDict()
_PID = None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def reject_symlinks(path):
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Shared recording paths cannot follow symbolic links")
    return path


def _array(file, key):
    import h5py
    parts = PurePosixPath(key).parts
    if not parts or key.startswith("/") or any(p in {".", ".."} for p in parts):
        raise ValueError("Invalid shared stream dataset key")
    current = file
    for part in parts:
        if not isinstance(current.get(part, getlink=True), h5py.HardLink):
            raise ValueError("Shared streams cannot follow external or symbolic HDF5 links")
        current = current[part]
    if not isinstance(current, h5py.Dataset) or current.is_virtual or current.external:
        raise ValueError("Shared streams must contain physical HDF5 arrays")
    return current


def stream_reference(path, dataset, sha256=None, **metadata):
    import h5py
    path = reject_symlinks(path)
    with h5py.File(path, "r") as file:
        array = _array(file, dataset)
        return dict(path=str(path), sha256=sha256 or digest(path), dataset=dataset,
                    shape=list(array.shape), dtype=str(array.dtype), **metadata)


def verify_reference(reference, *, steps=None, verified=None, verify_files=True):
    import h5py
    if not Path(reference["path"]).is_absolute():
        raise ValueError("Shared stream references require absolute paths")
    path = reject_symlinks(reference["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("Shared stream file is missing or unsafe")
    fingerprint = reference.get("sha256", "")
    if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("Shared stream checksum is missing")
    key = (str(path), fingerprint)
    if verify_files and (verified is None or key not in verified):
        if digest(path) != fingerprint:
            raise ValueError("Shared stream checksum differs from the dataset manifest")
        if verified is not None:
            verified.add(key)
    with h5py.File(path, "r") as file:
        array = _array(file, reference["dataset"])
        if list(array.shape) != reference["shape"] or str(array.dtype) != reference["dtype"]:
            raise ValueError("Shared stream shape or dtype differs from the manifest")
        if array.dtype.kind not in "biuf" or not array.shape or array.shape[0] < 1 or (steps is not None and array.shape[0] != steps):
            raise ValueError("Shared stream frames differ from the recording")
    return reference


def validate_manifest(manifest):
    if manifest.get("format") != FORMAT:
        raise ValueError("Select a shared recording dataset; old exported formats must be prepared again")
    episodes = manifest.get("episodes", [])
    if not episodes or not isinstance(manifest.get("contract"), str):
        raise ValueError("Recording dataset requires episodes and a policy data contract")
    split = manifest.get("split", {})
    train, validation = split.get("train", []), split.get("validation", [])
    if not train or any(type(i) is not int for i in train + validation) or sorted(train + validation) != list(range(len(episodes))):
        raise ValueError("Dataset splits must be disjoint, complete and contain training episodes")
    ids = []
    for i, episode in enumerate(episodes):
        if episode.get("index") != i or type(episode.get("steps")) is not int or episode["steps"] < 1:
            raise ValueError("Recording episodes need consecutive indices and positive lengths")
        if not episode.get("streams") or not episode.get("source", {}).get("sha256"):
            raise ValueError("Recording episode source and streams are required")
        ids.append(episode["id"])
    if len(set(ids)) != len(ids) or manifest.get("steps") != sum(e["steps"] for e in episodes):
        raise ValueError("Recording episode identities or total frame count are invalid")
    if "files" in manifest:
        raise ValueError("Recording manifests reference shared streams, never adapter payload inventories")
    return manifest


def verify_dataset(root, sha=None, *, verify_files=True):
    path = reject_symlinks(root)
    if path.is_dir():
        path = path / "manifest.json"
    if path.is_symlink() or not path.is_file() or (sha is not None and digest(path) != sha):
        raise ValueError("Recording dataset manifest differs from its pinned checksum")
    manifest = validate_manifest(json.loads(path.read_text()))
    verified = set()
    for episode in manifest["episodes"]:
        for reference in episode["streams"].values():
            verify_reference(reference, steps=episode["steps"], verified=verified, verify_files=verify_files)
        if "timestamps" in episode["streams"]:
            import h5py
            import numpy as np
            reference = episode["streams"]["timestamps"]
            with h5py.File(reference["path"], "r") as file:
                timestamps = _array(file, reference["dataset"])[...]
            dt = 1.0 / source_frequency(episode, manifest)
            if timestamps.ndim != 1 or not np.isfinite(timestamps).all() or not np.allclose(np.diff(timestamps), dt, rtol=1e-5, atol=1e-6):
                raise ValueError("Recording timestamps must be regular and match capture.step_dt")
    return manifest


def _handle(path):
    import h5py
    global _PID
    if _PID != os.getpid():
        for file in _HANDLES.values():
            file.close()
        _HANDLES.clear()
        _PID = os.getpid()
    if path not in _HANDLES:
        _HANDLES[path] = h5py.File(path, "r", rdcc_nbytes=2 * 1024 * 1024)
        while len(_HANDLES) > 32:
            _, old = _HANDLES.popitem(last=False)
            old.close()
    _HANDLES.move_to_end(path)
    return _HANDLES[path]


def close_handles():
    for file in _HANDLES.values():
        file.close()
    _HANDLES.clear()


class EpisodeReader:
    def __init__(self, episode, preprocessing=None, order=None, *, stride=1):
        self.metadata = episode
        self.source_steps = episode["steps"]
        self.stride = stride
        self.steps = (self.source_steps + stride - 1) // stride
        self.streams = episode["streams"]
        self.preprocessing = preprocessing or {}
        self.order = order

    def read(self, name, selection=slice(None)):
        import numpy as np
        reference = self.streams[name]
        array = _array(_handle(reference["path"]), reference["dataset"])
        if isinstance(selection, slice):
            start, stop, step = selection.indices(self.steps)
            selection = (slice(start * self.stride, stop * self.stride, step * self.stride)
                         if step > 0 else np.arange(start, stop, step))
        elif isinstance(selection, (int, np.integer)):
            index = int(selection)
            if index < 0:
                index += self.steps
            if index < 0 or index >= self.steps:
                raise IndexError("Frame selection is outside the recording")
            selection = index * self.stride
        if isinstance(selection, (list, tuple, np.ndarray)):
            indices = np.asarray(selection)
            if indices.size and indices.dtype.kind not in "iu":
                raise IndexError("Frame indices must be integers")
            indices = indices.astype(np.int64)
            if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.steps):
                raise IndexError("Frame selection is outside the recording")
            if not len(indices):
                return np.empty((0, *array.shape[1:]), dtype=array.dtype)
            unique, inverse = np.unique(indices * self.stride, return_inverse=True)
            values = array[unique.tolist()][inverse]
        else:
            values = array[selection]
        if "channels" in reference:
            values = values[..., reference["channels"]]
        return values

    def joint(self, name, selection=slice(None)):
        values = self.read(name, selection)
        return values if self.order is None else values[..., self.order]

    def rgb(self, name, selection, *, size=None, interpolation="linear", chw=False):
        import numpy as np
        values = self.read(name, selection)
        single = values.ndim == 3
        frames = values[None] if single else values
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("RGB streams must contain uint8 HWC frames")
        if size is not None and tuple(frames.shape[1:3]) != (size[1], size[0]):
            import cv2
            method = {"linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA}[interpolation]
            frames = np.stack([cv2.resize(frame, tuple(size), interpolation=method) for frame in frames])
        if chw:
            frames = np.moveaxis(frames, -1, -3)
        return frames[0] if single else frames


class RecordingDataset:
    def __init__(self, root, sha=None, *, verify_files=False, manifest=None, control_hz=None):
        self.root = str(root)
        self.manifest = (validate_manifest(manifest) if manifest is not None else
                         verify_dataset(root, sha, verify_files=verify_files))
        self.strides = [frequency_stride(source_frequency(e, self.manifest), control_hz)
                        if control_hz is not None else 1 for e in self.manifest["episodes"]]

    def __len__(self):
        return len(self.manifest["episodes"])

    def episode(self, index):
        episode = self.manifest["episodes"][index]
        return EpisodeReader(episode, self.manifest.get("preprocessing"),
                             episode.get("policy_to_source_indices", self.manifest.get("policy_to_source_indices")),
                             stride=self.strides[index])

    def episode_steps(self, index):
        return (self.manifest["episodes"][index]["steps"] + self.strides[index] - 1) // self.strides[index]

    def read(self, episode_index, name, selection=slice(None)):
        return self.episode(episode_index).read(name, selection)

    def normalization(self, *, minimum_std=0.01):
        import numpy as np
        result = {}
        for name in ("state", "action"):
            count, mean, squared = 0, None, None
            for index in self.manifest["split"]["train"]:
                values = self.episode(index).joint(name).astype(np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("Training states/actions contain nonfinite values")
                n, batch_mean = len(values), values.mean(axis=0)
                batch_squared = np.square(values - batch_mean).sum(axis=0)
                if mean is None:
                    count, mean, squared = n, batch_mean, batch_squared
                else:
                    if batch_mean.shape != mean.shape:
                        raise ValueError("Joint normalization requires one shared embodiment layout")
                    delta = batch_mean - mean
                    squared += batch_squared + np.square(delta) * count * n / (count + n)
                    mean += delta * n / (count + n)
                    count += n
            result[name + "_mean"] = mean.astype(np.float32)
            result[name + "_std"] = np.maximum(np.sqrt(squared / count), minimum_std).astype(np.float32)
        return result
