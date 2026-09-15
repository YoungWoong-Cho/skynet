"""Verify and read immutable observations as inputs to native dataset converters.

Frozen beside conversion entrypoints. File handles are short-lived, so instances
are safe to pickle for spawned workers.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath

ARTIFACT_SCHEMA = "skynet.observation-artifact/v1"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_artifact(reference, *, source_sha256=None, camera_id=None, modality=None, verify_files=True):
    root = Path(reference["path"])
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("Shared observation artifact directory is missing or unsafe")
    path = root / "manifest.json"
    if path.is_symlink() or digest(path) != reference["manifest_sha256"]:
        raise ValueError("Shared observation manifest differs from its pinned fingerprint")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != ARTIFACT_SCHEMA or manifest.get("artifact_key") != reference.get("artifact_key") or manifest.get("validation", {}).get("status") != "PASSED":
        raise ValueError("Shared observation artifact is incomplete or incompatible")
    spec = manifest.get("spec")
    if not isinstance(spec, dict) or hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() != manifest["artifact_key"]:
        raise ValueError("Shared observation content identity does not match its recipe")
    if any(spec.get(key) != manifest.get(key) for key in ("source_sha256", "camera_id", "modality", "recipe")):
        raise ValueError("Shared observation provenance differs from its pinned recipe")
    for key, expected in (("source_sha256", source_sha256), ("camera_id", camera_id), ("modality", modality)):
        if expected is not None and manifest.get(key) != expected:
            raise ValueError("Shared observation belongs to a different " + key)
    if "values.hdf5" not in manifest.get("files", {}):
        raise ValueError("Shared observation values are absent from the verified inventory")
    for name, receipt in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or not relative.parts or any(p in {".", ".."} for p in relative.parts) or "\\" in name:
            raise ValueError("Unsafe observation artifact file path")
        item = root / name
        if item.is_symlink() or not item.resolve().is_relative_to(root.resolve()) or not item.is_file():
            raise ValueError("Shared observation file is missing or unsafe")
        if item.stat().st_size != receipt["size_bytes"] or (verify_files and digest(item) != receipt["sha256"]):
            raise ValueError("Shared observation file failed checksum verification")
    return manifest


class ArtifactArray:
    def __init__(self, reference, *, source_sha256=None, camera_id=None, modality=None, verify_files=True):
        import h5py
        import numpy as np

        self.reference = dict(reference)
        self.manifest = verify_artifact(reference, source_sha256=source_sha256, camera_id=camera_id,
                                        modality=modality, verify_files=verify_files)
        self.path = str(Path(reference["path"]) / "values.hdf5")
        with h5py.File(self.path, "r") as file:
            if file.attrs.get("schema") != ARTIFACT_SCHEMA or not file.attrs.get("complete"):
                raise ValueError("Observation values were not completely published")
            for name in ("values", "timestamps", "frame_ids"):
                if not isinstance(file.get(name, getlink=True), h5py.HardLink):
                    raise ValueError("Observation arrays cannot contain external or symbolic links")
            array = file["values"]
            self.shape, self.dtype = array.shape, array.dtype
            if list(self.shape) != self.manifest.get("shape") or str(self.dtype) != self.manifest.get("dtype") or not self.shape or self.shape[0] < 1:
                raise ValueError("Observation values differ from their declared shape or dtype")
            if file["timestamps"].shape != (self.shape[0],) or file["frame_ids"].shape != (self.shape[0],):
                raise ValueError("Observation frame/timestamp alignment is missing")
            if not np.array_equal(file["frame_ids"][:], np.arange(self.shape[0])):
                raise ValueError("Observation frame IDs must preserve source action order")
            timing = self.manifest["timing"]
            dt = timing["step_dt"]
            if timing.get("alignment") != "pre_action_state" or not np.isfinite(dt) or dt <= 0 or not np.allclose(file["timestamps"][:], np.arange(self.shape[0]) * dt, rtol=0, atol=1e-9):
                raise ValueError("Observation timestamps must match original pre-action control steps")
            if self.manifest["modality"] in {"rgb", "depth"}:
                if self.manifest.get("calibration", {}).get("convention") != "ros_optical":
                    raise ValueError("Observation camera calibration convention is missing")
                for name, shape in (("calibration/intrinsics", (self.shape[0], 3, 3)),
                                    ("calibration/world_from_camera", (self.shape[0], 4, 4))):
                    if not isinstance(file.get(name, getlink=True), h5py.HardLink) or file[name].shape != shape:
                        raise ValueError("Observation camera calibration is missing or misaligned")
                    if not np.isfinite(file[name][:]).all():
                        raise ValueError("Observation camera calibration must be finite")
                matrices = file["calibration/intrinsics"][:]
                transforms = file["calibration/world_from_camera"][:]
                rotations = transforms[:, :3, :3]
                if (np.any(matrices[:, 0, 0] <= 0) or np.any(matrices[:, 1, 1] <= 0)
                        or not np.allclose(matrices[:, 2], [0, 0, 1])
                        or not np.allclose(transforms[:, 3], [0, 0, 0, 1])
                        or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5)
                        or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5)):
                    raise ValueError("Observation camera calibration is not a valid pinhole/rigid transform")

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        import h5py
        import numpy as np

        with h5py.File(self.path, "r") as file:
            if isinstance(index, (list, np.ndarray)):
                return np.stack([file["values"][int(i)] for i in index])
            return file["values"][index]
