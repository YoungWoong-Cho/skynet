"""Run OpenPI with an explicitly selected local LeRobot LIBERO dataset.

This capsule adapts the root argument omitted by OpenPI's data loader. It leaves
the upstream trainer, transforms and dataset classes intact and disallows dataset
downloads in this process. Source files and shared dataset caches are not edited.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read dataset metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def inspect_dataset(root: Path, norm_path: Path) -> dict:
    if not root.is_absolute() or not norm_path.is_absolute():
        raise ValueError("Dataset and normalization paths must be absolute compute-node paths")
    if norm_path.name != "norm_stats.json":
        raise ValueError("The normalization file must be named norm_stats.json")
    info = _json(root / "meta/info.json")
    if info.get("codebase_version") not in {"v2.0", "v2.1"}:
        raise ValueError("Unsupported LeRobot version: this bridge accepts v2.0 or v2.1")
    features = info.get("features", {})
    for key, shape in {"state": [8], "actions": [7]}.items():
        feature = features.get(key, {})
        if feature.get("shape") != shape or feature.get("dtype") not in {"float32", "float64"}:
            raise ValueError(f"Incompatible LIBERO dataset: {key} must be a floating-point vector of shape {shape}")
    for key in ("image", "wrist_image"):
        feature = features.get(key, {})
        shape = feature.get("shape", [])
        if feature.get("dtype") not in {"image", "video"} or len(shape) != 3 or 3 not in (shape[0], shape[-1]):
            raise ValueError(f"Incompatible LIBERO dataset: {key} must contain RGB images")
    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (float, int)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Dataset fps must be a positive finite number")
    for name in ("tasks.jsonl", "episodes.jsonl", "stats.json" if info["codebase_version"] == "v2.0" else "episodes_stats.jsonl"):
        if not (root / "meta" / name).is_file():
            raise ValueError(f"Selected dataset is incomplete: missing meta/{name}. Re-import it; automatic downloads are disabled.")
    chunk_size = info.get("chunks_size")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("Dataset chunks_size must be a positive integer")
    indices = set()
    total_frames = 0
    directory_files = {}
    def declared_file_exists(relative: PurePosixPath) -> bool:
        # One directory listing per chunk avoids thousands of network-filesystem
        # stat calls. LeRobot still opens/validates the actual observations.
        directory = root / relative.parent
        if directory not in directory_files:
            try:
                with os.scandir(directory) as entries:
                    directory_files[directory] = {entry.name for entry in entries if entry.is_file(follow_symlinks=False) or entry.is_symlink()}
            except OSError:
                directory_files[directory] = set()
        return relative.name in directory_files[directory]
    with (root / "meta/episodes.jsonl").open() as source:
        for line in source:
            episode = json.loads(line)
            index, length = episode.get("episode_index"), episode.get("length")
            if type(index) is not int or index < 0 or index in indices or type(length) is not int or length < 1:
                raise ValueError("Dataset episode indices and lengths must be valid and unique")
            indices.add(index)
            total_frames += length
            templates = [(info.get("data_path"), "")]
            templates += [(info.get("video_path"), key) for key, feature in features.items() if feature.get("dtype") == "video"]
            for template, key in templates:
                if not isinstance(template, str):
                    raise ValueError("Dataset is missing its data or video path template")
                relative = PurePosixPath(template.format(episode_chunk=index // chunk_size, episode_index=index, video_key=key))
                if relative.is_absolute() or ".." in relative.parts or not declared_file_exists(relative):
                    raise ValueError(f"Selected dataset is incomplete: missing or invalid file {relative}. Automatic downloads are disabled.")
    if not indices or indices != set(range(len(indices))) or len(indices) != info.get("total_episodes") or total_frames != info.get("total_frames"):
        raise ValueError("Dataset episode/frame totals do not match its manifest")
    stats = _json(norm_path).get("norm_stats", {})
    for key, length in (("state", 8), ("actions", 7)):
        values = stats.get(key, {})
        for name in ("mean", "std", "q01", "q99"):
            array = values.get(name)
            if not isinstance(array, list) or len(array) != length or any(type(value) not in (int, float) or not math.isfinite(value) for value in array):
                raise ValueError(f"Normalization {key}.{name} must contain {length} finite numbers; use stats computed for this dataset and config")
        if any(value < 0 for value in values["std"]) or any(low > high for low, high in zip(values["q01"], values["q99"])):
            raise ValueError(f"Normalization {key} contains invalid standard deviations or quantile bounds")
    return {"root": str(root), "episodes": len(indices), "frames": total_frames,
            "normalization_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest()}


def snapshot_normalization(path: Path, base: Path, asset_id: str, expected_sha: str) -> Path:
    relative = PurePosixPath(asset_id)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Unsupported OpenPI normalization asset identity")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha:
        raise ValueError("Normalization changed during validation; retry with the intended file")
    root = base / expected_sha
    destination = root / relative / "norm_stats.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as output:
            output.write(content)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise ValueError("Saved normalization snapshot has changed; restore it before retrying")
    return root


def bind_dataset(module, root: Path, repo_id: str) -> None:
    """Keep original class identities so multiprocessing can pickle datasets."""
    def no_download(*args, **kwargs):
        raise ValueError("The selected dataset is incomplete or unsupported. Automatic dataset fallback/download is disabled; re-import the selected version.")

    def rooted(original):
        signature = inspect.signature(original)
        if "root" not in signature.parameters or "repo_id" not in signature.parameters:
            raise ValueError("Unsupported LeRobot API: dataset constructors must expose root and repo_id")

        @functools.wraps(original)
        def initialize(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            if bound.arguments.get("repo_id") != repo_id:
                raise ValueError("OpenPI requested a different dataset than its selected data configuration")
            existing = bound.arguments.get("root")
            if existing is not None and Path(existing) != root:
                raise ValueError("Dataset root conflicts with the selected immutable bundle")
            if bound.arguments.get("force_cache_sync"):
                no_download()
            bound.arguments["root"] = root
            return original(*bound.args, **bound.kwargs)
        return initialize

    # Resolve and validate the API before changing any of its entry points.
    constructors = [(cls, rooted(cls.__init__)) for cls in (module.LeRobotDatasetMetadata, module.LeRobotDataset)]
    if not callable(getattr(module, "get_safe_version", None)) or not callable(getattr(module, "snapshot_download", None)):
        raise ValueError("Unsupported LeRobot download API; cannot guarantee immutable local loading")
    module.get_safe_version = no_download
    module.snapshot_download = no_download
    for cls, initialize in constructors:
        cls.__init__ = initialize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    summary = inspect_dataset(args.dataset_root, args.norm_stats)
    print("Selected OpenPI dataset: " + json.dumps(summary), flush=True)
    if args.check_only:
        return
    source = Path.cwd()
    if not (source / "scripts/train.py").is_file():
        raise ValueError("OpenPI scripts/train.py is missing from the pinned checkout")
    sys.path.insert(0, str(source / "src"))
    from openpi.training import config, data_loader
    training_args = args.training_args
    if training_args[:1] == ["--"]:
        training_args = training_args[1:]
    sys.argv = [str(source / "scripts/train.py"), *training_args]
    selected = config.cli()
    if not isinstance(selected.data, config.LeRobotLiberoDataConfig):
        raise ValueError("Unsupported selected-data configuration: choose an OpenPI LeRobot LIBERO config; other embodiments require their own data bridge")
    # Pin the explicitly chosen stats into the config used by the trainer and
    # checkpoint writer. The repository's default dataset stats are never used.
    asset_id = selected.data.assets.asset_id or selected.data.repo_id
    assets_root = snapshot_normalization(args.norm_stats,
        Path(selected.checkpoint_base_dir).parent / "adapter-support/dataset-assets",
        asset_id, summary["normalization_sha256"])
    assets = dataclasses.replace(selected.data.assets, assets_dir=str(assets_root), asset_id=asset_id)
    selected = dataclasses.replace(selected, data=dataclasses.replace(selected.data, assets=assets))
    bind_dataset(data_loader.lerobot_dataset, args.dataset_root, selected.data.repo_id)
    module_spec = importlib.util.spec_from_file_location("skynet_openpi_train", source / "scripts/train.py")
    trainer = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = trainer
    module_spec.loader.exec_module(trainer)
    trainer.main(selected)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"OpenPI selected dataset failed: {error}", file=sys.stderr)
        raise


OPENPI_DATASET_BRIDGE_SOURCE = Path(__file__).read_text(encoding="utf-8")
