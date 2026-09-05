import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from skynet_app.adapters.openpi_dataset_bridge import bind_dataset, inspect_dataset, snapshot_normalization


def dataset(tmp_path):
    root = tmp_path / "selected"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    info = {"codebase_version": "v2.0", "fps": 20, "chunks_size": 1000,
            "total_episodes": 1, "total_frames": 2, "data_path": "data/episode_{episode_index:06d}.parquet",
            "features": {"state": {"shape": [8], "dtype": "float32"}, "actions": {"shape": [7], "dtype": "float32"},
                         "image": {"shape": [224,224,3], "dtype": "image"}, "wrist_image": {"shape": [224,224,3], "dtype": "image"}}}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 2}) + "\n")
    (root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "Test"}\n')
    (root / "meta/stats.json").write_text('{}')
    # This test checks the metadata contract, not decoding Parquet observations.
    (root / "data/episode_000000.parquet").touch()
    norm = tmp_path / "norm_stats.json"
    norm.write_text(json.dumps({"norm_stats": {key: {"mean": [0]*n, "std": [1]*n, "q01": [-1]*n, "q99": [1]*n} for key,n in (("state",8),("actions",7))}}))
    return root, norm


def test_metadata_check_and_normalization_snapshot(tmp_path):
    root, norm = dataset(tmp_path)
    summary = inspect_dataset(root, norm)
    assert summary["frames"] == 2 and summary["episodes"] == 1
    output = snapshot_normalization(norm, tmp_path / "run", "physical-intelligence/libero", summary["normalization_sha256"])
    assert (output / "physical-intelligence/libero/norm_stats.json").read_bytes() == norm.read_bytes()
    assert snapshot_normalization(norm, tmp_path / "run", "physical-intelligence/libero", summary["normalization_sha256"]) == output
    norm.write_text('{}')
    with pytest.raises(ValueError, match="changed"):
        snapshot_normalization(norm, tmp_path / "run", "physical-intelligence/libero", summary["normalization_sha256"])


@pytest.mark.parametrize("change, expected", [
    (lambda root, norm: (root / "data/episode_000000.parquet").unlink(), "incomplete"),
    (lambda root, norm: (root / "meta/tasks.jsonl").unlink(), "incomplete"),
    (lambda root, norm: norm.write_text('{"norm_stats": {}}'), "Normalization"),
])
def test_incomplete_selected_files_are_rejected(tmp_path, change, expected):
    root, norm = dataset(tmp_path)
    change(root, norm)
    with pytest.raises(ValueError, match=expected):
        inspect_dataset(root, norm)


@pytest.mark.parametrize("field, value, expected", [
    ("codebase_version", "v3.0", "Unsupported"),
    ("fps", 0, "fps"),
    ("data_path", "../outside.parquet", "invalid file"),
    ("total_frames", 999, "totals"),
    ("features", {}, "Incompatible LIBERO"),
])
def test_wrong_dataset_contract_fails_early(tmp_path, field, value, expected):
    root, norm = dataset(tmp_path)
    path = root / "meta/info.json"
    info = json.loads(path.read_text()); info[field] = value; path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match=expected):
        inspect_dataset(root, norm)


def test_both_lerobot_classes_receive_exact_root_and_downloads_fail(tmp_path):
    class Metadata:
        def __init__(self, repo_id, root=None, revision=None, force_cache_sync=False):
            self.root = root
    class Dataset:
        def __init__(self, repo_id, root=None, force_cache_sync=False):
            self.root = root
            self.meta = module.LeRobotDatasetMetadata(repo_id, root)
    module = SimpleNamespace(LeRobotDatasetMetadata=Metadata, LeRobotDataset=Dataset,
                             get_safe_version=lambda: pytest.fail("network"), snapshot_download=lambda: pytest.fail("network"))
    bind_dataset(module, tmp_path, "selected/repo")
    data = module.LeRobotDataset("selected/repo")
    assert data.root == data.meta.root == tmp_path
    assert type(data) is Dataset  # Retains the upstream class for multiprocessing.
    with pytest.raises(ValueError, match="different dataset"):
        module.LeRobotDataset("other/repo")
    with pytest.raises(ValueError, match="conflicts"):
        module.LeRobotDataset("selected/repo", tmp_path / "other")
    with pytest.raises(ValueError, match="disabled"):
        module.LeRobotDataset("selected/repo", force_cache_sync=True)
    for method in (module.snapshot_download, module.get_safe_version):
        with pytest.raises(ValueError, match="disabled"):
            method()


def test_capsule_checks_dataset_without_openpi_or_gpu(tmp_path):
    root, norm = dataset(tmp_path)
    script = Path(__file__).parents[1] / "skynet_app/adapters/openpi_dataset_bridge.py"
    result = subprocess.run([sys.executable, str(script), "--dataset-root", str(root), "--norm-stats", str(norm), "--check-only"], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '"frames": 2' in result.stdout
