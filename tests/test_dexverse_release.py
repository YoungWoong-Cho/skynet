import hashlib
import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from skynet_app.dexverse_release import historical_task, release_profile
from skynet_app.trajectory import validate_identity, validate_recorded_identity


@pytest.fixture
def importer(monkeypatch):
    from skynet_app.live_xr_review import ArrayUnpickler

    arrays = ModuleType("arrays")
    arrays.ArrayUnpickler = ArrayUnpickler
    monkeypatch.setitem(sys.modules, "arrays", arrays)
    trajectory = ModuleType("trajectory")
    trajectory.validate_identity = validate_identity
    trajectory.validate_recorded_identity = validate_recorded_identity
    monkeypatch.setitem(sys.modules, "trajectory", trajectory)
    path = (
        Path(__file__).resolve().parents[1] / "ops/datasets/import_dexverse_release.py"
    )
    spec = importlib.util.spec_from_file_location("release_import_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source(tmp_path, version):
    task = f"Dexverse-OpenFaucet-v{version}"
    state = {
        "articulation": {
            "robot": {
                "joint_position": np.zeros((1, 28)),
                "root_pose": np.zeros((1, 7)),
            }
        }
    }
    episodes = [
        {
            "actions": np.full((2, 28), i, dtype=np.float32),
            "num_steps": 2,
            "states": [state, state, state],
            "initial_state": state,
            "success": True,
            "goal_pose": np.array([i, 1, 2]),
            "episode_index": i,
            "release_source": {"source": "original.pkl", "index": i},
        }
        for i in range(2)
    ]
    payload = {
        "format": "dexverse_trajectory",
        "task": task,
        "robot_type": "floating_shadow_right",
        "schema_version": 5 if version else 3,
        "num_episodes": 2,
        "episodes": episodes,
        "task_version": version,
        "benchmark_revision": "baseline-v1-2026-09-08",
    }
    if version:
        payload["action_layout"] = {
            "dimension": 28,
            "robot_joint_names": [str(i) for i in range(28)],
        }
        payload["reset_attempts"] = [{"seed": 7, "success": True}]
    name = f"demonstrations/v{version}/articulation/{task}/demos.pkl"
    path = tmp_path / "downloads" / name
    path.parent.mkdir(parents=True)
    path.write_bytes(pickle.dumps(payload))
    request = {
        "revision": "a" * 40,
        "datasets_root": str(tmp_path / "datasets"),
        "download_root": str(tmp_path / "downloads"),
        "created_at": "2026-09-14T00:00:00Z",
        "license": "cc-by-4.0",
        "attribution": "DexVerse",
        "simulation_replay_verified": False,
    }
    entry = {
        "task": task,
        "episodes": 2,
        "profile": {"task": task},
        "files": [
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "episodes": 2,
            }
        ],
    }
    return request, entry, payload, path


@pytest.mark.parametrize("version", [0, 1])
def test_import_preserves_every_episode_and_is_idempotent(tmp_path, importer, version):
    request, entry, original, _ = make_source(tmp_path, version)
    job = importer.import_task(request, entry)
    assert len(job["recordings"]) == 2
    assert job["profile"]["robot"] == original["robot_type"]
    assert job["archive"]["source_removed"] is True
    for i, name in enumerate(job["recordings"]):
        with (Path(job["archive"]["root"]) / name).open("rb") as stream:
            split = importer.ArrayUnpickler(stream).load()
        assert split["num_episodes"] == 1
        assert split["schema_version"] == original["schema_version"]
        episode = split["episodes"][0]
        assert episode["episode_index"] == i
        np.testing.assert_array_equal(
            episode["actions"], original["episodes"][i]["actions"]
        )
        np.testing.assert_array_equal(
            episode["goal_pose"], original["episodes"][i]["goal_pose"]
        )
        if version:
            assert split["reset_attempts"] == original["reset_attempts"]
            assert split["action_layout"] == original["action_layout"]
    assert importer.import_task(request, entry) == job
    first = Path(job["archive"]["root"]) / job["recordings"][0]
    first.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        importer.import_task(request, entry)


def test_corrupt_source_is_not_published(tmp_path, importer):
    request, entry, _, path = make_source(tmp_path, 0)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="manifest"):
        importer.import_task(request, entry)
    assert not list((tmp_path / "datasets").rglob("import.json"))


def test_unknown_pickle_global_is_rejected(importer):
    import io

    raw = pickle.dumps(Path("/should-not-be-instantiated"))
    with pytest.raises(ValueError, match="Unsupported recording object"):
        importer.ArrayUnpickler(io.BytesIO(raw)).load()


def test_release_names_do_not_confuse_changed_v1_goals():
    assert (
        historical_task("Dexverse-OpenFlatFolder-v0")["name"] == "Open flat folder · v0"
    )
    assert "Close folder" in historical_task("Dexverse-OpenFlatFolder-v1")["name"]
    assert historical_task("unknown") is None
    base = {"repository": "/cluster/repos/v0", "source_revision": "old", "task": "old", "work_root": "/cluster/workstation"}
    v0 = release_profile(base, "Dexverse-OpenFaucet-v0", "/cluster")
    v1 = release_profile(base, "Dexverse-OpenFaucet-v1", "/cluster")
    assert "work_root" not in v0 and "work_root" not in v1
    assert base["work_root"] == "/cluster/workstation"
    assert (
        v0["recording_schema_version"] == 3 and v0["repository"] == base["repository"]
    )
    assert (
        v1["recording_schema_version"] == 5 and v1["repository"] != base["repository"]
    )


@pytest.mark.parametrize("schema", [3, 4, 5])
def test_incomplete_v1_release_is_readable_without_inventing_layout(schema):
    payload = {"format": "dexverse_trajectory", "schema_version": schema,
               "task": "Dexverse-OpenFaucet-v1", "robot_type": "floating_shadow_right"}
    assert validate_recorded_identity(payload) == schema
    with pytest.raises(ValueError):
        validate_identity(payload)
