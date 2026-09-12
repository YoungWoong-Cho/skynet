"""Verify suite selection and the immutable recording boundary, not replayed labels."""
import copy
import hashlib
import importlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from skynet_app.adapters.egoverse_manifest import manifests
from skynet_app.database import Database, canonical_json
from skynet_app.evaluation_contracts import bind_suite_to_dataset
from skynet_app.experiments import get_evaluation_catalog
from skynet_app.recorded_evaluation import recorded_episode_sources
from skynet_app import live_xr_archive


def suite():
    config = next(s.model_dump(mode="json") for s in get_evaluation_catalog() if s.suite == "dexverse_training_episode")
    return {"config_json": config}


def spec(episodes=None):
    return {"data": {"bundle": {"assignments": [{"role": "training_data", "version": {"metadata": {
        "episodes": episodes if episodes is not None else [{"session_id": "session", "source_index": 0, "steps": 2, "sha256": "a" * 64}],
        "split": {"train": [0], "validation": []}, "capture": {"task": "Dexverse-PickCube-v0"},
    }}}]}}}


def test_single_episode_binds_its_own_task_without_creating_validation():
    original = spec()
    bound = bind_suite_to_dataset(suite(), original)["config_json"]
    assert bound["tasks"] == ["Dexverse-PickCube-v0"]
    assert bound["maximum_episodes_per_task"] == 1
    assert original == spec()
    for episodes in ([], [{}, {}]):
        with pytest.raises(ValueError, match="single training episode"):
            bind_suite_to_dataset(suite(), spec(episodes))
    heldout = next(s.model_dump(mode="json") for s in get_evaluation_catalog() if s.suite == "egoverse_held_out")
    with pytest.raises(ValueError, match="no held-out"):
        bind_suite_to_dataset({"config_json": heldout}, original)


def test_pins_the_source_checksum_not_the_latest_recording(tmp_path, monkeypatch):
    database = Database(tmp_path / "test.db")
    archive_manifest = {"schema": "skynet.live-archive/v1", "session_id": "session", "files": [
        {"path": "output/recordings/one.pkl", "sha256": "a" * 64},
        {"path": "output/recordings/two.pkl", "sha256": "b" * 64},
    ]}
    checksum = hashlib.sha256(canonical_json(archive_manifest).encode()).hexdigest()
    root = f"/datasets/raw/dexverse-live/session/{checksum}/output"
    job = {"id": "session", "recordings": ["recordings/one.pkl", "recordings/two.pkl"],
           "archive": {"state": "READY", "gateway": "sky2", "root": root,
                       "manifest": archive_manifest, "manifest_sha256": checksum}}
    with database.connection() as c:
        c.execute("INSERT INTO live_xr_sessions VALUES (?, ?)", ("session", json.dumps(job)))
    monkeypatch.setattr(live_xr_archive, "CLUSTER", SimpleNamespace(paths=SimpleNamespace(datasets="/datasets")))
    cluster = SimpleNamespace(candidates=lambda host: [host], _remote_path=lambda path: path)
    source = recorded_episode_sources(database, cluster, spec())[0]
    assert source["path"] == root + "/recordings/one.pkl"
    assert source["sha256"] == "a" * 64
    bad = spec(); bad["data"]["bundle"]["assignments"][0]["version"]["metadata"]["episodes"][0]["source_index"] = 1
    with pytest.raises(ValueError, match="differs"):
        recorded_episode_sources(database, cluster, bad)
    with database.connection() as c:
        c.execute("DELETE FROM live_xr_sessions")
    with pytest.raises(ValueError, match="no longer registered"):
        recorded_episode_sources(database, cluster, spec())


@pytest.fixture
def portable(tmp_path, monkeypatch):
    manifest = next(m for m in manifests() if m.slug == "egoverse-hpt")
    entry = next(e for e in manifest.evaluations if e.environment == "isaac_lab")
    for name, content in entry.command.capsule_files.items():
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(content)
    monkeypatch.syspath_prepend(str(tmp_path / "adapter-support"))
    for name in ("recorded_scene", "policy_simulator", "egoverse_simulation"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("recorded_scene")


def test_original_state_is_loaded_with_checksum_and_restricted_pickle(portable, tmp_path):
    state = {"articulation": {"robot": {"joint_position": np.array([[1., 2.]])}}}
    payload = {"format": "dexverse_trajectory", "schema_version": 3, "task": "cube", "robot_type": "shadow",
               "num_episodes": 1, "episodes": [{"num_steps": 2, "states": [state, {}, {}]}]}
    path = tmp_path / "episode.pkl"; path.write_bytes(pickle.dumps(payload))
    source = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "steps": 2}
    capture = {"task": "cube", "robot": "shadow"}
    actual = portable.load_initial_state(source, capture)
    np.testing.assert_array_equal(actual["articulation"]["robot"]["joint_position"], [[1., 2.]])
    with pytest.raises(ValueError, match="checksum"):
        portable.load_initial_state({**source, "sha256": "b" * 64}, capture)
    with pytest.raises(ValueError, match="training scene"):
        portable.load_initial_state(source, {**capture, "robot": "other"})
    path.write_bytes(pickle.dumps(Path("must-not-be-deserialized")))
    source["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Unsupported recording object"):
        portable.load_initial_state(source, capture)


def test_native_presets_only_declare_compatible_simulator():
    from test_egoverse_native_algorithms import native_spec
    from skynet_app.adapters import resolve_adapter_evaluation_plan
    for manifest in manifests():
        for model in ("hpt_joints", "hpt_bc_flow_aria") if manifest.slug == "egoverse-hpt" else ("act",) if manifest.slug == "egoverse-act" else ("pi",):
            entry = [e for e in manifest.evaluations if e.environment == "isaac_lab"]
            if manifest.slug == "egoverse-pi":
                assert not entry
                continue
            plan = resolve_adapter_evaluation_plan(native_spec(manifest, model), environment="isaac_lab", suite="dexverse_training_episode", context={}, manifest=manifest)
            if model == "hpt_bc_flow_aria":
                assert not plan.argv
            else:
                assert plan.argv[1].endswith("/egoverse_simulation.py")
                assert "adapter-support/arrays.py" in plan.capsule_files


def test_run_api_keeps_single_episode_simulation_and_explains_empty_results(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from skynet_app import pipeline_api as pipeline
    from skynet_app.tracking import SessionCredentialStore
    from test_egoverse_native_algorithms import native_spec

    manifest = next(m for m in manifests() if m.slug == "egoverse-hpt")
    document = native_spec(manifest, "hpt_joints").model_dump(mode="json", by_alias=True)
    assignment = spec()["data"]["bundle"]["assignments"][0]
    from test_evaluation_compatibility import metadata
    assignment["version"]["metadata"].update(metadata())
    assignment.update(position=0, resource={"provider":"fixture","namespace":"test","name":"one","kind":"dataset"})
    assignment["version"].update(revision="v1",format="egoverse-episodes-zarr/v1",path="/prepared",manifest_sha256="a"*64,status="READY")
    document["data"] = {"bundle": {"id":"one","name":"one","version":"v1","manifest_sha256":"a"*64,"assignments":[assignment]}}
    svc = pipeline.PipelineService(Database(tmp_path / "api.db"), object(),
        credential_store=SimpleNamespace(load=lambda _:None), session_credentials=SessionCredentialStore())
    monkeypatch.setattr(svc.database, "get_run", lambda _: {"resolved_spec_json": document})
    monkeypatch.setattr(pipeline, "service", svc)
    monkeypatch.setattr(pipeline, "recorded_episode_sources", lambda *_: [{"path":"/original.pkl"}])
    app = FastAPI(); app.include_router(pipeline.router)
    try:
        client = TestClient(app)
        response = client.get('/api/evaluation-suites?run_id=run')
        assert response.status_code == 200, response.text
        payload = response.json()
        returned = {s['name']: s for s in payload['suites']}
        assert len(payload['suites']) == len(svc.database.list_evaluation_suites())
        assert returned['dexverse_training_episode']['is_default']
        assert returned['dexverse_training_episode']['config_json']['maximum_episodes_per_task'] == 1
        assert returned['dexverse_recorded']['compatibility']['ready']
        assert returned['egoverse_held_out']['compatibility']['status'] == 'incompatible'
        assert any('no held-out' in s['reason'] for s in payload['unavailable_suites'])
        def missing(*_):
            raise ValueError("Original recording unavailable")
        monkeypatch.setattr(pipeline, "recorded_episode_sources", missing)
        payload = client.get('/api/evaluation-suites?run_id=run').json()
        returned = {s['name']: s for s in payload['suites']}
        assert returned['dexverse_training_episode']['compatibility']['status'] == 'unknown'
        assert returned['dexverse_recorded']['compatibility']['ready']
        assert any(s["reason"] == "Original recording unavailable" for s in payload["unavailable_suites"])
    finally:
        svc.stop()


def test_initial_state_suite_reuses_only_the_identical_simulator_readiness():
    from skynet_app.runtime_readiness import suite_contract_sha256
    base = next(s.model_dump(mode="json") for s in get_evaluation_catalog() if s.suite == "dexverse_recorded")
    bound = bind_suite_to_dataset(suite(), spec())["config_json"]
    assert suite_contract_sha256(bound) == suite_contract_sha256(base)
    for field, value in (("evaluator", "other"), ("tasks", ["DifferentTask"]), ("task_catalog_provenance", {})):
        changed = copy.deepcopy(bound); changed[field] = value
        with pytest.raises(ValueError, match="same simulator"):
            suite_contract_sha256(changed)
    assert "runtime_readiness_suite" not in base
