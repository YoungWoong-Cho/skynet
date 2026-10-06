"""Immutable adapter declarations drive Convert, its observations and loader check."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from skynet_app.adapters import builtin_adapter_manifests
from test_policy_exports import fixture_recording_manifest
from skynet_app.dataset_formats import catalog, resolve_adapter
from skynet_app.training_contracts import RECORDING_DATASET_FORMAT, RecordingConversion


def version(manifest, identifier="version-1", number=1):
    return dict(id=identifier, version_number=number, name=manifest.display_name,
                manifest_sha256=str(number) * 64, manifest=manifest.model_dump(mode="json"))


def registry(*versions):
    row = dict(id="adapter-1", name="Adapter", latest_version=versions[-1], versions=list(versions))
    return row, SimpleNamespace(list_adapter_registry=lambda **_: [row], get_adapter=lambda *a, **kw: row)


def test_catalog_and_selection_read_the_same_pinned_declaration():
    first = version(fixture_recording_manifest())
    newer = deepcopy(first)
    newer.update(id="version-2", version_number=2)
    newer["manifest"]["train"]["data_requirements"]["recording_conversion"]["presets"][0]["name"] = "Updated label"
    _, database = registry(first, newer)
    item, = catalog(database)
    assert item["adapter_id"] == "adapter-1"
    assert item["adapter_version_id"] == "version-2"
    assert "capsule_files" not in item
    selected = resolve_adapter(database, "adapter-1", "version-1", "state")
    assert selected["adapter_version_id"] == "version-1"
    assert selected["data_preset_name"] == "Joint states"
    assert selected["observation_requirements"]["streams"] == []
    assert selected["capsule_files"] == first["manifest"]["train"]["capsule_files"]
    assert selected["format"] == RECORDING_DATASET_FORMAT
    rgb = resolve_adapter(database, "adapter-1", "version-1", "rgb")
    assert rgb["observation_requirements"] == first["manifest"]["train"]["data_requirements"]["recording_conversion"]["presets"][1]["observation_requirements"]
    assert rgb["preprocessing"]["image_size"] == [640, 480]


def test_resolver_rejects_missing_versions_presets_and_archived_adapters():
    row, database = registry(version(fixture_recording_manifest()))
    for identifier, ver, preset, message in [
        ("wrong-adapter", "version-1", "state", "registered adapter"),
        ("adapter-1", "missing", "state", "version is unavailable"),
        ("adapter-1", "version-1", "invented", "declared adapter data preset"),
    ]:
        with pytest.raises(ValueError, match=message):
            resolve_adapter(database, identifier, ver, preset)
    row["archived_at"] = "2026-09-19"
    assert catalog(database) == []
    with pytest.raises(ValueError, match="archived"):
        resolve_adapter(database, "adapter-1", "version-1")


def test_adapters_without_a_converter_remain_explicitly_unavailable():
    declared = version(fixture_recording_manifest())
    del declared["manifest"]["train"]["data_requirements"]["recording_conversion"]
    _, database = registry(declared)
    entry, = catalog(database)
    assert not entry["available"] and entry["data_presets"] == []
    with pytest.raises(ValueError, match="does not declare"):
        resolve_adapter(database, "adapter-1", "version-1")


def test_all_recording_declarations_bind_shared_format_and_freeze_their_loader():
    convertible = []
    for manifest in builtin_adapter_manifests():
        requirements = manifest.train.data_requirements
        conversion = requirements.recording_conversion if requirements else None
        if not conversion:
            continue
        convertible.append(manifest.slug)
        assert conversion.format == RECORDING_DATASET_FORMAT
        bindings = [field.data_binding for field in manifest.train.input_fields if field.data_binding]
        for preset in conversion.presets:
            assert all(RECORDING_DATASET_FORMAT in binding.formats and preset.contract in binding.contracts for binding in bindings)
            assert "adapter-support/" + preset.loader_validation.script in manifest.train.capsule_files
        assert "adapter-support/recording_dataset.py" in manifest.train.capsule_files
    assert set(convertible) == {"xpolicylab-act", "xpolicylab-act-native", "egoverse-act", "egoverse-hpt", "human-policy-hat"}


def test_duplicate_and_undeclared_default_presets_are_rejected():
    declaration = fixture_recording_manifest().train.data_requirements.recording_conversion.model_dump(mode="json")
    declaration["presets"].append(deepcopy(declaration["presets"][0]))
    with pytest.raises(ValidationError, match="unique presets"):
        RecordingConversion.model_validate(declaration)
    declaration["presets"].pop()
    declaration["default_preset"] = "missing"
    with pytest.raises(ValidationError, match="declared default"):
        RecordingConversion.model_validate(declaration)


def test_convert_api_requires_exact_adapter_identity_and_rejects_old_format(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from skynet_app import policy_exports_api
    calls = []
    def create(*args, **kwargs):
        calls.append((args, kwargs))
        return {"id": "conversion-1"}
    monkeypatch.setattr(policy_exports_api, "service", SimpleNamespace(create=create))
    app = FastAPI()
    app.include_router(policy_exports_api.router)
    payload = dict(session_id="recording-1", adapter_id="adapter-1", adapter_version_id="version-1", adapter_data_preset="state", name="Training data")
    with TestClient(app) as client:
        assert client.post("/api/data/exports", json=payload).status_code == 202
        assert calls[0][0] == ("recording-1", "adapter-1", "Training data")
        assert calls[0][1]["adapter_version_id"] == "version-1"
        assert calls[0][1]["adapter_data_preset"] == "state"
        assert client.post("/api/data/exports", json={**payload, "format": "obsolete-recipe"}).status_code == 422
        payload.pop("adapter_version_id")
        assert client.post("/api/data/exports", json=payload).status_code == 422
        assert len(calls) == 1


def test_convert_api_rejects_other_workspaces_private_adapter_before_creation(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from skynet_app import pipeline_api, policy_exports_api
    from skynet_app.database import Database
    from skynet_app.workspaces import CURRENT_WORKSPACE, WorkspaceDirectory

    database = Database(tmp_path / "workspaces.db")
    directory = WorkspaceDirectory(database)
    alice, _ = directory.open("alice@example.com")
    bob, _ = directory.open("bob@example.com")
    private = database.for_workspace(alice["id"]).create_adapter(name="Alice's adapter", manifest={})
    bob_database = database.for_workspace(bob["id"])
    own = bob_database.create_adapter(name="Bob's adapter", manifest={})
    create = Mock(return_value={"id": "conversion-1"})
    monkeypatch.setattr(pipeline_api, "service", SimpleNamespace(database=bob_database))
    monkeypatch.setattr(policy_exports_api, "service", SimpleNamespace(create=create))
    app = FastAPI()
    app.include_router(policy_exports_api.router)
    payload = dict(session_id="recording-1", adapter_id=own["id"],
                   adapter_version_id=own["latest_version"]["id"], name="Training data")
    token = CURRENT_WORKSPACE.set(bob["id"])
    try:
        with TestClient(app) as client:
            assert client.post("/api/data/exports", json=payload).status_code == 202
            create.assert_called_once()
            for hidden_reference in (
                {"adapter_id": private["id"]},
                {"adapter_version_id": private["latest_version"]["id"]},
            ):
                response = client.post("/api/data/exports", json={**payload, **hidden_reference})
                assert response.status_code == 404
                assert response.json()["detail"] == "Record not found in this workspace"
                create.assert_called_once()
    finally:
        CURRENT_WORKSPACE.reset(token)
