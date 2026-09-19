"""Shared preparation lineage and adapter handoff regression checks."""
import json
from concurrent.futures import ThreadPoolExecutor
import pytest
from test_policy_exports import setup as setup, create, manifest_path
from skynet_app.adapters import builtin_adapter_manifests
from skynet_app.pipeline_api import PipelineService

def prepared(setup, kind="fixture-rgb"):
    service, session, source = setup
    job = create(service, session["id"], kind, "Hand demonstrations")
    service.prepare(job["id"])
    job = service.get(job["id"])
    assert job["state"] == "READY", job.get("error")
    return service, session, source, job

def test_archiving_one_dataset_keeps_siblings_and_new_conversions_available(setup):
    service, session, _, first = prepared(setup)
    second = create(service, session["id"], "act", "ACT demonstrations")
    service.prepare(second["id"])
    second = service.get(second["id"])
    assert second["state"] == "READY", second.get("error")
    service.database.update_dataset(first["version_id"], archived=True)
    visible = {dataset["id"] for dataset in service.database.list_datasets()}
    assert first["version_id"] not in visible
    assert second["version_id"] in visible
    assert manifest_path(service, first).exists()
    assert manifest_path(service, second).exists()
    assert not service.database.get_data_resource(first["resource_id"])["archived_at"]
    assert service.preparation_options(session["id"])["session"]["eligible"]
    third = create(service, session["id"], "fixture-state", "State demonstrations")
    assert third["state"] == "QUEUED"
    assert third["resource_id"] == first["resource_id"]
    assert third["source_version_id"] == first["source_version_id"]

def test_catalog_reads_do_not_wait_for_dataset_mutations(setup):
    service, session, _, _ = prepared(setup)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with service.lock:
            pending = pool.submit(service.preparation_options, session["id"])
            # The catalog reads committed registry snapshots; deletion holds this
            # mutation lock while cleaning remote files and must not hide the form.
            assert pending.result(timeout=2)["session"]["eligible"]

def test_selection_rejects_duplicates_but_allows_training_without_validation(setup):
    service, session, _ = setup
    with pytest.raises(ValueError, match="unique recording"):
        create(service,
            None,
            "fixture-rgb",
            "Duplicate",
            selections=[dict(session_id=session["id"], indices=[0, 0])],
        )
    single = create(service,
        None, "fixture-rgb", "One episode",
        selections=[dict(session_id=session["id"], indices=[0])],
    )
    assert single["split"]["train"] == [0] and single["split"]["validation"] == []
    all_training = create(service, session["id"], "fixture-rgb", "No validation", validation_percent=0)
    assert all_training["split"]["train"] == [0, 1]
    assert all_training["split"]["validation"] == []

def test_source_group_is_derived_from_the_selected_recording(setup):
    service, session, _ = setup
    resource = service.database.create_data_resource(
        category="dataset",
        provider="collection",
        namespace="datasets",
        source_key="another",
        kind="demonstrations",
        metadata={"session_id": "another"},
    )
    job = create(service, session["id"], "fixture-rgb", "Selected recording")
    group = service.database.get_data_resource(job["resource_id"])
    assert group["id"] != resource["id"]
    assert group["source_key"] == session["id"]
    assert group["metadata"]["session_id"] == session["id"]


def test_conversion_request_rejects_explicit_source_group(setup):
    from pydantic import ValidationError
    from skynet_app.policy_exports_api import ExportRequest

    service, session, _ = setup
    adapter = service.test_adapters["test-recording-inputs"]
    with pytest.raises(ValidationError, match="resource_id"):
        ExportRequest(
            session_id=session["id"], adapter_id=adapter["id"],
            adapter_version_id=adapter["latest_version"]["id"],
            name="Selected recording", resource_id="another-group",
        )

def test_declarative_act_plan_includes_frozen_training_files():
    from test_experiments import make_spec
    from skynet_app.adapters import ManifestAdapter
    from skynet_app.dataset_formats import XPL_REPOSITORY, XPL_COMMIT

    manifest = next(m for m in builtin_adapter_manifests() if m.slug == "xpolicylab-act")
    spec = make_spec(
        source={
            "repository": XPL_REPOSITORY,
            "revision": XPL_COMMIT,
            "adapter": "xpolicylab-act",
        },
        resources={
            "account": "rl2-lab",
            "partition": "rl2-lab",
            "gpu": {"mode": "explicit", "count": 1, "type": "l40s"},
        },
        train={
            "checkpoint":{"auto_resume":False},
            "batch": {"value": 8, "declared_semantics": "per_device"},
            "learning_rate": 0.0001,
            "seed": 42,
        },
        native={
            "config": {
                "dataset_path": "/cluster/prepared/example",
                "dataset_manifest_sha256": "a" * 64,
                "epochs": 1,
            }
        },
    )
    document = spec.model_dump(mode="python")
    PipelineService._apply_training_preset(document, manifest)
    PipelineService._apply_manifest_input_defaults(document, manifest)
    spec = type(spec).model_validate(document)
    plan = ManifestAdapter(manifest).resolve(spec)
    assert not plan.blockers
    assert plan.capsule_files == manifest.train.capsule_files
    assert "def main():" in plan.capsule_files["adapter-support/skynet_act_training.py"]


def test_adapters_share_original_streams_without_copies(setup):
    service, session, source, first = prepared(setup)
    before = {str(p):p.stat().st_mtime_ns for p in source.rglob('*') if p.is_file()}
    second = create(service, session['id'], 'act', 'ACT demonstrations')
    service.prepare(second['id'])
    second = service.get(second['id'])
    assert second['state'] == 'READY', second.get('error')
    assert second['source_version_id'] == first['source_version_id']
    assert second['resource_id'] == first['resource_id']
    assert second['split'] == first['split']
    assert second['name'] == 'ACT demonstrations'
    assert first['name'] == 'Hand demonstrations'
    a=json.loads(manifest_path(service,first).read_text())
    b=json.loads(manifest_path(service,second).read_text())
    assert [episode['streams'] for episode in a['episodes']] == [episode['streams'] for episode in b['episodes']]
    assert before == {str(p):p.stat().st_mtime_ns for p in source.rglob('*') if p.is_file()}
    assert not list(service.root.rglob('*.zip'))
    assert not list(service.root.rglob('*.zarr'))
    for job in (first,second):
        assert [p.name for p in (service.root/job['id']/'output').iterdir()] == ['manifest.json']
