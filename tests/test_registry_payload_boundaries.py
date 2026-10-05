"""Registry deletion inspects every receipt without downloading its body."""
import hashlib
import json
from pathlib import Path

import pytest
from test_payload_store import object_db
from test_postgres import pg

from skynet_app.maintenance import Maintenance
from skynet_app.metadata_objects import MetadataObjects
from skynet_app.payload_store import PayloadStore, _CACHE, _REGISTRY_FLAGS
from skynet_app.registry_reference_match import registry_reference_match
from skynet_app.registry_dependencies import inline_consumers


@pytest.mark.parametrize('value,expected', [
    ('wanted-id', False), (['wanted-id'], False),
    ([{'value': 'wanted-id'}], True), ({'values': ['wanted-id']}, True),
    ({'values': [{'no': 'prefix wanted-id suffix'}]}, False),
    ({'deep': [[{'source': {'adapter': 'old-slug'}}]]}, True),
])
def test_single_pass_matcher_preserves_container_semantics(value, expected):
    assert registry_reference_match(value, 'adapter', ['wanted-id'], ['old-slug']) is expected


@pytest.mark.parametrize("kind", ["adapter", "suite"])
def test_inline_specifications_return_only_exact_dependency_flags(pg, kind):
    db, _ = pg
    target = {"id": "wanted-id", "adapter_key": "wanted-key", "name": "cube",
              "seed_key": "old-slug", "manifest_json": "{}"}
    documents = [
        {}, {"nested": [{"arbitrary": "wanted-id"}]},
        {"source": {"adapter": "old-slug"}},
        {"source": {"adapter": "old-slug", "adapter_version_id": "other"}},
        *[{"source": {"adapter": "old-slug", "adapter_id": v}} for v in (None, False, 0, "", [], {}, True, [0], {"x": 0})],
        {"nested": {"evaluation": {"suites": ["cube"]}}},
        {"evaluation": [{"suites": ["cube"]}]},
        {"evaluation": {"suites": {"nested": "cube"}}},
        {"code": "prefix wanted-id suffix", "evaluation": {"suites": "not-cube"}},
    ]
    project = db.create_project("Inline references")
    expected = {}
    for i, body in enumerate(documents):
        body = {**body, "code": "large capsule without references\n" * 10000}
        e = db.create_experiment(project_id=project["id"], name=str(i), requested_spec=body)
        v = db.create_variant(e["latest_revision"]["id"], name="one", parameters={}, resolved_spec=body)
        match = registry_reference_match(body, kind, ["wanted-id", "wanted-key"] if kind == "adapter" else ["wanted-id"], ["old-slug"] if kind == "adapter" else ["cube"])
        expected[e["latest_revision"]["id"]] = expected[v["id"]] = match
    with db.connection() as c:
        original_execute = c.execute
        scans = []
        def count_scans(query, parameters=None):
            if 'AS body FROM' in query:
                scans.append(query)
            return original_execute(query, parameters)
        c.execute = count_scans
        for table in ("experiment_revisions", "variants"):
            result = inline_consumers(c, table, kind, [target])
            assert len(result) == len(documents)
            assert all(row["referenced"] is expected[row["id"]] for row in result)
            assert len(json.dumps(result)) < 10000, "Never return embedded executable code"
        assert len(scans) == 1, 'A revision and its unchanged variant share one immutable reference check'
        inline_consumers(c, 'experiment_revisions', kind, [target])
        assert len(scans) == 1, 'Deletion revalidation must not rescan unchanged large specifications'
    e = db.create_experiment(project_id=project['id'], name='new dependency', requested_spec={'new': {'id': 'wanted-id'}})
    with db.connection() as c:
        results = inline_consumers(c, 'experiment_revisions', kind, [target])
    assert next(row for row in results if row['id'] == e['latest_revision']['id'])['referenced'] is True


@pytest.mark.parametrize("kind", ["adapter", "suite"])
@pytest.mark.parametrize("location", ["stage", "attempt"])
def test_registry_dependency_preserved_without_downloading_execution_body(object_db, monkeypatch, kind, location):
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    target = (db.upsert_seed_adapter(seed_key="policy", name="Policy", manifest={"slug": "policy"})
              if kind == "adapter" else db.register_evaluation_suite(evaluator_adapter="isaac_lab", evaluator_version="1",
                  name="cube", suite_version="1", config={"tasks": ["cube"]}))
    project = db.create_project("References")
    experiment = db.create_experiment(project_id=project["id"], name="Consumer", requested_spec={})
    variant = db.create_variant(experiment["latest_revision"]["id"], name="one", parameters={}, resolved_spec={})
    run = db.create_run(variant["id"], seed=1, adapter_name="other", adapter_version="1", run_directory="/fixture/run")
    document = {"code": "unrelated code\n" * 200000, "deep": [{"pinned_identifier": target["id"]}]}
    stage = db.create_stage(run["id"], stage_type="TRAIN", name="train", resolved_config=document if location == "stage" else {})
    db.create_job_attempt(stage["id"], execution_snapshot_json=document if location == "attempt" else {})
    _CACHE.clear(); _REGISTRY_FLAGS.clear()
    original = db.payload_store.objects._exchange
    wire = []

    def measure(request):
        response = original(request)
        wire.append((request, response))
        # Bodies containing executable capsules must never leave the storage host.
        responses = response if isinstance(response, list) else [response]
        assert all(item.get("size", 0) < 100000 or "content" not in item for item in responses)
        return response

    monkeypatch.setattr(db.payload_store.objects, "_exchange", measure)
    manager = Maintenance(db, object())
    with db.connection() as c:
        _, _, blockers = manager._graph(c, kind, target["id"])
    assert any(b["id"] == run["id"] for b in blockers)
    assert sum(len(json.dumps(response)) for _, response in wire) < 15000
    first_count = len(wire)
    with db.connection() as c:
        manager._graph(c, kind, target["id"])
    assert len(wire) == first_count, "Unchanged receipt checks must reuse immutable flags"


@pytest.mark.parametrize("document,expected", [
    ({"nested": [{"source": {"adapter": "old-slug"}}]}, True),
    ({"source": {"adapter": "old-slug", "adapter_version_id": "other"}}, False),
    ({"nested": {"anything": "wanted-id"}}, True),
    ({"code": "prefix wanted-id suffix", "source": {"adapter": "other"}}, False),
])
def test_remote_matching_has_identical_semantics_and_integrity_checks(object_db, document, expected):
    db, _ = object_db
    objects = MetadataObjects(db)
    content = json.dumps(document).encode(); digest = hashlib.sha256(content).hexdigest()
    path = objects.put_many([content])[digest]
    ref = {"sha256": digest, "path": path, "size": len(content)}
    assert registry_reference_match(document, "adapter", ["wanted-id"], ["old-slug"]) is expected
    assert objects.registry_matches([ref], "adapter", ["wanted-id"], ["old-slug"]) == [expected]
    Path(path).write_text("tampered")
    with pytest.raises(OSError, match="checksum"):
        objects.registry_matches([ref], "adapter", ["wanted-id"], ["old-slug"])
