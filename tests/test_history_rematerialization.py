"""Saved experiment definitions remain runnable after deleting execution history."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from skynet_app.database import Database
from test_pipeline import FakeCluster, canonical_spec, make_pipeline_service
import skynet_app.pipeline_api as pipeline


@pytest.fixture
def cleaned(tmp_path, monkeypatch):
    database = Database(tmp_path / 'cleaned.db')
    cluster = FakeCluster()
    service = make_pipeline_service(database, cluster)
    monkeypatch.setattr(pipeline, 'LOCAL_CAPSULE_ROOT', tmp_path / 'capsules')
    experiment = service.create_experiment(canonical_spec())
    variants = database.list_variants(experiment['latest_revision']['id'])
    old_ids = [row['id'] for row in experiment['runs']]
    with database.transaction() as c:
        c.execute('DELETE FROM runs')
    return database, cluster, service, experiment, variants, old_ids


def test_reading_keeps_history_empty_then_explicit_submit_materializes_once(cleaned):
    database, cluster, service, experiment, variants, old_ids = cleaned
    assert service.experiment_detail(experiment['id'])['runs'] == []
    assert cluster.submit_count == 0
    submitted = service.submit_experiment(experiment['id'])
    assert len(submitted['submitted']) == 1 and cluster.submit_count == 1
    runs = database.list_runs()
    assert len(runs) == len(variants) == 1
    assert runs[0]['id'] not in old_ids
    assert runs[0]['variant_id'] == variants[0]['id']
    assert runs[0]['seed'] == variants[0]['resolved_spec_json']['train']['seed']
    assert len(database.get_run(runs[0]['id'])['stages']) == 1
    assert database.list_variants(experiment['latest_revision']['id']) == variants
    service.submit_experiment(experiment['id'])
    assert cluster.submit_count == 1
    assert [row['id'] for row in database.list_runs()] == [row['id'] for row in runs]


def test_invalid_preserved_plan_does_not_create_partial_history(cleaned, monkeypatch):
    database, cluster, service, experiment, _, _ = cleaned
    def failed_plan(spec):
        raise ValueError('The selected dataset was removed')
    monkeypatch.setattr(pipeline, 'resolve_adapter_plan', failed_plan)
    with pytest.raises(ValueError, match='dataset was removed'):
        service.submit_experiment(experiment['id'])
    assert database.list_runs() == [] and cluster.submit_count == 0


def test_two_service_instances_restore_one_atomic_graph(cleaned):
    database, cluster, service, experiment, variants, _ = cleaned
    second = pipeline.PipelineService.__new__(pipeline.PipelineService)
    second.database = Database(database.path)
    barrier = threading.Barrier(2)
    def restore(instance):
        barrier.wait(timeout=5)
        instance._materialize_cleaned_revision(experiment['latest_revision'], variants)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(restore, instance) for instance in (service, second)]
        for future in futures:
            future.result(timeout=10)
    assert len(database.list_runs()) == 1
    assert len(database.list_stages(database.list_runs()[0]['id'])) == 1
    assert cluster.submit_count == 0


def test_failed_stage_insert_rolls_back_new_run(cleaned, monkeypatch):
    database, _, service, experiment, variants, _ = cleaned
    original = database._insert
    def fail_stage(connection, table, values):
        if table == 'workflow_stages':
            raise ValueError('synthetic stage failure')
        return original(connection, table, values)
    monkeypatch.setattr(database, '_insert', fail_stage)
    with pytest.raises(ValueError, match='stage failure'):
        service._materialize_cleaned_revision(experiment['latest_revision'], variants)
    assert database.list_runs() == []
    assert database.list_variants(experiment['latest_revision']['id']) == variants
