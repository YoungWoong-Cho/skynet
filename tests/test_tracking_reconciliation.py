"""Optional telemetry never owns scheduler lifecycle reconciliation."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from skynet_app import pipeline_api
from skynet_app.database import Database
from test_pipeline import FakeCluster, canonical_spec, make_pipeline_service, _declare_checkpoint_output


def submitted(tmp_path, monkeypatch):
    db = Database(tmp_path / 'tracking-reconcile.db')
    cluster = FakeCluster()
    service = make_pipeline_service(db, cluster)
    monkeypatch.setattr(pipeline_api, 'LOCAL_CAPSULE_ROOT', tmp_path / 'capsules')
    experiment = service.create_experiment(canonical_spec())
    service.submit_experiment(experiment['id'])
    run_id = experiment['runs'][0]['id']
    _declare_checkpoint_output(db, run_id)
    cluster.log_content = json.dumps({
        'path': f'/coc/flash7/ycho420/jobs/runs/{run_id}/checkpoints/step-1.ckpt',
        'final': True, 'size_bytes': 12345, 'file_count': 1,
        'is_directory': False, 'sha256': 'a' * 64,
    })
    monkeypatch.setattr(service, '_ingest_training_progress', lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(pipeline_api, 'sync_gpu_statistics', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(service, '_flush_tracking_provider', lambda *_args, **_kwargs: {})
    return db, cluster, service, run_id


def test_blocked_tracking_does_not_delay_completed_run_or_checkpoint(tmp_path, monkeypatch):
    db, cluster, service, run_id = submitted(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def blocked(_run_id):
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(service, '_publish_training_progress_tracking', blocked)
    cluster.state = 'RUNNING'
    service.reconcile()
    with ThreadPoolExecutor(max_workers=2) as pool:
        tracking = pool.submit(service.reconcile_tracking)
        try:
            assert entered.wait(5)
            cluster.state = 'COMPLETED'
            result = pool.submit(service.reconcile).result(timeout=5)
            assert result['updated'] == 1
            run = db.get_run(run_id)
            assert run['status'] == run['stages'][0]['status'] == 'SUCCEEDED'
            assert run['checkpoints'][0]['is_selected_for_inference']
            assert db.get_experiment(run['experiment_id'])['status'] == 'SUCCEEDED'
            assert not tracking.done()
        finally:
            release.set()
        tracking.result(timeout=5)


def test_terminal_tracking_is_recovered_after_process_restart(tmp_path, monkeypatch):
    db, cluster, service, run_id = submitted(tmp_path, monkeypatch)
    db.upsert_tracking_binding('wandb', 'run', run_id, remote_id=run_id,
                               remote_url='https://example.invalid/run', status='CONNECTED')
    cluster.state = 'COMPLETED'
    service.reconcile()  # Simulates a crash before any optional finish event exists.
    assert db.get_run(run_id)['status'] == 'SUCCEEDED'
    assert db.list_tracking_bindings('run', run_id)[0]['status'] == 'CONNECTED'

    def recovered_service():
        restarted = make_pipeline_service(Database(db.path), cluster)
        monkeypatch.setattr(restarted, '_repair_missing_active_tracking_bindings', lambda _: 0)
        monkeypatch.setattr(restarted, '_ingest_training_progress', lambda *_args, **_kwargs: 0)
        monkeypatch.setattr(restarted, '_active_tracking_providers', lambda _: [{'provider': 'wandb'}])
        monkeypatch.setattr(restarted, '_flush_tracking_provider', lambda *_args, **_kwargs: {})
        return restarted

    restarted = recovered_service()
    monkeypatch.setattr(restarted, '_sync_tracking_outputs', lambda *_args, **_kwargs: (_ for _ in ()).throw(ConnectionError('temporary outage')))
    restarted.reconcile_tracking()
    assert db.get_run(run_id)['status'] == 'SUCCEEDED'
    assert db.list_tracking_bindings('run', run_id)[0]['status'] == 'CONNECTED'
    again = recovered_service()
    calls = []

    def finish(identifier, *, status, providers):
        calls.append((identifier, status, providers))
        db.upsert_tracking_binding('wandb', 'run', identifier, remote_id=identifier,
                                   remote_url='https://example.invalid/run', status=status,
                                   metadata={'terminal_sync_protocol': pipeline_api.WANDB_TERMINAL_SYNC_PROTOCOL})

    monkeypatch.setattr(again, '_sync_tracking_outputs', finish)
    again.reconcile_tracking()
    again.reconcile_tracking()
    assert calls == [(run_id, 'FINISHED', [{'provider': 'wandb'}])]
    assert db.list_tracking_bindings('run', run_id)[0]['status'] == 'FINISHED'


def test_tracking_workers_do_not_run_concurrently(tmp_path, monkeypatch):
    db, cluster, service, run_id = submitted(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def blocked(_run_id):
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(service, '_publish_training_progress_tracking', blocked)
    other = make_pipeline_service(Database(db.path), cluster)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(service.reconcile_tracking)
        try:
            assert entered.wait(5)
            assert other.reconcile_tracking()['skipped'] == 'already running'
        finally:
            release.set()
        first.result(timeout=5)


def completed_tracking(tmp_path, monkeypatch, *, native=False, initialize=True):
    """Real finalization/journals; only remote W&B and cluster reads are faked."""
    from types import SimpleNamespace
    from skynet_app.experiments import ExperimentSpec
    from skynet_app.tracking import WandBSettings
    from skynet_app.tracking_journal import TrackingJournal
    from test_tracking import FakeWandBBridge

    db, cluster, service, run_id = submitted(tmp_path, monkeypatch)
    cluster.state = 'COMPLETED'
    service.reconcile()
    run = db.get_run(run_id)
    providers = [{'provider': 'wandb', 'project': 'tracking-test'}]
    settings = WandBSettings(api_key='test-key', entity='team', auto_flush=True)
    remote, history, system = {}, [], []
    journal = TrackingJournal(db, run_id)

    def factory(capsule, bridge_settings=None):
        bridge = FakeWandBBridge(capsule, bridge_settings or settings, journal=journal)
        bridge.remote_runs, bridge.history_rows, bridge.system_rows = remote, history, system
        return bridge

    monkeypatch.setattr(service, '_wandb_bridge', factory)
    monkeypatch.setattr(service, '_wandb_settings', lambda _provider=None: settings)
    monkeypatch.setattr(service, '_active_tracking_providers', lambda _spec: providers)
    monkeypatch.setattr(service, '_native_tracking_provider_names', lambda _spec: {'wandb'} if native else set())
    monkeypatch.setattr(service, '_flush_tracking_provider',
                        pipeline_api.PipelineService._flush_tracking_provider.__get__(service))
    db.record_training_progress_sample(
        run_id, run['attempts'][0]['id'], restart_count=0, completed=1000,
        total=1000, source_kind='jsonl', evidence={'metrics': {'train_loss': 0.2}},
        recorded_at='2026-09-20T00:00:00Z',
    )
    if initialize:
        service._start_tracking(
            ExperimentSpec.model_validate(run['resolved_spec_json']), run,
            tmp_path / 'capsules' / run_id, run['attempts'][0]['slurm_job_id'],
            providers=providers,
        )
        assert db.list_tracking_bindings('run', run_id)[0]['status'] == 'CONNECTED'
    else:
        db.upsert_tracking_binding('wandb', 'run', run_id, remote_id=None, remote_url=None, status='QUEUED', metadata={
            'endpoint': settings.base_url, 'entity': 'team', 'project': 'tracking-test',
        })
    return SimpleNamespace(
        db=db, service=service, run_id=run_id, providers=providers,
        bridge=lambda: factory(tmp_path / 'capsules' / run_id), history=history, remote=remote,
    )


def test_native_finalization_drains_metadata_without_writing_history_or_finish(tmp_path, monkeypatch):
    from dataclasses import replace
    fixture = completed_tracking(tmp_path, monkeypatch, native=True)
    bridge = fixture.bridge()
    bridge.settings = replace(bridge.settings, auto_flush=False)
    for index in range(105):
        bridge.set_tags(fixture.run_id, {'metadata_index': index})
    fixture.service._sync_tracking_outputs(fixture.run_id, status='FINISHED')
    assert fixture.db.list_tracking_bindings('run', fixture.run_id)[0]['status'] == 'QUEUED'
    assert bridge.pending_count() > 0
    fixture.service._sync_tracking_outputs(fixture.run_id, status='FINISHED')
    assert fixture.db.list_tracking_bindings('run', fixture.run_id)[0]['status'] == 'FINISHED'
    assert bridge.pending_count() == 0
    assert fixture.history == []
    assert not any(event['operation'] == 'finish_run' for event in bridge._events_unlocked())
    assert fixture.remote[fixture.run_id]['state'] == 'running'
    assert 'metadata_index:104' in fixture.remote[fixture.run_id]['tags']


def test_missing_remote_binding_retries_creation_before_final_training_metrics(tmp_path, monkeypatch):
    fixture = completed_tracking(tmp_path, monkeypatch, initialize=False)
    fixture.service.reconcile_tracking()
    bridge = fixture.bridge()
    assert fixture.db.get_run(fixture.run_id)['status'] == 'SUCCEEDED'
    assert fixture.db.list_tracking_bindings('run', fixture.run_id)[0]['status'] != 'FINISHED'
    assert bridge.binding(fixture.run_id) is not None
    assert fixture.history == []
    assert not any(event['operation'] == 'finish_run' for event in bridge._events_unlocked())
    fixture.service.reconcile_tracking()
    assert fixture.db.list_tracking_bindings('run', fixture.run_id)[0]['status'] == 'FINISHED'
    assert [row['_step'] for row in fixture.history if 'train_loss' in row] == [1000]
    assert len([event for event in bridge._events_unlocked() if event['operation'] == 'finish_run']) == 1
    assert bridge.pending_count() == 0


@pytest.mark.parametrize('failure_phase', ['final_log_read', 'metric_enqueue', 'gpu_read'])
def test_final_tracking_prerequisite_failure_stays_retryable(tmp_path, monkeypatch, failure_phase):
    from skynet_app.cluster_runtime import ClusterError
    fixture = completed_tracking(tmp_path, monkeypatch)
    service = fixture.service
    failures = []

    def fail_read(*_args, **kwargs):
        failures.append(kwargs)
        assert kwargs.get('raise_on_error') is True
        raise ClusterError('temporary final telemetry read failure')

    with monkeypatch.context() as failing:
        if failure_phase == 'final_log_read':
            failing.setattr(service, '_ingest_training_progress', fail_read)
        elif failure_phase == 'gpu_read':
            failing.setattr(pipeline_api, 'sync_gpu_statistics', fail_read)
        else:
            from test_tracking import FakeWandBBridge
            def fail_enqueue(*_args, **_kwargs):
                failures.append({})
                raise OSError('temporary journal persistence failure')
            failing.setattr(FakeWandBBridge, 'log_metrics_batch', fail_enqueue)
        service.reconcile_tracking()
    assert len(failures) == 1
    assert fixture.db.get_run(fixture.run_id)['status'] == 'SUCCEEDED'
    assert fixture.db.list_tracking_bindings('run', fixture.run_id)[0]['status'] != 'FINISHED'
    assert not any(event['operation'] == 'finish_run' for event in fixture.bridge()._events_unlocked())
    service.reconcile_tracking()
    assert fixture.db.list_tracking_bindings('run', fixture.run_id)[0]['status'] == 'FINISHED'
    assert [row['_step'] for row in fixture.history if 'train_loss' in row] == [1000]
    assert fixture.bridge().pending_count() == 0


def test_legacy_finished_binding_gets_real_terminal_receipt_once(tmp_path, monkeypatch):
    fixture = completed_tracking(tmp_path, monkeypatch)
    bridge = fixture.bridge()
    with monkeypatch.context() as old:
        old.setattr(type(bridge), '_post_file_stream', lambda *_args: None)
        fixture.service._sync_tracking_outputs(fixture.run_id, status='FINISHED')
    binding = fixture.db.list_tracking_bindings('run', fixture.run_id)[0]
    fixture.db.upsert_tracking_binding('wandb', 'run', fixture.run_id,
        remote_id=binding['remote_id'], remote_url=binding['remote_url'],
        status='FINISHED', metadata={})
    assert fixture.remote[fixture.run_id]['state'] == 'running'
    bridge = fixture.bridge()
    events = bridge._events_unlocked()
    for event in events:
        key = event.get('payload', {}).get('idempotency_key', '')
        if key.endswith(':finish:file-stream-v1'):
            event['payload']['idempotency_key'] = key.removesuffix(':file-stream-v1')
    bridge._atomic_write(bridge.spool_path, b''.join(json.dumps(e).encode()+b'\n' for e in events))
    before = len(fixture.history)
    fixture.service.reconcile_tracking()
    fixture.service.reconcile_tracking()
    assert fixture.remote[fixture.run_id]['state'] == 'finished'
    assert len(fixture.history) == before
    assert fixture.db.list_tracking_bindings('run', fixture.run_id)[0]['metadata_json']['terminal_sync_protocol'] == pipeline_api.WANDB_TERMINAL_SYNC_PROTOCOL
    assert sum(e['operation'] == 'finish_run' for e in bridge._events_unlocked()) == 2


def test_completion_transport_failure_remains_queued_until_ack(tmp_path, monkeypatch):
    fixture = completed_tracking(tmp_path, monkeypatch)
    bridge = fixture.bridge()
    with monkeypatch.context() as fail:
        fail.setattr(type(bridge), '_post_file_stream', lambda *_args: (_ for _ in ()).throw(ConnectionError('completion unavailable')))
        fixture.service.reconcile_tracking()
    binding = fixture.db.list_tracking_bindings('run', fixture.run_id)[0]
    assert binding['status'] == 'ERROR'
    assert 'completion unavailable' in binding['last_error']
    assert 'terminal_sync_protocol' not in binding['metadata_json']
    assert bridge.pending_count() == 1
    fixture.service.reconcile_tracking()
    assert bridge.pending_count() == 0
    assert fixture.remote[fixture.run_id]['state'] == 'finished'
    assert fixture.db.list_tracking_bindings('run', fixture.run_id)[0]['status'] == 'FINISHED'


def test_evaluation_attempt_never_replaces_training_tracking_identity(tmp_path, monkeypatch):
    fixture = completed_tracking(tmp_path, monkeypatch)
    train_job = fixture.db.get_run(fixture.run_id)['attempts'][0]['slurm_job_id']
    stage = fixture.db.create_stage(fixture.run_id, stage_type='EVALUATE', name='rollout')
    fixture.db.create_job_attempt(stage['id'], status='PENDING', slurm_job_id='999999')
    fixture.service._sync_tracking_outputs(fixture.run_id, status='FINISHED')
    events = fixture.bridge()._events_unlocked()
    final = [event for event in events if event['operation'] == 'finish_run'][-1]
    assert final['payload']['idempotency_key'] == f'final:{fixture.run_id}:{train_job}:FINISHED:finish:file-stream-v1:through:{events[-2]["sequence"]}'
    assert fixture.remote[fixture.run_id]['state'] == 'finished'
    assert f'slurm.job_id:{train_job}' in fixture.remote[fixture.run_id]['tags']
    assert 'slurm.job_id:999999' not in fixture.remote[fixture.run_id]['tags']


def test_late_evaluation_outputs_preserve_terminal_training_tracking(tmp_path, monkeypatch):
    state=completed_tracking(tmp_path,monkeypatch)
    state.service._sync_tracking_outputs(state.run_id,status='FINISHED')
    monkeypatch.setattr(state.service,'_final_tracking_payload',lambda run: ({'evaluation/late/success':.75}, []))
    # Evaluation completion calls the same output path without passing training status.
    state.service._sync_tracking_outputs(state.run_id)
    binding=state.db.list_tracking_bindings('run',state.run_id)[0]
    assert binding['status']=='FINISHED'
    assert state.bridge().binding(state.run_id)['completion_sent']=='finished'
    events=state.bridge()._events_unlocked()
    assert events[-1]['operation']=='finish_run'
    assert events[-2]['operation']=='log_metrics'


def test_legacy_finished_marker_is_reconciled_after_late_stream_data(tmp_path, monkeypatch):
    fixture = completed_tracking(tmp_path, monkeypatch)
    fixture.service.reconcile_tracking()
    bridge = fixture.bridge()
    # Older builds retained FINISHED even when more file-stream data followed it.
    bridge.log_metrics(fixture.run_id, {'late/result': 0.5}, idempotency_key='legacy-late-result')
    bridge.drain_spool()
    fixture.remote[fixture.run_id]['state'] = 'crashed'
    binding = fixture.db.list_tracking_bindings('run', fixture.run_id)[0]
    fixture.db.upsert_tracking_binding(
        'wandb', 'run', fixture.run_id, remote_id=binding['remote_id'],
        remote_url=binding['remote_url'], status='FINISHED',
        metadata={**binding['metadata_json'], 'terminal_sync_protocol': 'file-stream-v1'},
    )
    history_count = len(fixture.history)
    fixture.service.reconcile_tracking()
    assert fixture.remote[fixture.run_id]['state'] == 'finished'
    assert len(fixture.history) == history_count  # Replayed data must not duplicate history.
    events = bridge._events_unlocked()
    assert events[-1]['operation'] == 'finish_run'
    assert ':through:' in events[-1]['payload']['idempotency_key']
    count = len(events)
    fixture.service.reconcile_tracking()
    assert len(bridge._events_unlocked()) == count


def test_final_metrics_published_under_another_job_are_not_a_new_history_row(tmp_path, monkeypatch):
    fixture = completed_tracking(tmp_path, monkeypatch)
    db, service = fixture.db, fixture.service
    monkeypatch.setattr(service, '_final_tracking_payload', lambda run: ({'final/success_rate': 0.5}, []))

    def final_rows():
        return [row for row in fixture.history if 'final/success_rate' in row]

    service._sync_tracking_outputs(fixture.run_id, status='FINISHED', providers=fixture.providers)
    assert len(final_rows()) == 1
    # The run's latest job is now a different one, which changes the key of every final event.
    db.update_job_attempt(db.get_run(fixture.run_id)['attempts'][0]['id'], slurm_job_id='424242')
    service._sync_tracking_outputs(fixture.run_id, status='FINISHED', providers=fixture.providers)
    assert len(final_rows()) == 1
    assert fixture.remote[fixture.run_id]['state'] == 'finished'


def test_retry_that_ended_without_a_launch_boundary_still_finishes_tracking(tmp_path, monkeypatch, caplog):
    from skynet_app.cluster_runtime import ClusterError
    from skynet_app.training_progress_log import BOUNDARY_NOT_READY
    fixture = completed_tracking(tmp_path, monkeypatch)
    db, service = fixture.db, fixture.service
    run = db.get_run(fixture.run_id)
    stage = run['stages'][0]
    stage['resolved_config_json']['plan']['progress'] = {'source': {
        'kind': 'jsonl', 'path': 'artifacts/logs.json.txt', 'completed_key': 'step', 'required_key': 'step'}}
    db.update_stage(stage['id'], resolved_config_json=stage['resolved_config_json'])
    # The retry ran under a launcher that published no receipt and ended long ago.
    db.create_job_attempt(stage['id'], status='SUCCEEDED', slurm_job_id='777', gateway='sky1',
                          finished_at='2026-09-11T20:35:36.000Z')
    reads = []

    def read_log(path, gateway='auto', **kwargs):
        reads.append((path, kwargs['execution_boundary']['required']))
        raise ClusterError(f'sky1: FileNotFoundError: {BOUNDARY_NOT_READY}')

    monkeypatch.setattr(service.cluster, 'read_log', read_log)
    monkeypatch.setattr(service, '_ingest_training_progress',
                        pipeline_api.PipelineService._ingest_training_progress.__get__(service))
    assert service.reconcile_tracking()['synced'] == 1
    binding = db.list_tracking_bindings('run', fixture.run_id)[0]
    assert binding['status'] == 'FINISHED'
    assert binding['metadata_json']['terminal_sync_protocol'] == pipeline_api.WANDB_TERMINAL_SYNC_PROTOCOL
    assert fixture.remote[fixture.run_id]['state'] == 'finished'
    # Nothing is left to retry: the run has left the durable queue.
    assert service.reconcile_tracking()['synced'] == 0
    assert reads == [(f"{run['run_directory']}/artifacts/logs.json.txt", True)]
    assert 'Run tracking reconciliation failed' not in caplog.text


def test_tracking_reconcile_reads_provider_requests_from_projections(tmp_path, monkeypatch):
    from skynet_app.db_backend import PostgresConnection
    db, cluster, service, run_id = submitted(tmp_path, monkeypatch)
    statements = []
    original = PostgresConnection.execute
    def recorded(self, statement, parameters=None):
        statements.append(statement)
        return original(self, statement, parameters)
    monkeypatch.setattr(PostgresConnection, 'execute', recorded)
    service.reconcile_tracking()
    assert any('tracking_providers' in statement for statement in statements)
    assert not any('resolved_spec_json::jsonb' in statement for statement in statements)
