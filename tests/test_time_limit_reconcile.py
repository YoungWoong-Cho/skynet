"""Only an attempt's verified interruption receipt (time limit or GPU preflight) permits a retry."""
import json

import pytest
from test_pipeline import FakeCluster, _exit_record, canonical_spec, make_pipeline_service

from skynet_app.database import Database
from skynet_app.gpu_preflight import (
    GPU_MISSING_EXIT_CODE, GPU_MISSING_MESSAGE, GPU_MISSING_REASON, GPU_MISSING_STATE, INTERRUPTION_RECEIPT,
    TIME_LIMIT_EXIT_CODE, TIME_LIMIT_REASON,
)
import skynet_app.pipeline_api as pipeline


@pytest.mark.parametrize('scenario', ['valid', 'missing', 'malformed', 'different_job', 'different_run', 'ordinary_failure', 'cancelled', 'budget_exhausted'])
def test_time_limit_warning_retries_only_the_matching_attempt(tmp_path, monkeypatch, scenario):
    database = Database(tmp_path / 'state.db')
    cluster = FakeCluster()
    service = make_pipeline_service(database, cluster)
    monkeypatch.setattr(pipeline, 'LOCAL_CAPSULE_ROOT', tmp_path / 'capsules')
    payload = canonical_spec()
    if scenario == 'budget_exhausted':
        payload['train']['checkpoint']['max_attempts'] = 1
    experiment = service.create_experiment(payload)
    service.submit_experiment(experiment['id'])
    run_id = experiment['runs'][0]['id']
    run = database.get_run(run_id)
    attempt = run['attempts'][0]
    receipt = dict(schema_version=1, job_id=attempt['slurm_job_id'], run_id=run_id,
                   reason=TIME_LIMIT_REASON, exit_code=TIME_LIMIT_EXIT_CODE)
    if scenario == 'different_job': receipt['job_id'] = 'another-job'
    if scenario == 'different_run': receipt['run_id'] = 'another-run'
    original_read = cluster.read_optional_file
    expected_path = f"{run['run_directory']}/attempts/{attempt['slurm_job_id']}/state/{INTERRUPTION_RECEIPT}"
    def read_optional_file(path, *args, **kwargs):
        if path == expected_path:
            return 'sky1', None if scenario == 'missing' else 'invalid' if scenario == 'malformed' else json.dumps(receipt)
        return original_read(path, *args, **kwargs)
    cluster.read_optional_file = read_optional_file
    cluster.state = 'CANCELLED' if scenario == 'cancelled' else 'FAILED'
    cluster.status_details = {'ExitCode': '1:0' if scenario == 'ordinary_failure' else f'{TIME_LIMIT_EXIT_CODE}:0'}
    service.reconcile()
    updated = database.get_run(run_id)
    first = next(a for a in updated['attempts'] if a['id'] == attempt['id'])
    if scenario == 'valid':
        assert first['status'] == 'TIMEOUT'
        assert first['slurm_state'] == 'FAILED'  # Preserve the actual scheduler result.
        assert cluster.submit_count == 2
        assert len(updated['attempts']) == 2
    else:
        assert cluster.submit_count == 1
        assert len(updated['attempts']) == 1
        assert updated['status'] == ('CANCELLED' if scenario == 'cancelled' else 'FAILED')


@pytest.mark.parametrize('scenario', ['retry', 'exit_record', 'receipt_missing', 'receipt_other_job', 'ordinary_failure', 'cancelled', 'budget_exhausted'])
def test_missing_gpu_preflight_exit_queues_a_new_attempt_until_the_budget_is_spent(tmp_path, monkeypatch, scenario):
    database = Database(tmp_path / 'state.db')
    cluster = FakeCluster()
    service = make_pipeline_service(database, cluster)
    monkeypatch.setattr(pipeline, 'LOCAL_CAPSULE_ROOT', tmp_path / 'capsules')
    payload = canonical_spec()
    if scenario == 'budget_exhausted':
        payload['train']['checkpoint']['max_attempts'] = 1
    experiment = service.create_experiment(payload)
    service.submit_experiment(experiment['id'])
    run_id = experiment['runs'][0]['id']
    run = database.get_run(run_id)
    attempt = run['attempts'][0]
    receipt = dict(schema_version=1, job_id='another-job' if scenario == 'receipt_other_job' else attempt['slurm_job_id'],
                   run_id=run_id, reason=GPU_MISSING_REASON, exit_code=GPU_MISSING_EXIT_CODE)
    original_read = cluster.read_optional_file
    expected_path = f"{run['run_directory']}/attempts/{attempt['slurm_job_id']}/state/{INTERRUPTION_RECEIPT}"
    def read_optional_file(path, *args, **kwargs):
        if path == expected_path:
            return 'sky1', None if scenario == 'receipt_missing' else json.dumps(receipt)
        return original_read(path, *args, **kwargs)
    cluster.read_optional_file = read_optional_file
    exit_code = '1:0' if scenario == 'ordinary_failure' else f'{GPU_MISSING_EXIT_CODE}:0'
    if scenario == 'exit_record':
        # Slurm already forgot the job; the exit record plus the receipt is unambiguous.
        cluster.accounting_error, cluster.forgotten = 'sacct: error: Connection refused', {attempt['slurm_job_id']}
        cluster.exit_records[attempt['slurm_job_id']] = _exit_record(attempt['slurm_job_id'], exit_code=GPU_MISSING_EXIT_CODE)
        service.reconcile()  # one scan that misses a job is not yet proof that Slurm forgot it
    else:
        cluster.state = 'CANCELLED' if scenario == 'cancelled' else 'FAILED'
        cluster.status_details = {'ExitCode': exit_code}
    service.reconcile()
    updated = database.get_run(run_id)
    first = next(a for a in updated['attempts'] if a['id'] == attempt['id'])
    if scenario in ('retry', 'exit_record'):
        assert (first['status'], first['slurm_state'], first['exit_code']) == (GPU_MISSING_STATE, 'FAILED', exit_code)
        assert first['slurm_reason'] == GPU_MISSING_MESSAGE
        assert cluster.submit_count == 2 and len(updated['attempts']) == 2
        queued = next(e for e in updated['events'] if e['event_type'] == 'AUTO_RESUME_QUEUED')
        assert queued['details_json']['reason'] == GPU_MISSING_MESSAGE
    else:
        assert cluster.submit_count == 1 and len(updated['attempts']) == 1
        assert updated['status'] == ('CANCELLED' if scenario == 'cancelled' else 'FAILED')
        if scenario == 'budget_exhausted':
            # The attempt keeps its classification; only the run and stage end FAILED.
            assert first['status'] == GPU_MISSING_STATE and first['slurm_reason'] == GPU_MISSING_MESSAGE
