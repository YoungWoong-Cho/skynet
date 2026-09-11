"""Only an attempt's verified time-limit receipt permits a warning-triggered retry."""
import json

import pytest
from test_pipeline import FakeCluster, canonical_spec, make_pipeline_service

from skynet_app.database import Database
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
                   reason='time_limit_warning', exit_code=124)
    if scenario == 'different_job': receipt['job_id'] = 'another-job'
    if scenario == 'different_run': receipt['run_id'] = 'another-run'
    original_read = cluster.read_log
    expected_path = f"{run['run_directory']}/attempts/{attempt['slurm_job_id']}/state/interruption.json"
    def read_log(path, *args, **kwargs):
        if path == expected_path and scenario != 'missing':
            return 'sky1', 'invalid' if scenario == 'malformed' else json.dumps(receipt)
        return original_read(path, *args, **kwargs)
    cluster.read_log = read_log
    cluster.state = 'CANCELLED' if scenario == 'cancelled' else 'FAILED'
    cluster.status_details = {'ExitCode': '1:0' if scenario == 'ordinary_failure' else '124:0'}
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
