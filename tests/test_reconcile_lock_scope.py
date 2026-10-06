"""Concurrency contracts: slow remote reads cannot own all lifecycle writes."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from test_evaluation_placement import evaluation_service, request_for
from test_tracking_reconciliation import submitted


def test_slow_slurm_read_allows_independent_submission(evaluation_service, monkeypatch):
    service, cluster, run, suite = evaluation_service
    service.create_evaluation(request_for(run, suite))
    entered, release = Event(), Event()
    original = cluster.job_statuses
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(15)
        return original(*args, **kwargs)
    monkeypatch.setattr(cluster, 'job_statuses', blocked)
    with ThreadPoolExecutor(max_workers=2) as pool:
        poll = pool.submit(service.reconcile)
        try:
            assert entered.wait(5)
            created = pool.submit(service.create_evaluation, request_for(run, suite)).result(timeout=8)
            assert created['submission']['job_id']
            assert not poll.done()
            assert service.reconcile()['skipped'] == 'already running'
        finally:
            release.set()
        poll.result(timeout=8)
    attempts = [a for a in service.database.get_run(run['id'])['attempts'] if a['stage_id'] == created['stage_id']]
    assert len(attempts) == 1


def test_cancellation_during_slurm_read_is_not_overwritten(tmp_path, monkeypatch):
    db, cluster, service, run_id = submitted(tmp_path, monkeypatch)
    run = db.get_run(run_id); stage = run['stages'][0]; attempt = run['attempts'][0]
    entered, release = Event(), Event()
    original = cluster.job_statuses
    def blocked(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(15)
        return result
    monkeypatch.setattr(cluster, 'job_statuses', blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        poll = pool.submit(service.reconcile)
        try:
            assert entered.wait(5)
            with service._reconcile_lock:
                db.transition_workflow_state(attempt_id=attempt['id'], attempt_updates={'status':'CANCELLED'},
                    stage_id=stage['id'], stage_updates={'status':'CANCELLED'},
                    run_id=run_id, run_updates={'status':'CANCELLED'})
        finally:
            release.set()
        poll.result(timeout=8)
    current = db.get_run(run_id)
    assert current['status'] == current['stages'][0]['status'] == current['attempts'][0]['status'] == 'CANCELLED'


def test_running_poll_does_not_load_full_run_or_experiment(tmp_path, monkeypatch):
    db, cluster, service, run_id = submitted(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError('Lifecycle status poll loaded a full execution payload')
    for name in ('get_run', 'get_experiment', 'list_runs', 'list_evaluations'):
        monkeypatch.setattr(db, name, forbidden)
    assert service.reconcile()['updated'] == 1


def test_tracking_delivery_continues_while_collection_is_blocked(tmp_path, monkeypatch):
    db, cluster, service, run_id = submitted(tmp_path, monkeypatch)
    entered, release = Event(), Event()
    delivered = []
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(15)
    monkeypatch.setattr(service, '_publish_training_progress_tracking', blocked)
    monkeypatch.setattr(service, '_flush_tracking_provider', lambda provider, **kw: delivered.append(provider))
    with ThreadPoolExecutor(max_workers=1) as pool:
        scan = pool.submit(service.reconcile_tracking)
        try:
            assert entered.wait(5)
            assert service.flush_tracking()['ok']
            assert delivered == ['wandb', 'mlflow']
            assert not scan.done()
        finally:
            release.set()
        scan.result(timeout=8)


def test_only_one_tracking_delivery_owner(tmp_path, monkeypatch):
    db, cluster, service, run_id = submitted(tmp_path, monkeypatch)
    entered, release = Event(), Event()
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(15)
    monkeypatch.setattr(service, '_flush_tracking_provider', blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        delivery = pool.submit(service.flush_tracking)
        try:
            assert entered.wait(5)
            assert service.flush_tracking()['skipped'] == 'already running'
        finally:
            release.set()
        delivery.result(timeout=8)
