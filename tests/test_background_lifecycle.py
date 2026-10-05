"""Background recovery uses real threads and isolated DBs; no cluster jobs."""

from contextlib import contextmanager
import threading
import time
from types import SimpleNamespace

import pytest

from skynet_app.background_owner import stop_background_services
from skynet_app.database import Database
from skynet_app.live_xr_archive import LiveArchiveService
from skynet_app.policy_exports import DATASET_FORMAT, PolicyExportService
from skynet_app.workspaces import CURRENT_WORKSPACE, LEGACY_WORKSPACE, WorkspaceServices


def eventually(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def test_group_stop_signals_every_service_and_attempts_all_cleanup():
    calls = []

    def stop(name):
        calls.append((name, "stop"))
        if name == "first":
            raise ConnectionError("Unavailable database")

    services = [SimpleNamespace(
        request_stop=lambda name=name: calls.append((name, "signal")),
        stop=lambda name=name: stop(name),
    ) for name in ("first", "second", "third")]
    with pytest.raises(ExceptionGroup, match="shutdown is incomplete"):
        stop_background_services(services)
    assert calls == [(name, operation) for operation in ("signal", "stop")
                     for name in ("first", "second", "third")]


def test_policy_monitor_recovers_independent_failures_and_restarts_without_db_lock(tmp_path, monkeypatch):
    database = Database(tmp_path / "policy-db")
    service = PolicyExportService(SimpleNamespace(live=SimpleNamespace(database=database, root=tmp_path)))
    service.monitor_interval = 0.005
    prepared = threading.Event()
    scans = []

    @contextmanager
    def unavailable_lock():
        raise ConnectionError("DB lock cannot be acquired during shutdown")
        yield

    def scan():
        scans.append(True)
        if len(scans) == 1 or prepared.is_set():
            return []
        if len(scans) == 2:
            raise ConnectionError("Transient pending-list read failure")
        return [{"id": "pending", "format": DATASET_FORMAT, "state": "PENDING"}]

    def observation_failure():
        raise ConnectionError("Independent observation job failure")

    monkeypatch.setattr(service, "lock", unavailable_lock())
    monkeypatch.setattr(service, "list", scan)
    monkeypatch.setattr(service.observations, "tick", observation_failure)
    monkeypatch.setattr(service, "_prepare_cluster", lambda _: prepared.set())
    try:
        service.start()
        first_monitor, first_executor = service.monitor_thread, service.executor
        service.start()
        assert service.monitor_thread is first_monitor
        assert prepared.wait(3), "Observation and list failures must not permanently stall pending work"
        service.stop()
        assert not first_monitor.is_alive()
        assert service.executor is None and not service.active
        service.start()
        assert service.monitor_thread is not first_monitor
        assert service.executor is not first_executor
        assert service.executor.submit(lambda: 7).result(timeout=2) == 7
        assert service.monitor_thread.is_alive()
    finally:
        service.stop()


def test_archive_stop_drains_worker_then_restarts_executor_and_monitor(monkeypatch):
    service = LiveArchiveService(SimpleNamespace(cluster=object(), list=lambda: []))
    entered, finished = threading.Event(), threading.Event()

    def archive(identifier, *, cleanup):
        entered.set()
        assert service.stopping.wait(3)
        finished.set()

    monkeypatch.setattr(service, "archive", archive)
    try:
        service.start(enabled=True)
        old_monitor, old_executor = service.monitor, service.executor
        service.start(enabled=True)
        assert service.monitor is old_monitor
        service.dispatch("first")
        assert entered.wait(3)
        service.stop()
        assert finished.is_set() and not service.active
        assert not old_monitor.is_alive() and service.executor is None
        service.start(enabled=True)
        assert service.monitor is not old_monitor and service.monitor.is_alive()
        assert service.executor is not old_executor
        assert service.executor.submit(lambda: 9).result(timeout=2) == 9
    finally:
        service.stop()


def test_workspace_loops_retry_discovery_and_tracking_cannot_block_lifecycle(tmp_path, monkeypatch):
    database = Database(tmp_path / "workspace-db")
    manager = WorkspaceServices(SimpleNamespace(database=database))
    manager.poll_interval = 0.005
    lifecycle, tracking, delivery = [], [], []
    tracking_entered, release_tracking = threading.Event(), threading.Event()
    child_stopped = threading.Event()
    restarts = []

    def reconcile():
        lifecycle.append(CURRENT_WORKSPACE.get())
        if len(lifecycle) == 1:
            raise ConnectionError("One workspace pass failed")

    def reconcile_tracking():
        tracking.append(CURRENT_WORKSPACE.get())
        tracking_entered.set()
        assert release_tracking.wait(3)

    def request_stop():
        child_stopped.set()
        release_tracking.set()

    manager._services[LEGACY_WORKSPACE] = SimpleNamespace(
        prepare_background_restart=lambda: restarts.append(True),
        request_stop=request_stop, stop=request_stop,
        reconcile=reconcile, reconcile_tracking=reconcile_tracking,
        flush_tracking=lambda: delivery.append(CURRENT_WORKSPACE.get()),
    )
    original_connection = database.connection
    failed_threads = set()

    @contextmanager
    def recovering_connection():
        name = threading.current_thread().name
        if name in {"skynet-workspaces", "skynet-workspace-tracking"} and name not in failed_threads:
            failed_threads.add(name)
            raise ConnectionError("Transient discovery outage")
        with original_connection() as connection:
            yield connection

    monkeypatch.setattr(database, "connection", recovering_connection)
    try:
        manager.start()
        first_threads = manager._worker_threads()
        manager.start()
        assert manager._worker_threads() == first_threads
        assert tracking_entered.wait(3)
        eventually(lambda: len(lifecycle) >= 3)
        assert len(tracking) == 1, "The blocked tracking call must run independently of lifecycle"
        eventually(lambda: len(delivery) >= 2)
        assert set(delivery) == {LEGACY_WORKSPACE}
        assert set(lifecycle) == {LEGACY_WORKSPACE} and tracking == [LEGACY_WORKSPACE]
        assert len(failed_threads) == 2
        manager.stop()
        assert child_stopped.is_set() and all(not thread.is_alive() for thread in first_threads)
        manager.start()
        assert all(new is not old for new, old in zip(manager._worker_threads(), first_threads))
        assert len(restarts) == 2
        eventually(lambda: len(tracking) > 1)
    finally:
        release_tracking.set()
        manager.stop()
