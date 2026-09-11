import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from skynet_app.credential_store import CredentialStoreUnavailable, StoredCredential
from skynet_app.db_backend import INTEGRITY_ERRORS
from skynet_app.database import Database
from skynet_app.slack_api import slack_router
from skynet_app.slack_notifications import (
    DeliveryError,
    SlackNotifications,
    post_to_slack,
    validate_webhook,
)
from skynet_app.workspaces import (
    WorkspaceDirectory,
    WorkspaceMiddleware,
    session_router,
)

WEBHOOK = "https://hooks.slack.com/services/TTEST/BTEST/secret_test_only"


class MemoryCredentials:
    def __init__(self):
        self.values = {}

    def load(self, provider):
        return self.values.get(provider)

    def save(self, provider, endpoint, credentials):
        self.values[provider] = StoredCredential(provider, endpoint, dict(credentials))

    def delete(self, provider):
        self.values.pop(provider, None)


@pytest.fixture
def setup(tmp_path):
    system = Database(tmp_path / "test.db")
    directory = WorkspaceDirectory(system)

    def create(email="alice@example.com"):
        owner, _ = directory.open(email)
        database = Database(system.path, workspace_id=owner["id"])
        sender = Mock()
        service = SlackNotifications(
            database, MemoryCredentials(), sender=sender, clock=lambda: 1000
        )
        return database, service, sender

    return system, directory, create


def graph(database, *, stage_type="TRAIN"):
    project = database.create_project("project")
    experiment = database.create_experiment(
        project_id=project["id"], name="Cube <@everyone>", requested_spec={}
    )
    variant = database.create_variant(
        experiment["latest_revision"]["id"], name="v1", parameters={}, resolved_spec={}
    )
    run = database.create_run(
        variant["id"],
        seed=1,
        adapter_name="egoverse-hpt",
        adapter_version="2",
        run_directory="/cluster/run",
    )
    stage = database.create_stage(run["id"], stage_type=stage_type, name=stage_type)
    attempt = database.create_job_attempt(stage["id"], status="SUBMITTING")
    if stage_type == "EVALUATE":
        database.create_evaluation(
            run["id"],
            stage_id=stage["id"],
            evaluator_adapter="egoverse-hpt",
            evaluator_version="2",
            suite_name="cube",
            suite_version="1",
            tasks=["cube"],
            seeds=[1],
            episodes_per_task=1,
        )
    return run, stage, attempt


def transition(db, stage, attempt, status, **fields):
    return db.transition_workflow_state(
        stage_id=stage["id"],
        stage_updates={"status": status},
        attempt_id=attempt["id"],
        attempt_updates={"status": status, **fields},
    )


def queued(db):
    with db.connection() as connection:
        return [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM notification_outbox WHERE owner_id=? ORDER BY id",
                (db.workspace_id,),
            )
        ]


def test_actual_transitions_all_events_idempotency_and_secret_storage(setup):
    _, _, create = setup
    db, slack, sender = create()
    _, stage, attempt = graph(db)
    slack.configure(webhook_url=WEBHOOK, app_url="http://skynet:8080")
    transition(db, stage, attempt, "SUBMITTED", slurm_job_id="123")
    transition(db, stage, attempt, "RUNNING")
    transition(db, stage, attempt, "RUNNING")
    transition(db, stage, attempt, "CANCELLING")
    assert [r["category"] for r in queued(db)] == ["submitted", "running"]
    transition(db, stage, attempt, "CANCELLED")
    transition(db, stage, attempt, "FAILED")
    transition(db, stage, attempt, "SUCCEEDED")
    assert [r["category"] for r in queued(db)] == [
        "submitted",
        "running",
        "cancelled",
        "failed",
        "completed",
    ]
    for _ in range(5):
        assert slack.deliver_one()
    assert not slack.deliver_one()
    assert sender.call_count == 5
    assert all(row["status"] == "delivered" for row in queued(db))
    assert "secret_test_only" not in json.dumps(slack.settings())
    with db.connection() as c:
        if getattr(c, "dialect", "sqlite") == "postgresql":
            tables = [row[0] for row in c.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")]
            stored = "\n".join(str(dict(row)) for table in tables for row in c.execute('SELECT * FROM "'+table+'"'))
        else:
            stored = "\n".join(c.iterdump())
        assert "secret_test_only" not in stored
    payload = sender.call_args_list[0].args[1]
    assert (
        payload["mrkdwn"] is False
        and payload["blocks"][0]["text"]["type"] == "plain_text"
    )
    assert payload["blocks"][-1]["elements"][0]["url"] == "http://skynet:8080/#runs"


def test_failed_workflow_does_not_report_successful_process_as_complete(setup):
    _, _, create = setup
    db, slack, sender = create()
    slack.configure(webhook_url=WEBHOOK)
    _, stage, attempt = graph(db)
    transition(db, stage, attempt, "SUBMITTED", slurm_job_id="123")
    db.transition_workflow_state(
        stage_id=stage["id"],
        stage_updates={"status": "FAILED"},
        attempt_id=attempt["id"],
        attempt_updates={
            "status": "SUCCEEDED",
            "slurm_state": "COMPLETED",
            "started_at": "2026-09-11T03:00:00Z",
        },
    )
    # Completion was observed without an intermediate poll. The dispatcher still
    # orders submitted -> started -> failed after checkpoint validation failed.
    for _ in range(3):
        slack.deliver_one()
    titles = [call.args[1]["text"] for call in sender.call_args_list]
    assert (
        "submitted" in titles[0]
        and "started running" in titles[1]
        and "failed" in titles[2]
    )
    assert "completed" not in "".join(titles)


@pytest.mark.parametrize("stage_type", ["TRAIN", "EVALUATE"])
def test_only_confirmed_attempts_notify_including_after_restart(setup, stage_type):
    system, _, create = setup
    db, slack, sender = create()
    slack.configure(webhook_url=WEBHOOK)
    _, stage, attempt = graph(db, stage_type=stage_type)
    for status in ["SUBMITTING", "FAILED", "BLOCKED", "CANCELLED"]:
        transition(db, stage, attempt, status, slurm_reason="Submission outcome unknown: SSH operation timed out")
    Database(system.path)
    assert queued(db) == []
    assert not slack.deliver_one()
    sender.assert_not_called()
    transition(db, stage, attempt, "SUBMITTED", slurm_job_id="123")
    transition(db, stage, attempt, "RUNNING")
    transition(db, stage, attempt, "FAILED")
    assert [row["category"] for row in queued(db)] == ["submitted", "running", "failed"]
    for _ in range(3):
        assert slack.deliver_one()
    assert sender.call_count == 3


def test_upgrade_skips_previously_queued_unconfirmed_errors(setup):
    system, _, create = setup
    db, slack, sender = create()
    slack.configure(webhook_url=WEBHOOK)
    _, stage, attempt = graph(db)
    with db.transaction() as c:
        for category in ['submission_unconfirmed', 'failed']:
            c.execute('INSERT INTO notification_outbox(owner_id,stage_id,attempt_key,category,job_status) VALUES(?,?,?,?,?)',
                (db.workspace_id,stage['id'],attempt['id'],category,'SUBMITTING'))
    Database(system.path)
    assert [row['status'] for row in queued(db)] == ['skipped', 'skipped']
    assert not slack.deliver_one()
    sender.assert_not_called()


def test_rollback_disabled_and_no_history_backfill(setup):
    _, _, create = setup
    db, slack, _ = create()
    _, stage, attempt = graph(db)
    transition(db, stage, attempt, "RUNNING", slurm_job_id="123")
    slack.configure(webhook_url=WEBHOOK)
    assert queued(db) == []
    with pytest.raises(RuntimeError), db.transaction() as c:
        c.execute(
            "UPDATE workflow_stages SET status='SUCCEEDED' WHERE id=?", (stage["id"],)
        )
        raise RuntimeError("rollback")
    assert queued(db) == []
    slack.configure(enabled=False)
    transition(db, stage, attempt, "SUCCEEDED")
    assert queued(db) == []


def test_evaluations_and_workspace_ownership(setup):
    _, _, create = setup
    db, a, sender = create()
    other, b, other_sender = create("bob@example.com")
    a.configure(webhook_url=WEBHOOK, app_url="http://skynet:8080")
    b.configure(webhook_url=WEBHOOK.replace("BTEST", "BOTHER"))
    _, stage, attempt = graph(db, stage_type="EVALUATE")
    transition(db, stage, attempt, "SUBMITTED", slurm_job_id="999")
    transition(db, stage, attempt, "RUNNING")
    transition(db, stage, attempt, "SUCCEEDED")
    assert not b.deliver_one() and not other_sender.called
    a.deliver_one()
    payload = sender.call_args.args[1]
    assert payload["text"].startswith("Evaluation submitted")
    assert payload["blocks"][-1]["elements"][0]["url"].endswith("#evaluations")
    with other.transaction() as c, pytest.raises(INTEGRITY_ERRORS):
        c.execute(
            "UPDATE slack_notifications SET enabled=0 WHERE owner_id=?",
            (db.workspace_id,),
        )
    with other.transaction() as c, pytest.raises(INTEGRITY_ERRORS):
        c.execute(
            "DELETE FROM notification_outbox WHERE owner_id=?", (db.workspace_id,)
        )
    a.disconnect()
    assert b.settings()["configured"] and queued(db) == []


def test_retry_survives_restart_and_preserves_fifo(setup):
    _, _, create = setup
    db, a, sender = create()
    a.configure(webhook_url=WEBHOOK)
    _, stage, attempt = graph(db)
    transition(db, stage, attempt, "SUBMITTED", slurm_job_id="123")
    transition(db, stage, attempt, "RUNNING")
    sender.side_effect = DeliveryError("Rate limited", retry_after=60)
    a.deliver_one()
    assert queued(db)[0]["due_at"] == 1060
    restarted = SlackNotifications(
        Database(db.path, workspace_id=db.workspace_id),
        a.credentials,
        sender=sender,
        clock=lambda: 1059,
    )
    assert not restarted.deliver_one()
    assert sender.call_count == 1
    sender.side_effect = None
    restarted.clock = lambda: 1060
    restarted.deliver_one()
    restarted.deliver_one()
    assert [r["status"] for r in queued(db)] == ["delivered", "delivered"]


def test_parallel_dispatchers_share_a_lease(setup):
    _, _, create = setup
    db, a, _ = create()
    a.configure(webhook_url=WEBHOOK)
    _, stage, attempt = graph(db)
    transition(db, stage, attempt, "SUBMITTED", slurm_job_id="123")
    entered = threading.Event()
    release = threading.Event()

    def slow(*args):
        entered.set()
        assert release.wait(5)

    a.sender = slow
    second = SlackNotifications(
        Database(db.path, workspace_id=db.workspace_id),
        a.credentials,
        sender=Mock(),
        clock=lambda: 1000,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(a.deliver_one)
        assert entered.wait(5)
        assert not second.deliver_one()
        release.set()
        assert future.result()
    assert not second.sender.called


def test_retry_attempts_and_webhook_replacement(setup):
    _, _, create = setup
    db, a, sender = create()
    a.configure(webhook_url=WEBHOOK)
    _, stage, attempt = graph(db)
    transition(db, stage, attempt, "SUBMITTED", slurm_job_id="123")
    sender.side_effect = DeliveryError("Offline", retry_after=1)
    for n in range(6):
        a.clock = lambda n=n: 1000 + n * 1000
        assert a.deliver_one()
    assert queued(db)[0]["status"] == "failed"
    assert not a.deliver_one() and sender.call_count == 6
    second = db.create_job_attempt(stage["id"], status="SUBMITTING")
    transition(db, stage, second, "SUBMITTED", slurm_job_id="124")
    a.configure(webhook_url=WEBHOOK.replace("BTEST", "BREPLACEMENT"))
    assert queued(db)[-1]["status"] == "skipped"
    assert not a.deliver_one()


@pytest.mark.parametrize(
    "url",
    [
        "http://hooks.slack.com/services/T/B/secret",
        "https://evil.test/services/T/B/secret",
        "https://hooks.slack.com.evil.test/services/T/B/secret",
        "https://user@hooks.slack.com/services/T/B/secret",
        "https://hooks.slack.com/services/T/B/secret?x=1",
        "https://hooks.slack.com:444/services/T/B/secret",
    ],
)
def test_only_official_secret_webhooks_are_allowed(url):
    with pytest.raises(ValueError) as caught:
        validate_webhook(url)
    assert url not in str(caught.value)


def test_save_does_not_send_and_pause_works_if_keyring_locked(setup):
    _, _, create = setup
    _db, a, sender = create()
    a.configure(webhook_url=WEBHOOK)
    assert not sender.called
    a.credentials.load = Mock(side_effect=CredentialStoreUnavailable("locked"))
    assert a.configure(enabled=False)["enabled"] is False


def test_http_transport_rate_limit_and_no_secret_error(monkeypatch):
    from urllib.error import HTTPError

    transport = Mock()
    monkeypatch.setattr(
        "skynet_app.slack_notifications.build_opener", lambda *args: transport
    )
    transport.open.side_effect = HTTPError(
        WEBHOOK, 429, "rate limited", {"Retry-After": "90"}, None
    )
    with pytest.raises(DeliveryError) as caught:
        post_to_slack(WEBHOOK, {"text": "job"})
    assert caught.value.retry_after == 90 and "secret_test_only" not in str(
        caught.value
    )
    transport.open.side_effect = HTTPError(
        WEBHOOK, 302, "redirect", {"Location": "https://evil.test"}, None
    )
    with pytest.raises(DeliveryError) as caught:
        post_to_slack(WEBHOOK, {"text": "job"})
    assert caught.value.retry_after is None


def test_api_scopes_settings_and_never_returns_webhook(setup, monkeypatch):
    from skynet_app.pipeline_api import PipelineService
    from skynet_app.workspaces import WorkspaceServices

    system, directory, create = setup
    db, a, sender = create()
    other, b, _ = create("bob@example.com")
    services = WorkspaceServices(
        PipelineService(system, cluster=Mock(), credential_store=MemoryCredentials())
    )
    services.for_workspace(db.workspace_id).notifications = a
    services.for_workspace(other.workspace_id).notifications = b
    app = FastAPI()
    app.add_middleware(WorkspaceMiddleware, services=services)
    app.include_router(session_router(directory))
    app.include_router(slack_router(services))
    alice, bob = TestClient(app), TestClient(app)
    assert alice.get("/api/notifications/slack").status_code == 401
    alice.post("/api/workspace/session", json={"email": "alice@example.com"})
    bob.post("/api/workspace/session", json={"email": "bob@example.com"})
    saved = alice.post(
        "/api/notifications/slack/connect", json={"webhook_url": WEBHOOK}
    )
    assert saved.status_code == 200 and saved.json()["configured"]
    assert "secret_test_only" not in saved.text
    assert not bob.get("/api/notifications/slack").json()["configured"]
    assert sender.call_count == 1
    again = alice.post("/api/notifications/slack/connect", json={"webhook_url": WEBHOOK})
    assert again.status_code == 200 and sender.call_count == 1
    invalid = alice.post(
        "/api/notifications/slack/connect",
        json={"webhook_url": "http://evil.test/secret_test_only"},
    )
    assert invalid.status_code == 422 and "secret_test_only" not in invalid.text
    assert alice.post("/api/notifications/slack/test").status_code == 404
    assert alice.post("/api/notifications/slack/retry").status_code == 404
    assert alice.put("/api/notifications/slack", json={}).status_code == 405
    assert bob.post("/api/notifications/slack/connect", json={}).status_code == 422
    assert bob.delete("/api/notifications/slack").status_code == 200
    assert alice.get("/api/notifications/slack").json()["enabled"]
    assert alice.delete("/api/notifications/slack").status_code == 200
    assert not a.settings()["configured"]


def test_job_retry_has_its_own_notifications_and_filters_are_applied(setup):
    _, _, create = setup
    db, a, _ = create()
    a.configure(webhook_url=WEBHOOK, events=["failed", "submitted"])
    _, stage, first = graph(db)
    transition(db, stage, first, "SUBMITTED", slurm_job_id="1")
    transition(db, stage, first, "RUNNING")
    transition(db, stage, first, "RETRY_PENDING")
    second = db.create_job_attempt(stage["id"], status="SUBMITTING")
    transition(db, stage, second, "SUBMITTED", slurm_job_id="2")
    transition(db, stage, second, "SUCCEEDED")
    assert [r["category"] for r in queued(db)] == ["submitted", "failed", "submitted"]
    assert [r["attempt_key"] for r in queued(db)] == [
        first["id"],
        first["id"],
        second["id"],
    ]
    a.configure(events=["failed"])
    assert [r["status"] for r in queued(db)] == ["skipped", "pending", "skipped"]


def test_unscoped_coordinator_cannot_configure_or_send(setup):
    system, _, _ = setup
    store = MemoryCredentials()
    service = SlackNotifications(system, store, sender=Mock())
    with pytest.raises(ValueError):
        service.configure(webhook_url=WEBHOOK)
    assert store.values == {}
    with pytest.raises(ValueError):
        service.deliver_one()


def test_failed_save_keeps_the_original_destination_and_pending_delivery(setup):
    _, _, create = setup
    db, slack, sender = create()
    slack.configure(webhook_url=WEBHOOK)
    _, stage, attempt = graph(db)
    transition(db, stage, attempt, "SUBMITTED", slurm_job_id="123")
    assert slack.settings()["pending_count"] == 1
    with db.transaction() as connection:
        if getattr(connection, "dialect", "sqlite") == "postgresql":
            connection.executescript("""CREATE FUNCTION reject_settings_fn() RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN RAISE EXCEPTION 'write rejected' USING ERRCODE='23514'; END; $$;
                CREATE TRIGGER reject_settings BEFORE UPDATE ON slack_notifications
                FOR EACH ROW EXECUTE FUNCTION reject_settings_fn();""")
        else:
            connection.execute("CREATE TRIGGER reject_settings BEFORE UPDATE ON slack_notifications BEGIN SELECT RAISE(ABORT,'write rejected'); END")
    with pytest.raises(INTEGRITY_ERRORS):
        slack.configure(webhook_url=WEBHOOK.replace("BTEST", "BNEW"))
    assert slack.credentials.load("slack").credentials["webhook_url"] == WEBHOOK
    assert queued(db)[0]["status"] == "pending"
    with db.transaction() as connection:
        connection.execute("DROP TRIGGER reject_settings ON slack_notifications" if db.is_postgres else "DROP TRIGGER reject_settings")
    sender.side_effect = None
    slack.deliver_one()
    assert sender.call_args.args[0] == WEBHOOK
    assert slack.settings()["failed_count"] == 0


def test_connect_validates_before_saving_or_enabling(setup):
    _, _, create = setup
    db, slack, sender = create()
    sender.side_effect = DeliveryError("Webhook revoked")
    with pytest.raises(DeliveryError, match="Webhook revoked"):
        slack.connect(webhook_url=WEBHOOK)
    assert not slack.settings()["configured"]
    assert not slack.settings()["enabled"]
    sender.side_effect = None
    connected = slack.connect(webhook_url=WEBHOOK)
    assert connected["enabled"] and len(connected["events"]) == 5
    assert sender.call_count == 2
    slack.connect(webhook_url=WEBHOOK)
    assert sender.call_count == 2
    slack.disconnect()
    assert not slack.settings()["configured"]
