"""Personal Slack settings and a durable, bounded incoming-webhook dispatcher."""

from __future__ import annotations

import json
import os
import re
from .db_backend import PostgresConnection, Record
from .db_backend import INTEGRITY_ERRORS, DATABASE_ERRORS
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .credential_store import CredentialStore, CredentialStoreError
from .database import Database
from .tracking import backoff_seconds
from .workflow_states import sql_list

# Notification categories in lifecycle order: (category, message verb, connect-message noun).
EVENT_TABLE = (
    ("submitted", "submitted", "submission"),
    ("running", "started running", "running"),
    ("cancelled", "cancelled", "cancellation"),
    ("failed", "failed", "failure"),
    ("completed", "completed", "completion"),
)
EVENTS = tuple(category for category, _, _ in EVENT_TABLE)
EVENT_VERBS = {category: verb for category, verb, _ in EVENT_TABLE}
# A later lifecycle message waits for the earlier ones of the same attempt; the
# terminal categories share one rank because an attempt reaches only one of them.
TERMINAL_RANK = 2
EVENT_RANKS = {category: min(index, TERMINAL_RANK) for index, category in enumerate(EVENTS)}
_RANK_CASE = "(CASE {column} " + " ".join(
    f"WHEN '{category}' THEN {rank}" for category, rank in EVENT_RANKS.items() if rank < TERMINAL_RANK
) + f" ELSE {TERMINAL_RANK} END)"
CONNECT_MESSAGE = ("Skynet connected. You’ll receive job "
                   + ", ".join(noun for _, _, noun in EVENT_TABLE[:-1]) + f" and {EVENT_TABLE[-1][2]}"
                   + " notifications here.")

MAX_ATTEMPTS = 6
HTTP_TIMEOUT_SECONDS = 5
# A 429 without a usable Retry-After waits this long; a stated one is clamped to the cap.
RATE_LIMIT_DEFAULT_RETRY_SECONDS = 30
RATE_LIMIT_MAX_RETRY_SECONDS = 3600
TRANSIENT_RETRY_SECONDS = 15
CREDENTIAL_RETRY_SECONDS = 60
# A sending row stays leased this long before another worker may take it over.
LEASE_SECONDS = 30
BACKOFF_BASE_SECONDS = 15
BACKOFF_CAP_SECONDS = 900
# Outbox statuses still awaiting delivery; a sending row whose lease lapses is claimed again.
QUEUED_STATUSES = frozenset({"pending", "sending"})
_QUEUED = sql_list(QUEUED_STATUSES)


def default_app_url() -> str:
    return os.environ.get("SKYNET_PUBLIC_URL", "")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def validate_webhook(value: str) -> str:
    value = value.strip()
    if not re.fullmatch(
        r"https://hooks\.slack(?:-gov)?\.com/services/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+",
        value,
    ):
        raise ValueError("Enter a Slack incoming webhook URL from hooks.slack.com.")
    return value


def validate_app_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        return ""
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or any(c in value for c in "<>\\\r\n\t ")
    ):
        raise ValueError(
            "App URL must be an HTTP or HTTPS address without credentials, a query or a fragment."
        )
    return value


class DeliveryError(Exception):
    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post_to_slack(webhook: str, payload: dict) -> None:
    request = Request(
        validate_webhook(webhook),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with build_opener(NoRedirect()).open(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            if response.status != 200 or response.read(256).strip() != b"ok":
                raise DeliveryError(
                    "Slack returned an unexpected response. Check the webhook."
                )
    except HTTPError as error:
        if error.code == 429:
            try:
                delay = max(1, min(RATE_LIMIT_MAX_RETRY_SECONDS,
                                   int(error.headers.get("Retry-After", str(RATE_LIMIT_DEFAULT_RETRY_SECONDS)))))
            except (ValueError, TypeError):
                delay = RATE_LIMIT_DEFAULT_RETRY_SECONDS
            raise DeliveryError(
                "Slack rate limit reached; retry scheduled.", retry_after=delay
            ) from None
        if error.code >= 500:
            raise DeliveryError(
                "Slack is temporarily unavailable; retry scheduled.", retry_after=TRANSIENT_RETRY_SECONDS
            ) from None
        raise DeliveryError(
            f"Slack rejected the notification (HTTP {error.code}). Check the webhook and channel permissions."
        ) from None
    except (URLError, TimeoutError, OSError):
        # Never expose exception strings: urllib errors may include the secret URL.
        raise DeliveryError(
            "Could not reach Slack; retry scheduled.", retry_after=TRANSIENT_RETRY_SECONDS
        ) from None


def due_owners(connection: PostgresConnection, timestamp: float) -> list[str]:
    """Workspaces with Slack delivery enabled and a queued message due by ``timestamp``.

    They are the only ones a delivery pass can act on, so idle workspaces cost
    the dispatcher no per-owner transaction.
    """
    return [row[0] for row in connection.execute(
        f"""SELECT s.owner_id FROM slack_notifications s
        WHERE s.enabled=1 AND EXISTS (
            SELECT 1 FROM notification_outbox n
            WHERE n.owner_id=s.owner_id AND n.status IN ({_QUEUED}) AND n.due_at<=?)
        ORDER BY s.owner_id""",
        (timestamp,),
    )]


class SlackNotifications:
    def __init__(
        self,
        database: Database,
        credential_store: CredentialStore,
        *,
        sender: Callable[[str, dict[str, Any]], None] = post_to_slack,
        clock: Callable[[], float] = time.time,
    ):
        self.database = database
        self.credentials = credential_store
        self.sender, self.clock = sender, clock
        self._lock = threading.RLock()

    @property
    def owner(self) -> str:
        # The unscoped coordinator must never send on behalf of another owner.
        if self.database.workspace_id is None:
            raise ValueError("Select a personal workspace for Slack notifications.")
        return self.database.workspace_id

    def _config(self, connection: PostgresConnection) -> dict[str, Any] | None:
        row = connection.execute(
            "SELECT * FROM slack_notifications WHERE owner_id=?", (self.owner,)
        ).fetchone()
        return dict(row) if row else None

    def settings(self) -> dict[str, Any]:
        with self.database.connection() as connection:
            config = self._config(connection)
            counts = dict(
                connection.execute(
                    "SELECT status,count(*) FROM notification_outbox WHERE owner_id=? GROUP BY status",
                    (self.owner,),
                ).fetchall()
            )
        store_error = None
        try:
            configured = self.credentials.load("slack") is not None
        except CredentialStoreError:
            configured = False
            store_error = "The server credential store is unavailable. Unlock it before sending notifications."
        return {
            "configured": configured,
            "enabled": bool(config and config["enabled"]),
            "events": json.loads(config["events_json"]) if config else list(EVENTS),
            "app_url": config["app_url"] if config else default_app_url(),
            "last_error": store_error or (config["last_error"] if config else None),
            "last_sent_at": config["last_sent_at"] if config else None,
            "pending_count": sum(counts.get(status, 0) for status in QUEUED_STATUSES),
            "failed_count": counts.get("failed", 0),
        }

    def configure(
        self,
        *,
        webhook_url: str | None = None,
        enabled: bool = True,
        events: Sequence[str] = EVENTS,
        app_url: str = "",
    ) -> dict[str, Any]:
        if not set(events) <= set(EVENTS) or len(events) != len(set(events)):
            raise ValueError("Choose only the supported job notification events.")
        if enabled and not events:
            raise ValueError("Select at least one job event.")
        _ = self.owner  # Reject unscoped callers before touching a credential store.
        app_url = validate_app_url(app_url)
        webhook = validate_webhook(webhook_url) if webhook_url else None
        with self._lock:
            old = self.credentials.load("slack") if enabled or webhook else None
            if enabled and not webhook and not old:
                raise ValueError(
                    "Add a Slack incoming webhook before enabling notifications."
                )
            if webhook:
                self.credentials.save(
                    "slack",
                    "https://" + urlsplit(webhook).hostname,
                    {"webhook_url": webhook},
                )
            try:
                with self.database.transaction() as connection:
                    # A new destination must not receive a backlog meant for the old
                    # channel. New connections start with future state changes only.
                    if not enabled or (
                        webhook
                        and old
                        and old.credentials.get("webhook_url") != webhook
                    ):
                        connection.execute(
                            f"UPDATE notification_outbox SET status='skipped',lease_token=NULL WHERE owner_id=? AND status IN ({_QUEUED})",
                            (self.owner,),
                        )
                    connection.execute(
                        """INSERT INTO slack_notifications(owner_id,enabled,events_json,app_url,updated_at)
                        VALUES (?,?,?,?,?) ON CONFLICT(owner_id) DO UPDATE SET enabled=excluded.enabled,
                        events_json=excluded.events_json,app_url=excluded.app_url,last_error=NULL,updated_at=excluded.updated_at""",
                        (
                            self.owner,
                            int(enabled),
                            json.dumps(list(events)),
                            app_url,
                            now_iso(),
                        ),
                    )
                    if events:
                        placeholders = ",".join("?" for _ in events)
                        connection.execute(
                            f"UPDATE notification_outbox SET status='skipped',lease_token=NULL WHERE owner_id=? AND status IN ({_QUEUED}) AND category NOT IN ({placeholders})",
                            (self.owner, *events),
                        )
            except DATABASE_ERRORS:
                if webhook:
                    if old:
                        self.credentials.save("slack", old.endpoint, old.credentials)
                    else:
                        self.credentials.delete("slack")
                raise
        return self.settings()

    def disconnect(self) -> dict[str, Any]:
        _ = self.owner
        with self._lock:
            self.credentials.delete("slack")
            with self.database.transaction() as connection:
                connection.execute(
                    "DELETE FROM slack_notifications WHERE owner_id=?", (self.owner,)
                )
                connection.execute(
                    "DELETE FROM notification_outbox WHERE owner_id=?", (self.owner,)
                )
        return self.settings()

    def connect(self, *, webhook_url: str) -> dict[str, Any]:
        _ = self.owner
        webhook = validate_webhook(webhook_url)
        with self._lock:
            previous = self.credentials.load("slack")
            current = self.settings()
            if (current["enabled"] and previous
                    and previous.credentials.get("webhook_url") == webhook
                    and not current["last_error"]):
                return current
            # Slack incoming webhooks have no read-only validation endpoint.
            # Connect sends this confirmation before saving or enabling delivery.
            self.sender(webhook, {"text": CONNECT_MESSAGE, "mrkdwn": False})
            return self.configure(webhook_url=webhook, enabled=True, events=EVENTS, app_url=default_app_url())

    def _payload(self, connection, item, config):
        row = connection.execute(
            """SELECT s.stage_type,r.id AS run_id,r.adapter_name,
            e.name AS experiment_name,a.slurm_job_id,a.attempt_number,a.slurm_state,a.exit_code,
            (SELECT id FROM evaluations WHERE stage_id=s.id LIMIT 1) AS evaluation_id
            FROM workflow_stages s JOIN runs r ON r.id=s.run_id
            JOIN variants v ON v.id=r.variant_id JOIN experiment_revisions er ON er.id=v.experiment_revision_id
            JOIN experiments e ON e.id=er.experiment_id
            LEFT JOIN job_attempts a ON a.id=? AND a.stage_id=s.id
            WHERE s.id=? AND r.owner_id=?""",
            (item["attempt_key"], item["stage_id"], self.owner),
        ).fetchone()
        if row is None or not row["slurm_job_id"] or item["category"] not in EVENTS:
            return None
        kind = (
            "Evaluation"
            if row["stage_type"] == "EVALUATE"
            else "Training"
            if row["stage_type"] == "TRAIN"
            else row["stage_type"].replace("_", " ").title()
        )
        title = f"{kind} {EVENT_VERBS[item['category']]}: {row['experiment_name']}"[:250]
        description = f"Adapter: {row['adapter_name']}\n"
        description += (
            f"Slurm job: {row['slurm_job_id']} · Attempt {row['attempt_number']}"
            if row["slurm_job_id"]
            else "Before cluster submission"
        )
        description += f"\nRecorded: {item['created_at']}"
        if item["job_status"] == "RETRY_PENDING":
            description += "\nAn automatic retry is queued."
        if item["category"] == "failed" and row["slurm_state"]:
            description += f"\nSlurm state: {row['slurm_state']} · Exit code: {row['exit_code'] or 'unknown'}"
        # Plain-text blocks prevent job names from triggering Slack mentions.
        payload = {
            "text": title,
            "mrkdwn": False,
            "unfurl_links": False,
            "unfurl_media": False,
            "blocks": [
                {"type": "section", "text": {"type": "plain_text", "text": title}},
                {
                    "type": "section",
                    "text": {"type": "plain_text", "text": description[:2500]},
                },
            ],
        }
        if config["app_url"]:
            url = config["app_url"] + (
                "/#evaluations" if row["evaluation_id"] else "/#runs"
            )
            payload["blocks"].append(
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Open Skynet"},
                            "url": url,
                        }
                    ],
                }
            )
        return payload

    def deliver_one(self) -> bool:
        """Claim once, send outside PostgreSQL, and persist retries across restarts."""
        with self._lock:
            timestamp = self.clock()
            with self.database.transaction() as connection:
                config = self._config(connection)
                if not config or not config["enabled"]:
                    return False
                row = connection.execute(
                    f"""SELECT n.* FROM notification_outbox n WHERE n.owner_id=? AND n.status IN ({_QUEUED})
                    AND NOT EXISTS (SELECT 1 FROM notification_outbox earlier
                        WHERE earlier.owner_id=n.owner_id AND earlier.stage_id=n.stage_id
                            AND earlier.attempt_key=n.attempt_key AND earlier.status IN ({_QUEUED})
                            AND {_RANK_CASE.format(column='earlier.category')}
                              < {_RANK_CASE.format(column='n.category')})
                    ORDER BY n.id LIMIT 1""",
                    (self.owner,),
                ).fetchone()
                if row is None or row["due_at"] > timestamp:
                    return False
                item = dict(row)
                if item["attempts"] >= MAX_ATTEMPTS:
                    connection.execute(
                        "UPDATE notification_outbox SET status='failed',lease_token=NULL,last_error='Delivery retry limit reached.' WHERE id=? AND owner_id=?",
                        (item["id"], self.owner),
                    )
                    connection.execute(
                        "UPDATE slack_notifications SET last_error='Delivery retry limit reached.' WHERE owner_id=?",
                        (self.owner,),
                    )
                    return True
                token = uuid.uuid4().hex
                payload = self._payload(connection, item, config)
                connection.execute(
                    "UPDATE notification_outbox SET status='sending',lease_token=?,due_at=?,attempts=attempts+1 WHERE id=? AND owner_id=?",
                    (token, timestamp + LEASE_SECONDS, item["id"], self.owner),
                )
            error = None
            try:
                record = self.credentials.load("slack")
                if not record:
                    raise DeliveryError(
                        "Saved Slack webhook is unavailable. Reconnect in Settings.",
                        retry_after=CREDENTIAL_RETRY_SECONDS,
                    )
                if payload:
                    self.sender(record.credentials["webhook_url"], payload)
            except CredentialStoreError:
                error = DeliveryError(
                    "Server credential store is unavailable; retry scheduled.",
                    retry_after=CREDENTIAL_RETRY_SECONDS,
                )
            except DeliveryError as caught:
                error = caught
            except (ValueError, KeyError):
                error = DeliveryError(
                    "Saved Slack settings are invalid. Reconnect in Settings."
                )
            attempts = item["attempts"] + 1
            retry = error and error.retry_after is not None and attempts < MAX_ATTEMPTS
            state = (
                "pending"
                if retry
                else "failed"
                if error
                else "delivered"
                if payload
                else "skipped"
            )
            due = (
                timestamp + max(error.retry_after,
                                backoff_seconds(attempts, base=BACKOFF_BASE_SECONDS, cap=BACKOFF_CAP_SECONDS))
                if retry
                else 0
            )
            with self.database.transaction() as connection:
                changed = connection.execute(
                    """UPDATE notification_outbox SET status=?,due_at=?,last_error=?,delivered_at=?,lease_token=NULL
                    WHERE id=? AND owner_id=? AND lease_token=?""",
                    (
                        state,
                        due,
                        str(error) if error else None,
                        now_iso() if state == "delivered" else None,
                        item["id"],
                        self.owner,
                        token,
                    ),
                ).rowcount
                if changed:
                    connection.execute(
                        "UPDATE slack_notifications SET last_error=?,last_sent_at=COALESCE(?,last_sent_at) WHERE owner_id=?",
                        (
                            str(error) if error else None,
                            now_iso() if state == "delivered" else None,
                            self.owner,
                        ),
                    )
            return True
