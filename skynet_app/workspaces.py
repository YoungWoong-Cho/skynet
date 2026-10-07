"""Email-only workspace selection for a trusted team; not identity verification."""

from __future__ import annotations

from contextvars import ContextVar
import hashlib
import logging
import secrets
import threading
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders

from .availability import DATABASE_CONNECTION_ERRORS, outage, unavailable_response
from .database import Database
from .workspace_schema import LEGACY_WORKSPACE, normalize_email


CURRENT_WORKSPACE: ContextVar[str | None] = ContextVar("skynet_workspace", default=None)
COOKIE = "skynet_workspace_session"
SESSION_SECONDS = 30 * 24 * 60 * 60
RECORD_PARAMETERS = {
    "project_id": "projects", "experiment_id": "experiments",
    "experiment_revision_id": "experiment_revisions", "variant_id": "variants",
    "run_id": "runs", "stage_id": "workflow_stages", "attempt_id": "job_attempts",
    "checkpoint_id": "checkpoints", "evaluation_id": "evaluations",
    "episode_id": "evaluation_episodes", "adapter_id": "adapters",
    "adapter_version_id": "adapters", "resume_checkpoint_id": "checkpoints",
}


class WorkspaceDirectory:
    def __init__(self, database: Database):
        self.database = database

    def open(self, email: str) -> tuple[dict[str, str], str]:
        email = normalize_email(email)
        token = secrets.token_urlsafe(32)
        with self.database.transaction() as connection:
            connection.execute("INSERT INTO workspaces(id,email) VALUES (?,?) ON CONFLICT(email) DO NOTHING", (str(uuid.uuid4()), email))
            row = connection.execute("SELECT id,email FROM workspaces WHERE email=?", (email,)).fetchone()
            connection.execute("DELETE FROM workspace_sessions WHERE expires_at <= ?", (int(time.time()),))
            connection.execute("INSERT INTO workspace_sessions VALUES (?,?,?)", (self._digest(token), row["id"], int(time.time()) + SESSION_SECONDS))
        return dict(row), token

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def resolve(self, token: str | None) -> dict[str, str] | None:
        if not token or len(token) > 128:
            return None
        with self.database.connection() as connection:
            row = connection.execute("""SELECT w.id,w.email FROM workspace_sessions s
                JOIN workspaces w ON w.id=s.workspace_id
                WHERE s.token_hash=? AND s.expires_at>?""", (self._digest(token), int(time.time()))).fetchone()
        return dict(row) if row else None

    def close(self, token: str | None) -> None:
        if token:
            with self.database.transaction() as connection:
                connection.execute("DELETE FROM workspace_sessions WHERE token_hash=?", (self._digest(token),))


class WorkspaceServices:
    """One service per workspace, with one coordinator for background reconciliation."""

    def __init__(self, system_service: Any):
        self.system = system_service
        self.directory = WorkspaceDirectory(system_service.database)
        self._services: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._notification_thread: threading.Thread | None = None
        self._tracking_thread: threading.Thread | None = None
        self._tracking_delivery_thread: threading.Thread | None = None
        self.poll_interval = 15

    def for_workspace(self, identifier: str):
        with self._lock:
            if identifier not in self._services:
                from .credential_store import KeyringCredentialStore
                from .tracking import SessionCredentialStore

                credential_store = (self.system.credential_store if identifier == LEGACY_WORKSPACE else
                                    KeyringCredentialStore(service_name=f"io.skynet-control.tracking.{identifier}"))
                credentials = self.system.credentials if identifier == LEGACY_WORKSPACE else SessionCredentialStore()
                self._services[identifier] = type(self.system)(
                    self.system.database.for_workspace(identifier),
                    cluster=self.system.cluster,
                    credential_store=credential_store, session_credentials=credentials,
                )
            return self._services[identifier]

    def current(self):
        identifier = CURRENT_WORKSPACE.get()
        return self.for_workspace(identifier) if identifier else self.system

    def __getattr__(self, name: str):
        return getattr(self.current(), name)

    def start(self):
        with self._lock:
            alive = any(thread and thread.is_alive() for thread in self._worker_threads())
            if alive:
                if self._stop.is_set():
                    raise RuntimeError("Workspace workers have not stopped")
                return
            for service in self._services.values():
                service.prepare_background_restart()
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="skynet-workspaces", daemon=True)
            self._thread.start()
            self._notification_thread = threading.Thread(target=self._notification_loop, name="skynet-notifications", daemon=True)
            self._notification_thread.start()
            self._tracking_thread = threading.Thread(target=self._tracking_loop, name="skynet-workspace-tracking", daemon=True)
            self._tracking_thread.start()
            self._tracking_delivery_thread = threading.Thread(target=self._tracking_delivery_loop, name="skynet-workspace-tracking-delivery", daemon=True)
            self._tracking_delivery_thread.start()

    def _worker_threads(self):
        return (self._thread, self._notification_thread, self._tracking_thread, self._tracking_delivery_thread)

    def _notification_loop(self):
        # Independent of cluster polling: Slack latency never delays job management.
        while not self._stop.wait(2):
            try:
                with self.system.database.connection() as connection:
                    owners = [
                        row[0] for row in connection.execute(
                            "SELECT owner_id FROM slack_notifications WHERE enabled=1 ORDER BY owner_id"
                        )
                    ]
            except Exception as error:
                logging.getLogger(__name__).error("Slack queue read failed (%s)", type(error).__name__)
                continue
            for owner in owners:
                if self._stop.is_set():
                    break
                try:
                    self.for_workspace(owner).notifications.deliver_one()
                except Exception as error:
                    # Do not log webhook URLs that may appear in transport errors.
                    logging.getLogger(__name__).error("Slack dispatcher failed (%s)", type(error).__name__)

    def _loop(self):
        self._workspace_loop("reconcile")

    def _tracking_loop(self):
        self._workspace_loop("reconcile_tracking")

    def _tracking_delivery_loop(self):
        self._workspace_loop("flush_tracking")

    def _workspace_loop(self, operation):
        while not self._stop.wait(self.poll_interval):
            try:
                with self.system.database.connection() as connection:
                    identifiers = [row[0] for row in connection.execute("SELECT id FROM workspaces ORDER BY created_at")]
            except Exception as error:
                logging.getLogger(__name__).error("Workspace discovery failed (%s)", type(error).__name__)
                continue
            # The legacy workspace stays reconciled even before it is claimed.
            for identifier in identifiers:
                if self._stop.is_set():
                    break
                token = CURRENT_WORKSPACE.set(identifier)
                try:
                    getattr(self.for_workspace(identifier), operation)()
                except Exception:
                    logging.getLogger(__name__).exception("Workspace %s failed: %s", operation, identifier)
                finally:
                    CURRENT_WORKSPACE.reset(token)

    def request_stop(self):
        self._stop.set()
        with self._lock:
            services = list(self._services.values())
        failures = []
        for service in services:
            try:
                service.request_stop()
            except Exception as error:
                failures.append(error)
        if failures:
            raise ExceptionGroup("Workspace stop signals are incomplete", failures)

    def stop(self):
        self.request_stop()
        for thread in self._worker_threads():
            if thread:
                thread.join(timeout=5)
        if any(thread and thread.is_alive() for thread in self._worker_threads()):
            raise RuntimeError("Workspace workers are still stopping")
        with self._lock:
            services = list(self._services.values())
        failures = []
        for service in services:
            try:
                service.stop()
            except Exception as error:
                failures.append(error)
        if failures:
            raise ExceptionGroup("Workspace service shutdown is incomplete", failures)


class WorkspaceMiddleware:
    def __init__(self, app, *, services: WorkspaceServices):
        self.app, self.services = app, services

    async def __call__(self, scope, receive, send):
        started = False

        async def track_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self._dispatch(scope, receive, track_send)
        except DATABASE_CONNECTION_ERRORS as error:
            if started or scope["type"] != "http":
                raise
            code, detail = outage(error)
            logging.getLogger(__name__).error("Workspace request failed (%s): %s", type(error).__name__, detail)
            await unavailable_response(code=code, detail=detail)(scope, receive, send)

    async def _dispatch(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        path = request.url.path
        if not path.startswith("/api/"):
            return await self.app(scope, receive, send)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                return await JSONResponse({"detail": "Use this app's own page to make changes"}, status_code=403)(scope, receive, send)
        if path == "/api/workspace/session":
            expected = request.headers.get("x-skynet-workspace")
            current = await run_in_threadpool(self.services.directory.resolve, request.cookies.get(COOKIE))
            if expected and expected != (current or {}).get("id"):
                return await JSONResponse({"detail": "The workspace changed in another tab. Reload this page.", "code": "workspace_changed"}, status_code=409)(scope, receive, send)
            return await self.app(scope, receive, send)
        workspace = await run_in_threadpool(self.services.directory.resolve, request.cookies.get(COOKIE))
        if workspace is None:
            return await JSONResponse({"detail": "Enter your email to open a workspace", "code": "workspace_required"}, status_code=401)(scope, receive, send)
        expected = request.headers.get("x-skynet-workspace")
        if expected and expected != workspace["id"]:
            return await JSONResponse({"detail": "The workspace changed in another tab. Reload this page.", "code": "workspace_changed"}, status_code=409)(scope, receive, send)
        # Settings and notes live in the central database and never touch the workspace's cluster storage.
        storage_free = path.startswith(("/api/workspace/", "/api/tracking/", "/api/notifications/", "/api/notes"))
        if (request.method not in {"GET", "HEAD", "OPTIONS", "DELETE"}
                and not storage_free and not path.endswith("/cancel")):
            work_root = await run_in_threadpool(
                lambda: self.services.for_workspace(workspace["id"]).storage.work_root
            )
            if work_root is None:
                from .workspace_storage import STORAGE_REQUIRED
                return await JSONResponse(
                    {"detail": STORAGE_REQUIRED, "code": "storage_required"}, status_code=409,
                )(scope, receive, send)
        scope.setdefault("state", {})["workspace"] = workspace
        token = CURRENT_WORKSPACE.set(workspace["id"])

        async def send_private(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["Cache-Control"] = "private, no-store"
            await send(message)

        try:
            await self.app(scope, receive, send_private)
        finally:
            CURRENT_WORKSPACE.reset(token)


async def require_workspace_records(request: Request):
    """Guard nested path, query and body references before handlers execute."""
    if CURRENT_WORKSPACE.get() is None:
        return  # Internal callers and standalone unit-test routers have no session layer.
    from .pipeline_api import service

    database = service.database

    def check(values):
        if isinstance(values, dict):
            for key, value in values.items():
                table = RECORD_PARAMETERS.get(key)
                if table and isinstance(value, str) and value and not database.owns(table, value):
                    raise HTTPException(404, "Record not found in this workspace")
                if isinstance(value, (dict, list)):
                    check(value)
        elif isinstance(values, list):
            for value in values:
                check(value)

    check(dict(request.path_params))
    check(dict(request.query_params))
    if request.method in {"POST", "PUT", "PATCH"} and "application/json" in request.headers.get("content-type", ""):
        try:
            check(await request.json())
        except ValueError:
            pass  # FastAPI returns the normal malformed JSON error.
    adapter_id = request.path_params.get("adapter_id")
    if adapter_id and request.method != "GET" and not request.url.path.endswith(("/clone", "/validate")):
        if not database.owns("adapters", adapter_id, writable=True):
            raise HTTPException(403, "This adapter belongs to another workspace and cannot be changed here.")


class EmailRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)


def session_router(directory: WorkspaceDirectory) -> APIRouter:
    router = APIRouter(prefix="/api/workspace", tags=["workspace"])

    @router.get("/session")
    def session(request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        workspace = directory.resolve(request.cookies.get(COOKIE))
        if workspace:
            from .workspace_storage import WorkspaceStorage
            storage = WorkspaceStorage(directory.database.for_workspace(workspace["id"]))
            workspace["storage_configured"] = storage.work_root is not None
        return {"workspace": workspace}

    @router.post("/session")
    def open_workspace(payload: EmailRequest, request: Request, response: Response):
        try:
            workspace, token = directory.open(payload.email)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        directory.close(request.cookies.get(COOKIE))
        response.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, httponly=True,
                            secure=request.url.scheme == "https", samesite="lax")
        response.headers["Cache-Control"] = "no-store"
        return {"workspace": workspace}

    @router.delete("/session")
    def close_workspace(request: Request, response: Response):
        directory.close(request.cookies.get(COOKIE))
        response.delete_cookie(COOKIE)
        response.headers["Cache-Control"] = "no-store"
        return {"workspace": None}

    return router
