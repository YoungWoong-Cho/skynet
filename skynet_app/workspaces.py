"""Email-only workspace selection for a trusted team; not identity verification."""

from __future__ import annotations

import argparse
from contextvars import ContextVar
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import threading
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.datastructures import MutableHeaders

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

    def claim_legacy(self, email: str) -> None:
        email = normalize_email(email)
        with self.database.transaction() as connection:
            row = connection.execute("SELECT email FROM workspaces WHERE id=?", (LEGACY_WORKSPACE,)).fetchone()
            if row["email"] not in (None, email):
                raise ValueError("The existing workspace already belongs to another email")
            existing = connection.execute("SELECT id FROM workspaces WHERE email=?", (email,)).fetchone()
            if existing and existing["id"] != LEGACY_WORKSPACE:
                raise ValueError("That email already has a workspace; migrate it explicitly before claiming legacy records")
            connection.execute("UPDATE workspaces SET email=? WHERE id=?", (email, LEGACY_WORKSPACE))

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
        owner_file = system_service.database.path.with_name("workspace-owner.json")
        owner_email = os.environ.get("SKYNET_LEGACY_OWNER_EMAIL")
        if not owner_email and owner_file.exists():
            owner_email = json.loads(owner_file.read_text())["email"]
        if owner_email:
            self.directory.claim_legacy(owner_email)
        self._services: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def for_workspace(self, identifier: str):
        with self._lock:
            if identifier not in self._services:
                from .credential_store import KeyringCredentialStore
                from .tracking import SessionCredentialStore

                credential_store = (self.system.credential_store if identifier == LEGACY_WORKSPACE else
                                    KeyringCredentialStore(service_name=f"io.skynet-control.tracking.{identifier}"))
                credentials = self.system.credentials if identifier == LEGACY_WORKSPACE else SessionCredentialStore()
                self._services[identifier] = type(self.system)(
                    Database(self.system.database.path, workspace_id=identifier),
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
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="skynet-workspaces", daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(15):
            with self.system.database.connection() as connection:
                identifiers = [row[0] for row in connection.execute("SELECT id FROM workspaces ORDER BY created_at")]
            # The legacy workspace stays reconciled even before it is claimed.
            for identifier in identifiers:
                if self._stop.is_set():
                    break
                token = CURRENT_WORKSPACE.set(identifier)
                try:
                    self.for_workspace(identifier).reconcile()
                except Exception:
                    logging.getLogger(__name__).exception("Workspace reconciliation failed: %s", identifier)
                finally:
                    CURRENT_WORKSPACE.reset(token)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        with self._lock:
            services = list(self._services.values())
        for service in services:
            service.stop()


class WorkspaceMiddleware:
    def __init__(self, app, *, services: WorkspaceServices):
        self.app, self.services = app, services

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        path = request.url.path
        if not path.startswith("/api/") or path == "/api/health":
            return await self.app(scope, receive, send)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                return await JSONResponse({"detail": "Use this app's own page to make changes"}, status_code=403)(scope, receive, send)
        if path == "/api/workspace/session":
            expected = request.headers.get("x-skynet-workspace")
            current = self.services.directory.resolve(request.cookies.get(COOKIE))
            if expected and expected != (current or {}).get("id"):
                return await JSONResponse({"detail": "The workspace changed in another tab. Reload this page.", "code": "workspace_changed"}, status_code=409)(scope, receive, send)
            return await self.app(scope, receive, send)
        workspace = self.services.directory.resolve(request.cookies.get(COOKIE))
        if workspace is None:
            return await JSONResponse({"detail": "Enter your email to open a workspace", "code": "workspace_required"}, status_code=401)(scope, receive, send)
        expected = request.headers.get("x-skynet-workspace")
        if expected and expected != workspace["id"]:
            return await JSONResponse({"detail": "The workspace changed in another tab. Reload this page.", "code": "workspace_changed"}, status_code=409)(scope, receive, send)
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
            raise HTTPException(403, "Shared adapters are read-only. Clone one into your workspace to change it.")


class EmailRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)


def session_router(directory: WorkspaceDirectory) -> APIRouter:
    router = APIRouter(prefix="/api/workspace", tags=["workspace"])

    @router.get("/session")
    def session(request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        return {"workspace": directory.resolve(request.cookies.get(COOKIE))}

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


def main():
    parser = argparse.ArgumentParser(description="Assign pre-workspace Skynet records to their owner")
    parser.add_argument("--database", required=True)
    parser.add_argument("--legacy-owner", required=True)
    parser.add_argument("--prepare-owner", action="store_true", help="Save the owner for the next app startup without opening or migrating the database")
    args = parser.parse_args()
    if args.prepare_owner:
        owner_file = Path(args.database).expanduser().resolve().with_name("workspace-owner.json")
        email = normalize_email(args.legacy_owner)
        if owner_file.exists() and json.loads(owner_file.read_text()).get("email") != email:
            parser.error("An owner is already configured; update it explicitly")
        owner_file.parent.mkdir(parents=True, exist_ok=True)
        owner_file.write_text(json.dumps({"email": email}) + "\n")
        print("Existing workspace owner saved for the next startup")
        return
    WorkspaceDirectory(Database(args.database)).claim_legacy(args.legacy_owner)
    print("Existing workspace assigned")


if __name__ == "__main__":
    main()
