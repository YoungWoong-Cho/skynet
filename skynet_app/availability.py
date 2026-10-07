"""Keep the local web UI available while the cluster-dependent API starts."""

from __future__ import annotations

import logging
import subprocess

import anyio
from psycopg import InterfaceError, OperationalError
from psycopg.errors import LockNotAvailable
from starlette.responses import JSONResponse

from .cluster_runtime import ClusterError
from .database_endpoint import DatabaseUnreachable


CLUSTER_UNAVAILABLE = (
    "Cannot access the Skynet cluster. "
    "Check your network or VPN connection and try again."
)
DATABASE_UNAVAILABLE = (
    "The Skynet database did not respond. "
    "Check that its service is running on the database host and try again."
)
DATABASE_BUSY = "Skynet is waiting for an earlier database operation to finish. Try again shortly."
STARTING = "Skynet is initializing. Please wait."
DATABASE_CONNECTION_ERRORS = (ConnectionError, OperationalError, InterfaceError)
CONNECTION_ERRORS = (
    *DATABASE_CONNECTION_ERRORS, subprocess.TimeoutExpired, ClusterError,
)
log = logging.getLogger(__name__)


def outage(error) -> tuple[str, str]:
    """Name the dependency that failed; never expose the exception's own text."""
    if isinstance(error, LockNotAvailable):
        # PostgreSQL is reachable: another operation holds a lock this one needs.
        return "database_busy", DATABASE_BUSY
    if isinstance(error, DatabaseUnreachable):
        return "database_unavailable", (
            f"Cannot reach the Skynet database through SSH host {error.host}. "
            f"Check that `ssh {error.host}` works on the machine running Skynet and try again."
        )
    if isinstance(error, (OperationalError, InterfaceError)):
        return "database_unavailable", DATABASE_UNAVAILABLE
    return "cluster_unavailable", CLUSTER_UNAVAILABLE


def unavailable_response(*, code="cluster_unavailable", detail=CLUSTER_UNAVAILABLE, **extra):
    return JSONResponse(
        {"detail": detail, "code": code, **extra},
        status_code=503,
        headers={"Cache-Control": "no-store", "Retry-After": "5"},
    )


class ClusterApplication:
    """Publish a complete API only after initialization succeeds; never replay requests."""

    def __init__(self, loader, *, retry_interval=5):
        self.loader = loader
        self.retry_interval = retry_interval
        self.application = None
        self.owner = None
        self.status, self.detail = "starting", STARTING

    def pending_response(self, **extra):
        return unavailable_response(code=self.status, detail=self.detail, **extra)

    async def connect(self):
        while self.application is None:
            try:
                # Building the services opens the central DB and may wait on SSH. Keep
                # them off the event loop, with exactly one attempt in flight.
                # Default cancellation waits for the worker before shutdown.
                application, owner = await anyio.to_thread.run_sync(self.loader)
            except CONNECTION_ERRORS as error:
                self.status, self.detail = outage(error)
                log.warning(
                    "Skynet startup is waiting (%s): %s Retrying.",
                    type(error).__name__, self.detail,
                )
                await anyio.sleep(self.retry_interval)
                continue
            self.owner = owner
            owner.start()
            self.application = application
            self.status = "ready"

    def stop(self):
        if self.owner is not None:
            self.owner.stop()

    async def __call__(self, scope, receive, send):
        if self.application is not None:
            return await self.application(scope, receive, send)
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 1013})
        await self.pending_response()(scope, receive, send)
