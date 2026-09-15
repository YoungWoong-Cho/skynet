"""Keep the local web UI available while the cluster-dependent API starts."""

from __future__ import annotations

import logging
import subprocess

import anyio
from psycopg import InterfaceError, OperationalError
from psycopg.errors import LockNotAvailable
from starlette.responses import JSONResponse

from .cluster_runtime import ClusterError


CLUSTER_UNAVAILABLE = (
    "Cannot access the Skynet cluster. "
    "Check your network or VPN connection and try again."
)
STARTING = "Skynet is initializing. Please wait."
STARTUP_BUSY = "Skynet is waiting for an earlier database operation to finish. Retrying automatically."
CONNECTION_ERRORS = (
    ConnectionError, OperationalError, InterfaceError,
    subprocess.TimeoutExpired, ClusterError,
)
log = logging.getLogger(__name__)


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
        self.status = "starting"

    def pending_response(self, **extra):
        detail = {
            "starting": STARTING,
            "startup_busy": STARTUP_BUSY,
        }.get(self.status, CLUSTER_UNAVAILABLE)
        return unavailable_response(code=self.status, detail=detail, **extra)

    async def connect(self):
        while self.application is None:
            try:
                # Imports initialize the central DB and may wait on SSH. Keep
                # them off the event loop, with exactly one attempt in flight.
                # Default cancellation waits for the worker before shutdown.
                application, owner = await anyio.to_thread.run_sync(self.loader)
            except LockNotAvailable:
                # PostgreSQL is reachable: another operation holds a lock that
                # initialization needs. This is not a cluster connection outage.
                self.status = "startup_busy"
                log.warning("Skynet startup waiting for an earlier database operation; retrying")
                await anyio.sleep(self.retry_interval)
                continue
            except CONNECTION_ERRORS as error:
                self.status = "cluster_unavailable"
                log.warning(
                    "Skynet cluster unavailable during startup (%s); retrying",
                    type(error).__name__,
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
