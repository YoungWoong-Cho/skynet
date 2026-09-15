"""Committed change hints. Snapshots stay authoritative; reconnect always resyncs."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from .workspaces import COOKIE

log = logging.getLogger(__name__)
CHANNEL = "skynet_changes"
TOPICS = frozenset({"data", "exports", "recordings", "adapters", "settings", "session"})


class Subscription:
    def __init__(self, workspace, loop):
        self.workspace, self.loop = workspace, loop
        self.queue = asyncio.Queue(maxsize=16)
        self.closed = False

    def push(self, event):
        if self.closed:
            return
        if self.queue.full():
            while not self.queue.empty():
                self.queue.get_nowait()
            # Overflow must recover the current state, not lose the last change.
            if event["event"] not in {"unavailable", "revalidate"}:
                event = {"event": "resync", "data": {"v": 1}}
        self.queue.put_nowait(event)


class ChangeFeed:
    """One dedicated LISTEN connection per app process, independent of job ownership."""

    def __init__(self, database, *, reconnect_delay=2, heartbeat=15):
        self.database = database
        self.reconnect_delay, self.heartbeat = reconnect_delay, heartbeat
        self.ready = threading.Event()
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.subscriptions = set()
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stopping.clear()
        self.thread = threading.Thread(target=self._run, name="skynet-changes", daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping.set()
        self._unavailable()
        if self.thread:
            self.thread.join(timeout=12)

    def subscribe(self, workspace):
        subscription = Subscription(workspace, asyncio.get_running_loop())
        # Register before publishing resync, so a commit during the subsequent
        # snapshot read cannot be missed. LISTEN is already committed here.
        with self.lock:
            if not self.ready.is_set():
                raise HTTPException(503, "Live updates are reconnecting")
            self.subscriptions.add(subscription)
            subscription.push({"event": "resync", "data": {"v": 1}})
        return subscription

    def unsubscribe(self, subscription):
        with self.lock:
            subscription.closed = True
            self.subscriptions.discard(subscription)

    def _send(self, event, scope="*"):
        with self.lock:
            for subscription in tuple(self.subscriptions):
                if scope not in {"*", subscription.workspace}:
                    continue
                try:
                    subscription.loop.call_soon_threadsafe(subscription.push, event)
                except RuntimeError:
                    self.subscriptions.discard(subscription)

    def _unavailable(self):
        with self.lock:
            self.ready.clear()
        self._send({"event": "unavailable", "data": {"v": 1}})

    def _notification(self, payload):
        try:
            value = json.loads(payload)
            if not isinstance(value, dict) or value.get("v") != 1 or not isinstance(value.get("scope"), str):
                raise ValueError("Unknown change envelope")
            if not isinstance(value.get("topics"), list) or not all(isinstance(topic, str) for topic in value["topics"]):
                raise ValueError("Invalid change topics")
            topics = set(value["topics"])
            if not topics or not topics <= TOPICS:
                raise ValueError("Unknown change topic")
        except (TypeError, ValueError, KeyError):
            # Unknown messages never acquire a wider data payload or scope.
            self._send({"event": "resync", "data": {"v": 1}})
            return
        if "session" in topics:
            self._send({"event": "revalidate", "data": {"v": 1}}, value["scope"])
            topics.discard("session")
        if topics:
            self._send({"event": "change", "data": {"v": 1, "topics": sorted(topics)}}, value["scope"])

    def _run(self):
        while not self.stopping.is_set():
            connection = None
            try:
                connection = self.database.backend.connect()
                connection.raw.execute("SET statement_timeout = '10s'")
                connection.raw.execute("LISTEN " + CHANNEL)
                with self.lock:
                    self.ready.set()
                last_check = time.monotonic()
                while not self.stopping.is_set():
                    for notification in connection.raw.notifies(timeout=1):
                        self._notification(notification.payload)
                        if self.stopping.is_set():
                            break
                    if time.monotonic() - last_check >= self.heartbeat:
                        connection.raw.execute("SELECT 1")
                        last_check = time.monotonic()
            except Exception as error:
                log.warning("Live update connection interrupted (%s)", type(error).__name__)
            finally:
                self._unavailable()
                if connection is not None:
                    connection.close()
            self.stopping.wait(self.reconnect_delay)


class ChangeResponse(StreamingResponse):
    def __init__(self, *args, cleanup, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            return await super().__call__(scope, receive, send)
        finally:
            self.cleanup()
            await self.body_iterator.aclose()


def change_router(feed, directory):
    router = APIRouter()

    @router.get("/api/changes")
    async def changes(request: Request, expected_workspace: str):
        workspace = getattr(request.state, "workspace", None)
        if not workspace:
            raise HTTPException(401, "Open a workspace first")
        if expected_workspace != workspace["id"]:
            raise HTTPException(409, "The workspace changed. Reload this page.")
        subscription = feed.subscribe(workspace["id"])
        token = request.cookies.get(COOKIE)

        async def stream():
            last_authorized = time.monotonic()
            try:
                while not await request.is_disconnected():
                    try:
                        event = await asyncio.wait_for(subscription.queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        event = None
                    if (event and event["event"] == "revalidate") or time.monotonic() - last_authorized >= 60:
                        current = await run_in_threadpool(directory.resolve, token)
                        if not current or current["id"] != subscription.workspace:
                            yield 'event: unavailable\ndata: {"v":1}\n\n'
                            return
                        last_authorized = time.monotonic()
                        if event and event["event"] == "revalidate":
                            continue
                    if event is None:
                        yield ": heartbeat\n\n"
                    else:
                        yield "event: " + event["event"] + "\ndata: " + json.dumps(event["data"], separators=(",", ":")) + "\n\n"
                        if event["event"] == "unavailable":
                            return
            finally:
                feed.unsubscribe(subscription)

        return ChangeResponse(stream(), cleanup=lambda: feed.unsubscribe(subscription), media_type="text/event-stream", headers={
            "Cache-Control": "private, no-store", "X-Accel-Buffering": "no",
        })

    return router
