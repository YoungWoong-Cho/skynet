"""One background coordinator for all app hosts using the same database."""

from __future__ import annotations

import logging
import threading

from .db_backend import lock_key

log = logging.getLogger(__name__)


class BackgroundOwner:
    def __init__(self, database, start_services, stop_services, *, interval=3):
        self.database = database
        self.start_services, self.stop_services = start_services, stop_services
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.state = "stopped"

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run, name="skynet-background-owner", daemon=True
        )
        self.thread.start()

    def _run(self):
        connection = None
        started = False
        try:
            self.state = "standby"
            while not self.stop_event.is_set():
                try:
                    connection = self.database.backend.connect()
                    acquired = connection.execute(
                        "SELECT pg_try_advisory_lock(?)",
                        (lock_key("background-owner"),),
                    ).fetchone()[0]
                except Exception as exc:
                    log.error(
                        "Background ownership unavailable (%s)", type(exc).__name__
                    )
                    acquired = False
                if acquired:
                    break
                if connection:
                    connection.close()
                    connection = None
                self.stop_event.wait(self.interval)
            if self.stop_event.is_set():
                return
            started = True
            self.start_services()
            self.state = "active"
            while not self.stop_event.wait(self.interval):
                if connection:
                    # A broken ownership session cannot silently keep running a
                    # second scheduler. Stop this host's workers on loss.
                    connection.execute("SELECT 1")
        except Exception as exc:
            self.state = "lost"
            log.error("Background coordinator stopped (%s)", type(exc).__name__)
        finally:
            try:
                if started:
                    self.stop_services()
            finally:
                if connection:
                    connection.close()
                if self.state != "lost":
                    self.state = "stopped"

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=30)
