"""One background coordinator for all app hosts using the same database."""

from __future__ import annotations

import logging
import threading

from .db_backend import lock_key

log = logging.getLogger(__name__)


def stop_background_services(services):
    """Signal every worker before joining any; cleanup must not require the DB."""
    failures = []
    for service in services:
        try:
            service.request_stop()
        except Exception as error:
            failures.append(error)
    for service in services:
        try:
            service.stop()
        except Exception as error:
            failures.append(error)
    if failures:
        raise ExceptionGroup("Background service shutdown is incomplete", failures)


class BackgroundOwner:
    def __init__(self, database, start_services, stop_services, *, interval=3, companions=()):
        self.database = database
        self.start_services, self.stop_services = start_services, stop_services
        self.interval = interval
        self.companions = companions
        self.stop_event = threading.Event()
        self.thread = None
        self.state = "stopped"

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        for companion in self.companions:
            companion.start()
        self.thread = threading.Thread(
            target=self._run, name="skynet-background-owner", daemon=True
        )
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            connection = None
            started = False
            try:
                self.state = "standby"
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
                if acquired and not self.stop_event.is_set():
                    started = True  # A partially started group also needs cleanup.
                    self.start_services()
                    self.state = "active"
                    while not self.stop_event.wait(self.interval):
                        # A lost session stops this host's workers before it may
                        # start another generation under a newly acquired lease.
                        connection.execute("SELECT 1")
            except Exception as exc:
                self.state = "lost"
                log.error("Background coordinator stopped (%s)", type(exc).__name__)
            finally:
                if started:
                    while True:
                        try:
                            self.stop_services()
                            break
                        except Exception as exc:
                            # Never start a second generation after incomplete
                            # cleanup. Retain a healthy lease while draining.
                            self.state = "lost"
                            log.error("Background cleanup incomplete (%s)", type(exc).__name__)
                            threading.Event().wait(self.interval)
                if connection:
                    try:
                        connection.close()
                    except Exception as exc:
                        log.error("Background ownership cleanup failed (%s)", type(exc).__name__)
            self.stop_event.wait(self.interval)
        self.state = "stopped"

    def stop(self):
        self.stop_event.set()
        try:
            if self.thread:
                self.thread.join(timeout=30)
        finally:
            for companion in self.companions:
                companion.stop()
