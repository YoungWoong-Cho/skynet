"""Serialize recording consumers and deletion across application hosts."""

from contextlib import ExitStack, contextmanager
from functools import wraps

from .db_backend import lock_key


def guarded_recording(method):
    @wraps(method)
    def wrapped(self, identifier, *args, **kwargs):
        live = getattr(self, "live", None) or self.reviews.live
        guard = getattr(live, "recording_guard", None)
        # Standalone review readers can be used without a database-backed service.
        if guard is None:
            return method(self, identifier, *args, **kwargs)
        ids = {identifier[0] if isinstance(identifier, tuple) else identifier}
        ids.update(s["session_id"] for s in kwargs.get("selections") or [])
        with ExitStack() as stack:
            for key in sorted(ids):
                stack.enter_context(guard(key))
            return method(self, identifier, *args, **kwargs)

    return wrapped


@contextmanager
def recording_guard(live, identifier):
    # Shared leases permit simultaneous readers/renderers. Deletion alone takes
    # the exclusive counterpart. Reuse a thread's lease for nested calls.
    readers = getattr(live.recording_readers, "ids", set())
    if identifier in readers:
        yield
        return
    with live.database.connection() as c:
        key = lock_key("recording:" + identifier)
        if not c.execute("SELECT pg_try_advisory_lock_shared(?)", (key,)).fetchone()[0]:
            raise ValueError(
                "This recording is being deleted. Try again after deletion finishes."
            )
        try:
            if c.execute(
                "SELECT 1 FROM maintenance_operations WHERE target_kind IN ('recording','recording-file') AND plan_json::jsonb->'records'->'live_xr_sessions' @> jsonb_build_array(?::text)",
                (identifier,),
            ).fetchone():
                raise ValueError(
                    "This recording is being deleted. Finish its pending deletion first."
                )
            live.get(identifier)
            live.recording_readers.ids = readers | {identifier}
            try:
                yield
            finally:
                live.recording_readers.ids = readers
        finally:
            c.execute("SELECT pg_advisory_unlock_shared(?)", (key,))
