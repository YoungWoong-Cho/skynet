"""In-process caches bounded by the byte size of what they hold."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from threading import RLock
from typing import Any

# What a put of an entry larger than the whole bound does: "skip" leaves it out
# (any older value for its key is still dropped); "evict_all" inserts it, so the
# eviction loop empties the cache, the new entry last.
OVERSIZE_POLICIES = frozenset({"skip", "evict_all"})


class ByteBoundedCache:
    """Entries bounded by their summed sizes, evicting the oldest entry first.

    The running size changes on insert and eviction, so an insert never re-sums
    every entry. With ``refresh_on_read`` a hit becomes the newest entry (least
    recently used eviction); without it eviction follows insertion order. The
    lock is re-entrant, so a caller may hold it across several calls.
    """

    def __init__(self, max_bytes: int, *, oversize: str, refresh_on_read: bool) -> None:
        if oversize not in OVERSIZE_POLICIES:
            raise ValueError(f"Unknown oversize policy: {oversize}")
        self.max_bytes = max_bytes
        self.oversize = oversize
        self.refresh_on_read = refresh_on_read
        self.entries: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()
        self.size_bytes = 0
        self.lock = RLock()

    def __contains__(self, key: Hashable) -> bool:
        with self.lock:
            return key in self.entries

    def get(self, key: Hashable) -> Any:
        """The cached value, or None."""
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                return None
            if self.refresh_on_read:
                self.entries.move_to_end(key)
            return entry[0]

    def put(self, key: Hashable, value: Any, size: int | None = None) -> None:
        """Cache ``value``, sized ``len(value)`` unless the caller states its size."""
        size = len(value) if size is None else size
        with self.lock:
            previous = self.entries.pop(key, None)
            if previous is not None:
                self.size_bytes -= previous[1]
            if size > self.max_bytes and self.oversize == "skip":
                return
            self.entries[key] = (value, size)
            self.size_bytes += size
            while self.size_bytes > self.max_bytes:
                _, (_, removed) = self.entries.popitem(last=False)
                self.size_bytes -= removed

    def clear(self) -> None:
        with self.lock:
            self.entries.clear()
            self.size_bytes = 0
