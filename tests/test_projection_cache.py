"""Large histories stay cached; concurrent browsers share immutable reads."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from skynet_app.payload_store import ImmutableProjectionCache


def test_large_history_and_planning_do_not_evict_each_other():
    cache = ImmutableProjectionCache()
    calls = []

    def read(refs):
        calls.append(len(refs))
        return [{"value": ref} for ref in refs]

    history = {("resources", i): i for i in range(735)}
    planning = {("context", i): i for i in range(945)}
    for _ in range(3):
        assert len(cache.load(history, read)) == 735
        assert len(cache.load(planning, read)) == 945
    assert calls == [735, 945], "Repeated list refreshes must not re-read all execution bodies"


def test_lru_is_byte_bounded_and_mutation_does_not_poison_cache():
    cache = ImmutableProjectionCache(max_bytes=340)
    calls = []

    def read(refs):
        calls.extend(refs)
        return [{"items": [ref]} for ref in refs]

    cache.load({"a": "a", "b": "b"}, read)
    cache.load({"a": "a"}, read)["a"]["items"].append("mutation")
    cache.load({"c": "c"}, read)
    assert cache.size_bytes <= cache.max_bytes
    assert list(cache.entries) == ["a", "c"]
    assert cache.load({"a": "a"}, read)["a"] == {"items": ["a"]}
    cache.load({"b": "b"}, read)
    assert calls == ["a", "b", "c", "b"]
    cache.clear()
    assert cache.size_bytes == 0 and not cache.entries


def test_oversize_value_is_returned_but_not_retained():
    cache = ImmutableProjectionCache(max_bytes=200)
    value = {"big": "x" * 1000}
    assert cache.load({"a": "a"}, lambda refs: [value])["a"] == value
    assert not cache.entries and cache.size_bytes == 0


@pytest.mark.parametrize("failure", [False, True])
def test_concurrent_readers_share_io_and_failure_releases_waiters(failure):
    cache = ImmutableProjectionCache()
    entered, release, joined = Event(), Event(), Event()
    calls = []

    def read(refs):
        calls.append(refs)
        entered.set()
        assert release.wait(5)
        if failure:
            raise ConnectionError("temporary transport failure")
        return [True]

    def second():
        joined.set()
        return cache.load({"same": 1}, read)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(cache.load, {"same": 1}, read)
        assert entered.wait(5)
        b = pool.submit(second)
        assert joined.wait(5)
        release.set()
        for future in (a, b):
            if failure:
                with pytest.raises(ConnectionError):
                    future.result(timeout=5)
            else:
                assert future.result(timeout=5) == {"same": True}
    assert calls == [[1]]
    assert not cache.pending
    if failure:
        assert cache.load({"same": 1}, lambda refs: [False]) == {"same": False}


def test_incomplete_batch_does_not_poison_later_reads():
    cache = ImmutableProjectionCache()
    with pytest.raises(ValueError, match="Incomplete"):
        cache.load({"a": 1, "b": 2}, lambda refs: [True])
    assert not cache.pending and not cache.entries
    assert cache.load({"b": 2}, lambda refs: [False]) == {"b": False}
