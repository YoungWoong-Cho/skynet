"""API modules build their database-backed services on first use, never at import."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from skynet_app.lazy_service import LazyService, resolve

ROOT = Path(__file__).resolve().parents[1]
# Nothing listens on port 1, so a connection that slipped past the guard fails fast.
UNREACHABLE_DATABASE = "postgresql://nobody@127.0.0.1:1/none"
# Record and refuse every way an import could reach a database before the script runs.
GUARD = r'''
import subprocess
import psycopg
from skynet_app import database

attempts = []

def refuse(kind):
    def refused(*args, **kwargs):
        attempts.append(kind)
        raise AssertionError(kind + " attempted")
    return refused

psycopg.connect = refuse("database connection")
database.Database.__init__ = refuse("Database()")
subprocess.Popen = refuse("external process")
'''


def run_isolated(script, data_root):
    """Run script in a fresh interpreter whose data root has no database.json."""
    result = subprocess.run(
        [sys.executable, "-c", GUARD + script],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "SKYNET_DATABASE_URL": UNREACHABLE_DATABASE,
            "SKYNET_DATA_ROOT": str(data_root),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_importing_api_modules_opens_no_database(tmp_path):
    output = run_isolated(r'''
import time
started = time.perf_counter()
from skynet_app import collection_api, live_xr_api, main, maintenance_api, pipeline_api, policy_exports_api
assert attempts == [], attempts
print(f"imported in {time.perf_counter() - started:.2f}s")
''', tmp_path)
    assert output.startswith("imported in"), output


def test_building_one_service_builds_only_what_it_needs(tmp_path):
    run_isolated(r'''
from skynet_app import collection, episode_previews, live_xr, live_xr_archive, live_xr_review, live_xr_video, policy_exports

built = []

def record(name, module, attribute):
    class Recorded:
        def __init__(self, *args):
            built.append(name)
            self.args, self.database, self.active = args, name + " database", set()
    setattr(module, attribute, Recorded)

record("collection", collection, "CollectionService")
record("live", live_xr, "LiveXRService")
record("archive", live_xr_archive, "LiveArchiveService")
record("reviews", live_xr_review, "LiveReviewService")
record("episode previews", episode_previews, "EpisodePreviews")
record("videos", live_xr_video, "LiveVideoService")
record("exports", policy_exports, "PolicyExportService")

from skynet_app import collection_api, live_xr_api, pipeline_api, policy_exports_api
from skynet_app.lazy_service import resolve

assert built == []
exports = resolve(policy_exports_api.service)
assert built == ["collection", "live", "archive", "reviews", "episode previews", "videos", "exports"], built
live, reviews = resolve(live_xr_api.service), resolve(live_xr_api.reviews)
assert live.args == ("collection database",)
assert reviews.args == (live,) and exports.args == (reviews,)
assert live.archive is resolve(live_xr_api.archive)
assert not live.archive.busy("session")
resolve(live_xr_api.videos).active.add(("session", 0))
assert live.archive.busy("session")
resolve(collection_api.service), resolve(live_xr_api.episode_previews)
assert len(built) == 7, built
# The pipeline service would have opened Database(); nothing needed it.
assert attempts == [], attempts
''', tmp_path)


class Service:
    def __init__(self):
        self._lock = "service lock"

    def resolve(self, value):
        return "service", value

    def preview(self):
        return "real preview"


def test_concurrent_first_use_builds_the_service_once():
    built = []
    workers = 8
    arrived = threading.Barrier(workers)

    def build():
        built.append(threading.get_ident())
        time.sleep(0.05)  # Keep the build open while the other threads arrive.
        return Service()

    service = LazyService(build)
    assert built == []

    def use(index):
        arrived.wait()
        return resolve(service) if index % 2 else service.preview.__self__

    with ThreadPoolExecutor(workers) as pool:
        instances = list(pool.map(use, range(workers)))
    assert len(built) == 1
    assert all(instance is instances[0] for instance in instances)


def test_attribute_reads_writes_and_deletes_reach_the_instance():
    service = LazyService(Service)
    service.extra = 1
    instance = resolve(service)
    assert instance.extra == 1 and service.extra == 1
    del service.extra
    assert not hasattr(instance, "extra")
    with pytest.raises(AttributeError):
        service.extra
    # The proxy's own state never hides the service's attributes.
    assert service._lock == "service lock"
    assert service.resolve(2) == ("service", 2)
    assert resolve(instance) is instance


def test_monkeypatch_on_the_proxy_patches_the_instance(monkeypatch):
    service = LazyService(Service)
    with monkeypatch.context() as patch:
        patch.setattr(service, "preview", lambda: "patched")
        assert resolve(service).preview() == "patched"
    assert resolve(service).preview() == "real preview"


def test_failed_build_is_retried_on_next_use():
    attempts = []

    def build():
        attempts.append("build")
        if len(attempts) == 1:
            raise ConnectionError("database unavailable")
        return Service()

    service = LazyService(build)
    with pytest.raises(ConnectionError):
        service.preview
    assert service.preview() == "real preview"
    assert attempts == ["build", "build"]
