"""Isolated browser QA server. Synthetic data; SSH and all GPU work are disabled.

Run this file, then open port 8091. It creates a disposable database automatically.
This exercises the actual conversion API, artifact download, and registry UI.
"""

# Environment must be configured before importing the application database.
# ruff: noqa: E402

import atexit
import uuid
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
qa_root = Path(tempfile.mkdtemp(prefix="skynet-browser-qa-"))
test_admin = os.environ.get("SKYNET_TEST_POSTGRES_ADMIN")
if not test_admin:
    raise RuntimeError("Set SKYNET_TEST_POSTGRES_ADMIN to a disposable PostgreSQL test server")
qa_database = "skynet_browser_" + uuid.uuid4().hex
with psycopg.connect(test_admin, autocommit=True) as connection:
    connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(qa_database)))
os.environ["SKYNET_DATABASE_URL"] = make_conninfo(test_admin, dbname=qa_database)


def remove_qa_database():
    with psycopg.connect(test_admin, autocommit=True) as connection:
        connection.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(qa_database)))


atexit.register(remove_qa_database)

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from skynet_app import database as database_module

# Keep real endpoint and object-store settings out of this synthetic server.
database_module.APP_ROOT = qa_root
os.environ["SKYNET_DATA_ROOT"] = str(qa_root / "data")

from skynet_app.database import canonical_json
from skynet_app import live_xr_api as api
from skynet_app.cluster_runtime import ClusterClient
from skynet_app.live_conversion import LiveConversionService, FORMAT
from skynet_app.pipeline_api import router as data_router
from skynet_app.collection_api import router as collection_router
from skynet_app.local_capture_api import router as local_router
from skynet_app.capture_processing.api import router as legacy_router
from skynet_app.main import APP_ROOT


def disabled_ssh(*_, **__):
    raise ValueError("External connections are disabled in the browser QA fixture")


ClusterClient.ssh = disabled_ssh
files, outcomes, polls = {}, {}, {}


class FixtureCluster:
    """Only scheduler and remote artifacts are simulated; never opens SSH."""

    def submit_script(self, script, identifier, gateway, **_):
        assert "#SBATCH --gres=gpu:rtx_6000:1" in script
        request = json.loads(
            (qa_root / "conversions" / identifier / "request.json").read_text()
        )
        root = request["root"]
        buf = io.BytesIO()
        with h5py.File(buf, "w") as h5:
            for i in range(len(request["sources"])):
                g = h5.create_group(f"data/demo_{i}")
                g["actions"] = np.ones((3, 28), dtype=np.float32) * i
                g["obs/proprio/joint_pos"] = np.ones((3, 28), dtype=np.float32)
            h5.attrs["fixture"] = "Synthetic browser QA, not collected training data"
        raw = buf.getvalue()
        meta = dict(
            format=FORMAT,
            task=request["profile"]["task"],
            robot=request["profile"]["robot"],
            episodes=len(request["sources"]),
            steps=3 * len(request["sources"]),
            action_dim=28,
            observation_shapes={"proprio/joint_pos": [28]},
            sources=request["sources"],
            converter_sha256=request["converter_sha256"],
            artifact=dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)),
        )
        manifest = canonical_json(meta).encode()
        files[root + "/dataset.hdf5"] = raw
        files[root + "/manifest.json"] = manifest
        outcomes[root] = dict(
            state="READY",
            metadata=meta,
            manifest=dict(
                sha256=hashlib.sha256(manifest).hexdigest(), size_bytes=len(manifest)
            ),
        )
        if "broken" in request["session_id"]:
            outcomes[root] = dict(
                state="FAILED",
                error="Synthetic QA failure: recording checksum changed. Original data was preserved.",
            )
        polls[root] = 0
        return SimpleNamespace(job_id=identifier)

    def job_statuses(self, identifiers, gateway):
        job = api.conversions.get(identifiers[0])
        root = job["dataset_root"]
        polls[root] += 1
        state = (
            "PENDING"
            if polls[root] < 2
            else "RUNNING"
            if polls[root] < 4
            else "COMPLETED"
        )
        return gateway, {identifiers[0]: {"State": state, "Reason": "Synthetic QA"}}

    def ssh(self, host, command, **_):
        root = next((r for r in outcomes if r in command), None)
        if not root:
            raise ValueError(
                "External connections are disabled in the browser QA fixture"
            )
        if polls[root] < 2:
            return "{}"
        if polls[root] < 4:
            return json.dumps(
                dict(
                    state="RUNNING",
                    detail="Synthetic QA: converting episode 1 of 2…",
                    total=2,
                    completed=1,
                )
            )
        return json.dumps(outcomes[root])

    def read_log(self, *_, **__):
        return "sky2", "Synthetic conversion log. No GPU or SSH connection was used."

    def file_size(self, path, host):
        return host, len(files[path])

    def stream_file_range(self, path, host, start, end):
        yield files[path][start : end + 1]


class FixtureConversion(LiveConversionService):
    def launch(self, job, transport):
        # The fixture has no physical recording files or external filesystem.
        # Exercise the real submission and polling paths after simulated staging.
        job = self.update(job["id"], staged=True)
        return super().launch(job, transport)


profile = dict(
    api.service.profile(),
    task="Dexverse-PickCube-v0",
    task_name="Synthetic QA cube",
    robot="floating_shadow_right",
    hand="right",
    hand_name="Shadow right · QA fixture",
)
sessions = [
    dict(
        id="qa-saved",
        profile=profile,
        state="CAPTURED",
        gateway="qa-offline",
        root="/home/qa/skynet/session",
        created_at="2026-09-08T00:00:00Z",
        recordings=["recordings/one.pkl", "recordings/two.pkl"],
        recording_summary={"episodes": 2, "steps": 6},
        recording_checksums={
            "recordings/one.pkl": "a" * 64,
            "recordings/two.pkl": "b" * 64,
        },
    ),
    dict(
        id="qa-broken",
        profile={**profile, "task_name": "Synthetic QA failure"},
        state="CAPTURED",
        gateway="qa-offline",
        root="/home/qa/skynet/broken",
        created_at="2026-09-07T00:00:00Z",
        recordings=["recordings/broken.pkl"],
        recording_summary={"episodes": 1, "steps": 3},
        recording_checksums={"recordings/broken.pkl": "c" * 64},
    ),
]
api.service.list = lambda: sessions
api.service.get = lambda identifier: next(s for s in sessions if s["id"] == identifier)
api.conversions = FixtureConversion(
    api.reviews, qa_root / "conversions", cluster=FixtureCluster()
)
# Resolve the real, read-only configuration without a simulated SSH connection.
api.conversions.cluster.candidates = ClusterClient().candidates
api.conversions.cluster._remote_path = ClusterClient()._remote_path
app = FastAPI()
app.include_router(api.router)
app.include_router(data_router)
app.include_router(collection_router)
app.include_router(local_router)
app.include_router(legacy_router)
app.mount("/static", StaticFiles(directory=APP_ROOT / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def page():
    html = (APP_ROOT / "static/index.html").read_text()
    return html.replace(
        "<title>Skynet / Experiment Console</title>",
        "<title>Skynet · Synthetic Browser QA</title>",
    ).replace(
        "<body>",
        '<body><div style="padding:12px;background:#ffdf8a;text-align:center;font-weight:bold">SYNTHETIC BROWSER QA — isolated database, no workstation connection</div>',
    )


@app.middleware("http")
async def read_only_elsewhere(request, call_next):
    if request.method != "GET" and not request.url.path.endswith("/conversions"):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            {"detail": "Only synthetic conversion is enabled in this QA fixture"},
            status_code=409,
        )
    return await call_next(request)


if __name__ == "__main__":
    import uvicorn

    print("Isolated fixture directory:", qa_root, flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8091, log_level="warning")
