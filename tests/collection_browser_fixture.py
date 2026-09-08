"""Isolated browser QA server. Synthetic data; SSH and all GPU work are disabled.

Run this file, then open port 8091. It creates a disposable database automatically.
This exercises the actual conversion API, artifact download, and registry UI.
"""

# Environment must be configured before importing the application database.
# ruff: noqa: E402

import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
qa_root = Path(tempfile.mkdtemp(prefix="skynet-browser-qa-"))
os.environ["SKYNET_DATABASE_PATH"] = str(qa_root / "database.sqlite")

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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


class Transport:
    def ssh(self, host, command, stdin=None, **_):
        if stdin:
            capsule = json.loads(stdin)
            request = json.loads(capsule["files"]["request.json"])
            root = request["root"]
            buf = io.BytesIO()
            with h5py.File(buf, "w") as h5:
                for i in range(len(request["sources"])):
                    g = h5.create_group(f"data/demo_{i}")
                    g["actions"] = np.ones((3, 28), dtype=np.float32) * i
                    g["obs/proprio/joint_pos"] = np.ones((3, 28), dtype=np.float32)
                h5.attrs["fixture"] = (
                    "Synthetic browser QA, not collected training data"
                )
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
                artifact=dict(
                    sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)
                ),
            )
            manifest = canonical_json(meta).encode()
            files[root + "/dataset.hdf5"] = raw
            files[root + "/manifest.json"] = manifest
            outcomes[root] = dict(
                state="READY",
                metadata=meta,
                manifest=dict(
                    sha256=hashlib.sha256(manifest).hexdigest(),
                    size_bytes=len(manifest),
                ),
            )
            if "broken" in request["session_id"]:
                outcomes[root] = dict(
                    state="FAILED",
                    error="Synthetic QA failure: recording checksum changed. Original data was preserved.",
                )
            polls[root] = 0
            return '{"returncode":0}'
        root = next((r for r in outcomes if r in command), None)
        if command.startswith("tail"):
            return "Synthetic conversion log. No GPU or SSH connection was used."
        if not root:
            raise ValueError("Unknown QA conversion")
        polls[root] += 1
        if polls[root] < 2:
            result = dict(state="QUEUED", detail="Synthetic QA: waiting for GPU…")
        elif polls[root] < 4:
            result = dict(
                state="RUNNING",
                detail="Synthetic QA: converting episode 1 of 2…",
                total=2,
                completed=1,
            )
        else:
            result = outcomes[root]
        return json.dumps(
            dict(
                result,
                service={
                    "ActiveState": "active",
                    "SubState": "running",
                    "ExecMainStatus": "0",
                },
            )
        )

    def file_size(self, path, host):
        return host, len(files[path])

    def stream_file_range(self, path, host, start, end):
        yield files[path][start : end + 1]


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
api.service.transport = lambda _: Transport()
api.conversions = LiveConversionService(api.reviews, qa_root / "conversions")
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
