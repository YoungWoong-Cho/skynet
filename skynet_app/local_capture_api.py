from __future__ import annotations

import platform
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool

from .collection_api import service as collection_service
from .local_capture import LocalCaptureService, MAX_CAPTURE_BYTES

router = APIRouter(prefix="/api/collection/local", tags=["collection"])
service = LocalCaptureService(collection_service.database)
APP_ROOT = Path(__file__).resolve().parent.parent
_setup_cache: tuple[float, dict] | None = None


@router.get("/setup")
def setup(force: bool = False) -> dict:
    global _setup_cache
    if not force and _setup_cache and time.monotonic() - _setup_cache[0] < 60:
        return _setup_cache[1]
    checks = []
    checks.append({"name": "Build computer", "status": "ready" if platform.system() == "Darwin" else "unsupported",
                   "detail": "A Mac with Xcode is required to install the native headset app. Recording runs on the Vision Pro."})
    try:
        result = subprocess.run(["xcodebuild", "-version"], capture_output=True, text=True, timeout=10, check=False)
        checks.append({"name": "Xcode", "status": "ready" if result.returncode == 0 else "needs_setup",
                       "detail": result.stdout.strip() if result.returncode == 0 else "Install and open Xcode, accept its license, then install the visionOS SDK in Xcode Settings → Components."})
    except (OSError, subprocess.TimeoutExpired):
        checks.append({"name": "Xcode", "status": "needs_setup", "detail": "Install Xcode with its visionOS SDK on a Mac."})
    checks.extend([
        {"name": "Headset installation", "status": "needs_user", "detail": "visionOS 2.0 or newer. Pair the headset in Xcode, enable Developer Mode, select your Apple signing team and install Skynet Capture."},
        {"name": "Tracking permission", "status": "needs_user", "detail": "On the headset, choose Start recording and allow hand tracking. A physical headset is required; simulator input is unsupported."},
        {"name": "DexVerse simulation", "status": "needs_gpu", "detail": "Separate pipeline: requires a compatible NVIDIA GPU host, Isaac Sim 5.1 / Isaac Lab 2.3.2 and CloudXR. The local recorder does not run a simulated robot."},
    ])
    result = {"checks": checks, "checked_at": time.time(), "cache_seconds": 60,
              "native_minimum_visionos": "2.0", "source_project": str(APP_ROOT / "visionpro/SkynetCapture.xcodeproj"),
              "storage_path": str(service.root), "headset_verified": False}
    _setup_cache = (time.monotonic(), result)
    return result


@router.get("/client-source")
def client_source() -> Response:
    import io
    source = APP_ROOT / "visionpro"
    if not (source / "SkynetCapture.xcodeproj/project.pbxproj").is_file():
        raise HTTPException(503, "Vision Pro app source is missing from this installation")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for file in source.rglob("*"):
            if "xcuserdata" not in file.parts and file.is_file() and file.suffix in {".swift", ".plist", ".pbxproj", ".md"}:
                bundle.write(file, Path("SkynetCapture") / file.relative_to(source))
    return Response(archive.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="SkynetCapture-source.zip"'})


@router.get("/captures")
def captures() -> dict:
    return {"captures": service.list()}


@router.post("/captures/{provider}", status_code=201)
async def import_capture(provider: str, request: Request) -> dict:
    path = None
    try:
        with tempfile.NamedTemporaryFile(dir=service.root, suffix=".upload", delete=False) as target:
            path = Path(target.name)
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_CAPTURE_BYTES:
                    raise HTTPException(413, "Recording exceeds the 512 MB import limit. Record shorter sessions.")
                target.write(chunk)
        return await run_in_threadpool(service.import_file, provider, path)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


@router.get("/captures/{digest}/download")
def download_capture(digest: str) -> FileResponse:
    try:
        return FileResponse(service.file(digest), media_type="application/x-ndjson", filename=f"capture-{digest[:12]}.jsonl")
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
