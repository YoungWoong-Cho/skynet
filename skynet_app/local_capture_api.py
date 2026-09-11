from __future__ import annotations

import platform
import subprocess
import threading
import time
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from .collection_api import service as collection_service
from .local_capture import LocalCaptureService, MAX_CAPTURE_BYTES
from .cluster_runtime import ClusterError

router = APIRouter(prefix="/api/collection/local", tags=["collection"])
service = LocalCaptureService(collection_service.database)
APP_ROOT = Path(__file__).resolve().parent.parent
_setup_cache: tuple[float, dict] | None = None
_import_slot = threading.BoundedSemaphore(1)


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
        {"name": "DexVerse simulation", "status": "needs_gpu", "detail": "Recording on the headset does not need a GPU. Saved-recording replay runs on the cluster using Isaac Sim 5.1 and Isaac Lab 2.3.2; open the DexVerse cycles view to use it. CloudXR is needed only for live simulator teleoperation."},
    ])
    result = {"checks": checks, "checked_at": time.time(), "cache_seconds": 60,
              "native_minimum_visionos": "2.0", "source_project": str(APP_ROOT / "visionpro/SkynetCapture.xcodeproj"),
              "storage_path": "sky2 · " + service.storage.root, "headset_verified": False}
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
    # One bounded memory upload per server, never a temporary collection file.
    if not _import_slot.acquire(blocking=False):
        raise HTTPException(429, "Another recording is being imported. Wait for it to finish and retry.")
    try:
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > MAX_CAPTURE_BYTES:
                raise HTTPException(413, "Recording exceeds the 512 MB import limit. Record shorter sessions.")
            data.extend(chunk)
        return await run_in_threadpool(service.import_bytes, provider, bytes(data))
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except (ClusterError, OSError) as error:
        raise HTTPException(503, "Cannot store the recording on sky2. Check the cluster connection and retry. No local copy was saved.") from error
    finally:
        _import_slot.release()


@router.get("/captures/{digest}/download")
def download_capture(digest: str, request: Request):
    try:
        return service.file(digest).response(request, media_type="application/x-ndjson", filename=f"capture-{digest[:12]}.jsonl")
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except ClusterError as error:
        raise HTTPException(503, "The recording on sky2 is temporarily unavailable. Retry when the cluster connection is restored.") from error
