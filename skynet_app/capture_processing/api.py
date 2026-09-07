from __future__ import annotations
import re
import shlex
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from skynet_app.local_capture_api import service as captures
from skynet_app.cluster_runtime import ClusterError, SubmissionOutcomeUnknown
from .service import ProcessingService

router = APIRouter(prefix="/api/collection/processing", tags=["collection"])
service = ProcessingService(captures)


class CycleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    capture_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pipeline_key: str = "dexverse-shadow-right"
    seed: int = Field(default=0, ge=0, lt=2**31 - 20)
    epochs: int = Field(default=100, ge=1, le=1000)
    eval_episodes: int = Field(default=1, ge=1, le=20)


@router.get("/pipelines")
def pipelines():
    return {"pipelines": service.catalog()}


@router.get("/jobs")
def jobs():
    return {"jobs": service.list()}


@router.post("/jobs", status_code=202)
def create_job(request: CycleRequest):
    try:
        data = request.model_dump()
        digest = data.pop("capture_sha256")
        return service.create(digest, **data)
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.get("/jobs/{identifier}")
def get_job(identifier: str):
    try:
        return service.refresh(identifier)
    except KeyError as error:
        raise HTTPException(404, str(error)) from error


@router.post("/jobs/{identifier}/recover")
def recover_job(identifier: str):
    try:
        return service.retry(identifier)
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except SubmissionOutcomeUnknown as error:
        raise HTTPException(409, str(error)) from error
    except ClusterError as error:
        raise HTTPException(503, str(error)) from error


@router.post("/jobs/{identifier}/cancel")
def cancel_job(identifier: str):
    try:
        job = service.get(identifier)
        if job["state"] not in ("RUNNING", "PENDING") or not job.get("job_id"):
            raise HTTPException(409, "Only a submitted, active job can be cancelled")
        service.transport(job).cancel(job["job_id"], job["gateway"])
        # Keep status until the scheduler confirms cancellation.
        service.update(identifier, cancellation_requested=True)
        return service.refresh(identifier, force=True)
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    except ClusterError as error:
        raise HTTPException(503, str(error)) from error


@router.get("/jobs/{identifier}/logs", response_class=PlainTextResponse)
def logs(identifier: str):
    try:
        job = service.get(identifier)
        if not job.get("root"):
            return "Preparation has not submitted a job yet. " + (
                job.get("error") or ""
            )
        command = "\n".join(
            f"if test -f {shlex.quote(job['root'] + '/' + name)}; then tail -c 24000 {shlex.quote(job['root'] + '/' + name)}; fi"
            for name in ("stderr.log", "stdout.log")
        )
        return (
            service.transport(job).ssh(job["gateway"], command, timeout=20)
            or "The job has not written any logs yet."
        )
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    except ClusterError as error:
        raise HTTPException(503, str(error)) from error


def byte_range(value, size):
    if not value:
        return 0, size - 1, False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if not match or not any(match.groups()):
        raise ValueError("Only a single byte range is supported")
    first, last = match.groups()
    if first:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
        if start >= size or end < start:
            raise ValueError("Byte range is outside the file")
    else:
        length = int(last)
        if length < 1:
            raise ValueError("Suffix range must be positive")
        start = max(0, size - length)
        end = size - 1
    return start, end, True


@router.get("/jobs/{identifier}/artifacts/{name}")
def artifact(identifier: str, name: str, request: Request):
    try:
        job = service.get(identifier)
        if job["state"] != "SUCCEEDED":
            raise HTTPException(
                409,
                "Artifacts are available after the completed cycle has been verified",
            )
        expected = job["result"]["artifacts"].get(name)
        if expected is None:
            raise HTTPException(404, "Artifact not found")
        path = job["root"] + "/output/" + name
        host, size = service.transport(job).file_size(path, job["gateway"])
        if size != expected["size_bytes"]:
            raise HTTPException(409, "Artifact size has changed since verification")
        try:
            start, end, partial = byte_range(request.headers.get("range"), size)
        except ValueError as error:
            raise HTTPException(
                416, str(error), headers={"Content-Range": f"bytes */{size}"}
            ) from error
        media = (
            "video/mp4"
            if name.endswith(".mp4")
            else "application/json"
            if name.endswith(".json")
            else "application/octet-stream"
        )
        disposition = (
            "inline"
            if media != "application/octet-stream"
            else f'attachment; filename="{name}"'
        )
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
            "Content-Disposition": disposition,
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, max-age=3600",
            "ETag": '"' + expected["sha256"] + '"',
        }
        if partial:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return StreamingResponse(
            service.transport(job).stream_file_range(path, host, start=start, end=end),
            status_code=206 if partial else 200,
            media_type=media,
            headers=headers,
        )
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    except ClusterError as error:
        raise HTTPException(503, str(error)) from error


@router.post("/setup")
def configure_pipeline():
    from .setup import configure

    try:
        profile = next(
            p for p in service.catalog() if p["key"] == "dexverse-shadow-right"
        )
        return configure(profile, service.cluster)
    except (ValueError, OSError, ClusterError) as error:
        raise HTTPException(503, str(error)) from error


@router.get("/guide", response_class=PlainTextResponse)
def pipeline_guide():
    from .service import APP_ROOT

    return (APP_ROOT / "docs/dexverse-recording-pipeline.md").read_text()
