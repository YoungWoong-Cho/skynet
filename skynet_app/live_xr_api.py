from typing import Literal
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse
from pydantic import BaseModel, ConfigDict
from .collection_api import service as collection
from .cluster_runtime import ClusterError
from .live_xr import LiveXRService
from .live_xr_catalog import catalog
from .live_xr_review import LiveReviewService

router = APIRouter(prefix="/api/collection/live", tags=["collection"])
service = LiveXRService(collection.database)
reviews = LiveReviewService(service)


class StartRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    accepted_license: bool = False
    task: str | None = None
    robot: str | None = None


def checked(call, *args):
    try:
        return call(*args)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ClusterError, OSError) as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("")
def overview():
    profile = checked(service.profile)
    return {
        "sessions": service.list(),
        "catalog": catalog(),
        "license": service.consent(),
        "target": {
            "execution": profile["execution"],
            "host": profile["gateway"],
            "duration_minutes": profile["duration_minutes"],
        },
    }


@router.post("/sessions", status_code=202)
def start(request: StartRequest):
    return checked(
        service.create, request.accepted_license, request.task, request.robot
    )


@router.get("/sessions/{identifier}")
def status(identifier: str):
    return checked(service.refresh, identifier)


@router.post("/sessions/{identifier}/stop")
def stop(identifier: str):
    return checked(service.stop, identifier)


class HeadsetObservation(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    result: Literal["NOT_TESTED", "PORT_UNREACHABLE", "PORT_REACHABLE", "SCENE_VISIBLE"]


@router.post("/sessions/{identifier}/headset-result")
def report_headset(identifier: str, request: HeadsetObservation):
    from .database import utc_now

    return checked(
        lambda: service.update(
            identifier, headset_result=request.result, headset_reported_at=utc_now()
        )
    )


@router.get("/sessions/{identifier}/logs", response_class=PlainTextResponse)
def logs(identifier: str):
    return checked(service.logs, identifier)


@router.get("/guide", response_class=PlainTextResponse)
def guide():
    return (service.root / "docs/live-dexverse.md").read_text()


@router.get("/sessions/{identifier}/recordings/{index}/review")
def review_status(identifier: str, index: int):
    return checked(reviews.status, identifier, index)


@router.post("/sessions/{identifier}/recordings/{index}/review", status_code=202)
def review_create(identifier: str, index: int):
    return checked(reviews.create, identifier, index)


@router.get("/sessions/{identifier}/recordings/{index}/{name}")
def review_file(identifier: str, index: int, name: str):
    path = checked(reviews.artifact, identifier, index, name)
    return FileResponse(
        path,
        media_type="application/json"
        if name.endswith(".json")
        else "application/octet-stream",
        filename=None if name == "review.json" else f"{identifier[:8]}-{index}-{name}",
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, max-age=3600",
        },
    )
