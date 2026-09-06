from typing import Literal
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict
from .collection_api import service as collection
from .cluster_runtime import ClusterError
from .live_xr import LiveXRService

router = APIRouter(prefix="/api/collection/live", tags=["collection"])
service = LiveXRService(collection.database)


class StartRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    accepted_license: bool = False


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
    return {"sessions": service.list(), "license": service.consent()}


@router.post("/sessions", status_code=202)
def start(request: StartRequest):
    return checked(service.create, request.accepted_license)


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
