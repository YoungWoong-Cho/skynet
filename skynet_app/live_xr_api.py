from typing import Literal
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, FileResponse
from pydantic import BaseModel, ConfigDict, model_validator
from .collection_api import service as collection
from .cluster_runtime import ClusterError
from .live_xr import LiveXRService
from .live_xr_catalog import catalog
from .live_xr_review import LiveReviewService
from .live_xr_video import LiveVideoService
from .live_conversion import LiveConversionService
from .live_xr_archive import LiveArchiveService
from .remote_artifacts import RemoteArtifact
from .episode_previews import EpisodePreviews

router = APIRouter(prefix="/api/collection/live", tags=["collection"])
service = LiveXRService(collection.database)
archive = service.archive = LiveArchiveService(service)
reviews = LiveReviewService(service)
episode_previews = EpisodePreviews(reviews)
videos = LiveVideoService(reviews)
conversions = LiveConversionService(reviews)
service.conversions = conversions
archive.busy = lambda identifier: (
    any(key[0] == identifier for key in reviews.active | videos.active)
)


class StartRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    accepted_license: bool = False
    task: str | None = None
    robot: str | None = None
    image_capture: bool = False
    retargeter: str = "dexpilot"


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
        "conversions": conversions.list(),
        "conversion_target": {
            key: value
            for key, value in conversions.target().items()
            if key in {"gateway", "account", "partition", "execution"}
        },
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
        service.create, request.accepted_license, request.task, request.robot, request.image_capture, request.retargeter
    )


@router.get("/sessions/{identifier}")
def status(identifier: str):
    return checked(service.status, identifier)


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


class ConversionRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    name: str

    @model_validator(mode="before")
    @classmethod
    def no_selection(cls, value):
        if isinstance(value, dict) and "indices" in value:
            raise ValueError(
                "Conversion includes every recording. Reload the page and try again."
            )
        return value


@router.post("/sessions/{identifier}/conversions", status_code=202)
def convert_session(identifier: str, request: ConversionRequest):
    return checked(conversions.create, identifier, request.name)


@router.get("/conversions/{identifier}")
def conversion_status(identifier: str):
    return checked(conversions.refresh, identifier)


@router.get("/conversions/{identifier}/logs", response_class=PlainTextResponse)
def conversion_logs(identifier: str):
    return checked(conversions.logs, identifier)


@router.get("/conversions/{identifier}/{name}")
def conversion_artifact(identifier: str, name: str, request: Request):
    artifact = checked(conversions.artifact, identifier, name)
    if isinstance(artifact, RemoteArtifact):
        return checked(lambda: artifact.response(request, filename=name))
    return FileResponse(
        artifact,
        filename=name,
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.get("/sessions/{identifier}/recordings/{index}/review")
def review_status(identifier: str, index: int):
    return checked(reviews.status, identifier, index)


@router.post("/sessions/{identifier}/recordings/{index}/review", status_code=202)
def review_create(identifier: str, index: int):
    return checked(reviews.create, identifier, index)


@router.get("/sessions/{identifier}/recordings/{index}/video")
def video_status(identifier: str, index: int, episode: int = 0):
    return checked(videos.status, identifier, index, episode)


@router.post("/sessions/{identifier}/recordings/{index}/video", status_code=202)
def video_create(identifier: str, index: int, episode: int = 0):
    return checked(videos.create, identifier, index, episode)


@router.delete("/sessions/{identifier}/recordings/{index}/video")
def video_cancel(identifier: str, index: int, episode: int = 0, generation: str | None = None):
    return checked(videos.cancel, identifier, index, episode, generation)


@router.get("/sessions/{identifier}/recordings/{index}/viewer")
def episode_viewer_status(identifier: str, index: int, episode: int = 0):
    return checked(episode_previews.status, identifier, index, episode)


@router.post("/sessions/{identifier}/recordings/{index}/viewer", status_code=202)
def episode_viewer_prepare(identifier: str, index: int, episode: int = 0):
    return checked(lambda: episode_previews.status(identifier, index, episode, start=True))


@router.get("/sessions/{identifier}/recordings/{index}/viewer/{name}")
def episode_viewer_artifact(identifier: str, index: int, name: str, request: Request, episode: int = 0):
    artifact = checked(episode_previews.artifact, identifier, index, episode, name)
    return checked(lambda: artifact.response(request, media_type="application/json" if name.endswith(".json") else "video/mp4"))


@router.get("/sessions/{identifier}/recordings/{index}/{name}")
def review_file(identifier: str, index: int, name: str, request: Request, episode: int = 0):
    if name == "video.mp4":
        artifact = checked(videos.artifact, identifier, index, episode)
        if isinstance(artifact, RemoteArtifact):
            return checked(lambda: artifact.response(request, media_type="video/mp4"))
        return FileResponse(
            artifact,
            media_type="video/mp4",
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, max-age=3600",
            },
        )
    path = checked(reviews.artifact, identifier, index, name)
    if isinstance(path, RemoteArtifact):
        return checked(lambda: path.response(
            request, media_type="application/json" if name.endswith(".json") else "application/octet-stream",
            filename=None if name == "review.json" else f"{identifier[:8]}-{index}-{name}"))
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
