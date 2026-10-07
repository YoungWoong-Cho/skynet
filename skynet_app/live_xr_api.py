from dataclasses import dataclass
from typing import Literal
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, FileResponse, Response
from pydantic import BaseModel, ConfigDict
from .collection_api import service as collection
from .cluster_runtime import ClusterError
from .live_xr import LiveXRService
from .live_xr_catalog import catalog
from .retargeting import DEFAULT as DEFAULT_RETARGETER
from .live_xr_review import LiveReviewService
from .live_xr_video import LiveVideoService
from .live_xr_archive import LiveArchiveService
from .remote_artifacts import RemoteArtifact
from .episode_previews import EpisodePreviews
from .lazy_service import LazyService, resolve

router = APIRouter(prefix="/api/collection/live", tags=["collection"])


@dataclass(frozen=True)
class LiveServices:
    """The live-collection services, which refer to each other and are built together."""

    service: LiveXRService
    archive: LiveArchiveService
    reviews: LiveReviewService
    episode_previews: EpisodePreviews
    videos: LiveVideoService


def _build_live_services() -> LiveServices:
    live = LiveXRService(collection.database)
    archive = live.archive = LiveArchiveService(live)
    reviews = LiveReviewService(live)
    episode_previews = EpisodePreviews(reviews)
    videos = LiveVideoService(reviews)
    archive.busy = lambda identifier: (
        any(key[0] == identifier for key in reviews.active | videos.active)
    )
    return LiveServices(live, archive, reviews, episode_previews, videos)


_live = LazyService(_build_live_services)
service = LazyService(lambda: resolve(_live).service)
archive = LazyService(lambda: resolve(_live).archive)
reviews = LazyService(lambda: resolve(_live).reviews)
episode_previews = LazyService(lambda: resolve(_live).episode_previews)
videos = LazyService(lambda: resolve(_live).videos)


class StartRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    accepted_license: bool = False
    task: str | None = None
    robot: str | None = None
    retargeter: str = DEFAULT_RETARGETER


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
        service.create, request.accepted_license, request.task, request.robot, request.retargeter
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


@router.get("/sessions/{identifier}/recordings/{index}/review")
def review_status(identifier: str, index: int):
    return checked(reviews.status, identifier, index)


@router.post("/sessions/{identifier}/recordings/{index}/review", status_code=202)
def review_create(identifier: str, index: int):
    return checked(reviews.create, identifier, index)


@router.get("/sessions/{identifier}/recordings/{index}/video")
def video_status(identifier: str, index: int, episode: int = 0):
    return checked(videos.status, identifier, index, episode)


@router.delete("/sessions/{identifier}/recordings/{index}/video")
def video_cancel(identifier: str, index: int, episode: int = 0, generation: str | None = None):
    return checked(videos.cancel, identifier, index, episode, generation)


@router.get("/sessions/{identifier}/recordings/{index}/hand/{path:path}")
def recorded_hand_asset(identifier: str, index: int, path: str, request: Request):
    transport, gateway, artifact = checked(episode_previews.hand_file, identifier, index, path)
    if "text" in artifact:
        return Response(artifact["text"], media_type="application/xml")
    if transport is None:
        return FileResponse(artifact["path"])
    remote = RemoteArtifact(transport, gateway, artifact["path"], artifact["size_bytes"])
    return checked(lambda: remote.response(request, media_type="application/octet-stream"))


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
        return checked(lambda: artifact.response(request, media_type="video/mp4"))
    artifact = checked(reviews.artifact, identifier, index, name)
    return checked(lambda: artifact.response(
        request, media_type="application/json" if name.endswith(".json") else "application/octet-stream",
        filename=None if name == "review.json" else f"{identifier[:8]}-{index}-{name}"))
