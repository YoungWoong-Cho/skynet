"""A single preview/confirm contract for history and storage cleanup."""

from typing import Literal
from subprocess import TimeoutExpired

import psycopg

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .cluster_runtime import ClusterError
from .maintenance import Maintenance

router = APIRouter(prefix="/api/maintenance")
Kind = Literal["experiment", "run", "evaluation", "adapter", "suite", "dataset", "prepared", "local-copy", "recording", "recording-file"]


class DeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(min_length=64, max_length=64)
    gateway: str = "auto"


class CleanupRequest(DeleteRequest):
    paths: list[str] = Field(min_length=1, max_length=500)


def manager():
    from .pipeline_api import LOCAL_CAPSULE_ROOT, service

    if not service.database.workspace_id:
        raise HTTPException(401, "Open an email workspace first")
    return Maintenance(
        service.database, service.cluster, local_capsules=LOCAL_CAPSULE_ROOT
    )


def recording_manager(kind="recording"):
    from .live_xr_api import service, reviews, videos, episode_previews, conversions
    from .recording_deletion import RecordingMaintenance
    from .recording_file_deletion import RecordingFileMaintenance
    implementation = RecordingFileMaintenance if kind == "recording-file" else RecordingMaintenance
    return implementation(manager().db, service, reviews, videos, episode_previews, conversions)


def invoke(operation):
    try:
        return operation()
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    except psycopg.IntegrityError as error:
        raise HTTPException(409, "Dependencies changed. Review deletion again.") from error
    except (psycopg.OperationalError, TimeoutExpired) as error:
        raise HTTPException(503, "The database or cluster connection was interrupted. Reconnect and review the operation again.") from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except (OSError, ClusterError) as error:
        raise HTTPException(
            503,
            "Storage operation did not finish. Retry after reconnecting. "
            + str(error)[-500:],
        ) from error


@router.get("/history/{kind}/{identifier}")
def preview(kind: Kind, identifier: str, gateway: str = "auto"):
    if kind in {"recording", "recording-file"}:
        return invoke(lambda: recording_manager(kind).preview(kind, identifier, gateway))
    if kind in {"dataset", "prepared", "local-copy"}:
        from . import prepared_deletion
        from .policy_exports_api import service
        return invoke(lambda: prepared_deletion.preview(service, manager().db, kind, identifier))
    return invoke(lambda: manager().preview(kind, identifier, gateway))


@router.delete("/history/{kind}/{identifier}")
def delete(kind: Kind, identifier: str, request: DeleteRequest):
    if kind in {"recording", "recording-file"}:
        return invoke(lambda: recording_manager(kind).delete(kind, identifier, request.token, request.gateway))
    if kind in {"dataset", "prepared", "local-copy"}:
        from . import prepared_deletion
        from .policy_exports_api import service
        return invoke(lambda: prepared_deletion.delete(service, manager().db, kind, identifier, request.token))
    return invoke(
        lambda: manager().delete(kind, identifier, request.token, request.gateway)
    )


@router.get("/storage")
def inspect(gateway: str = "auto"):
    return invoke(lambda: manager().inspect_storage(gateway))


@router.post("/storage/cleanup")
def cleanup(request: CleanupRequest):
    return invoke(
        lambda: manager().clean_storage(request.paths, request.token, request.gateway)
    )
