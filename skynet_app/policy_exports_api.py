"""Adapter-selected recording preparation API."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from .cluster_runtime import DEFAULT_GATEWAY
from .live_xr_api import checked, reviews
from .policy_exports import (
    DEFAULT_SPLIT_SEED, DEFAULT_VALIDATION_PERCENT, MAX_VALIDATION_PERCENT, SPLIT_SEED_LIMIT, PolicyExportService,
)
from .workspaces import require_workspace_records

router = APIRouter(
    prefix="/api/data/exports", tags=["data"],
    dependencies=[Depends(require_workspace_records)],
)
service = PolicyExportService(reviews)


class ExportRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    session_id: str
    adapter_id: str
    adapter_version_id: str
    adapter_data_preset: str | None = None
    name: str
    gateway: str = DEFAULT_GATEWAY
    validation_percent: int = Field(default=DEFAULT_VALIDATION_PERCENT, ge=0, le=MAX_VALIDATION_PERCENT)
    seed: int = Field(default=DEFAULT_SPLIT_SEED, ge=0, lt=SPLIT_SEED_LIMIT)
    overfit_episode: int | None = Field(default=None, ge=0)


class RetryRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


@router.get("/jobs")
def jobs():
    from .workspaces import CURRENT_WORKSPACE
    if CURRENT_WORKSPACE.get() is not None:
        from .pipeline_api import service as pipeline_service
        return checked(lambda: service.job_overview(workspace_database=pipeline_service.database))
    return checked(service.job_overview)


@router.get("/options/{session_id}")
def preparation_options(session_id: str):
    from .workspaces import CURRENT_WORKSPACE
    if CURRENT_WORKSPACE.get() is not None:
        from .pipeline_api import service as pipeline_service
        return checked(lambda: service.preparation_options(
            session_id, workspace_database=pipeline_service.database
        ))
    return checked(service.preparation_options, session_id)


@router.post("", status_code=202)
def create(request: ExportRequest):
    return checked(
        lambda: service.create(
            request.session_id,
            request.adapter_id,
            request.name,
            target="cluster",
            validation_percent=request.validation_percent,
            seed=request.seed,
            gateway=request.gateway,
            overfit_episode=request.overfit_episode,
            adapter_version_id=request.adapter_version_id,
            adapter_data_preset=request.adapter_data_preset,
        )
    )


@router.get("/{identifier}")
def status(identifier: str):
    return checked(service.status, identifier)


@router.post("/{identifier}/retry", status_code=202)
def retry(identifier: str, request: RetryRequest):
    return checked(service.retry, identifier, "cluster")


@router.get("/{identifier}/{name}")
def artifact(identifier: str, name: str, request: Request):
    remote = checked(service.remote_artifact, identifier, name)
    if remote:
        media = "application/json" if name == "manifest.json" else "text/plain"
        return checked(lambda: remote.response(request, media_type=media, filename=name))
    return FileResponse(
        checked(service.artifact, identifier, name),
        filename=name,
        headers={"X-Content-Type-Options": "nosniff"},
    )
