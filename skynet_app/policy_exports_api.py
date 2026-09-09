"""Preparation API; the historic exports URL remains a compatible entry point."""

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from .live_xr_api import checked, reviews
from .policy_exports import PolicyExportService

router = APIRouter(prefix="/api/data/exports", tags=["data"])
service = PolicyExportService(reviews)


class ExportRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    session_id: str
    format: str
    name: str
    resource_id: str | None = None
    gateway: str = "auto"
    validation_percent: int = Field(default=20, ge=0, le=50)
    seed: int = Field(default=42, ge=0, lt=2**31)


class RetryRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


@router.get("")
def overview():
    return checked(service.options)


@router.post("", status_code=202)
def create(request: ExportRequest):
    return checked(
        lambda: service.create(
            request.session_id,
            request.format,
            request.name,
            request.resource_id,
            target="cluster",
            validation_percent=request.validation_percent,
            seed=request.seed,
            gateway=request.gateway,
        )
    )


@router.delete("/datasets/{resource_id}")
def delete_dataset(resource_id: str):
    return checked(service.delete_dataset, resource_id)


@router.delete("/{identifier}")
def delete_format(identifier: str):
    return checked(lambda: service.delete_dataset(service.get(identifier)["resource_id"], identifier))


@router.get("/{identifier}")
def status(identifier: str):
    return checked(service.get, identifier)


@router.post("/{identifier}/retry", status_code=202)
def retry(identifier: str, request: RetryRequest):
    return checked(service.retry, identifier, "cluster")


@router.delete("/{identifier}/local-copy")
def remove_local_copy(identifier: str):
    return checked(service.remove_local_copy, identifier)


@router.get("/{identifier}/{name}")
def artifact(identifier: str, name: str):
    return FileResponse(
        checked(service.artifact, identifier, name),
        filename=name,
        headers={"X-Content-Type-Options": "nosniff"},
    )
