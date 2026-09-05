from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from .cluster_runtime import ClusterError
from .collection import (
    SESSION_STATES,
    CollectionAdapterManifest,
    CollectionCompleteRequest,
    CollectionErrorRequest,
    CollectionGatewayRequest,
    CollectionRestartRequest,
    CollectionService,
    CollectionSessionCreate,
    CollectionValidationError,
    compile_collection_sbatch,
)


router = APIRouter(prefix="/api/collection", tags=["collection"])
service = CollectionService()


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, KeyError):
        return HTTPException(status_code=404, detail=str(error).strip("'"))
    if isinstance(error, sqlite3.IntegrityError):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, (ValueError, CollectionValidationError)):
        return HTTPException(status_code=422, detail=str(error))
    if isinstance(error, ClusterError):
        return HTTPException(status_code=503, detail=str(error))
    return HTTPException(status_code=500, detail=str(error))


@router.get("/capabilities")
def collection_capabilities() -> dict[str, Any]:
    return {
        "adapter_manifest_schema": CollectionAdapterManifest.model_json_schema(),
        "session_create_schema": CollectionSessionCreate.model_json_schema(),
        "session_states": list(SESSION_STATES),
        "stream_kinds": ["action", "proprio", "sensor"],
        "raw_capture_policy": "preserve-native",
        "transforms": "register as data-registry derivation jobs",
    }


@router.get("/adapters")
def list_collection_adapters(
    include_archived: bool = Query(default=False),
) -> dict[str, Any]:
    return {"adapters": service.store.list_adapters(include_archived=include_archived),
            "templates": service.store.bundled_templates()}


@router.post("/adapters", status_code=201)
def create_collection_adapter(
    manifest: CollectionAdapterManifest,
) -> dict[str, Any]:
    try:
        return {"adapter": service.store.create_adapter(manifest)}
    except Exception as error:
        raise _http_error(error) from error


@router.get("/adapters/{adapter_id}")
def get_collection_adapter(adapter_id: str) -> dict[str, Any]:
    adapter = service.store.get_adapter(adapter_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail="Collection adapter not found")
    return {"adapter": adapter}


@router.put("/adapters/{adapter_id}")
@router.patch("/adapters/{adapter_id}")
def update_collection_adapter(
    adapter_id: str, manifest: CollectionAdapterManifest
) -> dict[str, Any]:
    try:
        return {"adapter": service.store.update_adapter(adapter_id, manifest)}
    except Exception as error:
        raise _http_error(error) from error


@router.delete("/adapters/{adapter_id}")
def archive_collection_adapter(adapter_id: str) -> dict[str, Any]:
    try:
        return {"adapter": service.store.set_adapter_archived(adapter_id, True)}
    except Exception as error:
        raise _http_error(error) from error


@router.post("/adapters/{adapter_id}/restore")
def restore_collection_adapter(adapter_id: str) -> dict[str, Any]:
    try:
        return {"adapter": service.store.set_adapter_archived(adapter_id, False)}
    except Exception as error:
        raise _http_error(error) from error


@router.get("/sessions")
def list_collection_sessions(
    adapter_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=10000),
) -> dict[str, Any]:
    try:
        return {
            "sessions": service.store.list_sessions(
                adapter_id=adapter_id, status=status, limit=limit
            )
        }
    except Exception as error:
        raise _http_error(error) from error


@router.post("/sessions", status_code=201)
def create_collection_session(request: CollectionSessionCreate) -> dict[str, Any]:
    try:
        return {"session": service.store.create_session(request)}
    except Exception as error:
        raise _http_error(error) from error


@router.get("/sessions/{session_id}")
def get_collection_session(session_id: str) -> dict[str, Any]:
    session = service.store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Collection session not found")
    return {"session": session}


@router.get("/sessions/{session_id}/manifest")
def get_collection_session_manifest(session_id: str) -> dict[str, Any]:
    session = service.store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Collection session not found")
    return {
        "manifest": session["canonical_manifest"],
        "sha256": session["manifest_sha256"],
    }


@router.post("/sessions/{session_id}/preflight")
def preflight_collection_session(session_id: str) -> dict[str, Any]:
    try:
        return {"preflight": service.preflight(session_id)}
    except Exception as error:
        raise _http_error(error) from error


@router.post("/sessions/{session_id}/prepare")
def prepare_collection_session(
    session_id: str, request: CollectionGatewayRequest
) -> dict[str, Any]:
    try:
        return service.prepare(
            session_id,
            gateway=request.gateway,
            remote_validate=request.remote_validate,
        )
    except Exception as error:
        raise _http_error(error) from error


@router.get("/sessions/{session_id}/sbatch")
def get_collection_sbatch(session_id: str) -> dict[str, Any]:
    try:
        session = service.store.get_session(session_id)
        if session is None:
            raise KeyError("Collection session not found")
        compiled = compile_collection_sbatch(session)
        return {"sbatch": compiled.model_dump()}
    except Exception as error:
        raise _http_error(error) from error


@router.post("/sessions/{session_id}/submit")
def submit_collection_session(
    session_id: str, request: CollectionGatewayRequest
) -> dict[str, Any]:
    try:
        return service.submit(session_id, gateway=request.gateway)
    except Exception as error:
        raise _http_error(error) from error


@router.post("/sessions/{session_id}/refresh")
def refresh_collection_session(session_id: str) -> dict[str, Any]:
    try:
        return {"session": service.refresh(session_id)}
    except Exception as error:
        raise _http_error(error) from error


@router.get("/sessions/{session_id}/logs", response_class=PlainTextResponse)
def get_collection_session_log(
    session_id: str,
    stream: str = Query(default="stderr", pattern="^(stdout|stderr)$"),
    lines: int = Query(default=500, ge=1, le=5000),
) -> str:
    try:
        return service.read_log(session_id, stream, lines)
    except Exception as error:
        raise _http_error(error) from error


@router.post("/sessions/{session_id}/cancel")
def cancel_collection_session(
    session_id: str, request: CollectionGatewayRequest
) -> dict[str, Any]:
    try:
        return {"session": service.cancel(session_id, request.gateway)}
    except Exception as error:
        raise _http_error(error) from error


@router.post("/sessions/{session_id}/error")
def report_collection_session_error(
    session_id: str, request: CollectionErrorRequest
) -> dict[str, Any]:
    try:
        return {"session": service.report_error(session_id, request)}
    except Exception as error:
        raise _http_error(error) from error


@router.post("/sessions/{session_id}/restart", status_code=201)
def restart_collection_session(
    session_id: str, request: CollectionRestartRequest
) -> dict[str, Any]:
    try:
        return {
            "session": service.store.restart_session(session_id, name=request.name)
        }
    except Exception as error:
        raise _http_error(error) from error


@router.post("/sessions/{session_id}/complete")
def complete_collection_session(
    session_id: str, request: CollectionCompleteRequest
) -> dict[str, Any]:
    try:
        return service.complete(session_id, request)
    except Exception as error:
        raise _http_error(error) from error


__all__ = ["router", "service"]
