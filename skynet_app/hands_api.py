import json
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, StrictFloat
from .hands import HandLibrary

router = APIRouter(prefix="/api/hands", tags=["Hands"])
library = HandLibrary()


def checked(call, *args):
    try:
        return call(*args)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("")
def catalog():
    return {"hands": library.list()}


@router.post("/{key}/{side}/install", status_code=202)
def install(key: str, side: str):
    return checked(library.start_install, key, side)


@router.get("/{key}/{side}/model")
def model(key: str, side: str):
    result = checked(library.model, key, side)
    return {k: v for k, v in result.items() if k != "files"}


@router.get("/{key}/{side}/urdf")
def urdf(key: str, side: str):
    checked(library.model, key, side)
    return FileResponse(
        library.directory(library.entry(key, side), side) / "model.urdf",
        media_type="application/xml",
    )


@router.get("/{key}/{side}/assets/{path:path}")
def asset(key: str, side: str, path: str):
    return FileResponse(checked(library.asset, key, side, path))


class Pose(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    revision: str
    joints: dict[str, StrictFloat]


@router.get("/{key}/{side}/poses")
def poses(key: str, side: str):
    return {"poses": checked(library.poses, key, side)}


@router.post("/{key}/{side}/poses", status_code=201)
def save_pose(key: str, side: str, pose: Pose):
    return checked(library.save_pose, key, side, pose.name, pose.joints, pose.revision)


@router.get("/{key}/{side}/export")
def export_pose(
    key: str, side: str, revision: str, joints: str = Query(max_length=16384)
):
    def render():
        values = json.loads(joints)
        if not isinstance(values, dict):
            raise ValueError("Pose joints must be an object")
        return library.export_pose(key, side, values, revision)

    pose = checked(render)
    return Response(
        json.dumps(pose, indent=2, allow_nan=False),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{key}-{side}-pose.json"',
            "Cache-Control": "no-store",
        },
    )
