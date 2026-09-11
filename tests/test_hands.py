import hashlib
import json
import pytest
from skynet_app.hands import HandLibrary, joint_metadata, mesh_path, parse_urdf

URDF = b"""<robot name="test"><link name="palm"><visual><geometry><mesh filename="../meshes/palm.stl"/></geometry></visual></link><link name="finger"/><link name="tip"/><joint name="bend" type="revolute"><parent link="palm"/><child link="finger"/><limit lower="0" upper="1.5"/></joint><joint name="tip_bend" type="revolute"><parent link="finger"/><child link="tip"/><limit lower="0" upper="1.5"/><mimic joint="bend" multiplier="0.5"/></joint></robot>"""


@pytest.fixture
def library(tmp_path, monkeypatch):
    config = tmp_path / "catalog.json"
    config.write_text(
        json.dumps(
            {
                "hands": [
                    {
                        "key": "test",
                        "name": "Test",
                        "repository": "vendor/model",
                        "revision": "a" * 40,
                        "sides": {"right": "urdf/right.urdf"},
                        "packages": {},
                        "license_path": "LICENSE",
                    }
                ]
            }
        )
    )
    lib = HandLibrary(tmp_path / "store", config)

    def download(url):
        if url.endswith(".urdf"):
            return URDF
        if url.endswith(".stl"):
            return b"mesh"
        return b"Model license"

    monkeypatch.setattr(lib, "_download", download)
    return lib


def test_install_is_pinned_atomic_and_reused(library, monkeypatch):
    model = library.install("test", "right")
    assert model["revision"] == "a" * 40
    assert (
        model["files"]["meshes/palm.stl"]["sha256"]
        == hashlib.sha256(b"mesh").hexdigest()
    )
    assert library.asset("test", "right", "meshes/palm.stl").read_bytes() == b"mesh"
    monkeypatch.setattr(
        library,
        "_download",
        lambda _: pytest.fail("Stored model should not need network"),
    )
    assert library.install("test", "right") == model
    assert library.list()[0]["variants"]["right"]["state"] == "READY"
    assert not list(library.root.rglob("*.partial-*"))


def test_failed_download_does_not_publish_partial_model(library, monkeypatch):
    def fail(url):
        if url.endswith(".stl"):
            raise OSError("network disconnected")
        return URDF if url.endswith(".urdf") else b"license"

    monkeypatch.setattr(library, "_download", fail)
    with pytest.raises(ValueError, match="palm.stl.*network disconnected"):
        library.install("test", "right")
    assert library.list()[0]["variants"]["right"]["state"] == "NOT_DOWNLOADED"
    assert not list(library.root.rglob("manifest.json"))
    assert not list(library.root.rglob("*.partial-*"))


def test_paths_and_unlisted_assets_cannot_leave_model(library):
    assert mesh_path("urdf/right.urdf", "../meshes/part.stl", {}) == "meshes/part.stl"
    assert (
        mesh_path("urdf/right.urdf", "package://hand/part.stl", {"hand": "body"})
        == "body/part.stl"
    )
    for path in (
        "../../secret",
        "/etc/passwd",
        "https://host/file",
        "file:///etc/passwd",
        "..\\secret",
    ):
        with pytest.raises(ValueError):
            mesh_path("urdf/right.urdf", path, {})
    library.install("test", "right")
    with pytest.raises(ValueError):
        library.asset("test", "right", "../../catalog.json")
    with pytest.raises(ValueError):
        library.asset("test", "right", "model.urdf")


def test_pose_persists_with_limits_and_exact_model_identity(library):
    library.install("test", "right")
    pose = library.save_pose("test", "right", "Grip", {"bend": 0.75}, "a" * 40)
    assert library.poses("test", "right") == [pose]
    for values in (
        {"bend": 2},
        {"bend": float("nan")},
        {"bend": float("inf")},
        {"bend": True},
        {"bend": 0.2, "tip_bend": 0.1},
        {},
    ):
        with pytest.raises(ValueError):
            library.save_pose("test", "right", "Bad", values, "a" * 40)
    with pytest.raises(ValueError, match="revision"):
        library.save_pose("test", "right", "Bad", {"bend": 0}, "b" * 40)
    assert len(library.poses("test", "right")) == 1


def test_unsupported_side_is_explicit(library):
    library.catalog[0]["sides"]["left"] = "urdf/left.urdf"
    library.catalog[0]["unsupported_sides"] = {"left": "Upstream thumb mesh is missing"}
    with pytest.raises(ValueError, match="thumb mesh"):
        library.start_install("test", "left")
    assert library.list()[0]["variants"]["left"]["state"] == "UNSUPPORTED"


def test_reject_invalid_limits_entities_and_mimic_cycles():
    with pytest.raises(ValueError):
        parse_urdf(b"<!DOCTYPE robot><robot/>")
    for replacement in (b'upper="nan"', b'upper="-1"'):
        with pytest.raises(ValueError):
            joint_metadata(parse_urdf(URDF.replace(b'upper="1.5"', replacement)))
    with pytest.raises(ValueError, match="mimic"):
        joint_metadata(parse_urdf(URDF.replace(b'joint="bend"', b'joint="tip_bend"')))


def test_lfs_payload_is_checksum_verified(library, monkeypatch):
    original = library._download

    def fetch(url):
        if "media.githubusercontent.com" in url:
            return b"bad!"
        if url.endswith(".stl"):
            return (
                b"version https://git-lfs.github.com/spec/v1\noid sha256:"
                + b"0" * 64
                + b"\nsize 4\n"
            )
        return original(url)

    monkeypatch.setattr(library, "_download", fetch)
    with pytest.raises(ValueError, match="checksum mismatch"):
        library.install("test", "right")
    assert not list(library.root.rglob("manifest.json"))


def test_api_pose_rejects_boolean_joint_values():
    from pydantic import ValidationError
    from skynet_app.hands_api import Pose

    with pytest.raises(ValidationError):
        Pose(name="Invalid", revision="a" * 40, joints={"bend": True})
    assert Pose(name="Valid", revision="a" * 40, joints={"bend": 0}).joints == {
        "bend": 0
    }


def test_unsupported_visual_format_is_explicit(library, monkeypatch):
    original = library._download
    monkeypatch.setattr(
        library,
        "_download",
        lambda url: (
            URDF.replace(b"palm.stl", b"palm.obj")
            if url.endswith(".urdf")
            else original(url)
        ),
    )
    with pytest.raises(ValueError, match="Unsupported visual mesh format"):
        library.install("test", "right")
    assert not list(library.root.rglob("manifest.json"))


def test_pose_export_validates_without_creating_saved_pose(library):
    library.install("test", "right")
    exported = library.export_pose("test", "right", {"bend": 0.75}, "a" * 40)
    assert exported["schema"] == "skynet.hand-pose/v1"
    assert exported["joints"] == {"bend": 0.75}
    assert library.poses("test", "right") == []
    with pytest.raises(ValueError, match="limits"):
        library.export_pose("test", "right", {"bend": 2}, "a" * 40)


@pytest.mark.parametrize(
    "key,side,expected",
    [
        ("leap-v1", "right", "skynet_leap_v1_right"),
        ("wuji-1", "left", "skynet_wuji_1_left"),
        ("inspire-rh56", "right", "skynet_inspire_rh56_right"),
        ("shadow", "left", "floating_shadow_left"),
        ("leap-v1", "left", None),
    ],
)
def test_model_api_links_to_its_exact_simulation_variant(
    monkeypatch, key, side, expected
):
    from skynet_app import hands_api

    monkeypatch.setattr(
        hands_api.library, "model", lambda *args: {"files": {}, "revision": "test"}
    )
    assert hands_api.model(key, side)["simulation_robot"] == expected


def test_pose_names_are_unique_and_maintenance_preserves_values(library):
    library.install("test", "right")
    pose = library.save_pose("test", "right", " Grip ", {"bend": 0.75}, "a" * 40)
    with pytest.raises(ValueError, match="already exists"):
        library.save_pose("test", "right", "grip", {"bend": 0.25}, "a" * 40)
    renamed = library.rename_pose("test", "right", pose["id"], "New grip")
    assert renamed["id"] == pose["id"]
    assert renamed["joints"] == pose["joints"]
    assert renamed["revision"] == pose["revision"]
    assert renamed["created_at"] == pose["created_at"]
    with pytest.raises(ValueError, match="name"):
        library.rename_pose("test", "right", pose["id"], "  ")
    other = library.save_pose("test", "right", "Other", {"bend": 0.25}, "a" * 40)
    with pytest.raises(ValueError, match="already exists"):
        library.rename_pose("test", "right", pose["id"], "Other")
    for invalid in ("../outside", "../" + pose["id"], "unknown"):
        with pytest.raises(ValueError, match="identifier"):
            library.delete_pose("test", "right", invalid)
    library.delete_pose("test", "right", pose["id"])
    assert [p["id"] for p in library.poses("test", "right")] == [other["id"]]
    with pytest.raises(ValueError, match="not found"):
        library.delete_pose("test", "right", pose["id"])


def test_concurrent_pose_saves_cannot_create_duplicate_names(library):
    from concurrent.futures import ThreadPoolExecutor

    library.install("test", "right")

    def save(_):
        try:
            return library.save_pose("test", "right", "Grip", {"bend": 0.75}, "a" * 40)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        result = list(executor.map(save, range(2)))
    assert sum(pose is not None for pose in result) == 1
    assert len(library.poses("test", "right")) == 1


def test_pose_maintenance_api(library, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from skynet_app import hands_api

    library.install("test", "right")
    monkeypatch.setattr(hands_api, "library", library)
    app = FastAPI()
    app.include_router(hands_api.router)
    client = TestClient(app)
    payload = {"name": "Grip", "joints": {"bend": 0.75}, "revision": "a" * 40}
    response = client.post("/api/hands/test/right/poses", json=payload)
    assert response.status_code == 201
    pose = response.json()
    route = f"/api/hands/test/right/poses/{pose['id']}"
    assert client.post("/api/hands/test/right/poses", json=payload).status_code == 400
    renamed = client.patch(route, json={"name": "Renamed"})
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Renamed"
    assert renamed.json()["joints"] == pose["joints"]
    assert client.patch(route, json={"name": "   "}).status_code == 400
    assert client.delete(route).json() == {"deleted": pose["id"]}
    assert client.get("/api/hands/test/right/poses").json() == {"poses": []}
