import json
from pathlib import Path
import runpy
import shlex
import subprocess

import numpy as np
import pytest

from skynet_app import simulation_hands as hands
from skynet_app.hands import HandLibrary, parse_urdf, joint_metadata

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = runpy.run_path(str(ROOT / "ops/xr/hands/runtime.py"))
URDF = b"""<robot name="test">
<link name="palm"><visual><geometry><mesh filename="palm.stl"/></geometry></visual></link>
<link name="finger"/><link name="tip"/>
<joint name="0.0" type="revolute"><parent link="palm"/><child link="finger"/><limit lower="0" upper="1.5"/></joint>
<joint name="coupled" type="revolute"><parent link="finger"/><child link="tip"/><limit lower="0" upper=".75"/><mimic joint="0.0" multiplier=".5"/></joint>
</robot>"""


@pytest.fixture
def source(tmp_path, monkeypatch):
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
                        "sides": {"right": "hand.urdf"},
                        "packages": {},
                        "license_path": "LICENSE",
                    }
                ]
            }
        )
    )
    library = HandLibrary(tmp_path / "library", config)
    monkeypatch.setattr(
        library,
        "_download",
        lambda url: URDF if url.endswith(".urdf") else b"source asset",
    )
    library.install("test", "right")
    spec = dict(
        key="test",
        name="Test right",
        side="right",
        revision="a" * 40,
        robot="skynet_test_right",
        palm="palm",
        tips=["tip"],
        alignment_rpy=[0, 0, 0],
        retargeting_scheme="vector",
    )
    monkeypatch.setattr(hands, "definition", lambda robot: dict(spec))
    return library, tmp_path / "bundles"


def build(source):
    return hands.build("skynet_test_right", library=source[0], output_root=source[1])


def test_real_source_names_mimic_and_floating_wrist_survive_bundle(source):
    path, manifest = build(source)
    assert manifest["finger_joints"] == ["h_0_0"]
    assert manifest["action_dimension"] == 7
    assert manifest["wrist_joints"] == [
        "skynet_x",
        "skynet_y",
        "skynet_z",
        "skynet_roll",
        "skynet_pitch",
        "skynet_yaw",
    ]
    xml = parse_urdf((path / "simulation.urdf").read_bytes())
    assert hands.validate_tree(xml) == "skynet_base"
    assert xml.find(".//mesh").get("filename") == "assets/palm.stl"
    assert (
        next(j for j in joint_metadata(xml) if j["name"] == "h_coupled")["mimic"][
            "joint"
        ]
        == "h_0_0"
    )
    assert RUNTIME["read_bundle"](path)[1] == manifest
    before = (path / "manifest.json").stat().st_mtime_ns
    assert build(source) == (path, manifest)
    assert (path / "manifest.json").stat().st_mtime_ns == before
    assert not list(source[1].rglob("*.partial"))


def test_modified_source_and_bundle_assets_fail_explicitly(source):
    path, _ = build(source)
    asset = path / "assets/palm.stl"
    asset.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        RUNTIME["read_bundle"](path)
    source[0].asset("test", "right", "palm.stl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        hands.build(
            "skynet_test_right", library=source[0], output_root=source[1] / "fresh"
        )
    assert not list((source[1] / "fresh").rglob("manifest.json"))


def test_disconnected_urdf_and_name_collisions_are_rejected(source):
    library = source[0]
    path = library.directory(library.entry("test", "right"), "right") / "model.urdf"
    original = path.read_bytes()
    path.write_bytes(original.replace(b"</robot>", b'<link name="orphan"/></robot>'))
    with pytest.raises(ValueError, match="root"):
        build(source)
    path.write_bytes(original.replace(b'name="coupled"', b'name="0_0"'))
    with pytest.raises(ValueError, match="collide"):
        build(source)


def test_joint_limits_reject_nonfinite_and_bound_optimizer_tolerance():
    clamp = RUNTIME["bounded_fingers"]
    assert np.allclose(clamp([-0.001, 1.001], [[0, 1], [0, 1]]), [0, 1])
    for values in ([np.nan], [np.inf], [], [[0.5]]):
        with pytest.raises(ValueError, match="missing or nonfinite"):
            clamp(values, [[0, 1]])


class LocalTransport:
    """Execute the actual SSH payload in a temporary directory without SSH."""

    def __init__(self, root):
        self.root, self.uploads = root.resolve(), 0

    def _remote_path(self, path):
        assert Path(path).resolve().is_relative_to(self.root)
        return path

    def ssh(self, gateway, command, *, stdin=None, **kwargs):
        if stdin is None:
            words = shlex.split(command)
            marker = Path(words[words.index("-f") + 1].rstrip(";"))
            return marker.read_text() if marker.exists() else ""
        self.uploads += 1
        completed = subprocess.run(
            shlex.split(command),
            input=stdin,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout


def test_remote_upload_checks_all_files_and_reuses_ready_bundle(source, tmp_path):
    path, manifest = build(source)
    remote_root = tmp_path / "remote"
    transport = LocalTransport(remote_root)
    remote = hands.upload(path, str(remote_root), transport, "local")
    assert RUNTIME["read_bundle"](remote)[1] == manifest
    assert hands.upload(path, str(remote_root), transport, "local") == remote
    assert transport.uploads == 1
    assert (Path(remote) / "READY").read_text() == manifest["digest"]


def test_remote_upload_rejects_corrupt_bundle_without_ready_marker(source, tmp_path):
    path, _ = build(source)
    (path / "assets/palm.stl").write_bytes(b"tampered")
    remote_root = tmp_path / "remote"
    with pytest.raises(subprocess.CalledProcessError) as error:
        hands.upload(path, str(remote_root), LocalTransport(remote_root), "local")
    assert "checksum mismatch" in error.value.stderr
    assert not list(remote_root.rglob("READY"))
    assert not list(remote_root.rglob(".hand-*"))


def test_catalog_exposes_ten_imported_variants_and_missing_mesh_reason():
    from skynet_app.live_xr_catalog import catalog, selection

    imported = [h for h in catalog()["hands"] if h.get("imported")]
    assert len(imported) == 10
    assert len({h["key"] for h in imported}) == 10
    for hand in imported:
        for task in catalog()["tasks"]:
            assert selection(task["key"], hand["key"])[1] == hand
    with pytest.raises(ValueError, match="missing thumb mesh"):
        selection(catalog()["default_task"], "skynet_allegro_v4_left")
