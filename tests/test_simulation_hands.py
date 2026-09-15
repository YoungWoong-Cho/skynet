import json
from pathlib import Path
import runpy
import shlex
import subprocess

import numpy as np
import pytest

from skynet_app import simulation_hands as hands
from skynet_app import hand_bundles
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
    monkeypatch.setattr(hand_bundles, "definition", lambda robot: dict(spec))
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
    for name in manifest["wrist_joints"][3:]:
        wrist = xml.find(f"joint[@name='{name}']")
        assert wrist.get("type") == "revolute"
        assert float(wrist.find("limit").get("lower")) < -1000
        assert float(wrist.find("limit").get("upper")) > 1000
    finger = xml.find("joint[@name='h_0_0']/limit")
    assert (finger.get("lower"), finger.get("upper")) == ("0", "1.5")
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


def test_catalog_exposes_thirteen_imported_variants_and_missing_mesh_reason():
    from skynet_app.live_xr_catalog import catalog, selection

    imported = [h for h in catalog()["hands"] if h.get("imported")]
    assert len(imported) == 13
    assert len({h["key"] for h in imported}) == 13
    for hand in imported:
        for task in catalog()["tasks"]:
            required = task.get("required_hand")
            if required and required != hand["side"]:
                with pytest.raises(ValueError, match="requires"):
                    selection(task["key"], hand["key"])
            else:
                assert selection(task["key"], hand["key"])[1] == hand
    with pytest.raises(ValueError, match="missing thumb mesh"):
        selection(catalog()["default_task"], "skynet_allegro_v4_left")


def test_visual_names_are_valid_usd_identifiers(source):
    library = source[0]
    path = library.directory(library.entry("test", "right"), "right") / "model.urdf"
    path.write_bytes(path.read_bytes().replace(b"<visual>", b'<visual name="link.0">'))
    directory, _ = build(source)
    xml = parse_urdf((directory / "simulation.urdf").read_bytes())
    assert xml.find(".//visual").get("name") == "h_link_0_0"


def test_collision_filter_excludes_mounts_but_keeps_other_fingers(tmp_path):
    path = tmp_path / "hand.urdf"
    path.write_text(
        '<robot><joint type="revolute"><parent link="palm"/><child link="index1"/></joint><joint type="revolute"><parent link="index1"/><child link="index2"/></joint><joint type="revolute"><parent link="index2"/><child link="index3"/></joint><joint type="fixed"><parent link="index3"/><child link="tip"/></joint><joint type="revolute"><parent link="palm"/><child link="thumb"/></joint></robot>'
    )
    pairs = RUNTIME["adjacent_collision_pairs"](path)
    assert ("index2", "palm") in pairs
    assert ("tip", "index1") in pairs
    assert ("tip", "palm") not in pairs
    assert not any("thumb" in pair and pair != ("thumb", "palm") for pair in pairs)


def test_mesh_filenames_are_safe_for_usd_prim_names(source, monkeypatch):
    library = source[0]
    model = library.model("test", "right")
    original = library.asset("test", "right", "palm.stl")
    model["files"]["link.0.stl"] = model["files"].pop("palm.stl")
    path = library.directory(library.entry("test", "right"), "right") / "model.urdf"
    path.write_bytes(path.read_bytes().replace(b"/palm.stl", b"/link.0.stl"))
    monkeypatch.setattr(library, "model", lambda *args: model)
    original_asset = library.asset
    monkeypatch.setattr(
        library,
        "asset",
        lambda key, side, name: (
            original if name == "link.0.stl" else original_asset(key, side, name)
        ),
    )
    directory, manifest = build(source)
    xml = parse_urdf((directory / "simulation.urdf").read_bytes())
    name = xml.find(".//mesh").get("filename")
    assert Path(name).stem.startswith("h_link_0_")
    assert "." not in Path(name).stem
    assert (directory / name).read_bytes() == original.read_bytes()
    assert manifest["source_assets"]["link.0.stl"] == name


def test_canonical_hand_has_no_simulator_or_teleop_code(source):
    path, hand = hand_bundles.build(
        "skynet_test_right", library=source[0], output_root=source[1] / "canonical"
    )
    assert hand["schema"] == "skynet.hand-bundle/v1"
    assert not any(n.endswith((".py", ".json")) for n in hand["files"])
    assert not any(
        k in hand for k in ("retargeting_scheme", "collision_neighbor_depth")
    )
    adapter_path, adapter = build(source)
    assert adapter["hand_asset"]["digest"] == hand["digest"]
    assert (adapter_path / "hand.urdf").read_bytes() == (
        path / "hand.urdf"
    ).read_bytes()
    retarget = json.loads((adapter_path / "retarget-right.json").read_text())[
        "retargeting"
    ]
    assert retarget["target_joint_names"] == hand["finger_joints"]


@pytest.mark.parametrize("side", ["right", "left"])
def test_official_shadow_palm_subtree_preserves_all_physical_properties(side):
    import xml.etree.ElementTree as ET

    original = parse_urdf(
        (ROOT / f"config/hand_models/shadow/{side}.urdf").read_bytes()
    )
    prefix = side[0] + "h_"
    tips = [prefix + finger + "tip" for finger in ("th", "ff", "mf", "rf", "lf")]
    physical = hand_bundles.palm_subtree(original, prefix + "palm", tips)
    joints = joint_metadata(physical)
    assert len(joints) == 22
    assert not any("WRJ" in j["name"] for j in joints)
    for element in physical:
        source = original.find(f"{element.tag}[@name='{element.get('name')}']")
        assert ET.tostring(element) == ET.tostring(source)
    assert (
        float(
            physical.find(f"link[@name='{prefix}ffdistal']/inertial/mass").get("value")
        )
        == 0.012
    )
    assert (
        float(
            physical.find(f"link[@name='{prefix}thdistal']/inertial/mass").get("value")
        )
        == 0.016
    )


def test_two_hands_preserve_action_order_across_serialization(source, monkeypatch):
    from types import SimpleNamespace

    library, _ = source
    spec = hand_bundles.definition("test")

    def definition(robot):
        side = (
            "both"
            if robot.endswith("bimanual")
            else "left"
            if robot.endswith("left")
            else "right"
        )
        return dict(spec, key="test", side=side, robot=robot)

    monkeypatch.setattr(hand_bundles, "definition", definition)
    # A second copy of the fixture model is enough to exercise actual assembly,
    # meshes, namespace references and ordering (not a mocked composed manifest).
    import shutil

    directory = library.directory(library.entry("test", "right"), "right")
    left = directory.with_name("left")
    shutil.copytree(directory, left)
    model = left / "model.urdf"
    model.write_text(model.read_text().replace("/right/assets/", "/left/assets/"))
    library.catalog[0]["sides"]["left"] = "hand.urdf"
    # The installed side is recorded inside the manifest.
    for filename in ("manifest.json",):
        mpath = left / filename
        data = json.loads(mpath.read_text())
        data["side"] = "left"
        mpath.write_text(json.dumps(data))
    path, m = hands.build(
        "skynet_test_bimanual", library=library, output_root=source[1]
    )
    loaded = RUNTIME["read_bundle"](path)[1]
    layouts = RUNTIME["hand_layouts"](loaded)
    assert list(layouts) == ["right", "left"]
    assert loaded["action_dimension"] == 14
    terms = RUNTIME["action_terms"](loaded, SimpleNamespace)
    assert [n for term in terms.values() for n in term.joint_names] == loaded[
        "wrist_joints"
    ] + loaded["finger_joints"]
    assert (
        len(terms["right_wrist"].joint_names)
        == len(terms["left_wrist"].joint_names)
        == 6
    )
    assert (
        terms["right_wrist"].use_default_offset
        and not terms["fingers"].use_default_offset
    )
    assert all(
        (path / x.get("filename")).is_file()
        for x in parse_urdf((path / "simulation.urdf").read_bytes()).findall(".//mesh")
    )
    assert set(loaded["neutral"]) <= set(loaded["finger_joints"])
    loaded["wrist_joints"] = list(reversed(loaded["wrist_joints"]))
    with pytest.raises(ValueError, match="order"):
        RUNTIME["hand_layouts"](loaded)


def test_historical_adapter_keeps_its_recorded_layout(source):
    _, m = build(source)
    for key in ("hands", "hand_order", "hand_asset"):
        m.pop(key)
    assert RUNTIME["hand_layouts"](m)["right"]["wrist_joints"] == m["wrist_joints"]


def test_adapter_code_changes_do_not_change_physical_identity(
    source, tmp_path, monkeypatch
):
    import shutil

    _, before = build(source)
    runtime_dir = tmp_path / "adapter-source" / "ops/xr/hands"
    runtime_dir.mkdir(parents=True)
    for name in ("runtime.py", "record.py", "anatomy.py"):
        shutil.copyfile(ROOT / "ops/xr/hands" / name, runtime_dir / name)
    with (runtime_dir / "runtime.py").open("a") as f:
        f.write("\n# A different adapter implementation.\n")
    monkeypatch.setattr(hands, "ROOT", tmp_path / "adapter-source")
    _, after = build(source)
    assert before["digest"] != after["digest"]
    assert before["hand_asset"] == after["hand_asset"]


def test_generated_runtime_cache_is_outside_the_watched_checkout(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    checkout = tmp_path / "checkout"
    monkeypatch.setattr(hands, "ROOT", checkout)
    target = hands.cache_root() / "robot" / "digest" / "runtime.py"
    assert not target.is_relative_to(checkout)
    assert target.is_relative_to(tmp_path / "cache")
    # A bare uvicorn --reload watches *.py recursively in the checkout. Keeping
    # generated code outside it also works for StatReload, which ignores excludes.
    from uvicorn.config import Config
    from uvicorn.supervisors.watchfilesreload import FileFilter

    assert FileFilter(Config("skynet_app.main:app", reload=True))(
        Path("data/simulation-hands/robot/digest/runtime.py")
    )


def test_bundle_lookup_supports_cache_and_original_recordings(tmp_path, monkeypatch):
    monkeypatch.setattr(hands, "cache_root", lambda: tmp_path / "cache")
    legacy = tmp_path / "app/data/simulation-hands/robot/old"
    legacy.mkdir(parents=True)
    (legacy / "manifest.json").write_text("{}")
    assert hands.find_bundle("robot", "old", app_root=tmp_path / "app") == legacy
    current = tmp_path / "cache/robot/new"
    current.mkdir(parents=True)
    (current / "manifest.json").write_text("{}")
    assert hands.find_bundle("robot", "new", app_root=tmp_path / "app") == current
    with pytest.raises(ValueError, match="original hand bundle"):
        hands.find_bundle("robot", "missing", app_root=tmp_path / "app")


def test_wuji_velocity_caps_follow_each_source_joint_and_leave_other_hands_alone(
    tmp_path,
):
    path = tmp_path / "hand.urdf"
    path.write_text(
        '<robot><joint name="finger.1"><limit velocity="8.11"/></joint><joint name="finger.2"><limit velocity="12.86"/></joint></robot>'
    )
    config = RUNTIME["finger_actuator_parameters"](
        {"hand_key": "wuji-2", "finger_joints": ["finger.2", "finger.1"]}, path
    )
    assert config == {
        "armature": 0.001,
        "velocity_limit_sim": {r"finger\.2": 12.86, r"finger\.1": 8.11},
    }
    assert RUNTIME["finger_actuator_parameters"]({"hand_key": "shadow"}, path) == {
        "armature": 0.001,
        "velocity_limit_sim": 5.0,
    }
    assert RUNTIME["finger_actuator_parameters"](
        {"hand_key": "inspire-rh56"}, path
    ) == {"armature": 0.01, "velocity_limit_sim": 5.0}
    path.write_text(
        '<robot><joint name="finger.1"><limit velocity="0"/></joint></robot>'
    )
    with pytest.raises(ValueError, match="positive finger velocity"):
        RUNTIME["finger_actuator_parameters"](
            {"hand_key": "wuji-2", "finger_joints": ["finger.1"]}, path
        )
