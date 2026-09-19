import copy
import json
import pickle

import h5py
import numpy as np
import pytest

from skynet_app.observation_contracts import (
    CONTRACT_SCHEMA, content_digest, plan_artifacts,
    rgb_requirements, validate_requirements,
)
from skynet_app.adapters.observation_artifacts import (
    ARTIFACT_SCHEMA, ArtifactArray, digest,
)


def artifact(root, camera="scene_front", *, source="a" * 64, offset=0, size=(4, 5)):
    root.mkdir()
    values = np.arange(3 * size[0] * size[1] * 3, dtype="uint8").reshape(3, *size, 3) + np.uint8(offset)
    with h5py.File(root / "values.hdf5", "w") as file:
        file["values"] = values
        file["timestamps"] = np.arange(3) / 60
        file["frame_ids"] = np.arange(3)
        file["calibration/intrinsics"] = np.tile(np.eye(3), (3, 1, 1))
        file["calibration/world_from_camera"] = np.tile(np.eye(4), (3, 1, 1))
        file.attrs.update(schema=ARTIFACT_SCHEMA, complete=True)
    spec = dict(source_sha256=source, camera_id=camera, modality="rgb", recipe={"camera": {"mount": "fixed_scene"}, "width": size[1], "height": size[0]}, fixture_offset=offset)
    manifest = dict(schema=ARTIFACT_SCHEMA, artifact_key=content_digest(spec), spec=spec,
        source_sha256=source, camera_id=camera, modality="rgb", shape=list(values.shape), dtype="uint8",
        recipe=spec["recipe"], calibration={"convention": "ros_optical"},
        timing={"alignment": "pre_action_state", "step_dt": 1 / 60, "frames": 3},
        validation={"status": "PASSED"}, files={"values.hdf5": {"sha256": digest(root / "values.hdf5"), "size_bytes": (root / "values.hdf5").stat().st_size}})
    (root / "manifest.json").write_text(json.dumps(manifest))
    return dict(path=str(root), artifact_key=manifest["artifact_key"], manifest_sha256=digest(root / "manifest.json")), values


def identity():
    return {"renderer_sha256": "r", "source_revision": "commit", "asset_sha256": "assets",
            "cameras": {name: {"sensor": name, "offset": [0, 0, 1]} for name in ("front", "side")}}


def cloud(channels="XYZRGB"):
    return {"schema": CONTRACT_SCHEMA, "timing": {"alignment": "pre_action_state", "stride": 1},
        "streams": [{"name": "scene", "modality": "point_cloud", "camera_ids": ["front", "side"],
            "width": 5, "height": 4, "channels": channels, "coordinate_frame": "world", "num_points": 8,
            "sampling": "farthest_point", "seed": 7, "processing_order": ["unproject", "world_transform", "merge", "crop", "sample"],
            "insufficient_points": "repeat_with_mask", "crop": None}]}


def test_state_declaration_requires_no_render():
    from test_policy_exports import fixture_recording_manifest
    declaration = fixture_recording_manifest().train.data_requirements.recording_conversion
    state = declaration.presets[0].model_dump(mode="json")
    assert declaration.format == "skynet.recording-dataset/v1"
    assert plan_artifacts("a" * 64, {}, state["observation_requirements"], {}) == []


def test_camera_reuse_does_not_depend_on_other_views_or_dataset_split():
    one = plan_artifacts("a" * 64, {"episode_key": "one", "split": "train"}, rgb_requirements(["front"], width=5, height=4), identity())
    many = plan_artifacts("a" * 64, {"episode_key": "another", "split": "validation"}, rgb_requirements(["front", "side"], width=5, height=4), identity())
    assert one[0]["artifact_key"] == many[0]["artifact_key"]
    changed = identity()
    changed["cameras"]["side"]["offset"] = [2, 0, 1]
    assert plan_artifacts("a" * 64, {}, rgb_requirements(["front"], width=5, height=4), changed)[0]["artifact_key"] == one[0]["artifact_key"]
    different = plan_artifacts("a" * 64, {"episode_index": 1}, rgb_requirements(["front"], width=5, height=4), identity())
    assert different[0]["artifact_key"] != one[0]["artifact_key"]
    assert all(content_digest(n["spec"]) == n["artifact_key"] for n in many)


def test_cloud_plan_merges_complete_camera_sources_before_sampling():
    nodes = plan_artifacts("a" * 64, {}, cloud(), identity())
    assert [n["modality"] for n in nodes] == ["depth", "rgb", "depth", "rgb", "point_cloud"]
    assert nodes[-1]["dependencies"] == [n["artifact_key"] for n in nodes[:-1]]
    xyz = plan_artifacts("a" * 64, {}, cloud("XYZ"), identity())
    assert [n["modality"] for n in xyz] == ["depth", "depth", "point_cloud"]
    bad = cloud()
    bad["streams"][0]["processing_order"] = ["unproject", "sample", "merge"]
    with pytest.raises(ValueError, match="merged before"):
        validate_requirements(bad)
    bad = cloud()
    bad["streams"][0]["coordinate_frame"] = "camera:front"
    validate_requirements(bad)
    renamed = cloud()
    renamed["streams"][0]["name"] = "another_model_input_slot"
    assert plan_artifacts("a" * 64, {}, renamed, identity())[-1]["artifact_key"] == nodes[-1]["artifact_key"]


def test_reader_checks_identity_alignment_and_is_spawn_safe(tmp_path):
    reference, expected = artifact(tmp_path / "front")
    array = ArtifactArray(reference, source_sha256="a" * 64, camera_id="scene_front", modality="rgb")
    np.testing.assert_array_equal(array[1], expected[1])
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(array))[[2, 0, 2]], expected[[2, 0, 2]])
    with pytest.raises(ValueError, match="different source"):
        ArtifactArray(reference, source_sha256="b" * 64)
    with h5py.File(tmp_path / "front/values.hdf5", "r+") as file:
        file["values"][0, 0, 0, 0] = 99
    with pytest.raises(ValueError, match="checksum"):
        ArtifactArray(reference)






def test_symlinked_artifact_inventory_is_rejected(tmp_path):
    reference, _ = artifact(tmp_path / "front")
    (tmp_path / "front/values.hdf5").rename(tmp_path / "outside.hdf5")
    (tmp_path / "front/values.hdf5").symlink_to(tmp_path / "outside.hdf5")
    with pytest.raises(ValueError, match="unsafe"):
        ArtifactArray(reference)


def test_eval_camera_contract_checks_geometry_without_legacy_image_hash():
    from skynet_app.adapters.policy_contract import observation_camera_recipe_matches
    wanted = {"scene_front": {"sensor": "front", "width": 256, "height": 256,
              "offset": {"pos": [1, 2, 3], "rot": [1, 0, 0, 0], "convention": "world"},
              "projection": {"focal_length": 24.0}}}
    actual = copy.deepcopy(wanted)
    actual["scene_front"]["projection"]["vertical_aperture"] = None
    assert observation_camera_recipe_matches(wanted, actual)
    actual["scene_front"]["offset"]["pos"][0] += .1
    assert not observation_camera_recipe_matches(wanted, actual)
    assert not observation_camera_recipe_matches(wanted, {})
