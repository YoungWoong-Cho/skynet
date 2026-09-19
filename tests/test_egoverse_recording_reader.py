"""Storage-leaf checks independent of a full EgoVerse GPU installation."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from skynet_app.adapters.egoverse_runtime import (
    DATASET_FORMAT, JOINT_CONTRACT, joint_data, joint_keymap,
    validate_checkpoint_receipt, validate_episode_split, validate_manifest,
)
from skynet_app.adapters.recording_dataset import canonical, digest, stream_reference, close_handles


def fixture(tmp_path):
    path = tmp_path / "shared.hdf5"
    with h5py.File(path, "w") as f:
        f["state"] = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        f["action"] = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
        for camera in ("scene_front", "scene_left", "scene_right"):
            f[camera] = np.full((3, 4, 5, 3), 128, np.uint8)
    root = tmp_path / "dataset"
    root.mkdir()
    streams = {key: stream_reference(path, key) for key in (
        "state", "action", "scene_front", "scene_left", "scene_right")}
    manifest = dict(format=DATASET_FORMAT, contract=JOINT_CONTRACT,
                    policy_to_source_indices=[1, 0], steps=3, capture={"step_dt": 1 / 60},
                    episodes=[dict(index=0, id="immutable-episode", steps=3, streams=streams,
                                   source=dict(path="/raw.pkl", sha256="a" * 64))],
                    split=dict(train=[0], validation=[], mode="training_only"))
    (root / "manifest.json").write_bytes(canonical(manifest))
    return root, manifest


def test_joint_config_uses_reference_resolver_and_preserves_explicit_split():
    config = joint_data("/manifest", 2, 0, horizon=30, manifest_sha="a" * 64)
    resolver = config["train_datasets"]["skynet_joints"]["resolver"]
    assert resolver["_target_"] == "egoverse_data.RecordingResolver"
    assert resolver["manifest_sha"] == "a" * 64
    assert resolver["key_map"]["horizon"] == 30
    assert config["valid_datasets"]["skynet_joints"]["resolver"]["split"] == "validation"


def test_shared_manifest_verification_and_explicit_old_checkpoint_rejection(tmp_path):
    root, manifest = fixture(tmp_path)
    sha = digest(root / "manifest.json")
    assert validate_manifest(root, sha, "hpt_joints") == manifest
    with pytest.raises(ValueError, match="predates"):
        validate_checkpoint_receipt({"model": "hpt_joints", "manifest_sha256": sha}, "hpt_joints", sha)
    receipt = dict(model="hpt_joints", manifest_sha256=sha, dataset_format=DATASET_FORMAT)
    validate_checkpoint_receipt(receipt, "hpt_joints", sha)
    with pytest.raises(ValueError, match="differs"):
        validate_checkpoint_receipt(receipt, "act", sha)
    manifest["episodes"].append(manifest["episodes"][0])
    manifest["split"] = dict(train=[0], validation=[1])
    with pytest.raises(ValueError, match="distinct stable"):
        validate_episode_split(root, manifest)


def test_leaf_reads_actual_shared_values_reorders_joints_and_pads_in_memory(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    # Native parent is irrelevant to the leaf. This leaves tensors and file reads real.
    monkeypatch.setitem(sys.modules, "egomimic.rldb.zarr.zarr_dataset_multi", SimpleNamespace(MultiDataset=object))
    source = Path(__file__).parents[1] / "skynet_app/adapters/egoverse_data.py"
    spec = importlib.util.spec_from_file_location("skynet_app.adapters._tested_egoverse_data", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root, _ = fixture(tmp_path)
    leaves = module.RecordingResolver(root, "train", joint_keymap(4), digest(root / "manifest.json")).resolve()
    assert list(leaves) == ["immutable-episode"]
    leaf = leaves["immutable-episode"]
    sample = leaf[1]
    np.testing.assert_array_equal(sample["joint_positions"].numpy(), [4, 3])
    np.testing.assert_array_equal(sample["actions_joints"].numpy(), [[40, 30], [60, 50], [60, 50], [60, 50]])
    assert sample["scene_front"].shape == (3, 4, 5)
    np.testing.assert_allclose(sample["scene_front"].numpy(), 128 / 255)
    assert sample["embodiment"] == 100
    assert sample["episode_hash"] == "immutable-episode"
    assert list(root.iterdir()) == [root / "manifest.json"]
    close_handles()


def test_external_native_inputs_remain_separate_from_retired_collection_format(tmp_path):
    from skynet_app.adapters.egoverse_runtime import NATIVE_DATASET_FORMAT
    root = tmp_path / "native"
    root.mkdir()
    payload = root / "data.yaml"
    payload.write_text("native: true\n")
    manifest = dict(format=NATIVE_DATASET_FORMAT, contract="egoverse.native-pi0.5_bc_eva/v1",
                    episodes=[dict(path="train/episode")], steps=1,
                    split=dict(train=[0], validation=[], mode="training_only"),
                    files={"data.yaml":dict(size_bytes=payload.stat().st_size, sha256=digest(payload))})
    (root / "manifest.json").write_bytes(canonical(manifest))
    sha = digest(root / "manifest.json")
    assert validate_manifest(root, sha, "pi0.5_bc_eva") == manifest
    validate_checkpoint_receipt(dict(model="pi0.5_bc_eva", manifest_sha256=sha), "pi0.5_bc_eva", sha)
    with pytest.raises(ValueError):
        validate_manifest(root, sha, "hpt_joints")
    manifest["contract"] = JOINT_CONTRACT
    (root / "manifest.json").write_bytes(canonical(manifest))
    with pytest.raises(ValueError, match="External native dataset contract"):
        validate_manifest(root, digest(root / "manifest.json"), "pi0.5_bc_eva")


def test_recording_resolver_samples_every_modality_on_the_same_source_frames(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "egomimic.rldb.zarr.zarr_dataset_multi", SimpleNamespace(MultiDataset=object))
    source = Path(__file__).parents[1] / "skynet_app/adapters/egoverse_data.py"
    spec = importlib.util.spec_from_file_location("skynet_app.adapters._sampled_egoverse_data", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root, manifest = fixture(tmp_path)
    leaf = module.RecordingResolver(root, "train", joint_keymap(4), digest(root / "manifest.json"), control_hz=30).resolve()["immutable-episode"]
    assert len(leaf) == leaf.metadata["total_frames"] == 2
    np.testing.assert_array_equal(leaf.reader.joint("state"), [[2, 1], [6, 5]])
    np.testing.assert_array_equal(leaf.reader.joint("action"), [[20, 10], [60, 50]])
    assert leaf.reader.read("scene_front").shape == (2, 4, 5, 3)
    assert json.loads((root / "manifest.json").read_text()) == manifest
    assert manifest["episodes"][0]["steps"] == 3
    close_handles()


def test_experiment_sampling_keeps_short_recordings_and_rejects_temporal_resume(tmp_path):
    from skynet_app.adapters.egoverse_runtime import experiment_sampling
    root, manifest = fixture(tmp_path)
    sampling = experiment_sampling(manifest, "act", control_hz=30, action_steps=100)
    assert sampling["episodes"][0]["sampled_steps"] == 2
    assert sampling["splits"]["train"]["windows"] == 2  # Native repeat-last padding.
    receipt = dict(model="act", manifest_sha256="a" * 64, dataset_format=DATASET_FORMAT, sampling=sampling)
    validate_checkpoint_receipt(receipt, "act", "a" * 64, sampling=sampling)
    for settings in ({"control_hz": 15, "action_steps": 100}, {"control_hz": 30, "action_steps": 30}):
        changed = experiment_sampling(manifest, "act", **settings)
        with pytest.raises(ValueError, match="frequency or action chunk"):
            validate_checkpoint_receipt(receipt, "act", "a" * 64, sampling=changed)
    with pytest.raises(ValueError, match="divide"):
        experiment_sampling(manifest, "hpt_joints", control_hz=29)


@pytest.mark.parametrize("setting", [{"control_hz": 30}, {"action_steps": 30}])
def test_external_native_data_rejects_recording_sampling_controls(setting):
    from skynet_app.adapters.egoverse_runtime import experiment_sampling
    with pytest.raises(ValueError, match="External native EgoVerse Zarr"):
        experiment_sampling({}, "hpt_bc_flow_eva", **setting)
    assert experiment_sampling({}, "pi0.5_bc_eva") is None


def test_conversion_verification_does_not_construct_experiment_windows(tmp_path, monkeypatch, capsys):
    from skynet_app.adapters import egoverse_runtime as runtime
    root, _ = fixture(tmp_path)
    monkeypatch.chdir(root)
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **kw: runtime.REVISION)
    def no_model():
        raise AssertionError("Conversion must not instantiate a training model")
    monkeypatch.setattr(runtime, "register_joint_domain", no_model)
    monkeypatch.setattr(sys, "argv", ["egoverse_runtime", "--repository", str(root), "--dataset", str(root), "--manifest-sha", digest(root / "manifest.json"), "--output", str(tmp_path / "output"), "--verify-only", "--control-hz", "29", "--action-steps", "10000"])
    runtime.main()
    assert json.loads(capsys.readouterr().out)["schema"] == "skynet.egoverse-loader-validation/v1"
    assert not (tmp_path / "output").exists()
