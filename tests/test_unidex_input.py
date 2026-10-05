"""Frozen input counts and full optimizer batches; no GPU performance claim."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from skynet_app.adapters.recording_dataset import canonical, close_handles, digest
from skynet_app.adapters.unidex_data import UniDexDataset, UniDexCollectionDataset, UniDexMixtureSampler, validate_manifest
from skynet_app.adapters.unidex_input import (
    checkpoint_pointcloud_recipe, collection_pointcloud_recipe, dataset_pointcloud_recipe,
    default_pointcloud_recipe, validate_pointcloud_recipe,
)
from skynet_app.adapters.unidex_manifest import recording_conversion
from skynet_app.adapters.unidex_runtime import RUN_SCHEMA, UniDexPolicy, training_identity
from skynet_app.observation_contracts import plan_artifacts
from tests.test_unidex_runtime import normalizer, recording_fixture


def selection(root, metadata, position=0):
    return dict(position=position, version_id=f"version-{position}", path=str(root),
                manifest_sha256=digest(root / "manifest.json"), metadata=metadata)


def receipt(selected):
    return dict(schema=RUN_SCHEMA, datasets=[{key: item[key] for key in (
        "position", "version_id", "manifest_sha256")} for item in selected])


@pytest.mark.parametrize("count", [1024, 10000])
def test_actual_recording_reader_uses_frozen_count_and_never_rewrites_files(tmp_path, count):
    root, metadata = recording_fixture(tmp_path, count=4, num_points=count)
    before = {path: digest(path) for path in tmp_path.rglob("*") if path.is_file()}
    verified = validate_manifest(root, digest(root / "manifest.json"))
    assert dataset_pointcloud_recipe(verified)["num_points"] == count
    dataset = UniDexDataset(root, verified, "train", normalizer(), control_hz=60, action_steps=2)
    try:
        sample = dataset[0]
        assert sample["pointcloud"].shape == (1, count, 6)
        np.testing.assert_array_equal(sample["pointcloud"][0, 0], [1, -1, -1, .5, .5, .5])
        assert {path: digest(path) for path in before} == before
    finally:
        close_handles()


def test_new_conversion_reuses_rgb_depth_but_has_a_new_pointcloud_artifact():
    requirement = recording_conversion().presets[0].observation_requirements
    assert requirement["streams"][0]["num_points"] == 10000
    old = copy.deepcopy(requirement)
    old["streams"][0]["num_points"] = 1024
    renderer = dict(renderer="frozen-simulator", cameras={"scene_front": {"physical": "same-camera"}})
    def artifacts(contract):
        return {node["modality"]: node["artifact_key"] for node in
                plan_artifacts("a" * 64, {}, contract, renderer)}
    new, existing = artifacts(requirement), artifacts(old)
    assert new["rgb"] == existing["rgb"] and new["depth"] == existing["depth"]
    assert new["point_cloud"] != existing["point_cloud"]


@pytest.mark.parametrize("defect", ["missing", "shape", "dtype", "mixed_episodes"])
def test_reject_missing_or_inconsistent_recipe_before_reading_windows(tmp_path, defect):
    root, metadata = recording_fixture(tmp_path, count=2)
    stream = metadata["episodes"][0]["streams"]["scene_front_pointcloud"]
    if defect == "missing":
        stream.pop("recipe")
    elif defect == "shape":
        stream["recipe"]["num_points"] = 10000
    elif defect == "dtype":
        stream["dtype"] = "float64"
    else:
        other = metadata["episodes"][1]["streams"]["scene_front_pointcloud"]
        other["recipe"]["num_points"] = other["shape"][1] = 10000
    with pytest.raises(ValueError, match="recipe"):
        dataset_pointcloud_recipe(metadata)


def test_different_dataset_counts_fail_before_collection_reads(tmp_path):
    a, ma = recording_fixture(tmp_path / "old", count=2)
    b, mb = recording_fixture(tmp_path / "new", count=2, num_points=10000)
    selected = [selection(a, ma), selection(b, mb, 1)]
    with pytest.raises(ValueError, match="different point-cloud recipes"):
        collection_pointcloud_recipe(selected)
    with pytest.raises(ValueError, match="different point-cloud recipes"):
        UniDexCollectionDataset(selected, "train", normalizer(), {})


@pytest.mark.parametrize("count", [1024, 10000])
def test_old_checkpoint_recipe_requires_verified_manifest_not_today_default(tmp_path, count):
    root, metadata = recording_fixture(tmp_path, count=2, num_points=count)
    selected = [selection(root, metadata)]
    saved = receipt(selected)
    # Runtime metadata is intentionally compact: use the pinned file instead.
    selected[0]["metadata"] = {}
    assert checkpoint_pointcloud_recipe(saved, selected)["num_points"] == count
    with pytest.raises(ValueError, match="lacks a point-cloud recipe"):
        checkpoint_pointcloud_recipe(saved)
    (root / "manifest.json").write_bytes(canonical({**metadata, "tampered": True}))
    with pytest.raises(ValueError, match="checksum mismatch"):
        checkpoint_pointcloud_recipe(saved, selected)


def test_new_checkpoint_pins_recipe_in_identity_and_compares_it_with_training_manifest(tmp_path):
    root, metadata = recording_fixture(tmp_path, count=2, num_points=10000)
    selected = [selection(root, metadata)]
    args = SimpleNamespace(batch_size=4, learning_rate=.0001, num_workers=1, seed=42,
                           precision="fp32", gpu_count=4, gradient_accumulation=8,
                           epochs=32, max_steps=30000)
    saved = training_identity(args, {"horizon_steps": 30}, {}, {}, selections=selected)
    assert saved["pointcloud_recipe"]["num_points"] == 10000
    assert checkpoint_pointcloud_recipe(saved) == default_pointcloud_recipe()
    saved["pointcloud_recipe"]["num_points"] = 1024
    with pytest.raises(ValueError, match="differs from its frozen training inputs"):
        checkpoint_pointcloud_recipe(saved, selected)
    saved["datasets"][0]["version_id"] = "other"
    with pytest.raises(ValueError, match="training datasets differ"):
        checkpoint_pointcloud_recipe(saved, selected)


@pytest.mark.parametrize("count", [None, True, 0, 511, 100001, 10000.0])
def test_input_count_never_falls_back_or_changes_native_feature_size(count):
    recipe = default_pointcloud_recipe()
    recipe["num_points"] = count
    with pytest.raises(ValueError, match="explicit count"):
        validate_pointcloud_recipe(recipe)


@pytest.mark.parametrize("count", [1024, 10000])
def test_inference_uses_checkpoint_input_count_without_changing_native_axes(tmp_path, count):
    torch = pytest.importorskip("torch")
    calls = []
    class Model:
        def infer_action(self, cloud, state, prompts):
            calls.append((cloud.clone(), state.clone(), prompts))
            return torch.zeros((1, 2, 82))
    policy = UniDexPolicy.__new__(UniDexPolicy)
    policy.pointcloud_recipe = {**default_pointcloud_recipe(), "num_points": count}
    policy.device, policy.model, policy.horizon, policy.normalizer = torch.device("cpu"), Model(), 2, normalizer()
    cloud = np.full((count, 6), .5, dtype=np.float32)
    result = policy.predict(cloud, np.zeros(82), "Pick up cube")
    assert result.shape == (2, 82) and calls[0][0].shape == (1, 1, count, 6)
    np.testing.assert_array_equal(calls[0][0][0, 0, 0].numpy(), [.5, -.5, -.5, .5, .5, .5])
    other = 10000 if count == 1024 else 1024
    with pytest.raises(ValueError, match="checkpoint recipe"):
        policy.predict(np.zeros((other, 6)), np.zeros(82), "Pick up cube")


@pytest.mark.parametrize("mixing", UniDexMixtureSampler.POLICIES)
def test_every_accumulated_update_has_global_batch_128_including_epoch_tail(mixing):
    replicas, microbatch, accumulation, windows = 4, 4, 8, 877
    samplers = [UniDexMixtureSampler(["wuji2"] * windows, mixing, rank=rank, replicas=replicas,
                batch_size=microbatch, gradient_accumulation_steps=accumulation)
                for rank in range(replicas)]
    shards = [list(sampler) for sampler in samplers]
    assert {len(shard) for shard in shards} == {224}
    global_order = [value for row in zip(*shards) for value in row]
    assert len(global_order) == 896
    if mixing == "window_proportional":
        assert sorted(global_order[:windows]) == list(range(windows))
    for start in range(0, len(shards[0]), microbatch * accumulation):
        assert sum(len(shard[start:start + microbatch * accumulation]) for shard in shards) == 128
    identity = samplers[0].identity()
    assert identity["complete_batch_size"] == 4
    assert identity["gradient_accumulation_steps"] == 8
    assert identity["optimizer_step_sample_alignment_per_rank"] == 32
    assert samplers[0].epoch_plan(4) == dict(replicas=4, draws_before_padding=877,
        draws_after_padding=896, padding_draws=19, samples_per_rank=224, microbatches_per_rank=56,
        optimizer_updates=7, global_batch_size=128)
    for _ in range(8):
        samplers[0].mark_consumed(4)
    saved = samplers[0].state_dict()
    resumed = UniDexMixtureSampler(["wuji2"] * windows, mixing, rank=0, replicas=4,
                                  batch_size=4, gradient_accumulation_steps=8)
    resumed.load_state_dict(saved)
    assert list(resumed) == shards[0][32:]
    changed = UniDexMixtureSampler(["wuji2"] * windows, mixing, rank=0, replicas=4, batch_size=4)
    with pytest.raises(ValueError, match="identity"):
        changed.load_state_dict(saved)
