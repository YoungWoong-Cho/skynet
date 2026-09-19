"""CPU contracts for native UniDex; no GPU/model-weight success is implied."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from ops.datasets.action_codecs.geometry import inverse_transform, pose9, rotation_xyz, transform9
from ops.datasets.action_codecs.unidex import (
    IDENTITY_POSE, anchor_faas_actions, encode_absolute, encode_recording,
    load_codec, supported_robots, validate_source,
)
from skynet_app.adapters.recording_dataset import FORMAT, canonical, digest, stream_reference, close_handles
from skynet_app.adapters.unidex_data import CONTRACT, NativeNormalizer, UniDexDataset, validate_manifest
from skynet_app.adapters.unidex_runtime import stable_digest, validate_resume
from skynet_app.adapters.unidex_weights import verify_weight_provenance


def normalizer():
    return NativeNormalizer({"norm_type": {"state": "minmax", "action": "minmax", "pointcloud": "identity"},
                             "norm_stats": {k: {"min": [-2.] * 82, "max": [2.] * 82} for k in ("state", "action")}})


def recording_fixture(tmp_path, robots=("floating_shadow_right", "floating_shadow_right")):
    root = tmp_path / "prepared"
    root.mkdir()
    store = tmp_path / "recordings"
    store.mkdir()
    episodes = []
    for index, robot in enumerate(robots):
        count = 128
        state = np.zeros((count, 82), dtype=np.float32)
        state[:, :9] = IDENTITY_POSE
        state[:, 9:18] = IDENTITY_POSE
        state[:, 0] = np.arange(count) / 10
        state[:, 18] = index
        action = state.copy()
        action[:, 0] += 0.03
        cloud = np.ones((count, 1024, 6), dtype=np.float32)
        cloud[..., 3:] = 0.5
        path = store / f"{index}.hdf5"
        with h5py.File(path, "w") as file:
            for key, values in {"faas_state_absolute": state, "faas_action_absolute": action,
                                "scene_front_pointcloud": cloud}.items():
                file.create_dataset(key, data=values)
        episodes.append({"index": index, "id": f"episode-{index}", "steps": count,
                         "source": {"path": "/preserved/raw.pkl", "sha256": str(index) * 64},
                         "prompt": "Pick up cube", "hand_id": robot,
                         "capture": {"step_dt": 1 / 60},
                         "action_representation": {"codec_sha256": load_codec(robot).digest, "frame": "camera_opengl"},
                         "streams": {k: stream_reference(path, k) for k in (
                             "faas_state_absolute", "faas_action_absolute", "scene_front_pointcloud")}})
    split_at = len(episodes) // 2
    manifest = {"format": FORMAT, "contract": CONTRACT, "episodes": episodes, "steps": count * len(episodes),
                "split": {"train": list(range(split_at)), "validation": list(range(split_at, len(episodes)))},
                "temporal": {"source_fps": 60., "control_hz": 15., "frame_stride": 4,
                             "action_horizon": 30, "execution_horizon": 1},
                "preprocessing": {"pointcloud_frame": "camera_ros_optical", "pointcloud_native_frame": "camera_opengl"},
                "action_representation": {"id": "skynet.unidex-faas/v1", "frame": "camera_opengl", "action_semantics": "controller_targets"}}
    (root / "manifest.json").write_bytes(canonical(manifest))
    return root, manifest


def test_shared_loader_fixes_episode_split_before_windows_and_one_action_anchor(tmp_path):
    root, expected = recording_fixture(tmp_path)
    manifest = validate_manifest(root, digest(root / "manifest.json"))
    assert manifest == expected
    norm = normalizer()
    train = UniDexDataset(root, manifest, "train", norm)
    valid = UniDexDataset(root, manifest, "validation", norm)
    assert {i for i, _ in train.windows} == {0}
    assert {i for i, _ in valid.windows} == {1}
    sample = train[1]  # source frame 4, all future actions anchored there.
    assert sample["pointcloud"].shape == (1, 1024, 6)
    assert sample["state"].shape == (1, 82)
    assert sample["action"].shape == (30, 82)
    expected_x = 0.03 + np.arange(30) * 0.4
    np.testing.assert_allclose(norm.unnormalize("action", sample["action"])[:, 0], expected_x, atol=2e-6)
    np.testing.assert_array_equal(sample["pointcloud"][0, 0, :3], [1, -1, -1])
    assert sorted(p.name for p in root.iterdir()) == ["manifest.json"]
    close_handles()


def test_mixed_shadow_wuji2_loader_preserves_native_commands_and_camera_frames(tmp_path):
    """Exercise real codecs with synthetic commands/clouds, not simulator rendering."""
    robots = ("floating_shadow_right", "skynet_wuji_2_right") * 2
    root, manifest = recording_fixture(tmp_path, robots)
    native = []
    for index, episode in enumerate(manifest["episodes"]):
        codec = load_codec(episode["hand_id"])
        count = episode["steps"]
        lower, upper = codec.spec.effective_lower.copy(), codec.spec.effective_upper.copy()
        lower[codec.wrist_indices], upper[codec.wrist_indices] = -0.2, 0.2
        phase = np.linspace(0.25, 0.65, count + 1)[:, None]
        q = lower + (upper - lower) * phase
        # Controller targets deliberately differ from both current/future measured states.
        targets = q[:-1].copy()
        targets[:, codec.wrist_indices[0]] += 0.025
        targets[:, codec.wrist_indices[4]] += 0.03
        finger_index = codec.action_names.index(codec.independent[0]["joint"])
        targets[:, finger_index] += 0.04
        actions = (targets - codec.action_offset) / codec.action_scale
        capture = codec.spec.capture
        capture["robot_joint_names"] = list(reversed(codec.action_names))
        raw = {"actions": actions, "states": [{"articulation": {"robot": {
            "joint_position": joints[::-1][None],
            "root_pose": np.array([[0.2, -0.1, 0.3, np.cos(0.15), 0., 0., np.sin(0.15)]])
        }}} for joints in q]}
        camera = np.repeat(np.eye(4)[None], count, axis=0)
        camera[:, :3, :3] = rotation_xyz(np.array([0.2, -0.3, 0.4]))
        camera[:, :3, 3] = [-0.3, 0.4, 0.8]
        arrays, representation = encode_recording(raw, capture, camera)
        path = Path(episode["streams"]["faas_state_absolute"]["path"])
        with h5py.File(path, "r+") as file:
            for name, values in arrays.items():
                file[name][:] = values
            # Synthetic metric XYZRGB: tests view convention only, never image quality.
            file["scene_front_pointcloud"][:] = [0.2, -0.3, 0.4, 0.1, 0.5, 0.9]
        episode.update(capture=capture, action_representation=representation)
        episode["streams"] = {name: stream_reference(path, name) for name in episode["streams"]}
        native.append((codec, q, actions, arrays))
    (root / "manifest.json").write_bytes(canonical(manifest))
    manifest = validate_manifest(root, digest(root / "manifest.json"))
    norm = normalizer()
    try:
        for split in ("train", "validation"):
            dataset = UniDexDataset(root, manifest, split, norm)
            assert {manifest["episodes"][i]["hand_id"] for i, _ in dataset.windows} == set(robots)
            assert len(dataset) == 6  # Three complete 30-step/stride-4 windows per episode.
            for episode_index in manifest["split"][split]:
                sample = dataset[dataset.windows.index((episode_index, 4))]
                codec, q, actions, arrays = native[episode_index]
                frame_indices = np.arange(4, 4 + 30 * 4, 4)
                expected_actions, _ = codec.encode_actions(actions[frame_indices], q[4])
                restored = norm.unnormalize("action", sample["action"])
                np.testing.assert_allclose(restored, expected_actions, atol=2e-6)
                np.testing.assert_allclose(codec.decode_actions(restored, q[4]), actions[frame_indices], atol=3e-6)
                np.testing.assert_allclose(norm.unnormalize("state", sample["state"])[0], arrays["faas_state_absolute"][4], atol=2e-6)
                np.testing.assert_allclose(restored[:, 9:18], np.broadcast_to(IDENTITY_POSE, (30, 9)), atol=2e-6)
                np.testing.assert_allclose(restored[:, 18:][:, ~codec.present_mask[18:]], 0, atol=2e-6)
                np.testing.assert_allclose(sample["pointcloud"][0, 0], [0.2, 0.3, -0.4, 0.1, 0.5, 0.9])
                assert sample["state"].shape == (1, 82) and sample["action"].shape == (30, 82)
                assert sample["pointcloud"].shape == (1, 1024, 6)
                assert all(np.isfinite(sample[key]).all() for key in ("state", "action", "pointcloud"))
    finally:
        close_handles()


@pytest.mark.parametrize("defect", ["split", "hand", "rate", "representation", "cloud_frame"])
def test_loader_refuses_leakage_unsupported_hands_or_implicit_geometry(tmp_path, defect):
    root, manifest = recording_fixture(tmp_path)
    if defect == "split":
        manifest["split"]["validation"] = [0]
    elif defect == "hand":
        manifest["episodes"][0]["hand_id"] = "skynet_wuji_1_right"
    elif defect == "rate":
        manifest["temporal"]["control_hz"] = 30
    elif defect == "representation":
        manifest["action_representation"]["action_semantics"] = "future_measured_state"
    else:
        manifest["preprocessing"] = {}
    (root / "manifest.json").write_bytes(canonical(manifest))
    with pytest.raises(ValueError):
        validate_manifest(root, digest(root / "manifest.json"))


def test_corrupt_sample_does_not_replace_it_with_random_other_episode(tmp_path):
    root, manifest = recording_fixture(tmp_path)
    path = Path(manifest["episodes"][0]["streams"]["scene_front_pointcloud"]["path"])
    with h5py.File(path, "r+") as file:
        file["scene_front_pointcloud"][0, 0, 0] = np.nan
    # Recomputed manifest isolates semantic corruption from file-integrity checks.
    for ref in manifest["episodes"][0]["streams"].values():
        ref["sha256"] = digest(path)
    (root / "manifest.json").write_bytes(canonical(manifest))
    dataset = UniDexDataset(root, manifest, "train", normalizer())
    with pytest.raises(ValueError, match="refusing random"):
        dataset[0]
    close_handles()


@pytest.mark.parametrize("robot", supported_robots())
def test_exact_hand_codec_roundtrip_fk_and_root_camera_frames(robot):
    codec = load_codec(robot)
    rng = np.random.default_rng(9)
    lower, upper = codec.spec.effective_lower.copy(), codec.spec.effective_upper.copy()
    lower[codec.wrist_indices[3:]], upper[codec.wrist_indices[3:]] = -0.5, 0.5
    q = lower + (upper - lower) * rng.uniform(0.25, 0.75, (4, codec.action_dim))
    action = (q - codec.action_offset) / codec.action_scale
    encoded, _ = codec.encode_actions(action, q[0])
    decoded = codec.decode_actions(encoded, q[0])
    np.testing.assert_allclose(decoded, action, atol=1e-7)
    poses = codec.spec.forward_kinematics(q[0])
    expected_wrist = poses[codec.spec.frame_link] @ np.asarray(codec.spec.data["link_to_frame"])
    np.testing.assert_allclose(codec.wrist_pose(q[0]), expected_wrist, atol=1e-9)
    capture = copy.deepcopy(codec.spec.capture)
    capture["robot_joint_names"] = list(codec.action_names)
    roots = np.tile([0.2, 0.3, 0.4, 1., 0., 0., 0.], (5, 1))
    episode = {"actions": action, "states": [{"articulation": {"robot": {
        "joint_position": q[min(i, 3)][None], "root_pose": roots[i][None]}}} for i in range(5)]}
    validate_source(episode, capture)
    camera = np.repeat(np.eye(4)[None], 4, axis=0)
    camera[:, 0, 3] = -0.5
    values, metadata = encode_recording(episode, capture, camera)
    assert metadata["frame"] == "camera_opengl"
    world_root = np.eye(4)
    world_root[:3, 3] = roots[0, :3]
    expected = np.diag([1., -1., -1., 1.]) @ inverse_transform(camera[0]) @ world_root @ expected_wrist
    np.testing.assert_allclose(transform9(values["faas_state_absolute"][0, :9]), expected, atol=1e-6)
    # Root translation cancels only when both current state and command share it.
    anchored = anchor_faas_actions(values["faas_state_absolute"][0], values["faas_action_absolute"])
    np.testing.assert_allclose(anchored, encoded, atol=2e-6)
    bad = copy.deepcopy(episode)
    bad["states"][0]["articulation"]["robot"]["root_pose"][0, 3] = 0
    with pytest.raises(ValueError, match="quaternion"):
        validate_source(bad, capture)


@pytest.mark.parametrize("robot", ["skynet_wuji_1_right", "skynet_sharpa_right"])
def test_unverified_mappings_fail_before_render(robot):
    with pytest.raises(ValueError, match="blocked"):
        load_codec(robot)


def test_resume_refuses_old_format_changed_inputs_and_weights_only_state():
    identity = {"schema": "skynet.unidex-run/v1", "dataset_format": FORMAT, "manifest_sha256": "a" * 64}
    saved = {"skynet": {**identity, "identity_sha256": stable_digest(identity)}, "optimizer_states": [{}], "lr_schedulers": [{}]}
    validate_resume(saved, identity)
    with pytest.raises(ValueError, match="identical"):
        validate_resume(saved, {**identity, "manifest_sha256": "b" * 64})
    with pytest.raises(ValueError, match="optimizer"):
        validate_resume({**saved, "optimizer_states": []}, identity)
    with pytest.raises(ValueError, match="Old or foreign"):
        validate_resume({"state_dict": {}}, identity)


def test_weight_provenance_requires_all_files_and_rejects_tampering(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"not loaded in this provenance-only test")
    (base / "tokenizer.json").write_text("{}")
    pc = tmp_path / "pc.pt"
    pc.write_bytes(b"point encoder")
    receipt = {"schema": "skynet.unidex-weights/v1",
               "base": {"origin": "official-pinned-base", "files": {p.name: digest(p) for p in base.iterdir()}},
               "pointcloud": {"origin": "official-pinned-uni3d", "sha256": digest(pc)}}
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(receipt))
    assert verify_weight_provenance(base, pc, path) == receipt
    (base / "tokenizer.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="checksum"):
        verify_weight_provenance(base, pc, path)


def test_codec_capsule_resolves_static_contracts_from_changed_working_directory(tmp_path):
    import shutil
    import subprocess
    import sys
    repo = Path(__file__).parents[1]
    capsule = tmp_path / "capsule"
    shutil.copytree(repo / "ops/datasets/action_codecs", capsule / "action_codecs", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(repo / "config/action_representations/unidex-faas-v1", capsule / "action_codecs/specs")
    script = capsule / "verify.py"
    script.write_text("from action_codecs.unidex import load_codec, supported_robots\n"
                      "assert all(load_codec(robot).action_dim > 0 for robot in supported_robots())\n")
    subprocess.run([sys.executable, str(script)], cwd=tmp_path, check=True)


def test_recorded_position_commands_are_not_clipped_to_physical_joint_limits():
    codec = load_codec("floating_shadow_right")
    q = (codec.spec.effective_lower + codec.spec.effective_upper) / 2
    q[codec.wrist_indices] = 0
    mapping = codec.independent[0]
    joint = codec.action_names.index(mapping["joint"])
    target = q.copy()
    target[joint] = codec.spec.lower[joint] - 0.001
    actions = ((target - codec.action_offset) / codec.action_scale)[None]
    capture = copy.deepcopy(codec.spec.capture)
    capture["robot_joint_names"] = list(codec.action_names)
    state = {"articulation": {"robot": {"joint_position": q[None],
             "root_pose": np.array([[0., 0., 0., 1., 0., 0., 0.]])}}}
    episode = {"actions": actions, "states": [state, copy.deepcopy(state)]}
    validate_source(episode, capture)
    values, _ = encode_recording(episode, capture, np.eye(4)[None])
    expected = target[joint] * mapping["scale"] + mapping["offset"]
    assert values["faas_action_absolute"][0,18+mapping["slot"]] == pytest.approx(expected)
    encoded, _ = codec.encode_actions(actions, q)
    np.testing.assert_allclose(codec.decode_actions(encoded, q), actions, atol=1e-9)
    bad = copy.deepcopy(episode)
    bad["actions"][0,joint] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_source(bad, capture)
