"""CPU contracts for native UniDex; no GPU/model-weight success is implied."""
import copy
from collections import Counter
import hashlib
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
from skynet_app.adapters.unidex_data import (
    CONTRACT, NativeNormalizer, UniDexDataset, UniDexCollectionDataset, UniDexMixtureSampler,
    validate_manifest, validate_recorded_values,
)
from skynet_app.adapters.unidex_runtime import (
    RUN_SCHEMA, checkpoint_sampling, stable_digest, training_identity, validate_resume,
    load_native_config, load_training_selections, training_limits,
)
from skynet_app.adapters.recording_time import resolve_collection_sampling
from skynet_app.adapters.unidex_weights import verify_weight_provenance
from skynet_app.adapters.unidex_input import default_pointcloud_recipe


def normalizer():
    return NativeNormalizer({"norm_type": {"state": "minmax", "action": "minmax", "pointcloud": "identity"},
                             "norm_stats": {k: {"min": [-2.] * 82, "max": [2.] * 82} for k in ("state", "action")}})


@pytest.fixture
def native_yaml_repository(tmp_path):
    """Minimal pinned-upstream YAML syntax, without downloading model weights."""
    pytest.importorskip("omegaconf")
    files = {
        "model/unidex.yaml": """defaults:
  - _self_
  - pointcloud_encoder: uni3d_l
_target_: src.unidex.unidex.PointCloudUniDexTrain
horizon_steps: 30
max_seq_len: 540
max_pointcloud_text_tokens: ${model.max_seq_len}
projector:
  _target_: src.openmodel.modules.PaliGemmaMultiModalProjector
  config:
    hidden_size: ${model.pointcloud_encoder.pc_feat_dim}
joint:
  _target_: src.openmodel.joint_model.JointModel
  config:
    rms_norm_eps: 1e-6
""",
        "model/pointcloud_encoder/uni3d_l.yaml": """_target_: src.pointcloud_encoder.uni3d.Uni3D
pc_feat_dim: 1024
""",
        "dataset/normalizer/base.yaml": """norm_type:
  state: minmax
  action: minmax
  pointcloud: identity
norm_stats:
  state:
    min: [-1.0]
    max: [1.0]
  action: ${norm_stats.state}
""",
        "train.yaml": """train:
  optimizer:
    _target_: torch.optim.AdamW
    lr: 1e-4
    betas: [0.9, 0.95]
    eps: 1e-8
    weight_decay: 1e-10
  scheduler:
    _target_: src.utils.schedulers.CosineDecaySchedule
    warmup_steps: 2000
    decay_steps: 200000
    decay_lr: 0.1
    peak_lr: 1.0
  trainer:
    max_epochs: 32
""",
    }
    for name, content in files.items():
        path = tmp_path / "config" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return tmp_path


def test_native_yaml_preserves_official_numeric_types_and_resolves_model_inputs(native_yaml_repository):
    root = native_yaml_repository
    before = {path: path.read_bytes() for path in (root / "config").rglob("*.yaml")}
    model, norm, training = load_native_config(root, 17, root / "paligemma", root / "uni3d.pt")
    # These exact exponent spellings occur in the official config. PyYAML's
    # safe_load silently turns them into strings, breaking AdamW and RMSNorm.
    for field, expected in {"lr": 1e-4, "eps": 1e-8, "weight_decay": 1e-10}.items():
        assert type(training["optimizer"][field]) is float
        assert training["optimizer"][field] == expected
    assert type(model["joint"]["config"]["rms_norm_eps"]) is float
    assert model["joint"]["config"]["rms_norm_eps"] == 1e-6
    assert model["max_pointcloud_text_tokens"] == 540
    assert model["projector"]["config"]["hidden_size"] == 1024
    assert model["horizon_steps"] == 17
    assert model["pretrained_model_path"] == model["tokenizer_path"] == str(root / "paligemma")
    assert model["pointcloud_encoder"]["pretrained_model_path"] == str(root / "uni3d.pt")
    assert model["projector"]["_target_"] == "src.unidex.modules.PaliGemmaMultiModalProjector"
    assert model["joint"]["_target_"] == "src.unidex.joint_model.JointModel"
    assert norm["norm_stats"]["action"] == {"min": [-1.0], "max": [1.0]}
    # The receipt/checkpoint identity must contain resolved ordinary JSON data.
    assert json.loads(json.dumps([model, norm, training])) == [model, norm, training]
    assert {path: path.read_bytes() for path in before} == before


def test_native_training_yaml_resolves_interpolated_scientific_values(native_yaml_repository):
    path = native_yaml_repository / "config/train.yaml"
    content = path.read_text().replace("train:\n", "train:\n  learning_rate: 1e-4\n", 1)
    path.write_text(content.replace("lr: 1e-4", "lr: ${train.learning_rate}", 1))
    _, _, training = load_native_config(native_yaml_repository, 30)
    assert type(training["optimizer"]["lr"]) is float
    assert training["optimizer"]["lr"] == 1e-4


def test_native_numeric_configuration_constructs_and_steps_real_adamw(native_yaml_repository):
    torch = pytest.importorskip("torch")
    hydra = pytest.importorskip("hydra")
    model, _, training = load_native_config(native_yaml_repository, 30)
    parameter = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
    optimizer = hydra.utils.instantiate(training["optimizer"], params=[parameter])
    # Exercise the float epsilon at both original failure boundaries: optimizer
    # construction and the RMSNorm arithmetic used inside the native model.
    eps = model["joint"]["config"]["rms_norm_eps"]
    normalized = parameter * torch.rsqrt(parameter.pow(2).mean() + eps)
    loss = normalized.sum()
    before = parameter.detach().clone()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert not torch.equal(parameter, before)
    assert optimizer.state[parameter]["step"].item() == 1


def recording_fixture(tmp_path, robots=("floating_shadow_right", "floating_shadow_right"), count=128, num_points=1024):
    root = tmp_path / "prepared"
    root.mkdir(parents=True)
    store = tmp_path / "recordings"
    store.mkdir()
    episodes = []
    for index, robot in enumerate(robots):
        state = np.zeros((count, 82), dtype=np.float32)
        state[:, :9] = IDENTITY_POSE
        state[:, 9:18] = IDENTITY_POSE
        state[:, 0] = np.arange(count) / 10
        state[:, 18] = index
        action = state.copy()
        action[:, 0] += 0.03
        cloud = np.ones((count, num_points, 6), dtype=np.float32)
        cloud[..., 3:] = 0.5
        path = store / f"{index}.hdf5"
        with h5py.File(path, "w") as file:
            for key, values in {"faas_state_absolute": state, "faas_action_absolute": action,
                                "scene_front_pointcloud": cloud}.items():
                file.create_dataset(key, data=values)
        episodes.append({"index": index, "id": f"episode-{index}", "steps": count,
                         "source": {"path": "/preserved/raw.pkl", "sha256": hashlib.sha256(f"{root}:{index}".encode()).hexdigest()},
                         "prompt": "Pick up cube", "hand_id": robot,
                         "capture": {"step_dt": 1 / 60},
                         "action_representation": {"codec_sha256": load_codec(robot).digest, "frame": "camera_opengl"},
                         "streams": {k: stream_reference(path, k) for k in (
                             "faas_state_absolute", "faas_action_absolute", "scene_front_pointcloud")}})
        episodes[-1]["streams"]["scene_front_pointcloud"]["recipe"] = {
            **default_pointcloud_recipe(), "num_points": num_points}
    split_at = max(1, len(episodes) // 2)
    manifest = {"format": FORMAT, "contract": CONTRACT, "episodes": episodes, "steps": count * len(episodes),
                "split": {"train": list(range(split_at)), "validation": list(range(split_at, len(episodes)))},
                "preprocessing": {"pointcloud_frame": "camera_ros_optical", "pointcloud_native_frame": "camera_opengl"},
                "action_representation": {"id": "skynet.unidex-faas/v1", "frame": "camera_opengl", "action_semantics": "controller_targets"}}
    (root / "manifest.json").write_bytes(canonical(manifest))
    return root, manifest


def test_shared_loader_fixes_episode_split_before_windows_and_one_action_anchor(tmp_path):
    root, expected = recording_fixture(tmp_path)
    manifest = validate_manifest(root, digest(root / "manifest.json"))
    assert manifest == expected
    norm = normalizer()
    train = UniDexDataset(root, manifest, "train", norm, control_hz=15, action_steps=30)
    valid = UniDexDataset(root, manifest, "validation", norm, control_hz=15, action_steps=30)
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


def test_one_frame_conversion_verification_is_independent_of_training_windows(tmp_path, monkeypatch):
    import sys
    from skynet_app.adapters import unidex_runtime

    root, expected = recording_fixture(tmp_path, robots=("skynet_wuji_2_right",), count=1)
    original = (root / "manifest.json").read_bytes()
    manifest = validate_manifest(root, digest(root / "manifest.json"))
    assert manifest == expected and manifest["split"]["validation"] == []
    validate_recorded_values(root, manifest)
    monkeypatch.setattr(unidex_runtime, "verify_repository", lambda _: None)
    monkeypatch.setattr(unidex_runtime, "resolve_sampling", lambda *a, **k: pytest.fail("Conversion must not select training windows"))
    monkeypatch.setattr(unidex_runtime, "load_native_config", lambda *a, **k: pytest.fail("Conversion must not instantiate model configuration"))
    output = tmp_path / "validation"
    monkeypatch.setattr(sys, "argv", ["unidex_runtime.py", "--repository", "/unused-pinned-source",
        "--dataset", str(root), "--manifest-sha", digest(root / "manifest.json"), "--output", str(output), "--verify-only"])
    try:
        unidex_runtime.main()
        report = json.loads((output / "loader-validation.json").read_text())
        assert report["status"] == "PASSED" and report["validation_scope"] == "recorded_streams"
        assert report["episodes"] == report["steps"] == 1 and "windows" not in report
        assert (root / "manifest.json").read_bytes() == original
        with pytest.raises(ValueError):
            UniDexDataset(root, manifest, "train", normalizer(), control_hz=15, action_steps=30)
    finally:
        close_handles()


@pytest.mark.parametrize("control_hz,action_steps,stride", [(None, 30, 1), (30, 30, 2), (15, 10, 4)])
def test_experiment_sampling_reuses_the_same_recording_artifacts(tmp_path, control_hz, action_steps, stride):
    root, manifest = recording_fixture(tmp_path, count=90)
    before = {path: digest(path) for path in tmp_path.rglob("*") if path.is_file()}
    norm = normalizer()
    try:
        dataset = UniDexDataset(root, manifest, "train", norm, control_hz=control_hz, action_steps=action_steps)
        assert dataset.sampling["control_hz"] == (60 if control_hz is None else control_hz)
        assert dataset.horizon == action_steps and dataset.strides[0] == stride
        assert dataset.windows[0] == (0, 0) and dataset.windows[1] == (0, stride)
        sample = dataset[1]
        actions = norm.unnormalize("action", sample["action"])
        np.testing.assert_allclose(actions[:, 0], .03 + np.arange(action_steps) * stride / 10, atol=2e-6)
        # Targets begin with the command at the observation frame, never future measured joints.
        assert actions[0, 0] == pytest.approx(.03, abs=2e-6)
        assert len(dataset) == len(range(0, 90 - (action_steps - 1) * stride, stride))
        assert {path: digest(path) for path in before} == before
        assert "temporal" not in manifest
    finally:
        close_handles()


def test_sampling_short_clips_only_block_experiment_windows(tmp_path):
    root, _ = recording_fixture(tmp_path, count=90)
    manifest = validate_manifest(root, digest(root / "manifest.json"))
    try:
        with pytest.raises(ValueError):
            UniDexDataset(root, manifest, "train", normalizer(), control_hz=15, action_steps=30)
        dataset = UniDexDataset(root, manifest, "train", normalizer(), control_hz=30, action_steps=30)
        assert len(dataset) == 16
    finally:
        close_handles()


def test_experiment_sampling_is_pinned_for_resume_and_inference(tmp_path):
    args = SimpleNamespace(manifest_sha="a" * 64, batch_size=4, learning_rate=.0001, num_workers=1,
                           seed=42, precision="fp32", gpu_count=1, gradient_accumulation=4)
    sampling = {"schema": "skynet.recording-sampling/v1", "control_hz": 30, "action_steps": 30,
                "episodes": [{"index": 0, "source_hz": 60, "stride": 2}]}
    root, manifest = recording_fixture(tmp_path, count=2)
    selected = [dict(position=0, version_id=None, manifest_sha256=digest(root / "manifest.json"), metadata=manifest)]
    identity = training_identity(args, {"horizon_steps": 30}, {}, sampling, selections=selected)
    assert identity["sampling"] == sampling and "temporal" not in identity
    saved = {"skynet": {**identity, "identity_sha256": stable_digest(identity)},
             "optimizer_states": [{}], "lr_schedulers": [{}],
             "loops": {"fit_loop": {"state_dict": {"combined_loader": [{"schema": "skynet.unidex-loader-state/v1"}]}}}}
    validate_resume(saved, identity)
    for field, value in (("control_hz", 60), ("action_steps", 15)):
        changed = copy.deepcopy(identity)
        changed["sampling"][field] = value
        with pytest.raises(ValueError, match="identical"):
            validate_resume(saved, changed)
    assert checkpoint_sampling(identity)["control_hz"] == 30
    bad = copy.deepcopy(identity)
    bad["sampling"]["action_steps"] = 15
    with pytest.raises(ValueError, match="model horizon"):
        checkpoint_sampling(bad)
    with pytest.raises(ValueError):
        checkpoint_sampling({**identity, "schema": "skynet.unidex-run/v1"})


def test_all_registered_hands_loader_preserves_native_commands_and_camera_frames(tmp_path):
    """Exercise real codecs with synthetic commands/clouds, not simulator rendering."""
    robots = tuple(supported_robots()) * 2
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
        episode["streams"] = {name: stream_reference(path, name, **({"recipe": reference["recipe"]}
                              if "recipe" in reference else {})) for name, reference in episode["streams"].items()}
        native.append((codec, q, actions, arrays))
    (root / "manifest.json").write_bytes(canonical(manifest))
    manifest = validate_manifest(root, digest(root / "manifest.json"))
    norm = normalizer()
    try:
        for split in ("train", "validation"):
            dataset = UniDexDataset(root, manifest, split, norm, control_hz=15, action_steps=30)
            assert {manifest["episodes"][i]["hand_id"] for i, _ in dataset.windows} == set(robots)
            assert len(dataset) == 3 * len(set(robots))  # Three complete windows per hand.
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
        manifest["episodes"][0]["hand_id"] = "unregistered_hand_right"
    elif defect == "rate":
        manifest["episodes"][0]["capture"]["step_dt"] = 0
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


def test_unregistered_mapping_fails_before_render():
    with pytest.raises(ValueError, match="No exact asset-bound FAAS contract"):
        load_codec("unregistered_hand_right")


def test_unverified_mapping_still_fails_before_render(tmp_path):
    import hashlib
    import shutil

    from ops.datasets.action_codecs.hand_contract import DEFAULT_SPEC_ROOT
    spec_root = tmp_path / "specs"
    shutil.copytree(DEFAULT_SPEC_ROOT, spec_root)
    path = spec_root / "skynet_sharpa_right.json"
    spec = json.loads(path.read_text())
    spec["mapping_status"] = "BLOCKED"
    spec["blocked_reason"] = "Deliberately unverified test asset"
    spec["spec_hash"] = hashlib.sha256(canonical({k: v for k, v in spec.items() if k != "spec_hash"})).hexdigest()
    path.write_bytes(canonical(spec))
    with pytest.raises(ValueError, match="Deliberately unverified test asset"):
        load_codec("skynet_sharpa_right", spec_root=spec_root)


def test_resume_refuses_old_format_changed_inputs_and_weights_only_state():
    identity = {"schema": RUN_SCHEMA, "dataset_format": FORMAT, "manifest_sha256": "a" * 64}
    saved = {"skynet": {**identity, "identity_sha256": stable_digest(identity)}, "optimizer_states": [{}], "lr_schedulers": [{}],
             "loops": {"fit_loop": {"state_dict": {"combined_loader": [{"schema": "skynet.unidex-loader-state/v1"}]}}}}
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


def selected_dataset(root, manifest, position):
    return {"position": position, "version_id": f"00000000-0000-4000-8000-{position + 1:012d}",
            "path": str(root), "manifest_sha256": digest(root / "manifest.json"), "metadata": manifest}


def frozen_data_document(selections):
    return {"data": {"bundle": {"assignments": [
        {"role": "training_data", "position": selection["position"],
         "version": {"format": FORMAT, "manifest_sha256": selection["manifest_sha256"],
                     "metadata": {**selection["metadata"], "registered_version_id": selection["version_id"]}},
         "config": {"location": {"kind": "cluster", "status": "AVAILABLE", "path": selection["path"],
                                  "manifest_sha256": selection["manifest_sha256"]}}}
        for selection in reversed(selections)]}}}


def test_runtime_uses_frozen_bundle_for_all_inputs_and_rechecks_actual_manifests(tmp_path, monkeypatch):
    import sys
    from skynet_app.adapters import unidex_runtime

    first = recording_fixture(tmp_path / "shadow", count=90)
    second = recording_fixture(tmp_path / "leap", robots=("skynet_leap_v1_right",) * 2, count=120)
    selections = [selected_dataset(*item, index) for index, item in enumerate((first, second))]
    spec = frozen_data_document(selections)
    spec["native"] = {"config": {"datasets": [{"path": "/not-the-frozen-selection"}]}}
    path = tmp_path / "resolved-spec.json"
    path.write_text(json.dumps(spec))
    args = SimpleNamespace(data_spec=str(path), dataset=None, manifest_sha=None)
    loaded = load_training_selections(args)
    assert loaded == selections
    output = tmp_path / "run"
    monkeypatch.setattr(unidex_runtime, "verify_repository", lambda _: None)
    monkeypatch.setattr(unidex_runtime, "load_native_config", lambda *a: ({"horizon_steps": 30}, normalizer().config, {}))
    monkeypatch.setattr(sys, "argv", ["unidex_runtime.py", "--repository", "/unused-pinned-source",
        "--data-spec", str(path), "--output", str(output), "--control-hz", "30", "--action-steps", "30",
        "--mixing-policy", "hand_balanced", "--max-steps", "13", "--config-only"])
    try:
        unidex_runtime.main()
        summary = json.loads((output / "sampling.json").read_text())
        assert summary["splits"]["train"]["windows"] == 47
        assert [row["version_id"] for row in summary["datasets"]] == [item["version_id"] for item in selections]
        assert json.loads((output / "mixture.json").read_text())["policy"] == "hand_balanced"
        second[0].joinpath("manifest.json").write_text("{}")
        with pytest.raises(ValueError, match="checksum"):
            load_training_selections(args)
    finally:
        close_handles()


def test_collection_preserves_each_hand_episode_split_anchor_and_store(tmp_path):
    first = recording_fixture(tmp_path / "shadow", count=90)
    second = recording_fixture(tmp_path / "leap", robots=("skynet_leap_v1_right",) * 2, count=120)
    selections = [selected_dataset(*item, index) for index, item in enumerate((first, second))]
    before = {path: digest(path) for path in tmp_path.rglob("*") if path.is_file()}
    sampling = resolve_collection_sampling(selections, control_hz=30, action_steps=30,
                                            window_policy="complete", require_validation=True)
    norm = normalizer()
    try:
        for split in ("train", "validation"):
            collection = UniDexCollectionDataset(selections, split, norm, sampling)
            assert len(collection) == 16 + 31
            assert Counter(collection.hands) == {"floating_shadow_right": 16, "skynet_leap_v1_right": 31}
            for part in collection.datasets:
                assert {index for index, _ in part.windows} == {0 if split == "train" else 1}
            np.testing.assert_allclose(norm.unnormalize("action", collection[16]["action"])[:, 0],
                                       .03 + np.arange(30) * .2, atol=2e-6)
            np.testing.assert_array_equal(collection[-1]["action"], collection[len(collection) - 1]["action"])
            with pytest.raises(IndexError):
                collection[len(collection)]
        assert {path: digest(path) for path in before} == before
    finally:
        close_handles()


def test_collection_accepts_empty_individual_validation_but_requires_aggregate(tmp_path):
    one = recording_fixture(tmp_path / "one", robots=("skynet_leap_v1_right",), count=90)
    two = recording_fixture(tmp_path / "two", count=90)
    selections = [selected_dataset(*item, index) for index, item in enumerate((one, two))]
    sampling = resolve_collection_sampling(selections, control_hz=30, action_steps=30, require_validation=True)
    try:
        valid = UniDexCollectionDataset(selections, "validation", normalizer(), sampling)
        assert len(valid) == 16 and set(valid.hands) == {"floating_shadow_right"}
        with pytest.raises(ValueError, match="validation"):
            resolve_collection_sampling(selections[:1], control_hz=30, action_steps=30, require_validation=True)
    finally:
        close_handles()


def test_collection_checks_source_overlap_and_one_physical_control_rate(tmp_path):
    first = recording_fixture(tmp_path / "a")
    second = recording_fixture(tmp_path / "b")
    selections = [selected_dataset(*item, index) for index, item in enumerate((first, second))]
    for episode in selections[1]["metadata"]["episodes"]:
        episode["capture"]["step_dt"] = 1 / 30
    with pytest.raises(ValueError, match="frequency|frequencies"):
        resolve_collection_sampling(selections, action_steps=10, require_validation=True)
    sampling = resolve_collection_sampling(selections, control_hz=15, action_steps=10, require_validation=True)
    assert [row["sampling"]["episodes"][0]["stride"] for row in sampling["datasets"]] == [4, 2]
    # The same source appearing in a different dataset's validation split must fail.
    selections[1]["metadata"]["episodes"][1]["source"] = selections[0]["metadata"]["episodes"][0]["source"]
    with pytest.raises(ValueError, match="(?i)source|recording|duplicate|overlap"):
        resolve_collection_sampling(selections, control_hz=15, action_steps=10, require_validation=True)


def test_sampler_distinguishes_upstream_window_weighting_from_hand_balance_and_replays_ddp():
    hands = ["leap"] * 5 + ["shadow"] * 96
    native = UniDexMixtureSampler(hands, seed=8)
    assert sorted(native.global_indices()) == list(range(101))
    assert Counter(hands[index] for index in native.global_indices()) == {"leap": 5, "shadow": 96}
    balanced = UniDexMixtureSampler(hands, "hand_balanced", seed=8)
    counts = Counter(hands[index] for index in balanced.global_indices())
    assert sorted(counts.values()) == [50, 51]
    for epoch in (0, 1, 12):
        balanced.set_epoch(epoch)
        global_order = balanced.global_indices()
        shards = []
        for rank in range(3):
            replica = UniDexMixtureSampler(hands, "hand_balanced", seed=8, rank=rank, replicas=3)
            replica.set_epoch(epoch)
            shards.append(list(replica))
            assert len(replica) == 34
        interleaved = [value for row in zip(*shards) for value in row]
        assert interleaved[:101] == global_order and interleaved[101:] == global_order[:1]
        resumed = UniDexMixtureSampler(hands, "hand_balanced", seed=8)
        resumed.set_epoch(epoch)
        assert resumed.global_indices() == global_order
    balanced.set_epoch(0)
    assert balanced.global_indices() != global_order
    assert native.identity()["policy"] == "window_proportional"


def test_resume_pins_every_dataset_mixing_policy_and_optimizer_budget(tmp_path):
    args = SimpleNamespace(manifest_sha=None, batch_size=4, learning_rate=.0001, num_workers=1,
                           seed=42, precision="fp32", gpu_count=2, gradient_accumulation=4,
                           epochs=32, max_steps=123)
    _, metadata = recording_fixture(tmp_path, count=2)
    selections = [dict(position=i, version_id=f"version-{i}", manifest_sha256=str(i) * 64, metadata=metadata) for i in range(2)]
    sampler = UniDexMixtureSampler(["leap", "shadow"], "hand_balanced")
    identity = training_identity(args, {"horizon_steps": 30}, {}, {}, selections=selections, mixture=sampler.identity())
    saved = {"skynet": {**identity, "identity_sha256": stable_digest(identity)},
             "optimizer_states": [{}], "lr_schedulers": [{}],
             "loops": {"fit_loop": {"state_dict": {"combined_loader": [{"schema": "skynet.unidex-loader-state/v1"}]}}}}
    validate_resume(saved, identity)
    assert training_limits(args) == {"max_epochs": -1, "max_steps": 123}
    for defect in ("fingerprint", "version", "order", "mixture", "budget"):
        changed = copy.deepcopy(identity)
        if defect == "fingerprint":
            changed["datasets"][1]["manifest_sha256"] = "a" * 64
        elif defect == "version":
            changed["datasets"][1]["version_id"] = "replacement"
        elif defect == "order":
            changed["datasets"].reverse()
        elif defect == "mixture":
            changed["mixture"]["policy"] = "window_proportional"
        else:
            changed["budget"]["max_steps"] += 1
        with pytest.raises(ValueError, match="identical"):
            validate_resume(saved, changed)
    args.max_steps = None
    assert training_limits(args) == {"max_epochs": 32, "max_steps": -1}
    args.max_steps = 0
    with pytest.raises(ValueError, match="budget"):
        training_limits(args)


@pytest.mark.parametrize("policy", UniDexMixtureSampler.POLICIES)
def test_sampler_resume_tracks_completed_samples_independently_of_prefetch(policy):
    hands = ["shadow"] * 7 + ["leap"] * 3
    first = UniDexMixtureSampler(hands, policy, seed=17)
    first.set_epoch(3)
    expected = list(first)
    # A worker can request every index before the first batch completes.
    prefetched = iter(first)
    assert list(prefetched) == expected
    assert first.state_dict()["consumed_samples"] == 0
    first.mark_consumed(2)
    first.mark_consumed(2)
    resumed = UniDexMixtureSampler(hands, policy, seed=17)
    resumed.load_state_dict(first.state_dict())
    resumed.set_epoch(3)
    assert len(resumed) == len(hands)
    assert list(resumed) == expected[4:]
    resumed.mark_consumed(6)
    assert list(resumed) == []
    resumed.set_epoch(4)
    assert resumed.consumed_samples == 0
    assert len(list(resumed)) == len(hands)
    assert list(resumed) != expected


def test_distributed_resume_uses_same_completed_count_for_each_rank_sequence():
    hands = ["shadow"] * 7 + ["leap"] * 4
    ranks = [UniDexMixtureSampler(hands, "hand_balanced", seed=19, rank=i, replicas=2) for i in range(2)]
    for sampler in ranks:
        sampler.set_epoch(2)
    expected = [list(sampler) for sampler in ranks]
    ranks[0].mark_consumed(4)
    saved = ranks[0].state_dict()
    for rank in range(2):
        resumed = UniDexMixtureSampler(hands, "hand_balanced", seed=19, rank=rank, replicas=2)
        resumed.load_state_dict(saved)
        assert list(resumed) == expected[rank][4:]
        assert len(resumed) == 6
    with pytest.raises(ValueError, match="world size"):
        UniDexMixtureSampler(hands, "hand_balanced", seed=19).load_state_dict(saved)
    with pytest.raises(ValueError, match="identity"):
        UniDexMixtureSampler(hands, "window_proportional", seed=19, rank=0, replicas=2).load_state_dict(saved)
    with pytest.raises(ValueError, match="position"):
        ranks[1].load_state_dict({**saved, "consumed_samples": 99})
    with pytest.raises(ValueError, match="sample count"):
        ranks[0].mark_consumed(3)


def test_sampler_rejects_incomplete_epoch_reset_instead_of_replaying_or_skipping_data():
    sampler = UniDexMixtureSampler(["shadow"] * 8, seed=42)
    sampler.mark_consumed(2)
    with pytest.raises(ValueError, match="incomplete data epoch"):
        sampler.set_epoch(1)
    assert sampler.epoch == 0 and sampler.consumed_samples == 2


def test_resume_rejects_checkpoint_without_completed_data_position():
    identity = {"schema": RUN_SCHEMA, "dataset_format": FORMAT}
    saved = {"skynet": {**identity, "identity_sha256": stable_digest(identity)},
             "optimizer_states": [{}], "lr_schedulers": [{}]}
    with pytest.raises(ValueError, match="completed training data position"):
        validate_resume(saved, identity)
