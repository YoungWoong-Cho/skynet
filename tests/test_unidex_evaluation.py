"""CPU checks for the Skynet FAAS simulator bridge; no GPU rollout is implied."""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from ops.datasets.action_codecs.geometry import inverse_transform
from ops.datasets.action_codecs.unidex import encode_absolute, load_codec, supported_robots
from ops.datasets.observation_geometry import point_cloud, pose_from_ros
from ops.datasets import observation_render as render
from skynet_app.adapters.policy_contract import contract_issues, unidex_contract, validate_observation
from skynet_app.adapters import unidex_evaluation as bridge
from skynet_app.adapters.unidex_runtime import RUN_SCHEMA, validate_checkpoint_inputs


def metadata(robot="skynet_leap_v1_right", *, points=1024):
    codec = load_codec(robot)
    capture = codec.spec.capture
    capture["cameras"] = {"scene_front": dict(sensor="third_person_camera", mount="fixed_scene", width=256, height=256,
        intrinsic_matrix=[[293.19970703125, 0., 128.], [0., 293.19970703125, 128.], [0., 0., 1.]],
        position_world=[-1.5, 0., 1.5],
        quaternion_world_ros=[-.4048279821872711, .5797538161277771, -.5797535181045532, .4048279821872711])}
    recipe = dict(camera_ids=["scene_front"], channels="XYZRGB", coordinate_frame="camera:scene_front",
        camera_convention="ros_optical", color_range="0_1", num_points=points, sampling="farthest_point", crop=None,
        depth_range=[.01, 5.], insufficient_points="repeat_with_mask", seed=0, width=256, height=256,
        dtype="float32", modality="point_cloud", processing_order=["unproject", "world_transform", "merge", "crop", "sample"])
    episode = dict(hand_id=robot, capture=capture, prompt="Pick up the cube", source={"sha256": "a" * 64},
        action_representation=dict(codec_sha256=codec.digest, frame="camera_opengl"),
        streams={"scene_front_pointcloud": {"recipe": recipe}})
    return dict(contract="skynet.unidex-pointcloud-faas/v1", capture=capture, episodes=[episode])


def context(manifest):
    return dict(policy={"native_config": {"datasets": [dict(position=0, version_id="train", path="/training",
        manifest_sha256="b" * 64, metadata=metadata("floating_shadow_right"))]}},
        target_dataset=dict(position=0, version_id="target", path="/target", manifest_sha256="c" * 64, metadata=manifest),
        checkpoint={"path": "/checkpoint.ckpt", "sha256": "d" * 64},
        compatibility={"io_contract": unidex_contract(manifest, control_hz=30)})


@pytest.mark.parametrize("robot", supported_robots())
@pytest.mark.parametrize("points", [1024, 10000])
def test_target_contract_retains_native_order_and_measured_camera(robot, points):
    manifest = metadata(robot, points=points)
    contract = unidex_contract(manifest, control_hz=30)
    assert contract_issues(contract) == []
    assert contract["joint_names"] == load_codec(robot).action_names
    assert contract["policy_to_source_indices"] == list(range(load_codec(robot).action_dim))
    assert contract["step_dt"] == 1 / 30 and contract["source_step_dt"] == 1 / 60
    assert contract["cameras"] == manifest["capture"]["cameras"]
    assert contract["pointcloud_recipe"]["num_points"] == points


@pytest.mark.parametrize("corrupt,message", [
    (lambda m: m["episodes"].append(metadata("floating_shadow_right")["episodes"][0]), "exactly one"),
    (lambda m: m["episodes"][0]["action_representation"].update(codec_sha256="0" * 64), "exact recorded FAAS"),
    (lambda m: m["episodes"][0]["streams"]["scene_front_pointcloud"]["recipe"].update(coordinate_frame="world"), "point-cloud recipe"),
    (lambda m: m["capture"]["cameras"]["scene_front"].update(mount="moving_wrist"), "fixed-scene"),
    (lambda m: m["capture"]["cameras"]["scene_front"].update(quaternion_world_ros=[0, 0, 0, 0]), "normalized"),
    (lambda m: m["capture"]["cameras"]["scene_front"].update(position_world=[-2., 0., 1.5]), "pinned observation camera"),
])
def test_target_contract_rejects_ambiguous_or_changed_geometry(corrupt, message):
    manifest = metadata()
    corrupt(manifest)
    with pytest.raises(ValueError, match=message):
        unidex_contract(manifest, control_hz=30)


@pytest.mark.parametrize("robot", supported_robots())
def test_faas_inference_decodes_exact_native_commands_from_measured_anchor(monkeypatch, robot):
    manifest = metadata(robot)
    codec = load_codec(robot)
    measured = np.linspace(.01, .1, codec.action_dim)
    commands = np.stack([measured + .01, measured + .02])
    encoded, _ = codec.encode_actions(commands, measured)
    calls = []

    class Policy:
        control_hz, horizon = 30, 2
        def __init__(self, *args, **kwargs):
            assert kwargs["expected_datasets"][0]["version_id"] == "train"
            self.pointcloud_recipe = unidex_contract(manifest)["pointcloud_recipe"]
        def predict(self, cloud, state, prompt):
            calls.append((cloud, state, prompt))
            return encoded

    monkeypatch.setattr(bridge, "UniDexPolicy", Policy)
    policy = bridge.RecordedPolicy(context(manifest), "/source", manifest, device="cpu")
    camera_pose = pose_from_ros([-1.5, 0., 1.5], [1., 0., 0., 0.])
    root_pose = pose_from_ros([.3, .2, .1], [1., 0., 0., 0.])
    observation = dict(state=measured, images={"scene_front": np.zeros((256, 256, 3), dtype="u1")},
        pointcloud=np.zeros((1024, 6), dtype="f4"), world_from_camera=camera_pose, world_from_root=root_pose)
    assert policy.step(observation, False) is None and not calls
    actual = policy.step(observation, True)
    np.testing.assert_allclose(actual, commands, atol=1e-6)
    expected_state = encode_absolute(codec, measured, camera_from_root=np.diag([1., -1., -1., 1.]) @ inverse_transform(camera_pose) @ root_pose)
    np.testing.assert_allclose(calls[0][1], expected_state, atol=1e-6)
    assert calls[0][2] == "Pick up the cube"


def test_invalid_target_instruction_and_unseen_claim_fail_before_loading_model(monkeypatch):
    monkeypatch.setattr(bridge, "UniDexPolicy", lambda *a, **kw: pytest.fail("must reject before model load"))
    manifest = metadata()
    ctx = context(manifest)
    ctx["unseen_embodiment"] = True
    ctx["policy"]["native_config"]["datasets"][0]["metadata"] = manifest
    with pytest.raises(ValueError, match="unseen hand"):
        bridge.RecordedPolicy(ctx, "/source", manifest)
    ctx["unseen_embodiment"] = False
    manifest["episodes"][0]["prompt"] = ""
    with pytest.raises(ValueError, match="task instruction"):
        bridge.RecordedPolicy(ctx, "/source", manifest)


def test_camera_reconstruction_preserves_task_events_and_success():
    contract = unidex_contract(metadata(), control_hz=30)
    measured = contract["cameras"]["scene_front"]
    recipe = render.camera_recipe_from_capture(measured, contract["source_revision"])
    k = np.array(measured["intrinsic_matrix"])
    assert recipe["projection"]["horizontal_aperture"] * k[0, 0] == pytest.approx(256 * 24.)
    assert recipe["offset"]["convention"] == "ros"
    cameras = render.default_scene_camera_recipes(contract["source_revision"])
    scene = NS()
    for camera in cameras.values():
        setattr(scene, camera["sensor"], NS(prim_path=camera["prim_path"], offset=NS(),
            spawn=NS(**camera["projection"], vertical_aperture=0.)))
    cfg = NS(scene=scene, _apply_observation_preset=lambda *_: None, _apply_multiview_cameras=lambda *_: None,
        events={"reset": "randomization"}, terminations={"success": "lift"}, observations={"state": "original"})
    bridge.configure_pointcloud_scene(cfg, contract)
    assert cfg.events == {"reset": "randomization"} and cfg.terminations == {"success": "lift"}
    assert scene.third_person_camera.data_types == ["distance_to_image_plane", "rgb"]
    assert scene.third_person_camera_left is None
    assert scene.third_person_camera.offset.pos == tuple(measured["position_world"])


@pytest.mark.parametrize("points", [1024, 10000])
def test_live_observation_reuses_conversion_geometry_and_rejects_camera_drift(monkeypatch, points):
    monkeypatch.setitem(sys.modules, 'observation_render', render)
    contract = unidex_contract(metadata(points=points), control_hz=30)
    camera = contract["cameras"]["scene_front"]
    depth = np.zeros((256, 256), dtype="f4")
    depth[100:103, 127:130] = 1.2
    rgb = np.full((256, 256, 3), 127, dtype="u1")
    sensor = NS(_annotators={"rgb": NS(get_data=lambda: rgb)}, update=lambda *a, **kw: None,
        data=NS(intrinsic_matrices=np.array([camera["intrinsic_matrix"]]), pos_w=np.array([camera["position_world"]]),
        quat_w_ros=np.array([camera["quaternion_world_ros"]]), output={"rgb": rgb[None], "distance_to_image_plane": depth[None]}))
    n = len(contract["joint_names"])
    robot = NS(data=NS(joint_pos=np.arange(n)[None], root_pos_w=np.array([[.1, .2, .3]]), root_quat_w=np.array([[1, 0, 0, 0]])))
    env = NS(scene={"third_person_camera": sensor, "robot": robot}, sim=NS(render=lambda: None, render_mode="test"))
    captured = []
    monkeypatch.setattr(render, "capture_current_frame", lambda env: captured.append(env))
    observation = bridge.observation_from_sensors(env, contract, list(range(n)), identity="episode", frame_id=4)
    view = dict(camera_id="scene_front", depth=depth, rgb=rgb, intrinsics=np.array(camera["intrinsic_matrix"]),
        world_from_camera=pose_from_ros(camera["position_world"], camera["quaternion_world_ros"]))
    expected, _ = point_cloud([view], contract["pointcloud_recipe"], identity="episode", frame_id=4)
    np.testing.assert_array_equal(observation["pointcloud"], expected)
    assert observation["pointcloud"].shape == (points, 6)
    np.testing.assert_array_equal(observation["world_from_root"][:3, 3], [.1, .2, .3])
    assert captured == [env]
    sensor.data.intrinsic_matrices[0, 0, 0] += 1
    with pytest.raises(ValueError, match="Live camera calibration"):
        bridge.observation_from_sensors(env, contract, list(range(n)), identity="episode", frame_id=5)


def test_shared_render_scheduler_does_not_advance_physics(monkeypatch):
    calls, time = [], [4.]
    orchestrator = NS(step=lambda **kw: calls.append(kw))
    timeline = NS(get_timeline_interface=lambda: NS(get_current_time=lambda: time[0]))
    replicator = NS(core=NS(orchestrator=orchestrator))
    monkeypatch.setitem(sys.modules, "omni", NS(replicator=replicator, timeline=timeline))
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", replicator.core)
    monkeypatch.setitem(sys.modules, "omni.timeline", timeline)
    env = NS(physics_dt=1 / 120, scene=NS(update=lambda **kw: calls.append(kw)))
    render.capture_current_frame(env)
    assert calls == [dict(rt_subframes=4, pause_timeline=False, delta_time=0., wait_for_render=True), dict(dt=1 / 120)]
    orchestrator.step = lambda **kw: time.__setitem__(0, time[0] + .01)
    with pytest.raises(RuntimeError, match="advanced the simulation timeline"):
        render.capture_current_frame(env)


def test_frozen_hand_install_uses_private_verified_copy(monkeypatch, tmp_path):
    original = tmp_path / "source"
    original.mkdir()
    contents = b"frozen asset"
    (original / "simulation.urdf").write_bytes(contents)
    manifest = dict(digest="hand-id", files={"simulation.urdf": dict(sha256=hashlib.sha256(contents).hexdigest(), size_bytes=len(contents))})
    (original / "manifest.json").write_text(json.dumps(manifest))
    profile = dict(hand_bundle=dict(root=str(original), digest="hand-id", manifest_sha256=hashlib.sha256((original / "manifest.json").read_bytes()).hexdigest()))
    installations = []
    monkeypatch.setitem(sys.modules, "runtime", NS(install=lambda p: installations.append(p) or manifest, validate_environment=lambda *_: None))
    receipts = {}
    work, installed, validate = render.install_frozen_hand(profile, staging_root=str(tmp_path / "private"), asset_receipts=receipts)
    try:
        assert installations == [Path(work.name)] and installed == manifest
        assert Path(work.name) != original and (Path(work.name) / "simulation.urdf").read_bytes() == contents
        (Path(work.name) / "generated.usd").write_bytes(b"generated")
        assert not (original / "generated.usd").exists()
        assert str(Path(work.name) / "simulation.urdf") in receipts
    finally:
        work.cleanup()
    (original / "simulation.urdf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen hand asset changed"):
        render.install_frozen_hand(profile, staging_root=str(tmp_path / "private"))


def test_checkpoint_verifies_every_training_identity_and_legacy_single_input():
    selected = [dict(position=i, version_id=f"v{i}", manifest_sha256=str(i) * 64) for i in range(2)]
    receipt = dict(schema=RUN_SCHEMA, datasets=copy.deepcopy(selected))
    validate_checkpoint_inputs(receipt, selected)
    selected[1]["version_id"] = "replacement"
    with pytest.raises(ValueError, match="training dataset"):
        validate_checkpoint_inputs(receipt, selected)
    validate_checkpoint_inputs(dict(schema="skynet.unidex-run/v2", manifest_sha256="1" * 64), [selected[1]])


def test_frozen_evaluator_capsule_imports_without_the_skynet_checkout(tmp_path):
    from skynet_app.adapters.unidex_manifest import manifest
    from skynet_app.evaluation_compatibility import compose_evaluator
    original = manifest()
    composed = compose_evaluator({"source": {"adapter": "unidex"}}, original,
                                  {"name": "dexverse_recorded", "evaluator_adapter": "isaac_lab"})
    assert original.evaluations == []
    assert "adapter-support/unidex_input.py" in composed.evaluations[0].command.capsule_files
    for name, content in composed.evaluations[0].command.capsule_files.items():
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    source = "import sys; sys.path.insert(0, sys.argv[1]); import unidex_evaluation, recorded_policy_evaluation, dexverse_evaluation; from action_codecs.unidex import supported_robots; assert len(supported_robots()) == 7"
    subprocess.run([sys.executable, "-I", "-c", source, str(tmp_path / "adapter-support")],
                   cwd=tmp_path, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("points,other", [(1024, 10000), (10000, 1024)])
def test_live_observation_shape_follows_frozen_recipe_not_current_default(points, other):
    contract = unidex_contract(metadata(points=points), control_hz=30)
    obs = dict(state=np.zeros(len(contract["joint_names"])),
               images={"scene_front": np.zeros((256, 256, 3), dtype="u1")},
               pointcloud=np.zeros((points, 6), dtype="f4"),
               world_from_camera=np.eye(4), world_from_root=np.eye(4))
    validate_observation(contract, obs)
    obs["pointcloud"] = np.zeros((other, 6), dtype="f4")
    with pytest.raises(ValueError, match=f"XYZRGB{points}"):
        validate_observation(contract, obs)
    del obs["pointcloud"]
    validate_observation(contract, obs, require_pointcloud=False)


@pytest.mark.parametrize("training_points,target_points", [(1024, 10000), (10000, 1024)])
def test_loaded_checkpoint_recipe_must_match_target_before_any_prediction(monkeypatch, training_points, target_points):
    target = metadata(points=target_points)

    class Policy:
        def __init__(self, *args, **kwargs):
            self.pointcloud_recipe = unidex_contract(metadata(points=training_points))["pointcloud_recipe"]
        def predict(self, *args):
            pytest.fail("Incompatible checkpoint inputs must not reach inference")

    monkeypatch.setattr(bridge, "UniDexPolicy", Policy)
    with pytest.raises(ValueError, match="Checkpoint point-cloud recipe differs"):
        bridge.RecordedPolicy(context(target), "/source", target, device="cpu")


@pytest.mark.parametrize("training_points,target_points,compatible", [
    (10000, 10000, True), (1024, 1024, True), (10000, 1024, False), (1024, 10000, False),
])
def test_submission_compares_frozen_training_and_target_pointcloud_recipes(training_points, target_points, compatible):
    from skynet_app.evaluation_compatibility import inspect_compatibility
    from skynet_app.adapters.unidex_manifest import manifest as adapter_manifest

    def prepared(hand, points):
        value = metadata(hand, points=points)
        value["capture"]["task"] = "Dexverse-PickCube-v0"
        first = value["episodes"][0]
        value["episodes"] = []
        for index in range(2):
            episode = copy.deepcopy(first)
            episode.update(index=index, id=f"{hand}-{index}", steps=120,
                           source={"sha256": hashlib.sha256(f"{hand}-{index}".encode()).hexdigest()})
            value["episodes"].append(episode)
        value.update(format="skynet.recording-dataset/v1", validation={"status": "PASSED"},
                     split={"train": [0], "validation": [1]})
        return value

    training = prepared("floating_shadow_right", training_points)
    target = prepared("skynet_wuji_2_right", target_points)
    spec = dict(source={"adapter": "unidex"}, native={"config": {"control_hz": 30, "action_steps": 30}},
                data={"bundle": {"assignments": [dict(role="training_data", position=0,
                    version=dict(format="skynet.recording-dataset/v1", manifest_sha256="a" * 64, metadata=training),
                    config={"location": dict(kind="cluster", status="AVAILABLE", path="/training", manifest_sha256="a" * 64)})]}})
    suite = dict(name="dexverse_recorded", evaluator_adapter="isaac_lab", config_json=dict(
        tasks=["Dexverse-PickCube-v0"], dataset_task_binding={"role": "training_data", "metadata_path": "capture.task"},
        unseen_embodiment=True, target_dataset=dict(metadata=target, path="/target", manifest_sha256="b" * 64)))
    report, _ = inspect_compatibility(spec, adapter_manifest(), suite,
                                      checkpoint={"status": "AVAILABLE", "sha256": "c" * 64})
    assert report["ready"] is compatible, report
    assert report["io_contract"]["pointcloud_recipe"]["num_points"] == target_points
    if not compatible:
        assert report["status"] == "incompatible"
        assert any("point-cloud recipe differs" in message for message in report["messages"])
