"""CPU acceptance checks for incremental camera artifacts and derived clouds."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import sys

import h5py
import numpy as np
import pytest

from ops.datasets import observation_geometry as geometry
from ops.datasets import observation_prepare as worker
from ops.datasets.observation_render import configure_cameras, default_scene_camera_recipes, render_identity
from skynet_app.observation_contracts import CONTRACT_SCHEMA, PREPARE_SCHEMA, content_digest, plan_artifacts, rgb_requirements


REVISION = "30cc673e27684b9f10186fa6bea731aed246bc9f"


def profile():
    return dict(repository="/pinned/dexverse", source_revision=REVISION, task="Dexverse-PickCube-v0", robot="floating_shadow_right", hand="right")


def cloud_contract(cameras=("scene_front",), *, channels="XYZ", count=6, frame="world"):
    return dict(schema=CONTRACT_SCHEMA, timing={"alignment": "pre_action_state", "stride": 1}, streams=[dict(
        name="cloud", modality="point_cloud", camera_ids=list(cameras), width=3, height=2,
        channels=channels, coordinate_frame=frame, num_points=count, sampling="farthest_point", seed=7,
        processing_order=["unproject", "world_transform", "merge", "crop", "sample"],
        insufficient_points="repeat_with_mask", crop=None, depth_range=[0, 2.5])])


def source(tmp_path, *, index=0):
    payload = dict(format="dexverse_trajectory", schema_version=3, episode_name=f"episode-{index}", task=profile()["task"], robot_type=profile()["robot"],
                   num_episodes=1, skynet_step_dt=1/60, episodes=[dict(num_steps=2, actions=np.zeros((2, 7), dtype="f4"),
                   states=[{"n": np.array([n], dtype="f4")} for n in range(3)], success=True)])
    path = tmp_path / f"episode-{index}.pkl"
    path.write_bytes(pickle.dumps(payload))
    sha = worker.file_digest(path)
    return dict(episode_key=f"episode-{index}", recording=str(path), sha256=sha, profile=profile(), episode_index=0,
                render_recipe=render_identity(profile(), "a"*64))


def requests(tmp_path, src, contract):
    nodes = plan_artifacts(src["sha256"], src, contract, src["render_recipe"])
    for node in nodes:
        node["output_dir"] = str(tmp_path / node["artifact_key"])
        node["dependencies"] = [{"artifact_key": key, "path": str(tmp_path / key), "manifest_sha256": "0"*64} for key in node["dependencies"]]
    return nodes


def request(srcs, jobs, mode="render"):
    return dict(schema=PREPARE_SCHEMA, mode=mode, sources=srcs, requests=jobs, request_id="request-1", attempt_token="attempt-2")


class FakeRenderer:
    starts, frames, episodes = 0, [], []
    def __init__(self, sources, jobs, *, progress=None, staging_root=None):
        self.sources, self.jobs = sources, jobs
        self.progress = progress or (lambda phase, **fields: None)
        self.capture_metadata = dict(task=profile()["task"], robot=profile()["robot"], action_joint_names=["x"]*7, step_dt=1/60)
    def __enter__(self):
        type(self).starts += 1
        return self
    def __exit__(self, *_):
        pass
    def begin_episode(self, source, payload, episode, step_dt):
        type(self).episodes.append(source["episode_key"])
    def capture(self, state, jobs):
        frame = int(state["n"][0])
        type(self).frames.append((frame, [(j["camera_id"], j["modality"]) for j in jobs]))
        values = {}
        for job in jobs:
            recipe = job["recipe"]
            camera = values.setdefault(job["camera_id"], dict(intrinsics=np.eye(3), world_from_camera=np.eye(4)))
            if job["modality"] == "rgb":
                camera["rgb"] = np.full((recipe["height"], recipe["width"], 3), frame+10, dtype="u1")
            else:
                camera["depth"] = np.ones((recipe["height"], recipe["width"]), dtype="f4")
        return values


@pytest.fixture(autouse=True)
def reset_fake():
    FakeRenderer.starts, FakeRenderer.frames, FakeRenderer.episodes = 0, [], []


def test_batch_receipt_is_committed_before_renderer_exit(tmp_path):
    srcs = [source(tmp_path, index=n) for n in range(2)]
    jobs = [job for src in srcs for job in requests(tmp_path, src, rgb_requirements(['scene_front'], width=3, height=2))]
    receipts = []
    class ClosingFailure(FakeRenderer):
        def __exit__(self, *_):
            assert len(receipts) == 1
            assert receipts[0]['state'] == 'READY'
            assert {item['artifact_key'] for item in receipts[0]['artifacts']} == {job['artifact_key'] for job in jobs}
            for job in jobs:
                worker.verify_artifact(job['output_dir'], request=job)
            raise RuntimeError('native close failed')
    result = worker.run_request(request(srcs, jobs), renderer_factory=ClosingFailure,
                                publish_result=lambda value: receipts.append(deepcopy(value)))
    assert result['state'] == 'READY' and 'native close failed' in result['cleanup_warning']
    assert len(result['artifacts']) == 2 and 'error' not in result


def test_partial_capture_never_commits_success(tmp_path):
    srcs = [source(tmp_path, index=n) for n in range(2)]
    jobs = [job for src in srcs for job in requests(tmp_path, src, rgb_requirements(['scene_front'], width=3, height=2))]
    receipts = []
    class SecondEpisodeFailure(FakeRenderer):
        def begin_episode(self, src, *args):
            if src['episode_key'] == srcs[1]['episode_key']:
                raise RuntimeError('second episode failed')
            return super().begin_episode(src, *args)
    result = worker.run_request(request(srcs, jobs), renderer_factory=SecondEpisodeFailure,
                                publish_result=receipts.append)
    assert result['state'] == 'FAILED' and not receipts
    assert len(result['artifacts']) == 1
    assert 'second episode failed' in result['error']


def test_commit_failure_cannot_report_ready(tmp_path):
    src = source(tmp_path)
    jobs = requests(tmp_path, src, rgb_requirements(['scene_front'], width=3, height=2))
    def cannot_commit(_):
        raise OSError('receipt write failed')
    result = worker.run_request(request([src], jobs), renderer_factory=FakeRenderer, publish_result=cannot_commit)
    assert result['state'] == 'FAILED' and 'receipt write failed' in result['error']
    assert len(result['artifacts']) == 1  # The immutable file remains reusable.


def test_unprojection_metric_z_skew_and_pixel_color_pairing():
    depth = np.array([[2, np.inf], [np.nan, 4]], dtype="f4")
    k = np.array([[2, .5, 0], [0, 4, 0], [0, 0, 1]], dtype=float)
    rgb = np.arange(12, dtype="u1").reshape(2, 2, 3)
    xyz, colors = geometry.unproject(depth, k, rgb=rgb)
    np.testing.assert_allclose(xyz, [[0, 0, 2], [1.75, 1, 4]])
    np.testing.assert_array_equal(colors, [rgb[0, 0], rgb[1, 1]])


def test_merge_transforms_all_points_before_crop_sampling():
    transform = np.eye(4)
    transform[0, 3] = 100
    views = [dict(camera_id="front", depth=np.ones((1, 3), dtype="f4"), intrinsics=np.eye(3), world_from_camera=np.eye(4)),
             dict(camera_id="left", depth=np.ones((1, 3), dtype="f4"), intrinsics=np.eye(3), world_from_camera=transform)]
    recipe = dict(coordinate_frame="world", channels="XYZ", crop={"min": [99, -1, .5], "max": [103, 1, 2]},
                  num_points=4, sampling="farthest_point", seed=1, insufficient_points="repeat_with_mask")
    points, valid = geometry.point_cloud(views, recipe, identity="test", frame_id=0)
    np.testing.assert_allclose(points[:3, 0], [100, 101, 102])
    np.testing.assert_array_equal(valid, [True, True, True, False])
    assert tuple(points[3]) in {tuple(p) for p in points[:3]}


def test_camera_frame_cloud_and_color_normalization():
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    view = dict(camera_id="front", depth=np.ones((1, 1), dtype="f4"), rgb=np.array([[[255, 128, 0]]], dtype="u1"),
                intrinsics=np.eye(3), world_from_camera=transform)
    recipe = dict(coordinate_frame="camera:front", channels="XYZRGB", color_range="0_1", num_points=1, sampling="fps",
                  crop={"min": [-.5, -.5, .5], "max": [.5, .5, 1.5]})
    points, valid = geometry.point_cloud([view], recipe, identity="test", frame_id=0)
    np.testing.assert_allclose(points, [[0, 0, 1, 1, 128/255, 0]])
    assert valid.all()


def test_fps_is_deterministic_independent_of_global_rng_and_preserves_colors():
    points = np.column_stack([np.arange(10), np.zeros(10), np.ones(10), np.arange(10), np.zeros(10), np.ones(10)])
    first, _ = geometry.sample_points(points, 3, method="fps", seed=14)
    np.random.seed(100)
    np.random.random(500)
    second, _ = geometry.sample_points(points, 3, method="fps", seed=14)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 0], first[:, 3])
    assert len(np.unique(first[:, 0])) == 3


def test_rgb_then_front_depth_then_remaining_depth_only_missing_work(tmp_path):
    src = source(tmp_path)
    rgb_jobs = requests(tmp_path, src, rgb_requirements(width=3, height=2))
    first = worker.run_request(request([src], rgb_jobs), renderer_factory=FakeRenderer)
    assert first["state"] == "READY", first
    assert first["request_id"] == "request-1" and first["attempt_token"] == "attempt-2"
    frozen = {r["artifact_key"]: r["manifest_sha256"] for r in first["artifacts"]}
    all_nodes = requests(tmp_path, src, cloud_contract(channels="XYZRGB"))
    capture_jobs = [j for j in all_nodes if j["modality"] != "point_cloud"]
    FakeRenderer.frames = []
    second = worker.run_request(request([src], capture_jobs), renderer_factory=FakeRenderer)
    assert second["state"] == "READY", second
    assert all(jobs == [("scene_front", "depth")] for _, jobs in FakeRenderer.frames)
    reused = next(a for a in second["artifacts"] if a["reused"])
    assert frozen[reused["artifact_key"]] == reused["manifest_sha256"]
    third_nodes = requests(tmp_path, src, cloud_contract(("scene_front", "scene_left", "scene_right")))
    third_capture = [j for j in third_nodes if j["modality"] == "depth"]
    FakeRenderer.frames = []
    third = worker.run_request(request([src], third_capture), renderer_factory=FakeRenderer)
    assert third["state"] == "READY", third
    assert all(set(jobs) == {("scene_left", "depth"), ("scene_right", "depth")} for _, jobs in FakeRenderer.frames)
    assert FakeRenderer.starts == 3
    assert not any(p.name.startswith(".") and ".preparing-" in p.name for p in tmp_path.iterdir())


def test_offline_derive_reads_verified_depth_without_simulator(tmp_path):
    src = source(tmp_path)
    nodes = requests(tmp_path, src, cloud_contract(channels="XYZRGB"))
    rendered = worker.run_request(request([src], [j for j in nodes if j["modality"] != "point_cloud"]), renderer_factory=FakeRenderer)
    assert rendered["state"] == "READY", rendered
    job = nodes[-1]
    receipts = {a["artifact_key"]: a for a in rendered["artifacts"]}
    job["dependencies"] = [{k: receipts[d["artifact_key"]][k] for k in ("artifact_key", "path", "manifest_sha256")} for d in job["dependencies"]]
    src["recording"] = "/does/not/need/original.pkl"
    before = FakeRenderer.starts
    result = worker.run_request(request([src], [job], "derive"), renderer_factory=lambda *_: pytest.fail("CPU derivation launched simulator"))
    assert result["state"] == "READY", result
    assert FakeRenderer.starts == before
    with h5py.File(Path(result["artifacts"][0]["path"]) / "values.hdf5") as file:
        np.testing.assert_array_equal(file["values"][0, :, 3:], np.full((6, 3), 10))
        assert file["values"].shape == (2, 6, 6)
        np.testing.assert_array_equal(file["frame_ids"][:], [0, 1])
    assert json.loads((Path(result["artifacts"][0]["path"]) / "manifest.json").read_text())["capture"]["robot"] == profile()["robot"]


def test_corrupt_reused_bytes_fail_without_overwrite_or_render(tmp_path):
    src = source(tmp_path)
    jobs = requests(tmp_path, src, rgb_requirements(cameras=["scene_front"], width=3, height=2))
    first = worker.run_request(request([src], jobs), renderer_factory=FakeRenderer)
    assert first["state"] == "READY"
    path = Path(jobs[0]["output_dir"]) / "values.hdf5"
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    second = worker.run_request(request([src], jobs), renderer_factory=FakeRenderer)
    assert second["state"] == "FAILED" and "checksum" in second["error"]
    assert FakeRenderer.starts == 1
    assert path.read_bytes().endswith(b"corrupt")


def test_backend_failure_publishes_no_partial_artifact(tmp_path):
    src = source(tmp_path)
    jobs = requests(tmp_path, src, rgb_requirements(cameras=["scene_front"], width=3, height=2))
    class Failing(FakeRenderer):
        def capture(self, state, jobs):
            if state["n"][0] == 1:
                raise RuntimeError("renderer stopped")
            return super().capture(state, jobs)
    result = worker.run_request(request([src], jobs), renderer_factory=Failing)
    assert result["state"] == "FAILED" and "renderer stopped" in result["error"]
    assert not Path(jobs[0]["output_dir"]).exists()
    assert not list(tmp_path.glob(".*.preparing-*"))


def test_invalid_source_stops_before_expensive_backend(tmp_path):
    src = source(tmp_path)
    jobs = requests(tmp_path, src, rgb_requirements(cameras=["scene_front"], width=3, height=2))
    Path(src["recording"]).write_bytes(b"changed")
    result = worker.run_request(request([src], jobs), renderer_factory=FakeRenderer)
    assert result["state"] == "FAILED" and "checksum" in result["error"]
    assert FakeRenderer.starts == 0


def test_two_episode_batch_uses_single_renderer_and_only_preaction_states(tmp_path):
    sources = [source(tmp_path, index=i) for i in range(2)]
    # Make source bytes distinct while retaining a valid trajectory.
    raw = pickle.loads(Path(sources[1]["recording"]).read_bytes())
    raw["episode_name"] = "second"
    Path(sources[1]["recording"]).write_bytes(pickle.dumps(raw))
    sources[1]["sha256"] = worker.file_digest(sources[1]["recording"])
    jobs = sum([requests(tmp_path, s, rgb_requirements(cameras=["scene_front"], width=3, height=2)) for s in sources], [])
    result = worker.run_request(request(sources, jobs), renderer_factory=FakeRenderer)
    assert result["state"] == "READY", result
    assert FakeRenderer.starts == 1 and FakeRenderer.episodes == ["episode-0", "episode-1"]
    assert [frame for frame, _ in FakeRenderer.frames] == [0, 1, 0, 1]


def test_source_pickle_does_not_execute_arbitrary_globals(tmp_path):
    src = source(tmp_path)
    class Untrusted:
        def __reduce__(self):
            return (eval, ("1+1",))
    path = Path(src["recording"])
    path.write_bytes(pickle.dumps(Untrusted()))
    src["sha256"] = worker.file_digest(path)
    with pytest.raises(ValueError, match="Unsupported recording object"):
        worker._load_source(src)


def test_camera_recipe_identity_excludes_paths_but_pins_asset_bytes():
    one = dict(profile(), runtime="/one/isaacsim", asset_bundle="/one/version-1",
               asset_bundle_manifest_sha256="b"*64, asset_inventory_sha256="c"*64)
    two = dict(one, runtime="/two/isaacsim", repository="/two/dexverse", asset_bundle="/two/version-1")
    assert render_identity(one, "a"*64) == render_identity(two, "a"*64)
    two["asset_inventory_sha256"] = "d"*64
    assert render_identity(one, "a"*64) != render_identity(two, "a"*64)
    with pytest.raises(ValueError, match="pinned"):
        render_identity(dict(profile(), asset_bundle="/unpinned"), "a"*64)


def test_camera_selection_does_not_keep_unrequested_depth_or_wrist():
    from types import SimpleNamespace as NS
    cameras = default_scene_camera_recipes(REVISION)
    scene = NS()
    for recipe in cameras.values():
        setattr(scene, recipe["sensor"], NS(prim_path=recipe["prim_path"], offset=NS(), spawn=NS(**recipe["projection"])))
    scene.wrist_camera = NS()
    scene.wrist_camera_body = NS()
    cfg = NS(scene=scene, _apply_observation_preset=lambda *_: None, _apply_multiview_cameras=lambda *_: None)
    jobs = [dict(camera_id="scene_front", modality="rgb", recipe=dict(camera=cameras["scene_front"], width=3, height=2))]
    selected = configure_cameras(cfg, jobs)
    assert set(selected) == {"scene_front"}
    assert scene.third_person_camera.data_types == ["rgb"]
    assert scene.third_person_camera_left is None and scene.third_person_camera_right is None and scene.wrist_camera is None
    assert scene.wrist_camera_body is not None
    assert cfg.events == {} and cfg.observations == {}


def test_spatially_accelerated_fps_matches_full_distance_reference():
    points = np.random.default_rng(24).uniform(size=(2100, 3))
    actual, _ = geometry.sample_points(points, 80, method="fps", seed=3)
    distances = np.full(len(points), np.inf)
    candidate = int(np.random.default_rng(3).integers(len(points)))
    selected = []
    for _ in range(80):
        selected.append(candidate)
        distances = np.minimum(distances, ((points - points[candidate])**2).sum(axis=1))
        distances[selected] = -1
        candidate = int(np.argmax(distances))
    np.testing.assert_array_equal(actual, points[selected].astype("f4"))


def test_xyzrgb_rejects_camera_calibration_mismatch(tmp_path):
    src = source(tmp_path)
    nodes = requests(tmp_path, src, cloud_contract(channels="XYZRGB"))
    capture_jobs = [j for j in nodes if j["modality"] != "point_cloud"]
    class MismatchedCamera(FakeRenderer):
        def capture(self, state, jobs):
            result = super().capture(state, jobs)
            if jobs[0]["modality"] == "rgb":
                result["scene_front"]["world_from_camera"][0, 3] = .01
            return result
    receipts = {}
    for job in capture_jobs:
        result = worker.run_request(request([src], [job]), renderer_factory=MismatchedCamera)
        assert result["state"] == "READY", result
        receipts[job["artifact_key"]] = result["artifacts"][0]
    cloud = nodes[-1]
    cloud["dependencies"] = [{k: receipts[d["artifact_key"]][k] for k in ("artifact_key", "path", "manifest_sha256")} for d in cloud["dependencies"]]
    result = worker.run_request(request([src], [cloud], "derive"))
    assert result["state"] == "FAILED" and "calibration is not aligned" in result["error"]
    assert not Path(cloud["output_dir"]).exists()


def test_recorded_metadata_can_come_from_verified_legacy_images(tmp_path):
    src = source(tmp_path)
    payload = pickle.loads(Path(src["recording"]).read_bytes())
    payload.pop("skynet_step_dt")
    Path(src["recording"]).write_bytes(pickle.dumps(payload))
    src["sha256"] = worker.file_digest(src["recording"])
    path = tmp_path / "legacy.hdf5"
    with h5py.File(path, "w") as file:
        file.attrs["metadata"] = json.dumps(dict(source_sha256=src["sha256"], task=profile()["task"], robot=profile()["robot"], step_dt=1/60))
    src.update(images=str(path), image_sha256=worker.file_digest(path))
    _, _, step_dt = worker._load_source(src)
    assert step_dt == 1/60
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        worker._load_source(src)


def test_release_without_timestamps_uses_explicit_pinned_renderer_timing(tmp_path):
    src = source(tmp_path)
    payload = pickle.loads(Path(src["recording"]).read_bytes())
    payload.pop("skynet_step_dt")
    Path(src["recording"]).write_bytes(pickle.dumps(payload))
    src["sha256"] = worker.file_digest(src["recording"])
    jobs = requests(tmp_path, src, rgb_requirements(cameras=["scene_front"], width=3, height=2))
    result = worker.run_request(request([src], jobs), renderer_factory=FakeRenderer)
    assert result["state"] == "READY", result
    manifest = worker.verify_artifact(result["artifacts"][0]["path"])
    assert manifest["capture"]["step_dt_source"] == "pinned_renderer"
    assert manifest["timing"]["step_dt"] == 1/60


def test_producer_scoped_staging_is_cleaned_on_failure(tmp_path):
    src = source(tmp_path)
    jobs = requests(tmp_path, src, rgb_requirements(cameras=["scene_front"], width=3, height=2))
    stage = tmp_path / "producer" / "staging"
    req = request([src], jobs)
    req["staging_root"] = str(stage)
    class FailingRenderer(FakeRenderer):
        def capture(self, state, jobs):
            assert len(list(stage.iterdir())) == 1
            raise ValueError("capture failed")
    result = worker.run_request(req, renderer_factory=FailingRenderer)
    assert result["state"] == "FAILED" and "capture failed" in result["error"]
    assert list(stage.iterdir()) == []
    assert not Path(jobs[0]["output_dir"]).exists()
    link = tmp_path / "linked-staging"
    link.symlink_to(stage, target_is_directory=True)
    req["staging_root"] = str(link)
    result = worker.run_request(req, renderer_factory=FakeRenderer)
    assert result["state"] == "FAILED" and "nonsymlink" in result["error"]


def test_scene_assets_require_pinned_receipts_or_private_generated_files(tmp_path):
    from types import SimpleNamespace as NS
    from ops.datasets.observation_render import DexVerseRenderer
    asset = tmp_path / 'source' / 'object.usd'
    asset.parent.mkdir()
    asset.write_text('original asset')
    render = DexVerseRenderer({}, [])
    render.profile = {'runtime': str(tmp_path/'runtime')}
    render.cameras, render.asset_receipts, render.scene_asset_digests = {}, {}, {}
    cfg = NS(scene=NS(object=NS(to_dict=lambda: {'usd_path':str(asset)})))
    with pytest.raises(ValueError, match='pinned inventory'):
        render._scene_identity(cfg)
    render.asset_receipts[str(asset)] = {'sha256':worker.file_digest(asset), 'size_bytes':asset.stat().st_size}
    assert len(render._scene_identity(cfg)) == 64
    asset.write_text('mutated asset!')
    with pytest.raises(ValueError, match='changed'):
        render._scene_identity(cfg)


def test_hand_manifest_bytes_are_part_of_render_identity():
    hand = dict(root='/bundle',digest='b'*64,manifest_sha256='c'*64)
    one = render_identity(dict(profile(),hand_bundle=hand),'a'*64)
    two = render_identity(dict(profile(),hand_bundle=dict(hand,manifest_sha256='d'*64)),'a'*64)
    assert one != two
    with pytest.raises(ValueError,match='pinned'):
        render_identity(dict(profile(),hand_bundle=dict(root='/bundle',digest='b'*64)),'a'*64)


def test_generated_hand_scene_identity_excludes_private_temporary_path(tmp_path):
    from types import SimpleNamespace as NS
    from ops.datasets.observation_render import DexVerseRenderer
    hashes = []
    for name in ['rgb-job', 'depth-job']:
        root = tmp_path / name / 'usd'
        root.mkdir(parents=True)
        asset = root / 'hand.usd'
        asset.write_text('USD with private source path ' + str(root))
        render = DexVerseRenderer({}, [])
        render.generated_asset_root = root
        render.profile = {'runtime':str(tmp_path/'runtime'), 'hand_bundle': {'manifest_sha256':'a'*64}}
        render.cameras, render.asset_receipts, render.scene_asset_digests = {}, {}, {}
        cfg = NS(scene=NS(robot=NS(to_dict=lambda: {'usd_path':str(asset)})))
        hashes.append(render._scene_identity(cfg))
    assert hashes[0] == hashes[1]


def test_missing_source_asset_binds_only_to_explicit_verified_bundle(tmp_path):
    from types import SimpleNamespace as NS
    from ops.datasets.observation_render import bind_scene_assets
    original = tmp_path / 'source' / 'hand.usd'
    frozen = tmp_path / 'bundle' / 'hand.usd'
    frozen.parent.mkdir()
    frozen.write_text('pinned USD')
    scene = NS(robot=NS(spawn=NS(usd_path=str(original))), camera=NS(prim_path='/World/Camera'))
    bind_scene_assets(scene, {str(original): {'path':str(frozen), 'sha256':worker.file_digest(frozen)}})
    assert scene.robot.spawn.usd_path == str(frozen)
    assert scene.camera.prim_path == '/World/Camera'


def test_native_capture_deadline_reports_failure_and_restores_alarm():
    import signal
    import time
    from ops.datasets.observation_render import capture_deadline

    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError, match="Camera capture timed out"):
        with capture_deadline(0.01):
            time.sleep(1)
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
