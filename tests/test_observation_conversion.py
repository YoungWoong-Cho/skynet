"""Real CPU conversion parity: shared render inputs, self-contained v1 outputs."""
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest
import zarr

from skynet_app.dataset_formats import RECIPES
from skynet_app.live_xr_review import ArrayUnpickler
from skynet_app.adapters.observation_artifacts import digest
from test_observation_contracts import artifact

ROOT = Path(__file__).parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "ops/xr"))
    from images import ImageWriter
    from recording_metadata import joint_layout
    worker = tmp_path / "worker"
    worker.mkdir()
    for name in ("policy_export.py", "egoverse_export.py", "egoverse_zarr_writer.py", "egoverse-provenance.json", "artifacts.py"):
        shutil.copyfile(ROOT / "ops/datasets" / name, worker / name)
    for name in ("images.py", "recording_metadata.py"):
        shutil.copyfile(ROOT / "ops/xr" / name, worker / name)
    for name in ("egoverse_splits.py", "observation_artifacts.py"):
        shutil.copyfile(ROOT / "skynet_app/adapters" / name, worker / name)
    shutil.copyfile(ROOT / "skynet_app/trajectory.py", worker / "trajectory.py")
    shutil.copytree(ROOT / "ops/datasets/xpolicylab", worker / "xpolicylab")
    (worker / "arrays.py").write_text("import pickle\nimport numpy as np\n" + inspect.getsource(ArrayUnpickler))
    (worker / "formats.json").write_text(json.dumps({key: value["format"] for key, value in RECIPES.items()}))
    names = [f"{axis}_translation_joint" for axis in "xyz"] + [f"{axis}_rotation_joint" for axis in "zyx"] + ["finger_a", "finger_b"]
    cameras = ["scene_front", "scene_left", "scene_right"]
    sources, legacy, original = [], [], []
    for index in range(2):
        location = tmp_path / f"episode-{index}"
        location.mkdir()
        metadata = dict(robot="floating_shadow_hand", task="test-task", hand="right", source_revision="abc",
            action_joint_names=names, robot_joint_names=names, groups=joint_layout(names, "right"),
            action_scale=[1.0] * 8, action_offset=[0.0] * 8,
            action_semantics="raw_joint_position_command; target = action * scale + offset", step_dt=1/60,
            color_space="RGB", alignment="image and state before action; legacy prose",
            cameras={name: dict(mount="fixed_scene", width=256, height=256, intrinsic_matrix=np.eye(3).tolist(),
                position_world=[0, 0, 0], quaternion_world_ros=[1, 0, 0, 0]) for name in cameras})
        actions = np.arange(24, dtype="float32").reshape(3, 8) / 100 + index
        states = actions + 10
        frames = {camera: np.arange(3*256*256*3, dtype="uint8").reshape(3, 256, 256, 3) + np.uint8(i * 10) for i, camera in enumerate(cameras)}
        writer = ImageWriter(location, metadata)
        for step in range(3):
            writer.append(states[step], actions[step], {camera: images[step] for camera, images in frames.items()}, step/60, 1000+step/60)
        receipt = writer.finish()
        episode = dict(actions=actions, num_steps=3, success=True, skynet_images=receipt,
            states=[{"articulation": {"robot": {"joint_position": value[None]}}} for value in [*states, states[-1]+1]])
        payload = dict(format="dexverse_trajectory", schema_version=3, robot_type=metadata["robot"], task=metadata["task"],
            num_episodes=1, episodes=[episode])
        if index == 0:
            payload["skynet_state_metadata"] = dict(metadata, alignment="state before action; new prose")
        recording = location / "episode.pkl"
        recording.write_bytes(pickle.dumps(payload))
        source = dict(recording=str(recording), sha256=digest(recording), index=index, session_id="session")
        legacy.append(dict(source, images=str(location / receipt["path"]), image_sha256=receipt["sha256"]))
        references = {}
        for i, camera in enumerate(cameras):
            reference, _ = artifact(location / camera, camera, source=source["sha256"], offset=i*10, size=(256, 256))
            manifest_path = Path(reference["path"]) / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["capture"] = dict(metadata, alignment="pre_action_state", render_mode="saved_states", step_dt_source="renderer")
            manifest_path.write_text(json.dumps(manifest))
            reference["manifest_sha256"] = digest(manifest_path)
            references[camera] = {"rgb": reference}
        sources.append(dict(source, observation_artifacts=references,
                            observation_streams={camera: value["rgb"] for camera, value in references.items()}))
        original.append((actions, states, frames))
    return worker, sources, legacy, original


def convert(worker, output, kind, sources, python):
    request = dict(format=kind, output=str(output), sources=sources, split={"train": [0], "validation": [1]},
        contract=RECIPES[kind]["contract"], source_revision="abc", converter_sha256="a" * 64)
    path = output.with_suffix(".json")
    path.write_text(json.dumps(request))
    result = subprocess.run([python, str(worker / "policy_export.py"), str(path)], capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((output / "manifest.json").read_text())


@pytest.mark.parametrize("kind", ["dp", "act", "egoverse"])
def test_shared_inputs_produce_native_v1_arrays_with_legacy_parity(prepared, tmp_path, kind):
    worker, sources, legacy, originals = prepared
    python = sys.executable
    if kind == "egoverse":
        python = os.environ.get("SKYNET_EGOVERSE_TEST_PYTHON")
        if not python:
            pytest.skip("Set SKYNET_EGOVERSE_TEST_PYTHON to an existing Zarr3/simplejpeg environment")
    expected_root, actual_root = tmp_path / "legacy", tmp_path / "shared"
    expected = convert(worker, expected_root, kind, legacy, python)
    actual = convert(worker, actual_root, kind, sources, python)
    assert actual["format"] == expected["format"] == RECIPES[kind]["format"]
    assert actual["episodes"][0]["observation_artifacts"] == sources[0]["observation_artifacts"]
    assert actual["episodes"][0]["observation_streams"] == sources[0]["observation_streams"]
    assert all(ep["image_sha256"] is None for ep in actual["episodes"])
    assert "storage_schema" not in actual
    # Both original RGB metadata and renderer-recovered metadata can be used.
    assert actual["capture"]["observation_camera_recipes"].keys() == sources[0]["observation_artifacts"].keys()
    # The completed native package must remain usable after shared inputs move.
    for source in sources:
        for modalities in source["observation_artifacts"].values():
            path = Path(modalities["rgb"]["path"])
            path.rename(path.with_name(path.name + ".unavailable"))
    compare = r'''
import sys, json
from pathlib import Path
import h5py, numpy as np, zarr
from artifacts import verify, digest
a,b,kind=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
manifest=verify(b,digest(b/'manifest.json'))
def arrays(x,y):
    for name in x.array_keys():
        np.testing.assert_array_equal(x[name][:],y[name][:])
    for name in x.group_keys():
        arrays(x[name],y[name])
if kind=='dp':
    arrays(zarr.open_group(str(a/'dataset/demonstrations.zarr'),mode='r'),zarr.open_group(str(b/'dataset/demonstrations.zarr'),mode='r'))
elif kind=='act':
    for i in range(2):
        with h5py.File(a/f'dataset/episode_{i}.hdf5','r') as x,h5py.File(b/f'dataset/episode_{i}.hdf5','r') as y:
            for key in ['action','observations/qpos',*['observations/images/'+c for c in ['cam_head','cam_left_wrist','cam_right_wrist']]]:
                np.testing.assert_array_equal(x[key][:],y[key][:])
else:
    for ep in manifest['episodes']:
        arrays(zarr.open_group(str(a/ep['path']),mode='r'),zarr.open_group(str(b/ep['path']),mode='r'))
print('native v1 arrays identical; package verifies without shared input paths')
'''
    result = subprocess.run([python, "-c", compare, str(expected_root), str(actual_root), kind], capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(worker), "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
