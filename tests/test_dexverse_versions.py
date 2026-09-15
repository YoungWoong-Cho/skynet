"""V1 provenance, recording persistence and portable replay boundaries."""

import copy
import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from skynet_app.dexverse_versions import (
    V1_REVISION,
    collection_tasks,
    environment_profile,
)
from skynet_app.live_xr_catalog import selection
from skynet_app.live_xr_review import inspect
from skynet_app.trajectory import validate_identity
from tests.test_live_review import (
    payload as payload,  # noqa: PLC0414 - re-export pytest fixture
)
from tests.test_live_xr import (
    service as service,  # noqa: PLC0414 - re-export pytest fixture
)

ROOT = Path(__file__).resolve().parents[1]


def worker(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "ops/xr" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "task,robot,side",
    [
        ("Dexverse-OpenFaucet-v1", "skynet_shadow_right", "right"),
        ("Dexverse-BimanualLiftCarton-v1", "skynet_shadow_bimanual", "both"),
    ],
)
def test_session_pins_v1_recorder_and_source(service, task, robot, side):
    session = service.create(True, task=task, robot=robot)
    p = session["profile"]
    assert p["source_revision"] == V1_REVISION
    assert p["repository"].endswith("/repos/DexVerse-ce974ac6/" + V1_REVISION)
    assert p["recording_schema_version"] == 5
    assert p["hand"] == side
    assert p["task_name"].endswith(" · v1")
    assert service.get(session["id"])["profile"] == p
    assert service.create(task=task, robot=robot)["id"] == session["id"]


def test_existing_v0_profile_stays_on_original_source():
    original = {
        "repository": "/old/repo",
        "source_revision": "a" * 40,
        "work_root": "/workspace",
    }
    resolved = environment_profile(original, "Dexverse-PickCube-v0")
    assert resolved["recording_schema_version"] == 3
    assert resolved["repository"] == original["repository"]
    assert resolved["source_revision"] == original["source_revision"]
    assert "recording_schema_version" not in original


def test_v1_bimanual_and_right_hand_tasks_reject_wrong_embodiments():
    tasks = collection_tasks()
    assert len(tasks) == len({t["key"] for t in tasks}) == 20
    for task in tasks:
        wrong = (
            "skynet_shadow_right"
            if task["required_hand"] == "both"
            else "skynet_shadow_bimanual"
        )
        with pytest.raises(ValueError, match="requires"):
            selection(task["key"], wrong)


def schema5(payload):
    payload.update(
        schema_version=5,
        task="Dexverse-OpenFaucet-v1",
        task_version=1,
        robot_type="skynet_shadow_right",
        benchmark_revision="dexverse-baseline-v1",
        action_layout={
            "dimension": 28,
            "robot_joint_names": [f"joint{i}" for i in range(28)],
        },
    )
    payload["episodes"][0]["task_state"] = {
        "commands": {
            "object_pose": {"pose_command_b": np.array([0, 0, 0, 1, 0, 0, 0.0])}
        },
        "env_buffers": {"_goal_face": np.array([3])},
    }
    return payload


def test_v1_recording_retains_scene_frames_and_checks_action_layout(payload):
    schema5(payload)
    profile = {
        "task": payload["task"],
        "robot": payload["robot_type"],
        "recording_schema_version": 5,
    }
    result = inspect(pickle.dumps(payload), profile)
    assert len(result["episodes"][0]["frames"]) == 4
    assert (
        result["episodes"][0]["frames"][1]["action"]
        == payload["episodes"][0]["actions"][0].tolist()
    )
    with pytest.raises(ValueError, match="format"):
        inspect(pickle.dumps(payload), {**profile, "recording_schema_version": 3})
    payload["action_layout"]["dimension"] = 29
    with pytest.raises(ValueError, match="layout"):
        inspect(pickle.dumps(payload), profile)


@pytest.mark.parametrize("schema", [3, 5])
def test_recorder_flushes_resets_without_duplicate_or_lost_episodes(
    tmp_path, payload, schema
):
    if schema == 5:
        schema5(payload)
    module = worker("collection")
    meta = {k: v for k, v in payload.items() if k not in {"episodes", "num_episodes"}}
    store = module.EpisodeStore(tmp_path, meta)
    recorder = SimpleNamespace(_metadata=meta, _episodes=[], _reset_attempts=[])
    module.flush_recorder(recorder, store)  # No completed episode on initial reset.
    for index in range(2):
        episode = copy.deepcopy(payload["episodes"][0])
        episode.update(episode_index=index, reset_id=index)
        recorder._episodes.append(episode)
        recorder._reset_attempts.append({"reset_id": index, "outcome": "success"})
        module.flush_recorder(recorder, store)
        module.flush_recorder(
            recorder, store
        )  # A subsequent reset must not re-save it.
        assert not recorder._episodes
        assert len(store.receipts) == index + 1
    paths = sorted((tmp_path / "recordings/live").glob("*.pkl"))
    assert len(paths) == 2
    loaded = [pickle.loads(p.read_bytes()) for p in paths]
    assert [p["episodes"][0]["episode_index"] for p in loaded] == [0, 1]
    for p in loaded:
        assert validate_identity(p) == schema
    if schema == 5:
        assert loaded[-1]["num_reset_attempts"] == 2
        assert (
            loaded[-1]["episodes"][0]["task_state"]["env_buffers"]["_goal_face"][0] == 3
        )


def test_v1_converter_identity_does_not_depend_on_a_local_source_checkout(monkeypatch):
    from skynet_app.dexverse_versions import V1_CONVERTER_SHA256
    from skynet_app.live_conversion import pinned_converter_digest

    def unexpected(*args, **kwargs):
        raise AssertionError(
            "The pinned v1 converter must not require local git objects"
        )

    monkeypatch.setattr("subprocess.check_output", unexpected)
    assert (
        pinned_converter_digest({"source_revision": V1_REVISION}) == V1_CONVERTER_SHA256
    )


@pytest.mark.parametrize("sides", [("right",), ("right", "left")])
def test_v1_configs_use_selected_hand_before_native_default(monkeypatch, sides):
    import runpy
    import sys
    from types import ModuleType

    runtime = runpy.run_path(str(ROOT / "ops/xr/hands/runtime.py"))
    robot = "skynet_test_" + ("bimanual" if len(sides) == 2 else "right")
    layouts = {
        side: {
            "wrist_joints": [f"{side}_{i}" for i in range(6)],
            "finger_joints": [side + "_finger"],
        }
        for side in sides
    }
    manifest = {
        "robot": robot,
        "hands": layouts,
        "hand_order": list(sides),
        "wrist_joints": [n for h in layouts.values() for n in h["wrist_joints"]],
        "finger_joints": [n for h in layouts.values() for n in h["finger_joints"]],
        "action_dimension": len(sides) * 7,
    }
    init = SimpleNamespace(
        _SINGLE_FLOATING_JOINTS={"native": "original"},
        _FLOATING_BIMANUAL_JOINTS={},
        _SINGLE_ARM_FLOATING_TRANSLATION_ROBOTS=("native",),
        _FLOATING_BIMANUAL_TRANSLATION_ROBOTS=(),
        _bimanual_hand_mount_offsets=lambda _: "original",
    )
    entries = {
        key: SimpleNamespace(kwargs={"env_cfg_entry_point": "native"})
        for key in ["Dexverse-OpenFaucet-v1", "Dexverse-BimanualLiftTray-v1"]
    }
    cfg = ModuleType("dexverse.baseline_v1.config")
    cfg.robot_init = init
    benchmark = ModuleType("dexverse.benchmark")
    benchmark.V1_CONFIGS = {key: ("fixture", "Config") for key in entries}
    task_module = ModuleType("dexverse.baseline_v1.config.fixture")
    task_module.Config = lambda **kwargs: kwargs
    gym = ModuleType("gymnasium")
    gym.spec = entries.__getitem__
    for name, module in [
        (cfg.__name__, cfg),
        (benchmark.__name__, benchmark),
        (task_module.__name__, task_module),
        (gym.__name__, gym),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    runtime["register_v1_hand"](manifest)
    chosen = (
        "Dexverse-BimanualLiftTray-v1" if len(sides) == 2 else "Dexverse-OpenFaucet-v1"
    )
    assert entries[chosen].kwargs["env_cfg_entry_point"]() == {"robot_type": robot}
    other = next(key for key in entries if key != chosen)
    assert entries[other].kwargs["env_cfg_entry_point"] == "native"
    assert init._SINGLE_FLOATING_JOINTS["native"] == "original"
    if len(sides) == 2:
        assert init._bimanual_hand_mount_offsets(robot) == {
            side: (0.0, 0.0, 0.0) for side in sides
        }
        assert init._bimanual_hand_mount_offsets("native") == "original"
    else:
        assert init._SINGLE_FLOATING_JOINTS[robot] == (
            tuple(layouts["right"]["wrist_joints"][:3]),
            tuple(layouts["right"]["wrist_joints"][3:]),
        )
