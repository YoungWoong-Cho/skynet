"""Exercise recorded-policy boundaries without requiring a local GPU."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from skynet_app.adapters import builtin_adapter_manifests
from skynet_app.evaluation_contracts import bind_suite_to_dataset
from skynet_app.experiments import (
    evaluation_task_catalog_sha256,
    get_evaluation_catalog,
)

SUPPORT = Path(__file__).parents[1] / "skynet_app" / "adapters"


@pytest.fixture
def portable(monkeypatch):
    monkeypatch.syspath_prepend(str(SUPPORT))
    import act_training
    import dexverse_evaluation
    import xpolicy_runtime

    return SimpleNamespace(
        act=act_training, sim=dexverse_evaluation, data=xpolicy_runtime
    )


def manifest():
    return {
        "policy_to_source_indices": [1, 0],
        "episodes": [{"steps": 3}, {"steps": 2}],
        "split": {"train": [1], "validation": [0]},
        "capture": {
            "action_joint_names": ["wrist", "finger"],
            "action_semantics": "raw_joint_position_command; target = action * scale + offset",
        },
    }


def test_registered_split_and_joint_mapping_are_enforced(portable):
    good = manifest()
    assert portable.data.validate_manifest(good) == good
    for split in [
        {"train": [0], "validation": [0]},
        {"train": [], "validation": [0, 1]},
    ]:
        bad = copy.deepcopy(good)
        bad["split"] = split
        with pytest.raises(ValueError, match="split|episodes"):
            portable.data.validate_manifest(bad)
    bad = copy.deepcopy(good)
    bad["policy_to_source_indices"] = [0, 0]
    with pytest.raises(ValueError, match="permutation"):
        portable.data.validate_manifest(bad)


def test_act_observation_matches_its_command_and_padding_is_masked(portable, tmp_path):
    (tmp_path / "dataset").mkdir()
    with h5py.File(tmp_path / "dataset/episode_0.hdf5", "w") as f:
        f["observations/qpos"] = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        f["action"] = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
        for camera in portable.act.CAMERAS:
            f.create_dataset(
                "observations/images/" + camera,
                shape=(3, 480, 640, 3),
                dtype="uint8",
                compression="gzip",
                fillvalue=123,
            )
    stats = dict(
        state_mean=np.array([1, 2]),
        state_std=np.array([2, 2]),
        action_mean=np.array([10, 20]),
        action_std=np.array([10, 10]),
    )
    qpos, images, actions, mask = portable.act.read_sample(tmp_path, 0, 1, stats, 4)
    np.testing.assert_array_equal(qpos, [1, 1])
    np.testing.assert_array_equal(actions, [[2, 2], [4, 4], [0, 0], [0, 0]])
    np.testing.assert_array_equal(mask, [False, False, True, True])
    assert images.shape == (3, 480, 640, 3)
    assert images[0, 0, 0, 0] == 123


def test_normalization_rejects_validation_leakage(portable, tmp_path):
    data = manifest()
    stats = {
        "fit": "training_episodes_only",
        "train": [1],
        "statistics": {
            k: {"mean": [1.0, 2.0], "std": [0.0, 2.0]} for k in ["state", "action"]
        },
    }
    path = tmp_path / "normalization.json"
    path.write_text(json.dumps(stats))
    np.testing.assert_allclose(
        portable.data.normalization(tmp_path, data)["action_std"], [0.01, 2.0]
    )
    stats["train"] = [0, 1]
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="training split"):
        portable.data.normalization(tmp_path, data)


def test_evaluation_task_is_frozen_from_training_bundle_not_catalog_mutated():
    catalog = next(
        s for s in get_evaluation_catalog() if s.suite == "dexverse_recorded"
    )
    suite = {"config_json": catalog.model_dump(mode="json")}
    spec = {
        "data": {
            "bundle": {
                "assignments": [
                    {
                        "role": "training_data",
                        "version": {
                            "metadata": {"capture": {"task": "Dexverse-PickCube-v0"}}
                        },
                    }
                ]
            }
        }
    }
    bound = bind_suite_to_dataset(suite, spec)
    assert bound["config_json"]["tasks"] == suite["config_json"]["tasks"]
    assert bound["config_json"]["default_tasks"] == ["Dexverse-PickCube-v0"]
    assert bound["config_json"]["task_catalog_complete"]
    assert bound["config_json"][
        "task_catalog_sha256"
    ] == evaluation_task_catalog_sha256(catalog.tasks)
    assert "default_tasks" not in suite["config_json"]
    legacy = copy.deepcopy(suite)
    legacy["config_json"].update(tasks=[], task_options=[], task_selection_mode="all_only")
    assert bind_suite_to_dataset(legacy, spec)["config_json"]["tasks"] == ["Dexverse-PickCube-v0"]
    with pytest.raises(ValueError, match="registered training dataset"):
        bind_suite_to_dataset(suite, {})
    spec["data"]["bundle"]["assignments"][0]["version"]["metadata"] = {}
    with pytest.raises(ValueError, match="simulation task"):
        bind_suite_to_dataset(suite, spec)


def test_existing_catalogs_do_not_gain_an_empty_binding():
    for suite in get_evaluation_catalog():
        if suite.dataset_task_binding is None:
            assert "dataset_task_binding" not in suite.model_dump(mode="json")


def test_dataset_bound_runtime_evidence_pins_the_simulator_not_one_dataset():
    from skynet_app.runtime_readiness import (
        build_readiness_contract,
        suite_contract_sha256,
    )

    contract, source = build_readiness_contract(
        "isaacsim-5.1.0_isaaclab-2.3.2_py311", "dexverse_recorded"
    )
    assert "AppLauncher" in source and "frame.max()" in source
    assert contract["runtime"]["source_dir"].endswith(
        "30cc673e27684b9f10186fa6bea731aed246bc9f"
    )
    original = next(
        s for s in get_evaluation_catalog() if s.suite == "dexverse_recorded"
    ).model_dump(mode="json")
    bound = copy.deepcopy(original)
    bound.update(
        tasks=["Dexverse-PickCube-v0"],
        task_options=[{"id": "Dexverse-PickCube-v0"}],
        task_catalog_sha256=evaluation_task_catalog_sha256(["Dexverse-PickCube-v0"]),
    )
    assert suite_contract_sha256(bound) == contract["suite_contract_sha256"]
    bound["task_catalog_provenance"]["revision"] = "changed"
    assert suite_contract_sha256(bound) != contract["suite_contract_sha256"]
    # Static benchmark suites continue to pin their exact task selection.
    original.pop("dataset_task_binding")
    bound = dict(original, tasks=["different"])
    assert suite_contract_sha256(bound) != suite_contract_sha256(original)


def test_both_policies_have_training_and_same_real_rollout_contract():
    policies = {m.slug: m for m in builtin_adapter_manifests()}
    for slug in ["xpolicylab-dp", "xpolicylab-act"]:
        model = policies[slug]
        assert model.train.progress.unit == "epoch"
        assert model.evaluations[0].suites == ["dexverse_recorded", "dexverse_training_episode"]
        files = model.evaluations[0].command.capsule_files
        assert "adapter-support/dexverse_evaluation.py" in files
        assert "adapter-support/images.py" in files


def test_simulator_rejects_changed_actions_and_uses_dataset_order(portable):
    term_type = type("JointPositionAction", (), {})
    term = term_type()
    term.cfg = SimpleNamespace(asset_name="robot")
    term._joint_names, term._scale, term._offset = (
        ["wrist", "finger"],
        2.0,
        np.array([[0.0, 1.0]]),
    )
    env = SimpleNamespace(
        action_manager=SimpleNamespace(
            active_terms=["joints"], get_term=lambda _: term
        ),
        step_dt=1 / 60,
        scene={"robot": SimpleNamespace(joint_names=["finger", "wrist"])},
    )
    capture = dict(
        action_joint_names=["wrist", "finger"],
        action_scale=[2.0, 2.0],
        action_offset=[0.0, 1.0],
        step_dt=1 / 60,
    )
    assert portable.sim.validate_layout(env, capture, [1, 0]) == [0, 1]
    term._scale = 3.0
    with pytest.raises(ValueError, match="actions differ"):
        portable.sim.validate_layout(env, capture, [1, 0])


def test_episode_resume_requires_same_run_and_existing_video(portable, tmp_path):
    video = tmp_path / "rollout.mp4"
    video.write_bytes(b"video")
    episode = dict(task="cube", seed=0, episode_index=0, video_path=str(video))
    row = dict(identity="identity", status="SUCCEEDED", episode=episode)
    ledger = tmp_path / "progress.jsonl"
    ledger.write_text(json.dumps(row) + "\n")
    assert portable.sim.completed_episodes(ledger, "identity") == {
        ("cube", 0, 0): episode
    }
    with pytest.raises(ValueError, match="different checkpoint"):
        portable.sim.completed_episodes(ledger, "other")
    video.unlink()
    with pytest.raises(ValueError, match="missing rollout video"):
        portable.sim.completed_episodes(ledger, "identity")
