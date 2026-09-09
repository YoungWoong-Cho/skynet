import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skynet_app.adapters import ManifestAdapter
from skynet_app.adapters.egoverse_manifest import manifests, CONTRACT, REVISION
from skynet_app.adapters.egoverse_runtime import (
    ACTDataInterface,
    joint_keymap,
    map_joint_model,
    parser,
)
from skynet_app.adapters.egoverse_evaluation import result
from skynet_app.training_contracts import data_contract_error
from test_experiments import make_spec
from test_policy_exports import setup as setup


def test_every_native_model_forwards_values_and_supports_four_gpus():
    for manifest in manifests():
        native = {
            f.path.removeprefix("native.config."): copy.deepcopy(f.default)
            for f in manifest.train.input_fields
            if f.default is not None
        }
        native.update(
            dataset_path="/prepared", dataset_manifest_sha256="a" * 64, epochs=3
        )
        if "pi05" in manifest.slug:
            native["weights"] = "/weights"
        spec = make_spec(
            source={
                "repository": manifest.default_repository,
                "revision": REVISION,
                "adapter": manifest.slug,
            },
            native={"config": native},
            train={
                "learning_rate": 0.001,
                "batch": {"value": 2, "declared_semantics": "per_device"},
                "num_workers_per_rank": 0,
                "precision": "fp32",
            },
            resources={"gpu": {"count": 4, "mode": "explicit"}},
        )
        plan = ManifestAdapter(manifest).resolve(spec)
        args = parser().parse_args(plan.argv[2:])
        assert args.epochs == 3 and args.gpu_count == 4 and args.learning_rate == 0.001
        assert args.dataset == "/prepared" and args.batch_size == 2
        assert args.precision == "fp32"
        assert not any("GPU" in b for b in plan.blockers)
        assert "--checkpoint" in plan.resume_argv
        assert manifest.train.checkpoint_globs
        assert manifest.capabilities.maximum_gpus == 16


def test_act_compatibility_does_not_normalize_twice():
    normalized = {"joint_positions": [0.25]}
    stats = SimpleNamespace(
        embodiments={100},
        norm_stats={100: {}},
        keys_of_type=lambda kind, emb: [kind, emb],
        zarr_key_to_keyname=lambda key, emb: key,
        unnormalize=lambda data, emb: {"raw": data},
    )
    interface = ACTDataInterface(stats)
    assert interface.keys_of_type("camera_keys") == ["camera_keys", 100]
    assert interface.normalize_data(normalized, 100) is normalized
    assert interface.unnormalize_data(normalized, 100) == {"raw": normalized}
    assert "scene_front" not in joint_keymap(norm_mode=True)
    assert joint_keymap(horizon=16)["actions_joints"]["horizon"] == 16


def test_offline_results_do_not_claim_simulation_success():
    context = dict(
        run_id="test",
        checkpoint=dict(path="/ckpt", sha256="a" * 64),
        evaluator=dict(adapter="egoverse", version="1"),
        suite=dict(name="held_out", version="1"),
    )
    output = result(
        context,
        [
            dict(
                task="held_out",
                seed=42,
                episode_index=i,
                success=None,
                status="SUCCEEDED",
                metrics={"joint_mse": value},
            )
            for i, value in enumerate([1.0, 3.0])
        ],
    )
    assert output["aggregate"][0]["mean"] == 2
    assert output["aggregate"][0]["sample_count"] == 2
    assert all(ep["success"] is None for ep in output["episodes"])
    assert not any(metric["metric"] == "success_rate" for metric in output["aggregate"])


def test_native_writer_matches_pinned_provenance():
    import hashlib

    root = Path(__file__).parents[1] / "ops/datasets"
    receipt = json.loads((root / "egoverse-provenance.json").read_text())
    assert (
        hashlib.sha256((root / "egoverse_zarr_writer.py").read_bytes()).hexdigest()
        == receipt["files"]["egomimic/rldb/zarr/zarr_writer.py"]
    )
    assert receipt["revision"] == REVISION


def test_incompatible_native_policies_do_not_accept_recorded_joint_contract():
    accepted = []
    for manifest in manifests():
        binding = manifest.train.input_fields[0].data_binding
        if CONTRACT in binding.contracts:
            accepted.append(manifest.slug)
    assert accepted == ["egoverse-act", "egoverse-hpt-joints", "egoverse-dp-joints"]


def test_preparation_preserves_dataset_name_when_freezing_native_assets(setup):
    service, session, _ = setup
    job = service.create(session["id"], "egoverse", "My recorded demonstrations")
    assert job["name"] == "My recorded demonstrations"
    assert (
        service.database.get_data_resource(job["resource_id"])["metadata"][
            "display_name"
        ]
        == job["name"]
    )
    worker = service.root / job["id"] / "worker"
    assert (worker / "egoverse-LICENSE").is_file()
    assert (worker / "egoverse_data.py").is_file()


def test_offline_evaluator_does_not_require_an_isaac_runtime(monkeypatch):
    from skynet_app.pipeline_api import PipelineService

    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    suite = json.loads(
        (
            Path(__file__).parents[1]
            / "config/evaluation_suites/egoverse-held-out.json"
        ).read_text()
    )
    runtimes, blockers = PipelineService._resolve_dual_evaluation_runtimes(
        {}, "egoverse-native", suite
    )
    assert blockers == []
    assert runtimes["evaluator_runtime"]["profile"] == "egoverse-native"


def test_held_out_episode_limit_is_bound_to_the_pinned_dataset():
    from skynet_app.evaluation_contracts import bind_suite_to_dataset

    suite = {
        "config_json": {
            "dataset_episode_binding": {
                "role": "training_data",
                "metadata_path": "split.validation",
            }
        }
    }
    spec = {
        "data": {
            "bundle": {
                "assignments": [
                    {
                        "role": "training_data",
                        "version": {"metadata": {"split": {"validation": [1, 4]}}},
                    }
                ]
            }
        }
    }
    bound = bind_suite_to_dataset(suite, spec)
    assert bound["config_json"]["maximum_episodes_per_task"] == 2
    assert "maximum_episodes_per_task" not in suite["config_json"]
    spec["data"]["bundle"]["assignments"][0]["version"]["metadata"]["split"][
        "validation"
    ] = []
    with pytest.raises(ValueError, match="no held-out episodes"):
        bind_suite_to_dataset(suite, spec)


def test_native_readiness_uses_gpu_and_media_checks_without_simulator_consent():
    from skynet_app.runtime_readiness import (
        build_readiness_contract,
        render_readiness_sbatch,
    )

    contract, _ = build_readiness_contract("egoverse-native", "egoverse_held_out")
    assert contract["verification"]["requires_gpu"]
    assert (
        "video_encode_decode"
        in contract["verification"]["compute_smoke"]["required_checks"]
    )
    script = render_readiness_sbatch("egoverse-native", "egoverse_held_out")
    assert "#SBATCH --gres=gpu:a40:1" in script
    assert "${OMNI_KIT_ACCEPT_EULA:-}" not in script


def test_native_split_rejects_duplicate_leaked_or_unconfined_episodes(tmp_path):
    from skynet_app.adapters.egoverse_runtime import validate_episode_split

    manifest = {
        "episodes": [{"path": "train/a.zarr"}, {"path": "valid/b.zarr"}],
        "split": {"train": [0], "validation": [1]},
    }
    validate_episode_split(tmp_path, manifest)
    for path in ("train/a.zarr", "../outside.zarr"):
        broken = copy.deepcopy(manifest)
        broken["episodes"][1]["path"] = path
        with pytest.raises(ValueError, match="distinct paths"):
            validate_episode_split(tmp_path, broken)
    manifest["split"]["validation"] = [0]
    with pytest.raises(ValueError, match="exactly one"):
        validate_episode_split(tmp_path, manifest)


def test_finite_input_check_ignores_optional_native_metadata(monkeypatch):
    import importlib.util
    import sys
    import numpy as np

    monkeypatch.setitem(
        sys.modules,
        "egomimic.rldb.zarr.zarr_dataset_multi",
        SimpleNamespace(MultiDataset=object),
    )
    file = Path(__file__).parents[1] / "skynet_app/adapters/egoverse_data.py"
    spec = importlib.util.spec_from_file_location("egoverse_data_check", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset = module.JointDataset.__new__(module.JointDataset)
    dataset.reject_outliers = False
    dataset.zarr_keys = {100: {"state": "joint_positions"}}
    sample = {
        "embodiment": 100,
        "joint_positions": np.array([1.0]),
        "intrinsics": np.array([np.nan]),
    }
    assert dataset._check_bounds(sample, None, 0, "episode") is None
    sample["joint_positions"][0] = np.inf
    with pytest.raises(ValueError, match="joint_positions"):
        dataset._check_bounds(sample, None, 0, "episode")
