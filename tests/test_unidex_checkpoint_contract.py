"""Common Skynet execution settings reach the UniDex runtime unchanged."""

import pytest

from skynet_app.adapters import ManifestAdapter, resolve_gpu_count, resolve_gpu_type
from skynet_app.adapters.unidex_manifest import REPOSITORY, REVISION, manifest
from skynet_app.adapters.unidex_runtime import parser
from skynet_app.experiments import ExperimentSpec
from skynet_app.pipeline_api import PipelineService
from test_experiments import make_spec


def checkpoint_plan(checkpoint, explicit, *, resources=None, native=None):
    declared = manifest()
    payload = make_spec(native={}).model_dump(mode="json", by_alias=True)
    payload["source"] = {
        "repository": REPOSITORY, "revision": REVISION, "adapter": "unidex",
    }
    payload["train"] = {"checkpoint": checkpoint}
    payload["intent"] = {"explicit_parameters": explicit}
    if resources is not None:
        payload["resources"] = {"account": "rl2-lab", "partition": "rl2-lab", **resources}
    payload["native"]["config"] = {
        "datasets": [{"path": "/prepared/recording", "manifest_sha256": "a" * 64}],
        "base_weights": "/weights/paligemma",
        "pointcloud_weights": "/weights/uni3d.pth",
        "weights_provenance": "/weights/provenance.json",
        **(native or {}),
    }
    PipelineService._apply_canonical_manifest_defaults(payload, declared)
    spec = ExperimentSpec.model_validate(payload)
    # Exercise the persisted canonical form, not a direct CLI-only shortcut.
    restored = ExperimentSpec.model_validate(spec.model_dump(mode="json", by_alias=True))
    plan = ManifestAdapter(declared).resolve(restored)
    return declared, restored, plan


def runtime_arguments(plan):
    assert plan.runnable, plan.blockers
    return parser().parse_args(plan.argv[2:])


def test_explicit_common_checkpoint_policy_reaches_native_runtime():
    declared, spec, plan = checkpoint_plan(
        {"save_every_steps": 250, "keep_last": 2},
        ["train.checkpoint.save_every_steps", "train.checkpoint.keep_last"],
    )
    args = runtime_arguments(plan)
    assert args.checkpoint_every_steps == spec.train.checkpoint.save_every_steps == 250
    assert args.keep_checkpoints == spec.train.checkpoint.keep_last == 2
    assert "artifacts/checkpoints/step-*.ckpt" in plan.checkpoint_globs
    assert "artifacts/checkpoints/last.ckpt" not in plan.checkpoint_globs
    assert declared.train.parameter_flags["train.checkpoint.save_every_steps"].flag == "--checkpoint-every-steps"
    assert declared.train.parameter_flags["train.checkpoint.keep_last"].flag == "--keep-checkpoints"


def test_inherited_checkpoint_policy_matches_declared_and_runtime_defaults():
    declared, spec, plan = checkpoint_plan({}, [])
    args = runtime_arguments(plan)
    assert "--checkpoint-every-steps" not in plan.argv
    assert "--keep-checkpoints" not in plan.argv
    assert args.checkpoint_every_steps == spec.train.checkpoint.save_every_steps == declared.defaults.checkpoint.save_every_steps == 1000
    assert args.keep_checkpoints == spec.train.checkpoint.keep_last == declared.defaults.checkpoint.keep_last == 3


@pytest.mark.parametrize("field,value", [("save_every_steps", 250), ("keep_last", 2)])
def test_changed_checkpoint_policy_missing_explicit_intent_is_blocked(field, value):
    _, _, plan = checkpoint_plan({field: value}, [])
    assert not plan.runnable
    assert any(f"train.checkpoint.{field}" in item and "not marked explicit" in item
               for item in plan.blockers)


def test_fresh_submission_resolves_validated_skynet_execution_defaults():
    declared, spec, plan = checkpoint_plan({}, [], resources={})
    args = runtime_arguments(plan)
    assert spec.resources.gpu.mode == "auto"
    assert spec.resources.gpu.profile == "recommended"
    assert resolve_gpu_count(spec, plan) == args.gpu_count == 4
    assert resolve_gpu_type(spec, plan) == "l40s"
    assert (spec.resources.cpus_per_task, spec.resources.memory_gb) == (12, 128)
    assert spec.native.config["distributed_strategy"] == args.distributed_strategy == "fsdp"
    assert plan.argv[plan.argv.index("--distributed-strategy") + 1] == "fsdp"
    assert args.batch_size == spec.train.batch.value == 4
    assert args.gradient_accumulation == spec.train.batch.gradient_accumulation_steps == 1
    assert args.precision == spec.train.precision == "fp32"
    assert declared.capabilities.minimum_gpus == 1
    assert (plan.capabilities.minimum_gpus, plan.capabilities.recommended_gpus) == (4, 4)


def test_auto_any_gpu_uses_the_existing_conditional_l40s_recommendation():
    _, spec, plan = checkpoint_plan({}, [], resources={"gpu": {"mode": "auto", "type": "any"}})
    args = runtime_arguments(plan)
    assert resolve_gpu_count(spec, plan) == args.gpu_count == 4
    assert resolve_gpu_type(spec, plan) == "l40s"


def test_explicit_non_l40s_single_gpu_ddp_is_preserved():
    # This verifies explicit selection preservation, not memory feasibility.
    gpu_type = "a40"
    _, spec, plan = checkpoint_plan({}, [], resources={
        "gpu": {"mode": "explicit", "count": 1, "type": gpu_type},
        "cpus_per_task": 8, "memory_gb": 96,
    }, native={"distributed_strategy": "ddp"})
    args = runtime_arguments(plan)
    assert args.distributed_strategy == spec.native.config["distributed_strategy"] == "ddp"
    assert args.gpu_count == resolve_gpu_count(spec, plan) == 1
    assert resolve_gpu_type(spec, plan) == gpu_type
    assert (spec.resources.cpus_per_task, spec.resources.memory_gb) == (8, 96)
    assert plan.capabilities.minimum_gpus == 1
