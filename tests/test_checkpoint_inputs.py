from __future__ import annotations

import copy
import json

import pytest

from skynet_app.adapters import (
    AdapterInputField, CommandTemplate, builtin_adapter_manifests, resolve_adapter_plan,
)
from skynet_app.pipeline_api import PipelineService
from skynet_app.slurm import compile_sbatch
from test_experiments import make_spec


CHECKPOINT = "/verified/run/artifacts/checkpoints/step-00000500.ckpt"


@pytest.fixture
def manifest():
    result = next(item for item in builtin_adapter_manifests() if item.slug == "generic").model_copy(deep=True)
    result.legacy_handler = None
    result.capabilities.supports_resume = True
    result.train = CommandTemplate(
        argv=["python", "train.py"],
        resume_argv=["--resume", "{{tokens.resume_checkpoint}}"],
        strict_native_config=True,
    )
    return result


def resume_document():
    return {
        "native": {"config": {"initial_checkpoint": CHECKPOINT, "initial_checkpoint_mode": "resume"}},
        "train": {"checkpoint": {"auto_resume": True}},
    }


def test_strict_adapter_accepts_only_valid_runner_resume_keys(manifest):
    document = resume_document()
    before = copy.deepcopy(document)
    PipelineService._validate_manifest_input_fields(document, manifest)
    assert document == before
    del document["native"]["config"]["initial_checkpoint_mode"]
    PipelineService._validate_manifest_input_fields(document, manifest)
    document["native"]["config"]["typo"] = 123
    with pytest.raises(ValueError, match="Unsupported adapter settings: typo"):
        PipelineService._validate_manifest_input_fields(document, manifest)


@pytest.mark.parametrize("path", [None, "", "relative.ckpt", " /abs.ckpt", "/a/../b.ckpt", "/a\n.ckpt", 12])
def test_runner_resume_requires_nonempty_absolute_checkpoint_path(manifest, path):
    document = resume_document()
    document["native"]["config"]["initial_checkpoint"] = path
    with pytest.raises(ValueError, match="nonempty absolute path"):
        PipelineService._validate_manifest_input_fields(document, manifest)


def test_undeclared_weights_mode_cannot_be_silently_resumed(manifest):
    document = resume_document()
    document["native"]["config"]["initial_checkpoint_mode"] = "weights"
    with pytest.raises(ValueError, match="weights-only initialization"):
        PipelineService._validate_manifest_input_fields(document, manifest)


def test_runner_resume_requires_adapter_capability(manifest):
    manifest.capabilities.supports_resume = False
    with pytest.raises(ValueError, match="does not support full-state"):
        PipelineService._validate_manifest_input_fields(resume_document(), manifest)


@pytest.mark.parametrize("argv", [[], ["--resume"]])
def test_runner_resume_requires_command_that_consumes_path(manifest, argv):
    manifest.train.resume_argv = argv
    with pytest.raises(ValueError, match="consumes the checkpoint path"):
        PipelineService._validate_manifest_input_fields(resume_document(), manifest)


def test_runner_resume_cannot_be_ignored_by_auto_resume_false(manifest):
    document = resume_document()
    document["train"]["checkpoint"]["auto_resume"] = False
    with pytest.raises(ValueError, match="would ignore it"):
        PipelineService._validate_manifest_input_fields(document, manifest)
    document["train"]["checkpoint"] = {}
    manifest.defaults.checkpoint.auto_resume = False
    with pytest.raises(ValueError, match="would ignore it"):
        PipelineService._validate_manifest_input_fields(document, manifest)


def test_existing_declared_handlers_and_nonstrict_adapters_are_unchanged(manifest):
    document = resume_document()
    document["native"]["config"]["initial_checkpoint"] = "handler-relative.ckpt"
    document["native"]["config"]["initial_checkpoint_mode"] = "weights"
    document["train"]["checkpoint"]["auto_resume"] = False
    manifest.capabilities.supports_resume = False
    manifest.train.resume_argv = []
    manifest.train.input_fields = [
        AdapterInputField(path=f"native.config.{name}", label=name, kind="string")
        for name in ("initial_checkpoint", "initial_checkpoint_mode")
    ]
    PipelineService._validate_manifest_input_fields(document, manifest)
    manifest.train.input_fields = []
    manifest.train.strict_native_config = False
    PipelineService._validate_manifest_input_fields(document, manifest)


def test_existing_strict_override_rejection_is_preserved(manifest):
    document = resume_document()
    document["native"]["overrides"] = {"arbitrary": True}
    with pytest.raises(ValueError, match="only accepts its declared"):
        PipelineService._validate_manifest_input_fields(document, manifest)


@pytest.mark.parametrize("frontend", [False, True], ids=["canonical", "submit-form"])
def test_resume_input_survives_normalization_and_frozen_runner_execution(manifest, frontend):
    # Real normalization/planning/compiler; only repository network inspection is stubbed.
    service = PipelineService.__new__(PipelineService)
    service._resolve_source_and_runtime = lambda source, runtime, _manifest, _gateway: (source, runtime)
    source = make_spec().source.model_dump(mode="json")
    source["adapter_manifest"] = manifest.model_dump(mode="json")
    source.pop("adapter_manifest_sha256", None)
    if frontend:
        payload = {
            "name": "resume-500", "adapter": manifest.slug, "source": source,
            "runtime": {"backend": "existing", "bootstrap_uv": False},
            "checkpoint": {"mode": "resume", "path": CHECKPOINT},
            "auto_resume": True,
            "resources": {"queue_policy": "normal", "gpu_mode": "explicit", "gpus_per_node": 1},
        }
    else:
        payload = make_spec(source=source, **resume_document()).model_dump(mode="json", by_alias=True)
    normalized = service.normalize_spec(payload)
    assert normalized.native.config["initial_checkpoint"] == CHECKPOINT
    assert normalized.native.config["initial_checkpoint_mode"] == "resume"
    assert normalized.train.checkpoint.auto_resume is True
    plan = resolve_adapter_plan(normalized)
    assert plan.resume_argv == ["--resume", "{{SKYNET_RESUME_CHECKPOINT}}"]
    execution = json.loads(compile_sbatch(normalized, plan, run_id="resume-input-test").files["execution.json"])
    assert execution["initial_checkpoint"] == CHECKPOINT
    assert execution["auto_resume"] is True
    assert execution["resume_argv"] == ["--resume", "{{SKYNET_RESUME_CHECKPOINT}}"]
    # The same public path also rejects a resume that the runner would ignore.
    rejected = copy.deepcopy(payload)
    if frontend:
        rejected["auto_resume"] = False
    else:
        rejected["train"]["checkpoint"]["auto_resume"] = False
    with pytest.raises(ValueError, match="would ignore it"):
        service.normalize_spec(rejected)
