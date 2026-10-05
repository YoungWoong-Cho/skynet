"""The execution wrapper owns the boundary of each shared JSONL producer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skynet_app.adapters import (
    TrainingProgressContract, resolve_adapter_plan, resolve_training_progress_contract,
)
from skynet_app.experiments import ExperimentSpec
from skynet_app.slurm import RUNNER_SOURCE, _execution_files, _training_progress_source
from skynet_app.training_progress_log import read_execution_log


@pytest.fixture
def wrapper(tmp_path, monkeypatch):
    namespace = {"__name__": "skynet_runtime_boundary_test"}
    exec(compile(RUNNER_SOURCE, "runtime-wrapper.py", "exec"), namespace)
    run = tmp_path / "run"
    run.mkdir()
    capsule = run / "attempts" / "123"
    capsule.mkdir(parents=True)
    path = run / "artifacts" / "logs.json.txt"
    path.parent.mkdir()
    source = {"kind": "jsonl", "path": "artifacts/logs.json.txt"}
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    return SimpleNamespace(namespace=namespace, run=run, capsule=capsule, path=path,
                           execution={"training_progress_source": source})


def capture(wrapper):
    return wrapper.namespace["capture_training_progress_start"](
        wrapper.execution, wrapper.run, wrapper.capsule
    )


def test_receipt_excludes_old_resets_and_keeps_identical_replayed_rows(wrapper):
    # An old reset is deliberately present before a third execution starts.
    old = b'{"global_step":100,"train_loss":1}\n{"global_step":50,"train_loss":1}\n'
    wrapper.path.write_bytes(old)
    receipt = capture(wrapper)
    assert receipt["schema_version"] == "skynet.training-progress-start/v1"
    assert receipt["job_id"] == "123" and receipt["restart_count"] == 0
    assert receipt["start_byte_offset"] == len(old)
    assert receipt["path"] == str(wrapper.path)
    assert receipt["device"] == wrapper.path.stat().st_dev
    assert receipt["inode"] == wrapper.path.stat().st_ino
    assert receipt["prefix_bytes"] == 64
    assert receipt["prefix_sha256"] == hashlib.sha256(old[-64:]).hexdigest()
    assert receipt["at_line_boundary"] is True
    assert receipt["captured_at"].endswith("Z")
    saved = wrapper.capsule / "state/training-progress-start-0.json"
    assert json.loads(saved.read_text()) == receipt
    replayed = b'{"global_step":50,"train_loss":1}\n'
    with wrapper.path.open("ab") as stream:
        stream.write(replayed)
    assert wrapper.path.read_bytes()[receipt["start_byte_offset"]:] == replayed
    assert json.loads(saved.read_text()) == receipt
    assert read_execution_log(wrapper.path, saved, job_id="123", restart_count=0) == replayed.decode()


def test_missing_progress_file_starts_at_zero_without_creating_it(wrapper):
    receipt = capture(wrapper)
    assert receipt["start_byte_offset"] == receipt["prefix_bytes"] == 0
    assert receipt["device"] is None and receipt["inode"] is None
    assert receipt["prefix_sha256"] == hashlib.sha256(b"").hexdigest()
    assert receipt["at_line_boundary"] is True
    assert not wrapper.path.exists()
    wrapper.path.write_text('{"global_step":1}\n')
    assert wrapper.path.read_bytes()[receipt["start_byte_offset"]:]


def test_partial_old_line_is_explicit_in_boundary(wrapper):
    wrapper.path.write_bytes(b'{"global_step":40')
    receipt = capture(wrapper)
    assert receipt["at_line_boundary"] is False
    assert receipt["prefix_bytes"] == wrapper.path.stat().st_size


def test_same_execution_never_replaces_its_receipt(wrapper):
    wrapper.path.write_bytes(b"old\n")
    receipt = capture(wrapper)
    wrapper.path.write_bytes(b"old\nnew\n")
    with pytest.raises(RuntimeError, match="already exists"):
        capture(wrapper)
    assert json.loads((wrapper.capsule / "state/training-progress-start-0.json").read_text()) == receipt


def test_same_job_restart_has_its_own_start_offset(wrapper, monkeypatch):
    wrapper.path.write_bytes(b"old\n")
    first = capture(wrapper)
    with wrapper.path.open("ab") as stream:
        stream.write(b"new\n")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    second = capture(wrapper)
    assert first["start_byte_offset"] == 4
    assert second["start_byte_offset"] == 8
    assert second["restart_count"] == 1
    assert json.loads((wrapper.capsule / "state/training-progress-start-0.json").read_text()) == first
    assert json.loads((wrapper.capsule / "state/training-progress-start-1.json").read_text()) == second
    with wrapper.path.open("ab") as stream:
        stream.write(b"resumed\n")
    assert read_execution_log(wrapper.path, wrapper.capsule / "state/training-progress-start-1.json",
                              job_id="123", restart_count=1) == "resumed\n"


@pytest.mark.parametrize("path", ["/tmp/elsewhere", "../elsewhere", "artifacts/../elsewhere",
                                  "artifacts\\elsewhere", "artifacts/./log", "", "log\nfile"])
def test_source_path_must_stay_inside_run(wrapper, path):
    wrapper.execution["training_progress_source"]["path"] = path
    with pytest.raises(RuntimeError, match="run directory"):
        capture(wrapper)
    assert not (wrapper.capsule / "state").exists()


def test_source_symlink_cannot_escape_run(wrapper, tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("private\n")
    wrapper.path.symlink_to(outside)
    with pytest.raises(RuntimeError, match="escapes the run directory"):
        capture(wrapper)


def test_non_regular_source_fails_without_waiting(wrapper):
    wrapper.namespace["os"].mkfifo(wrapper.path)
    with pytest.raises(RuntimeError, match="regular file"):
        capture(wrapper)


def test_missing_declaration_does_not_create_a_receipt(wrapper):
    wrapper.execution.clear()
    assert capture(wrapper) is None
    assert not (wrapper.capsule / "state").exists()


def test_main_captures_before_child_is_launched(wrapper, monkeypatch):
    execution = {**wrapper.execution, "stage": "train", "auto_resume": False,
                 "argv": ["trainer"], "resume_argv": []}
    execution_path = wrapper.capsule / "execution.json"
    execution_path.write_text(json.dumps(execution))
    monkeypatch.setenv("SKYNET_RUN_DIR", str(wrapper.run))
    monkeypatch.setenv("SKYNET_CAPSULE_DIR", str(wrapper.capsule))
    monkeypatch.setenv("SKYNET_PROJECT_DIR", str(wrapper.run))
    monkeypatch.setenv("SKYNET_SOURCE_DIR", str(wrapper.run))
    monkeypatch.setattr(wrapper.namespace["sys"], "argv", ["wrapper", str(execution_path)])
    for function in ("write_runtime_manifest", "snapshot_checkpoints", "run_preparation_steps",
                     "ensure_native_tracking_resume_ids", "check_timeout_warning"):
        wrapper.namespace[function] = lambda *_args, **_kwargs: None
    wrapper.namespace["signal"] = SimpleNamespace(SIGTERM=15, SIGINT=2, SIGUSR1=10,
                                                   signal=lambda *_args: None)
    wrapper.namespace["wait_for_child"] = lambda: 0
    launched = []

    def popen(argv, **kwargs):
        receipt = json.loads((wrapper.capsule / "state/training-progress-start-0.json").read_text())
        assert receipt["start_byte_offset"] == 0
        wrapper.path.write_text('{"global_step":1}\n')
        launched.append(argv)
        monkeypatch.delenv("SLURM_JOB_ID")  # No GPU sampler is needed in this wrapper test.
        return object()

    wrapper.namespace["subprocess"] = SimpleNamespace(Popen=popen)
    assert wrapper.namespace["main"]() == 0
    assert launched == [["trainer"]]


def progress(path="artifacts/epochs.jsonl"):
    return TrainingProgressContract.model_validate({
        "unit": "epoch", "total_path": "native.config.epochs",
        "source": {"kind": "jsonl", "path": path, "completed_key": "epoch", "required_key": "loss"},
        "step_source": {"kind": "jsonl", "path": "artifacts/steps.jsonl",
                        "completed_key": "step", "required_key": "loss"},
    })


def spec_and_plan():
    spec = ExperimentSpec.model_validate({
        "identity": {"project": "boundary", "experiment": "test"},
        "source": {"repository": "https://example.com/repo", "revision": "c" * 40, "adapter": "generic"},
        "runtime": {"backend": "existing", "bootstrap_uv": False},
        "native": {"argv": ["python", "train.py"], "resume_argv": ["--resume", "{{SKYNET_RESUME_CHECKPOINT}}"]},
    })
    return spec, resolve_adapter_plan(spec)


@pytest.mark.parametrize("max_steps,expected", [(None, "epoch"), (8000, "step")])
def test_pinned_plan_selects_same_reader_for_epoch_and_step_budgets(max_steps, expected):
    spec, plan = spec_and_plan()
    spec.train.max_steps = max_steps
    plan.progress = progress()
    selected = _training_progress_source(spec, plan)
    assert selected["completed_key"] == expected
    files = _execution_files(spec, plan, "run", stage="train", stage_auto_resume=True,
                             evaluation_progress_path="/eval/progress.jsonl")
    assert json.loads(files["execution.json"])["training_progress_source"] == selected
    evaluation = _execution_files(spec, plan, "run", stage="eval", stage_auto_resume=True,
                                  evaluation_progress_path="/eval/progress.jsonl")
    assert json.loads(evaluation["execution.json"])["training_progress_source"] is None


def test_pinned_plan_precedes_manifest_and_manifest_precedes_builtin():
    spec, plan = spec_and_plan()
    spec.train.max_steps = None
    spec.source.adapter_manifest = {"train": {"progress": progress("artifacts/pinned.jsonl").model_dump()}}
    assert _training_progress_source(spec, plan)["path"] == "artifacts/pinned.jsonl"
    plan.progress = progress("artifacts/plan.jsonl")
    assert _training_progress_source(spec, plan)["path"] == "artifacts/plan.jsonl"


def test_builtin_fallback_uses_the_selected_step_source():
    spec, plan = spec_and_plan()
    plan.adapter = "unidex"
    spec.train.max_steps = 8000
    source = _training_progress_source(spec, plan)
    assert source["path"] == "artifacts/logs.json.txt"
    assert source["completed_key"] == "global_step"


@pytest.mark.parametrize("max_steps,uses_steps", [
    (8000, True), (1, True), (8000.0, True), ("8000", True), ("8e3", True),
    (None, False), (True, False), (False, False), (0, False), (-1, False),
    (1.5, False), ("1.5", False), ("unknown", False), (float("inf"), False),
    (float("nan"), False), ({}, False), ([], False),
])
@pytest.mark.parametrize("encoded", [False, True])
def test_shared_step_reader_selection_preserves_validated_budget_behavior(max_steps, uses_steps, encoded):
    from skynet_app.pipeline_api import _resolved_training_total
    contract = progress()
    original = contract.model_dump()
    spec = {"train": {"max_steps": max_steps}}
    if encoded:
        spec = json.dumps(spec)
    resolved = resolve_training_progress_contract(original, spec)
    assert (_resolved_training_total(spec, "train.max_steps") is not None) == uses_steps
    assert resolved.source.completed_key == ("step" if uses_steps else "epoch")
    assert resolved.unit == ("step" if uses_steps else "epoch")
    assert contract.model_dump() == original


@pytest.mark.parametrize("spec", [None, "invalid json", "[]", {"train": []}, {}])
def test_shared_reader_preserves_epoch_source_for_missing_budget(spec):
    contract = progress()
    assert resolve_training_progress_contract(contract, spec) is contract


def test_observer_and_producer_share_the_same_resolver_and_builtin_step_source():
    from skynet_app import pipeline_api, slurm
    assert pipeline_api.resolve_training_progress_contract is resolve_training_progress_contract
    assert slurm.resolve_training_progress_contract is resolve_training_progress_contract
    declared, origin = pipeline_api._training_progress_contract({
        "adapter_name": "unidex", "resolved_spec_json": json.dumps({"train": {"max_steps": 8000}}),
    })
    assert origin == "builtin_compatibility"
    assert declared.unit == "step" and declared.source.completed_key == "global_step"
