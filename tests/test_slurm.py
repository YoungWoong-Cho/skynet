from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from skynet_app.adapters import PreparationStep, resolve_adapter_plan
from skynet_app.experiments import ExperimentSpec
from skynet_app.slurm import (
    RUNNER_SOURCE,
    SlurmCompileError,
    compile_sbatch,
    resolve_slurm_log_path,
    resolve_slurm_log_paths_from_sbatch,
)


COMMIT = "c" * 40


def make_spec(*, account="rl2-lab", node=None, runtime=None):
    return ExperimentSpec.model_validate(
        {
            "identity": {"project": "tests", "experiment": "canonical"},
            "source": {
                "repository": "https://github.com/example/smoke.git",
                "revision": COMMIT,
                "adapter": "generic",
            },
            "runtime": runtime or {"backend": "existing", "bootstrap_uv": False},
            "resources": {
                "account": account,
                "partition": account,
                "node": node or {"mode": "auto"},
                "gpu": {"mode": "explicit", "count": 4, "type": "l40s"},
                "time_limit": "04:00:00" if account == "rl2-lab" else "1-00:00:00",
            },
            "native": {
                "argv": ["python", "smoke.py", "--value", "a value; not shell"],
                "resume_argv": ["--checkpoint", "{{SKYNET_RESUME_CHECKPOINT}}"],
            },
        }
    )


def test_compiler_emits_canonical_directives_and_capsule():
    spec = make_spec()
    compiled = compile_sbatch(spec, resolve_adapter_plan(spec), run_id="run-001")
    script = compiled.script
    assert "#SBATCH --partition=rl2-lab" in script
    assert "#SBATCH --account=rl2-lab" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --gres=gpu:l40s:4" in script
    assert "#SBATCH --error=/coc/flash7/ycho420/logs/%x-%j.err" in script
    assert "#SBATCH --signal=B:USR1@300" in script
    assert "#SBATCH --requeue" in script
    assert "#SBATCH --nodelist" not in script
    assert "git clone --no-checkout" in script
    assert f"checkout --detach --force {COMMIT}" in script
    assert "hydra/launcher=submitit" not in script
    assert "argv.json" in compiled.files
    assert "native-config.json" in compiled.files
    assert "source-manifest.json" in script
    assert "runtime-manifest.json" in script
    assert "system-manifest.json" in script
    assert "capsule-manifest.json" in script
    assert 'SKYNET_CAPSULE_DIR="$SKYNET_RUN_DIR/attempts/$SLURM_JOB_ID"' in script
    assert '"$SKYNET_CAPSULE_DIR/job.sbatch"' in script
    assert "importlib.metadata.distributions()" in compiled.files["runtime-wrapper.py"]
    assert "--query-gpu=index,uuid,name,driver_version,memory.total" in script
    assert compiled.script_sha256
    assert compiled.job_name == "canonical-train"
    assert compiled.stdout_path_template == "/coc/flash7/ycho420/logs/canonical-train-%j.out"
    assert compiled.stderr_path_template == "/coc/flash7/ycho420/logs/canonical-train-%j.err"
    assert resolve_slurm_log_path(compiled.stdout_path_template, "8123") == (
        "/coc/flash7/ycho420/logs/canonical-train-8123.out"
    )


def test_sbatch_log_path_resolution_is_structural_and_rejects_unsupported_tokens():
    script = """#!/usr/bin/env bash
#SBATCH --job-name=demo-train
#SBATCH --output=/coc/flash7/ycho420/logs/%x-%j.out
#SBATCH --error=/coc/flash7/ycho420/logs/%x-%j.err

echo ready
"""
    assert resolve_slurm_log_paths_from_sbatch(script, "8123") == (
        "/coc/flash7/ycho420/logs/demo-train-8123.out",
        "/coc/flash7/ycho420/logs/demo-train-8123.err",
    )
    unsupported = script.replace("%x-%j.out", "%A.out")
    assert resolve_slurm_log_paths_from_sbatch(unsupported, "8123") == (
        None,
        "/coc/flash7/ycho420/logs/demo-train-8123.err",
    )


def test_compiler_exports_and_creates_shared_xdg_cache_root():
    spec = make_spec()
    script = compile_sbatch(spec, resolve_adapter_plan(spec), run_id="cache-001").script
    assert "export XDG_CACHE_HOME=/coc/flash7/ycho420/.cache" in script
    assert 'mkdir -p "$XDG_CACHE_HOME" "$WORK_ROOT"/.cache/{uv,huggingface,torch}' in script


def test_overcap_pair_and_manual_node():
    spec = make_spec(account="overcap", node={"mode": "manual", "name": "node123"})
    compiled = compile_sbatch(spec, resolve_adapter_plan(spec), run_id="overcap-001")
    assert "#SBATCH --partition=overcap" in compiled.script
    assert "#SBATCH --account=overcap" in compiled.script
    assert "#SBATCH --time=1-00:00:00" in compiled.script
    assert "#SBATCH --nodelist=node123" in compiled.script


def test_uv_runtime_has_fallback_and_pinned_bootstrap():
    spec = make_spec(runtime={"backend": "uv", "bootstrap_uv": True, "uv_version": "0.8.14"})
    compiled = compile_sbatch(spec, resolve_adapter_plan(spec), run_id="uv-001")
    assert "$WORK_ROOT/.local/bin/uv" in compiled.script
    assert "python3 -m uv" in compiled.script
    assert "uv==0.8.14" in compiled.script
    assert "--frozen" in compiled.script


def test_blocked_adapter_cannot_compile():
    payload = make_spec().model_dump(mode="json", by_alias=True)
    payload["source"]["adapter"] = "dexverse"
    payload["runtime"] = {"backend": "existing", "bootstrap_uv": False}
    payload["native"] = {"argv": [], "resume_argv": [], "config": {}, "overrides": {}}
    spec = ExperimentSpec.model_validate(payload)
    plan = resolve_adapter_plan(spec)
    assert not plan.runnable
    with pytest.raises(SlurmCompileError, match="DexVerse defines Isaac Lab environments"):
        compile_sbatch(spec, plan, run_id="blocked-001")


def test_checkpoint_retention_keeps_three_then_selected_only():
    namespace = {"__name__": "skynet_runtime_test"}
    exec(compile(RUNNER_SOURCE, "runtime-wrapper.py", "exec"), namespace)
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory) / "run"
        project_dir = Path(directory) / "project"
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        project_dir.mkdir()
        for step in range(1, 6):
            checkpoint = checkpoint_dir / f"step-{step}.ckpt"
            checkpoint.write_text(str(step))
            os.utime(checkpoint, (step, step))
        execution = {
            "checkpoint_globs": ["checkpoints/*.ckpt"],
            "checkpoint": {"keep_last": 3},
        }

        namespace["snapshot_checkpoints"](run_dir, project_dir, execution)
        assert [path.name for path in sorted(checkpoint_dir.glob("*.ckpt"))] == [
            "step-3.ckpt",
            "step-4.ckpt",
            "step-5.ckpt",
        ]

        namespace["snapshot_checkpoints"](run_dir, project_dir, execution, final=True)
        assert [path.name for path in checkpoint_dir.glob("*.ckpt")] == ["step-5.ckpt"]
        selected = checkpoint_dir / "selected-for-inference.json"
        assert selected.is_file()
        payload = json.loads(selected.read_text())
        assert payload["selection"]["actual"] == "latest"
        assert payload["resumable"] is False
        assert payload["sha256"]
        assert payload["size_bytes"] == 1
        assert payload["file_count"] == 1


def test_compiler_snapshots_adapter_checkpoint_lifecycle_fields():
    spec = make_spec()
    plan = resolve_adapter_plan(spec).model_copy(
        update={
            "checkpoint_globs": ["checkpoints/*/*/*"],
            "checkpoint_candidate_kind": "directory",
            "checkpoint_basename_regex": r"\d+",
            "checkpoint_prune_globs": ["train_state"],
            "checkpoint_inference_required_globs": ["params/*", "assets/*"],
        },
        deep=True,
    )

    compiled = compile_sbatch(spec, plan, run_id="checkpoint-contract")
    execution = json.loads(compiled.files["execution.json"])

    assert execution["schema_version"] == 5
    assert execution["checkpoint_globs"] == ["checkpoints/*/*/*"]
    assert execution["checkpoint_candidate_kind"] == "directory"
    assert execution["checkpoint_basename_regex"] == r"\d+"
    assert execution["checkpoint_prune_globs"] == ["train_state"]
    assert execution["checkpoint_inference_required_globs"] == ["params/*", "assets/*"]


def test_native_tracking_secret_is_loaded_out_of_band_and_resume_id_is_declared():
    from skynet_app.adapters import NativeTrackingIntegration

    spec = make_spec()
    plan = resolve_adapter_plan(spec).model_copy(
        update={
            "native_tracking": [
                NativeTrackingIntegration(
                    provider="wandb",
                    parameter_paths={"enabled": "native.config.wandb_enabled"},
                    run_id_file="{{SKYNET_RUN_DIR}}/checkpoints/debug/smoke/wandb_id.txt",
                )
            ]
        },
        deep=True,
    )
    secret_path = (
        "/coc/flash7/ycho420/jobs/runs/native-run/"
        "state/runtime-secrets/stage-1/wandb-api-key"
    )
    compiled = compile_sbatch(
        spec,
        plan,
        run_id="native-run",
        runtime_environment={
            "WANDB_ENTITY": "team",
            "WANDB_PROJECT": "central-project",
            "WANDB_RUN_ID": "native-run",
            "WANDB_RESUME": "must",
            "WANDB_DIR": "/coc/flash7/ycho420/jobs/runs/native-run",
        },
        secret_environment_files={"WANDB_API_KEY": secret_path},
        native_tracking_run_ids={"wandb": "native-run"},
    )
    assert "sentinel-secret" not in compiled.script
    assert f'export WANDB_API_KEY="$(cat -- {secret_path})"' in compiled.script
    assert f"rm -f -- {secret_path}" in compiled.script
    assert "export WANDB_RUN_ID=native-run" in compiled.script
    execution = json.loads(compiled.files["execution.json"])
    assert execution["native_tracking"] == [
        {
            "provider": "wandb",
            "remote_id": "native-run",
            "run_id_file": "{{SKYNET_RUN_DIR}}/checkpoints/debug/smoke/wandb_id.txt",
        }
    ]

    with pytest.raises(SlurmCompileError, match="below the run directory"):
        compile_sbatch(
            spec,
            plan,
            run_id="native-run",
            secret_environment_files={
                "WANDB_API_KEY": "/tmp/not-run-scoped"
            },
        )


def test_directory_checkpoint_lifecycle_selects_best_prunes_only_training_state():
    namespace = {"__name__": "skynet_runtime_test"}
    exec(compile(RUNNER_SOURCE, "runtime-wrapper.py", "exec"), namespace)
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory) / "run"
        project_dir = Path(directory) / "project"
        experiment_dir = run_dir / "checkpoints" / "pi05_libero" / "experiment"
        project_dir.mkdir()
        scores = {1: 0.1, 2: 0.9, 3: 0.4}
        for step, score in scores.items():
            root = experiment_dir / str(step)
            (root / "params").mkdir(parents=True)
            (root / "assets").mkdir()
            (root / "train_state").mkdir()
            (root / "params" / "weights").write_bytes(f"weights-{step}".encode())
            (root / "assets" / "norm.json").write_text("{}")
            (root / "train_state" / "optimizer").write_bytes(b"optimizer")
            (root / "skynet-checkpoint.json").write_text(
                json.dumps({"selection": {"score": score, "mode": "max"}})
            )
            os.utime(root, (step, step))
        temporary = experiment_dir / "3.orbax-checkpoint-tmp-0"
        temporary.mkdir()
        execution = {
            "checkpoint_globs": ["checkpoints/*/*/*"],
            "checkpoint_candidate_kind": "directory",
            "checkpoint_basename_regex": r"\d+",
            "checkpoint_prune_globs": ["train_state"],
            "checkpoint_inference_required_globs": ["params/*", "assets/*"],
            "checkpoint": {
                "keep_last": 3,
                "final_selector": "best",
                "remove_training_state_after_success": True,
            },
        }

        namespace["snapshot_checkpoints"](run_dir, project_dir, execution)
        assert sorted(path.name for path in experiment_dir.iterdir() if path.name.isdigit()) == [
            "1",
            "2",
            "3",
        ]

        selected = namespace["snapshot_checkpoints"](
            run_dir, project_dir, execution, final=True
        )
        assert selected == (experiment_dir / "2").resolve()
        assert not (experiment_dir / "1").exists()
        assert not (experiment_dir / "3").exists()
        assert temporary.exists()
        assert (selected / "params" / "weights").is_file()
        assert (selected / "assets" / "norm.json").is_file()
        assert not (selected / "train_state").exists()

        descriptor = json.loads(
            (run_dir / "checkpoints" / "selected-for-inference.json").read_text()
        )
        assert descriptor["selection"]["requested"] == "best"
        assert descriptor["selection"]["actual"] == "best"
        assert descriptor["selection"]["score"] == 0.9
        assert descriptor["cleanup"]["performed"] is True
        assert descriptor["cleanup"]["removed"] == ["train_state"]
        assert descriptor["resumable"] is False
        assert descriptor["is_directory"] is True
        assert descriptor["sha256"] == namespace["checkpoint_identity"](selected)["sha256"]
        assert descriptor["size_bytes"] > 0
        assert descriptor["file_count"] == 3


def test_checkpoint_candidates_require_complete_atomic_roots_and_deduplicate():
    namespace = {"__name__": "skynet_runtime_test"}
    exec(compile(RUNNER_SOURCE, "runtime-wrapper.py", "exec"), namespace)
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory) / "run"
        project_dir = Path(directory) / "project"
        complete = run_dir / "checkpoints" / "config" / "experiment" / "1"
        incomplete = run_dir / "checkpoints" / "config" / "experiment" / "2"
        (complete / "params").mkdir(parents=True)
        (complete / "assets").mkdir()
        (complete / "params" / "weights").write_text("weights")
        (complete / "assets" / "norm.json").write_text("{}")
        (incomplete / "train_state").mkdir(parents=True)
        (incomplete / "train_state" / "optimizer").write_text("optimizer")
        project_dir.mkdir()
        execution = {
            "checkpoint_globs": [
                "checkpoints/*/*/*",
                "checkpoints/config/experiment/*",
            ],
            "checkpoint_candidate_kind": "directory",
            "checkpoint_basename_regex": r"\d+",
            "checkpoint_inference_required_globs": ["params", "assets"],
        }

        candidates = namespace["checkpoint_candidates"](
            run_dir, project_dir, execution
        )

        assert candidates == [complete.resolve()]
        assert incomplete.is_dir()


def test_checkpoint_overlap_and_prune_required_overlap_fail_before_deletion():
    namespace = {"__name__": "skynet_runtime_test"}
    exec(compile(RUNNER_SOURCE, "runtime-wrapper.py", "exec"), namespace)
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory) / "run"
        project_dir = Path(directory) / "project"
        selected = run_dir / "checkpoints" / "config" / "experiment" / "1"
        ancestor = selected.parent
        (ancestor / "params").mkdir(parents=True)
        (ancestor / "params" / "ancestor-weights").write_text("ancestor")
        (selected / "params").mkdir(parents=True)
        (selected / "params" / "weights").write_text("weights")
        project_dir.mkdir()
        overlapping_candidates = {
            "checkpoint_globs": ["checkpoints/**/*"],
            "checkpoint_candidate_kind": "directory",
            "checkpoint_inference_required_globs": ["params/*"],
            "checkpoint": {"keep_last": 1},
        }
        with pytest.raises(RuntimeError, match="atomic and non-overlapping"):
            namespace["snapshot_checkpoints"](
                run_dir, project_dir, overlapping_candidates, final=True
            )
        assert (selected / "params" / "weights").is_file()
        assert (ancestor / "params" / "ancestor-weights").is_file()

        unsafe_cleanup = {
            "checkpoint_globs": ["checkpoints/*/*/*"],
            "checkpoint_candidate_kind": "directory",
            "checkpoint_basename_regex": r"\d+",
            "checkpoint_prune_globs": ["params"],
            "checkpoint_inference_required_globs": ["params/*"],
            "checkpoint": {
                "keep_last": 1,
                "final_selector": "latest",
                "remove_training_state_after_success": True,
            },
        }
        with pytest.raises(RuntimeError, match="overlap required inference"):
            namespace["snapshot_checkpoints"](
                run_dir, project_dir, unsafe_cleanup, final=True
            )
        assert (selected / "params" / "weights").is_file()


def test_success_without_declared_checkpoint_candidate_fails_wrapper(monkeypatch):
    namespace = {"__name__": "skynet_runtime_test"}
    exec(compile(RUNNER_SOURCE, "runtime-wrapper.py", "exec"), namespace)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run_dir = root / "run"
        project_dir = root / "project"
        capsule_dir = root / "attempt"
        for path in (run_dir, project_dir, capsule_dir):
            path.mkdir()
        execution = {
            "schema_version": 3,
            "run_id": "missing-checkpoint",
            "argv": [sys.executable, "-c", "pass"],
            "preparation_steps": [],
            "resume_argv": [],
            "auto_resume": False,
            "checkpoint": {"keep_last": 3, "final_selector": "latest"},
            "checkpoint_globs": ["checkpoints/*/*/*"],
            "checkpoint_candidate_kind": "directory",
            "checkpoint_basename_regex": r"\d+",
            "checkpoint_prune_globs": [],
            "checkpoint_inference_required_globs": ["params"],
            "initial_checkpoint": None,
        }
        execution_path = root / "execution.json"
        execution_path.write_text(json.dumps(execution))
        monkeypatch.setenv("SKYNET_RUN_DIR", str(run_dir))
        monkeypatch.setenv("SKYNET_SOURCE_DIR", str(project_dir))
        monkeypatch.setenv("SKYNET_PROJECT_DIR", str(project_dir))
        monkeypatch.setenv("SKYNET_CAPSULE_DIR", str(capsule_dir))
        monkeypatch.setattr(sys, "argv", ["runtime-wrapper.py", str(execution_path)])

        with pytest.raises(RuntimeError, match="declared checkpoint globs"):
            namespace["main"]()


def test_evaluation_disabled_resume_ignores_training_checkpoint_and_runs_base_once(
    monkeypatch,
):
    namespace = {"__name__": "skynet_runtime_test"}
    exec(compile(RUNNER_SOURCE, "runtime-wrapper.py", "exec"), namespace)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run_dir = root / "run"
        project_dir = root / "project"
        capsule_dir = root / "attempt"
        marker = root / "calls.txt"
        for path in (run_dir, project_dir, capsule_dir):
            path.mkdir()
        checkpoint = run_dir / "checkpoints" / "training-step.ckpt"
        checkpoint.parent.mkdir()
        checkpoint.write_text("training checkpoint")
        (checkpoint.parent / "latest.json").write_text(
            json.dumps({"path": str(checkpoint)})
        )
        base_argv = [
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; "
                "assert len(sys.argv) == 2; p=Path(sys.argv[1]); "
                "p.write_text((p.read_text() if p.exists() else '') + 'called\\n')"
            ),
            str(marker),
        ]
        progress = root / "evaluation" / "progress.jsonl"
        execution = {
            "schema_version": 4,
            "run_id": "evaluation-resume-disabled",
            "stage": "eval",
            "resume_mode": "evaluation_ledger",
            "argv": base_argv,
            "preparation_steps": [],
            "resume_argv": list(base_argv),
            "auto_resume": False,
            "checkpoint": {},
            "checkpoint_globs": [],
            "checkpoint_candidate_kind": "any",
            "checkpoint_basename_regex": None,
            "checkpoint_prune_globs": [],
            "checkpoint_inference_required_globs": [],
            "initial_checkpoint": None,
            "evaluation_resume": {
                "enabled": False,
                "progress_path": str(progress),
                "episode_identity": [],
            },
        }
        execution_path = root / "execution.json"
        execution_path.write_text(json.dumps(execution))
        monkeypatch.setenv("SKYNET_RUN_DIR", str(run_dir))
        monkeypatch.setenv("SKYNET_SOURCE_DIR", str(project_dir))
        monkeypatch.setenv("SKYNET_PROJECT_DIR", str(project_dir))
        monkeypatch.setenv("SKYNET_CAPSULE_DIR", str(capsule_dir))
        monkeypatch.setattr(sys, "argv", ["runtime-wrapper.py", str(execution_path)])

        assert namespace["main"]() == 0
        assert marker.read_text() == "called\n"
        assert json.loads((capsule_dir / "state/effective-argv.json").read_text()) == base_argv
        assert os.environ["SKYNET_EVALUATION_RESUME"] == "0"
        assert os.environ["SKYNET_EVAL_PROGRESS_PATH"] == str(progress)


def test_evaluation_compile_uses_stage_resume_and_canonical_progress_path():
    spec = make_spec()
    plan = resolve_adapter_plan(spec)
    base_argv = ["python", "evaluate.py", "--context", "context.json"]
    progress_path = "/coc/flash7/ycho420/eval/runs/eval-stage/progress.jsonl"
    plan = plan.model_copy(
        update={
            "argv": base_argv,
            "resume_argv": list(base_argv),
            "native_config": {
                **plan.native_config,
                "canonical_evaluation": {"progress_path": progress_path},
            },
        },
        deep=True,
    )

    compiled = compile_sbatch(
        spec,
        plan,
        run_id="evaluation-stage-resume",
        stage="eval",
        stage_auto_resume=False,
    )
    execution = json.loads(compiled.files["execution.json"])

    assert execution["stage"] == "eval"
    assert execution["resume_mode"] == "evaluation_ledger"
    assert execution["auto_resume"] is False
    assert execution["argv"] == base_argv
    assert execution["resume_argv"] == []
    assert execution["initial_checkpoint"] is None
    assert execution["evaluation_resume"]["enabled"] is False
    assert execution["evaluation_resume"]["progress_path"] == progress_path
    assert "#SBATCH --requeue" not in compiled.script
    assert f"export SKYNET_EVAL_PROGRESS_PATH={progress_path}" in compiled.script


def test_preparation_runs_before_training_and_reuses_only_verified_outputs(monkeypatch):
    namespace = {"__name__": "skynet_runtime_test"}
    exec(compile(RUNNER_SOURCE, "runtime-wrapper.py", "exec"), namespace)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run_dir = root / "run"
        project_dir = root / "project"
        first_capsule = root / "attempt-1"
        second_capsule = root / "attempt-2"
        for path in (run_dir, project_dir, first_capsule, second_capsule):
            path.mkdir(parents=True)
        order_path = run_dir / "order.txt"
        preparation = PreparationStep(
            id="fixture",
            argv=[
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "p=Path('{{SKYNET_RUN_DIR}}/order.txt'); "
                    "p.write_text((p.read_text() if p.exists() else '') + 'prepare\\n'); "
                    "Path('ready.txt').write_text('ready')"
                ),
            ],
            working_directory="artifacts/preparation/fixture",
            output_globs=["ready.txt"],
            expected_output_sha256={
                "ready.txt": hashlib.sha256(b"ready").hexdigest()
            },
        )
        execution = {
            "schema_version": 2,
            "run_id": "fixture",
            "argv": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "p=Path('{{SKYNET_RUN_DIR}}/order.txt'); "
                    "p.write_text(p.read_text() + 'train\\n')"
                ),
            ],
            "preparation_steps": [preparation.model_dump(mode="json")],
            "resume_argv": [],
            "auto_resume": False,
            "checkpoint": {"keep_last": 3},
            "checkpoint_globs": [],
            "initial_checkpoint": None,
        }
        execution_path = root / "execution.json"
        execution_path.write_text(json.dumps(execution))
        monkeypatch.setenv("SKYNET_RUN_DIR", str(run_dir))
        monkeypatch.setenv("SKYNET_SOURCE_DIR", str(project_dir))
        monkeypatch.setenv("SKYNET_PROJECT_DIR", str(project_dir))
        monkeypatch.setenv("SKYNET_CAPSULE_DIR", str(first_capsule))
        monkeypatch.setattr(sys, "argv", ["runtime-wrapper.py", str(execution_path)])

        assert namespace["main"]() == 0
        first_state = json.loads(
            (first_capsule / "state" / "preparation.json").read_text()
        )
        assert first_state["steps"][0]["status"] == "completed"
        assert first_state["steps"][0]["outputs"]["ready.txt"]
        assert order_path.read_text() == "prepare\ntrain\n"

        monkeypatch.setenv("SKYNET_CAPSULE_DIR", str(second_capsule))
        assert namespace["main"]() == 0
        second_state = json.loads(
            (second_capsule / "state" / "preparation.json").read_text()
        )
        assert second_state["steps"][0]["status"] == "skipped"
        assert order_path.read_text() == "prepare\ntrain\ntrain\n"

        execution["preparation_steps"][0]["expected_output_sha256"]["ready.txt"] = "0" * 64
        execution_path.write_text(json.dumps(execution))
        third_capsule = root / "attempt-3"
        third_capsule.mkdir()
        monkeypatch.setenv("SKYNET_CAPSULE_DIR", str(third_capsule))
        with pytest.raises(RuntimeError, match="output SHA-256 mismatch"):
            namespace["main"]()
        third_state = json.loads(
            (third_capsule / "state" / "preparation.json").read_text()
        )
        assert third_state["steps"][0]["status"] == "failed"
