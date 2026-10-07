"""A GPU job the cluster started without a GPU stops before its workload, with a receipt."""
import json
import os
import subprocess

import pytest
from test_pipeline import FakeCluster, canonical_spec, make_pipeline_service
from test_runtime_readiness import PROFILE, SUITE
from test_slurm import make_spec

from skynet_app import data_imports, isaac_job, runtime_readiness, slurm
from skynet_app.adapters import resolve_adapter_plan
from skynet_app.cluster_config import CLUSTER
from skynet_app.database import Database
from skynet_app.gpu_preflight import (
    GPU_MISSING_EXIT_CODE, GPU_MISSING_REASON, GPU_MISSING_STATE, INTERRUPTION_RECEIPT, TIME_LIMIT_EXIT_CODE,
    TIME_LIMIT_REASON, gpu_missing_exit, gpu_missing_receipt, gpu_preflight_lines, time_limit_receipt,
)
import skynet_app.pipeline_api as pipeline
from skynet_app.preparation_states import TRANSIENT_STATES

PREFLIGHT = "\n".join(gpu_preflight_lines(1))


def test_only_gpu_jobs_get_a_preflight_and_only_its_exit_code_counts():
    assert gpu_preflight_lines(0) == []
    assert f"exit {GPU_MISSING_EXIT_CODE}" in PREFLIGHT and INTERRUPTION_RECEIPT not in PREFLIGHT
    assert INTERRUPTION_RECEIPT in "\n".join(gpu_preflight_lines(2, receipt_dir='"$DIR"'))
    assert gpu_missing_exit({"ExitCode": f"{GPU_MISSING_EXIT_CODE}:0"})
    assert not gpu_missing_exit({"ExitCode": "1:0"}) and not gpu_missing_exit(None)
    gpu_receipt = {"reason": GPU_MISSING_REASON, "exit_code": GPU_MISSING_EXIT_CODE}
    time_limit = {"reason": TIME_LIMIT_REASON, "exit_code": TIME_LIMIT_EXIT_CODE}
    assert gpu_missing_receipt(gpu_receipt) and not gpu_missing_receipt(time_limit) and not gpu_missing_receipt(None)
    assert time_limit_receipt(time_limit) and not time_limit_receipt(gpu_receipt) and not time_limit_receipt(None)
    # Each receipt is one reason with its own code; a mixed pair is neither.
    assert not time_limit_receipt({"reason": TIME_LIMIT_REASON, "exit_code": GPU_MISSING_EXIT_CODE})
    assert GPU_MISSING_STATE in TRANSIENT_STATES


@pytest.mark.parametrize("scenario", ["nothing_visible", "cuda_visible", "slurm_job_gpus", "partial", "not_requested"])
def test_preflight_exits_under_bash_only_when_a_requested_gpu_is_missing(tmp_path, scenario):
    requested = {"not_requested": 0, "partial": 2}.get(scenario, 1)
    env = {"PATH": os.environ["PATH"], "SLURM_JOB_ID": "4020377", "SKYNET_RUN_ID": "run-1"}
    if scenario == "cuda_visible":
        env["CUDA_VISIBLE_DEVICES"] = "0"
    if scenario == "slurm_job_gpus":
        env["SLURM_JOB_GPUS"] = "2,3"
    if scenario == "partial":
        env["CUDA_VISIBLE_DEVICES"] = "GPU-abc"
    script = "\n".join(["set -Eeuo pipefail", *gpu_preflight_lines(requested, receipt_dir=f'"{tmp_path}/state"'),
                        "echo workload", "exit 0"])
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    receipt = tmp_path / "state" / INTERRUPTION_RECEIPT
    if scenario in ("nothing_visible", "partial"):
        assert result.returncode == GPU_MISSING_EXIT_CODE and "workload" not in result.stdout
        assert GPU_MISSING_REASON in result.stderr
        assert json.loads(receipt.read_text()) == {
            "schema_version": 1, "job_id": "4020377", "run_id": "run-1",
            "reason": GPU_MISSING_REASON, "exit_code": GPU_MISSING_EXIT_CODE,
        }
        assert gpu_missing_receipt(json.loads(receipt.read_text()))
    else:
        assert (result.returncode, result.stdout.strip(), receipt.exists()) == (0, "workload", False), result.stderr


def test_training_script_runs_the_preflight_after_its_traps_and_before_any_work():
    spec = make_spec()
    compiled = slurm.compile_sbatch(spec, resolve_adapter_plan(spec), run_id="preflight")
    expected = "\n".join(gpu_preflight_lines(compiled.gpu_count, receipt_dir='"$SKYNET_CAPSULE_DIR/state"'))
    assert compiled.gpu_count > 0 and compiled.script.count(expected) == 1
    script = compiled.script
    # The EXIT trap records the exit in final.json and the warning handler clears
    # a stale receipt before the preflight may write a new one.
    assert (script.index("trap finalize_attempt EXIT") < script.index(slurm.BATCH_WARNING_HANDLER)
            < script.index(expected) < script.index("sha256sum -c checksums.sha256"))


def test_every_other_gpu_builder_runs_the_preflight_and_cpu_builders_do_not():
    root = CLUSTER.paths.work_root
    default = CLUSTER.queue(CLUSTER.defaults.queue_policy)
    profile = {"partition": default.partition, "runtime": f"{root}/env",
               "repository": f"{root}/repo", "gpu_type": "any"}
    isaac = isaac_job.compile_isaac_job(profile, f"{root}/jobs/probe", "probe", ["true"], checks=["verify_source"])
    assert isaac.count(PREFLIGHT) == 1 and isaac.index("umask 077") < isaac.index(PREFLIGHT) < isaac.index("verify_source")
    readiness = runtime_readiness.render_readiness_sbatch(PROFILE, SUITE)
    assert readiness.count(PREFLIGHT) == 1 and readiness.index("set -euo pipefail") < readiness.index(PREFLIGHT)
    assert readiness.index(PREFLIGHT) < readiness.index("probe_root=")
    from test_data_imports import RESOURCE, import_request
    importing = data_imports.build_huggingface_import_job("a" * 36, RESOURCE, import_request().model_dump()).script
    assert "--gres" not in importing and "GPU preflight" not in importing


def test_invariant_repair_shares_the_transient_states_with_reconcile(tmp_path, monkeypatch):
    assert pipeline.TRANSIENT_STATES is TRANSIENT_STATES
    database = Database(tmp_path / "state.db")
    service = make_pipeline_service(database, FakeCluster())
    monkeypatch.setattr(pipeline, "LOCAL_CAPSULE_ROOT", tmp_path / "capsules")
    experiment = service.create_experiment(canonical_spec())
    service.submit_experiment(experiment["id"])
    run_id = experiment["runs"][0]["id"]
    attempt = database.get_run(run_id)["attempts"][0]
    # REVOKED is transient for reconcile; the repair used to keep a narrower private copy.
    database.update_job_attempt(attempt["id"], status="REVOKED")
    database.repair_workflow_state_invariants()
    repaired = database.get_run(run_id)
    assert repaired["status"] == "RETRY_PENDING"
    assert all(stage["status"] == "RETRY_PENDING" for stage in repaired["stages"] if stage["stage_type"] == "TRAIN")
