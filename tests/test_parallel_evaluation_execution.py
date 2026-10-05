"""Concurrent evaluations must never replace another stage's runnable files."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from skynet_app.adapters import resolve_adapter_plan
from skynet_app.pipeline_api import _evaluation_busy_reason
from skynet_app.slurm import _capsule_materializers, compile_sbatch, SlurmCompileError
from test_submission_capsule import local_client
from test_slurm import make_spec
from test_evaluation_placement import evaluation_service, request_for


def test_two_evaluations_materialize_into_disjoint_live_execution_directories(tmp_path, monkeypatch):
    root, client = local_client(tmp_path, monkeypatch)
    spec = make_spec()
    pinned = []
    for key in (None, "evaluation-a", "evaluation-b", "evaluation-a"):
        plan = resolve_adapter_plan(spec)
        if key:
            plan.native_config["canonical_evaluation"] = {"execution_key": key}
        job = compile_sbatch(spec, plan, run_id="same-trained-run", work_root=str(root),
            stage="eval" if key else "train",
            capsule_files={"adapter-support/evaluation-context.json": json.dumps({"target": key})})
        client.write_capsule_files(job.run_id, job.upload_files, "sky2", immutable=True)
        run_root = client.run_directory(job.run_id)
        commands = _capsule_materializers(run_root, job.files, execution_directory=job.run_directory)
        assert "\n".join(commands) in job.script
        assert 'export SKYNET_CAPSULE_DIR="$SKYNET_RUN_ROOT/attempts/$SLURM_JOB_ID"' in job.script
        assert f"{job.run_directory}/runtime-wrapper.py" in job.script
        # GNU install -D is not provided by macOS. Preserve its copy behavior.
        installer = 'install() { local source="${@: -2:1}"; local target="${@: -1}"; mkdir -p "$(dirname "$target")"; cp "$source" "$target"; }'
        result = subprocess.run(["bash", "-c", "set -Eeuo pipefail\n" + installer + "\n" + "\n".join(commands)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        expected = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in job.files.items()}
        pinned.append((Path(job.run_directory), expected))
        for directory, hashes in pinned:
            assert {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in hashes} == hashes
        if key:
            assert job.run_directory == f"{run_root}/evaluations/{key}"
    assert len({str(directory) for directory, _ in pinned}) == 3


@pytest.mark.parametrize("key", ["../training", "/absolute", "", "a/b", None])
def test_evaluation_execution_key_is_validated_and_historical_plans_stay_pinned(key):
    spec = make_spec()
    plan = resolve_adapter_plan(spec)
    plan.native_config["canonical_evaluation"] = {"execution_key": key}
    if key is None:
        job = compile_sbatch(spec, plan, run_id="historical", stage="eval")
        assert job.run_directory.endswith("/jobs/runs/historical")
    else:
        with pytest.raises(SlurmCompileError, match="execution key"):
            compile_sbatch(spec, plan, run_id="historical", stage="eval")


def test_busy_guard_permits_only_isolated_evaluation_stages():
    isolated = {"id": "a", "stage_type": "EVALUATE", "status": "RUNNING", "resolved_config_json": {"context": {"execution_key": "a"}}}
    assert _evaluation_busy_reason({"stages": [isolated]}) is None
    for unsafe in ({**isolated, "stage_type": "TRAIN"}, {**isolated, "resolved_config_json": {}},
                   {**isolated, "resolved_config_json": {"context": {"execution_key": "another-stage"}}}):
        assert _evaluation_busy_reason({"stages": [isolated, unsafe]})


def test_same_checkpoint_can_create_two_active_isolated_evaluations(evaluation_service):
    service, cluster, run, suite = evaluation_service
    first = service.create_evaluation(request_for(run, suite))
    second = service.create_evaluation(request_for(run, suite))
    assert first['stage_id'] != second['stage_id']
    updated = service.database.get_run(run['id'])
    assert _evaluation_busy_reason(updated) is None
    for evaluation in (first, second):
        stage_id = evaluation['stage_id']
        stage = next(item for item in updated['stages'] if item['id'] == stage_id)
        assert stage['resolved_config_json']['context']['execution_key'] == stage_id
        attempt = next(item for item in updated['attempts'] if item['stage_id'] == stage_id)
        assert attempt['execution_snapshot_json']['storage']['run_directory'].endswith('/evaluations/' + stage_id)
    assert cluster.submit_count == 3  # Fixture training plus both evaluations.


def test_independent_evaluations_can_wait_for_slurm_concurrently(evaluation_service, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    service, cluster, run, suite = evaluation_service
    rendezvous = Barrier(2)
    original = cluster.submit_script

    def delayed_submission(*args, **kwargs):
        # Holding the global planning lock during SSH would prevent the second
        # evaluation from reaching this point and break the rendezvous.
        rendezvous.wait(timeout=10)
        return original(*args, **kwargs)

    monkeypatch.setattr(cluster, "submit_script", delayed_submission)
    with ThreadPoolExecutor(max_workers=2) as pool:
        evaluations = list(pool.map(lambda _: service.create_evaluation(request_for(run, suite)), range(2)))
    assert len({item["stage_id"] for item in evaluations}) == 2
    assert all(item["submission"]["job_id"] for item in evaluations)
    for item in evaluations:
        attempts = [attempt for attempt in service.database.get_run(run["id"])["attempts"]
                    if attempt["stage_id"] == item["stage_id"]]
        assert len(attempts) == 1


def test_remote_evaluation_planning_does_not_hold_workspace_lock(evaluation_service, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    service, cluster, run, suite = evaluation_service
    rendezvous = Barrier(2)
    original = service._resolve_evaluation_implementation

    def delayed_planning(*args, **kwargs):
        rendezvous.wait(timeout=10)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_resolve_evaluation_implementation", delayed_planning)
    with ThreadPoolExecutor(max_workers=2) as pool:
        evaluations = list(pool.map(lambda _: service.create_evaluation(request_for(run, suite)), range(2)))
    assert len({item["stage_id"] for item in evaluations}) == 2
    assert all(item["submission"]["job_id"] for item in evaluations)
