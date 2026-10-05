"""Separate submission files stay pinned across verification and later attempts."""

import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import pytest

from skynet_app import cluster_runtime
from skynet_app.adapters import resolve_adapter_plan
from skynet_app.cluster_runtime import ClusterClient, ClusterError
from skynet_app.slurm import _capsule_materializers, compile_sbatch
from test_slurm import make_spec


def local_client(tmp_path, monkeypatch):
    root = tmp_path.resolve() / "cluster"
    monkeypatch.setattr(cluster_runtime, "WORK_ROOT", str(root))
    client = ClusterClient(("sky2",))

    def local_ssh(host, command, *, stdin=None, timeout=30):
        result = subprocess.run(command, shell=True, input=stdin, text=True,
                                capture_output=True, timeout=timeout)
        if result.returncode:
            raise ClusterError(result.stderr)
        return result.stdout

    monkeypatch.setattr(client, "ssh", local_ssh)
    return root, client


def file_hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("damage", ["missing", "changed", "rewritten_manifest"])
def test_staged_capsule_verification_stops_before_materializing_or_running(tmp_path, monkeypatch, damage):
    root, client = local_client(tmp_path, monkeypatch)
    spec = make_spec()
    job = compile_sbatch(spec, resolve_adapter_plan(spec), run_id="verified",
                         work_root=str(root), capsule_files={"adapter-support/input.json": '{"frozen":true}\n'})
    client.write_capsule_files(job.run_id, job.upload_files, "sky2", immutable=True)
    run = Path(client.run_directory(job.run_id))
    staged = run / job.capsule_relative_directory
    target = staged / "adapter-support/input.json"
    if damage == "missing":
        target.unlink()
    else:
        original = hashlib.sha256(target.read_bytes()).hexdigest()
        target.write_text('{"frozen":false}\n')
        if damage == "rewritten_manifest":
            manifest = staged / "checksums.sha256"
            replacement = hashlib.sha256(target.read_bytes()).hexdigest()
            manifest.write_text(manifest.read_text().replace(original, replacement))

    installed, started = tmp_path / "installed", tmp_path / "worker-started"
    # Spy on materialization to prove verification happens first, independent
    # of the host's install implementation (macOS does not support install -D).
    commands = _capsule_materializers(str(run), job.files)
    assert "\n".join(commands) in job.script
    script = "\n".join([
        "set -Eeuo pipefail",
        f"install() {{ touch {shlex.quote(str(installed))}; }}",
        *commands,
        f"touch {shlex.quote(str(started))}",
    ])
    result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
    assert result.returncode != 0
    assert ("No such file" if damage == "missing" else "FAILED") in result.stdout + result.stderr
    assert not installed.exists()
    assert not started.exists()


def test_retry_and_evaluation_stage_independent_capsules_preserve_prior_files(tmp_path, monkeypatch):
    root, client = local_client(tmp_path, monkeypatch)
    spec = make_spec()
    initial_plan = resolve_adapter_plan(spec)
    first = compile_sbatch(spec, initial_plan, run_id="shared-run", work_root=str(root),
                           capsule_files={"attempt-snapshot.json": json.dumps({"attempt": 1})})
    retry_plan = initial_plan.model_copy(deep=True)
    retry_plan.native_config["initial_checkpoint"] = "/checkpoints/step-100.ckpt"
    retry = compile_sbatch(spec, retry_plan, run_id="shared-run", work_root=str(root),
                           capsule_files={"attempt-snapshot.json": json.dumps({"attempt": 2})})
    evaluation = compile_sbatch(spec, initial_plan, run_id="shared-run", work_root=str(root), stage="eval",
                                capsule_files={"attempt-snapshot.json": json.dumps({"stage": "evaluation"})})
    assert len({job.capsule_relative_directory for job in (first, retry, evaluation)}) == 3

    frozen = []
    for job in (first, retry, evaluation):
        client.write_capsule_files(job.run_id, job.upload_files, "sky2", immutable=True)
        run = Path(client.run_directory(job.run_id))
        directory = run / job.capsule_relative_directory
        frozen.append((directory, file_hashes(directory)))
        # Execute the same manifest and file verification used by each script.
        checks = _capsule_materializers(str(run), job.files)[:2]
        result = subprocess.run(["bash", "-c", "set -Eeuo pipefail\n" + "\n".join(checks)],
                                text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        for prior, hashes in frozen:
            assert file_hashes(prior) == hashes

    # Recovery can replay the original submission without replacing later files.
    client.write_capsule_files(first.run_id, first.upload_files, "sky2", immutable=True)
    for directory, hashes in frozen:
        assert file_hashes(directory) == hashes
