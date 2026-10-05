"""Checkpoint receipt verification and the real manual-resume entry point."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from skynet_app import pipeline_api
from skynet_app.adapters import AdapterManifest
from skynet_app.cluster_runtime import ClusterError
from skynet_app.database import Database
from skynet_app.slurm import RUNNER_SOURCE
from tests.test_pipeline import (
    FakeCluster, _mark_training_run_cancelled, canonical_spec,
    make_pipeline_service, seed_repository_choices, submitted_execution,
)


def _receipt_fixture(root, *, run_id="run", plan=None, attempt_id="original", job_id="250"):
    pattern = (plan or {}).get("checkpoint_globs", ["artifacts/checkpoints/step-*.ckpt"])[0]
    target = root / pattern.replace("**", "artifacts").replace("*", "250")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"full checkpoint fixture")
    plan = plan or {}
    contract = {
        "candidate_globs": plan.get("checkpoint_globs", ["artifacts/checkpoints/step-*.ckpt"]),
        "candidate_kind": plan.get("checkpoint_candidate_kind", "any"),
        "basename_regex": plan.get("checkpoint_basename_regex"),
        "inference_required_globs": plan.get("checkpoint_inference_required_globs", []),
    }
    receipt = {
        "schema_version": 2, "run_id": run_id, "path": str(target),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "size_bytes": target.stat().st_size, "file_count": 1,
        "is_directory": False, "final": False, "resumable": True,
        "cleanup": {"performed": False}, "contract": contract,
    }
    request = {"root": str(root), "run_id": run_id, "contract": contract,
               "attempts": [{"id": attempt_id, "job_id": job_id}]}
    _save_receipt(root, receipt, job_id)
    return target, receipt, request


def _save_receipt(root, receipt, job_id="250"):
    for path in (root / "checkpoints/latest.json", root / "attempts" / job_id / "checkpoints/latest.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt))


def _probe(request, *, prelude=""):
    return subprocess.run([sys.executable, "-", json.dumps(request)],
                          input=prelude + pipeline_api._MANUAL_RESUME_CHECKPOINT_PROBE,
                          text=True, capture_output=True, timeout=10)


def _scandir_failure(path, *, error="PermissionError", fail_on=1):
    # Inject a portable permission/I/O failure even when tests run as root.
    return f'''import os
_original_scandir = os.scandir
_scan_count = 0
def _failing_scandir(path):
    global _scan_count
    if str(path) == {str(path)!r}:
        _scan_count += 1
        if _scan_count >= {fail_on}:
            raise {error}("unreadable checkpoint directory")
    return _original_scandir(path)
os.scandir = _failing_scandir
'''


def test_resume_receipt_binds_only_a_proven_single_attempt(tmp_path):
    _, _, request = _receipt_fixture(tmp_path)
    result = _probe(request)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["producer"] == "original"
    request["attempts"].append({"id": "later-failed", "job_id": "251"})
    result = _probe(request)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["producer"] is None


def test_missing_receipt_permits_replay_only_when_no_checkpoint_exists(tmp_path):
    target, _, request = _receipt_fixture(tmp_path)
    (tmp_path / "checkpoints/latest.json").unlink()
    result = _probe(request)
    assert result.returncode != 0
    assert "Checkpoint files exist" in result.stderr
    target.unlink()
    assert json.loads(_probe(request).stdout) == {"present": False}
    (tmp_path / "checkpoints/latest.json").write_text("null")
    assert _probe(request).returncode != 0


@pytest.mark.parametrize("error", ["PermissionError", "OSError"])
def test_unreadable_candidate_directory_is_not_treated_as_no_checkpoint(tmp_path, error):
    target, _, request = _receipt_fixture(tmp_path)
    (tmp_path / "checkpoints/latest.json").unlink()
    result = _probe(request, prelude=_scandir_failure(target.parent, error=error))
    assert result.returncode != 0
    assert "unreadable checkpoint directory" in result.stderr
    assert '"present": false' not in result.stdout


def test_missing_candidate_directory_still_permits_initial_replay(tmp_path):
    target, _, request = _receipt_fixture(tmp_path)
    (tmp_path / "checkpoints/latest.json").unlink()
    target.unlink()
    target.parent.rmdir()
    result = _probe(request)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"present": False}


@pytest.mark.parametrize("pattern", ["**/step-*.ckpt", "artifacts/**/step-*.ckpt", "artifacts/checkpoints/step-???.ckpt"])
def test_strict_checkpoint_glob_preserves_supported_matching(tmp_path, pattern):
    _, receipt, request = _receipt_fixture(tmp_path)
    request["contract"]["candidate_globs"] = [pattern]
    _save_receipt(tmp_path, receipt)
    result = _probe(request)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["present"] is True


def test_recursive_candidate_search_propagates_nested_directory_errors(tmp_path):
    target, _, request = _receipt_fixture(tmp_path)
    request["contract"]["candidate_globs"] = ["artifacts/**/*.ckpt"]
    (tmp_path / "checkpoints/latest.json").unlink()
    result = _probe(request, prelude=_scandir_failure(target.parent))
    assert result.returncode != 0
    assert "unreadable checkpoint directory" in result.stderr


def test_symlink_in_candidate_parent_is_rejected(tmp_path):
    target, _, request = _receipt_fixture(tmp_path)
    real_parent = target.parent.with_name("real-checkpoints")
    target.parent.rename(real_parent)
    target.parent.symlink_to(real_parent, target_is_directory=True)
    (tmp_path / "checkpoints/latest.json").unlink()
    result = _probe(request)
    assert result.returncode != 0
    assert "symlink" in result.stderr


@pytest.mark.parametrize("key,value", [
    ("run_id", "other"), ("sha256", "a" * 64), ("size_bytes", 1),
    ("resumable", False), ("final", True), ("contract", {}),
    ("cleanup", {"performed": True}), ("is_directory", True),
    ("path", "/outside/checkpoint.ckpt"), ("training_step", True),
])
def test_invalid_resume_receipt_fails_closed(tmp_path, key, value):
    _, receipt, request = _receipt_fixture(tmp_path)
    receipt[key] = value
    _save_receipt(tmp_path, receipt)
    assert _probe(request).returncode != 0


def test_partial_or_symlink_checkpoint_fails_closed(tmp_path):
    target, _, request = _receipt_fixture(tmp_path / "run")
    target.write_bytes(b"partial")
    assert _probe(request).returncode != 0
    target.unlink()
    outside = tmp_path / "other"
    outside.write_bytes(b"full checkpoint fixture")
    target.symlink_to(outside)
    assert _probe(request).returncode != 0


def test_checkpoint_glob_is_root_anchored(tmp_path):
    target, receipt, request = _receipt_fixture(tmp_path)
    wrong = tmp_path / "other" / target.relative_to(tmp_path)
    wrong.parent.mkdir(parents=True)
    target.rename(wrong)
    receipt["path"] = str(wrong)
    _save_receipt(tmp_path, receipt)
    result = _probe(request)
    assert result.returncode != 0
    assert "glob" in result.stderr.lower()


@pytest.mark.parametrize("pattern", ["**/../*.ckpt", "../*.ckpt", "/absolute/*", ""])
def test_unsafe_checkpoint_glob_is_rejected_before_receipt_lookup(tmp_path, pattern):
    _, receipt, request = _receipt_fixture(tmp_path)
    request["contract"]["candidate_globs"] = [pattern]
    (tmp_path / "checkpoints/latest.json").unlink()
    assert _probe(request).returncode != 0


def test_directory_identity_matches_existing_runner_schema(tmp_path):
    target, receipt, request = _receipt_fixture(tmp_path)
    target.unlink()
    target.mkdir()
    (target / "model").write_bytes(b"weights")
    (target / "optimizer").mkdir()
    (target / "optimizer/state").write_bytes(b"momentum")
    # Local fixture only: the production probe never imports/executes a runner.
    namespace = {"__name__": "checkpoint_identity_fixture"}
    exec(RUNNER_SOURCE, namespace)
    identity = namespace["checkpoint_identity"](target)
    receipt.update(identity)
    _save_receipt(tmp_path, receipt)
    result = _probe(request)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["identity"] == identity


@pytest.mark.parametrize("fail_on", [1, 3, 5])
def test_directory_fingerprint_and_hash_propagate_listing_errors(tmp_path, fail_on):
    target, receipt, request = _receipt_fixture(tmp_path)
    target.unlink()
    target.mkdir()
    (target / "model").write_bytes(b"weights")
    namespace = {"__name__": "checkpoint_identity_fixture"}
    exec(RUNNER_SOURCE, namespace)
    receipt.update(namespace["checkpoint_identity"](target))
    _save_receipt(tmp_path, receipt)
    # Two scans per directory walk: fingerprint, identity, final fingerprint.
    result = _probe(request, prelude=_scandir_failure(target, fail_on=fail_on))
    assert result.returncode != 0
    assert "unreadable checkpoint directory" in result.stderr


class LocalCheckpointCluster(FakeCluster):
    """Execute the fixed read-only probe against temporary fixture files."""

    def run_with_fallback(self, command, gateway="auto", **kwargs):
        if "SKYNET_MANUAL_RESUME_CHECKPOINT" in command:
            self.run_commands.append(command)
            assert kwargs["stdin"] == pipeline_api._MANUAL_RESUME_CHECKPOINT_PROBE
            result = subprocess.run([sys.executable, "-", shlex.split(command)[2]],
                                    input=kwargs["stdin"], text=True, capture_output=True, timeout=10)
            if result.returncode:
                raise ClusterError(result.stderr)
            return "sky2", result.stdout
        return super().run_with_fallback(command, gateway, **kwargs)


@pytest.fixture
def cancelled_training(tmp_path, monkeypatch):
    database = Database(tmp_path / "test.db")
    cluster = LocalCheckpointCluster()
    service = make_pipeline_service(database, cluster)
    adapter = next(item for item in database.list_adapter_registry() if item["seed_key"] == "pipeline_fixture")
    manifest = AdapterManifest.model_validate(adapter["latest_version"]["manifest"])
    manifest.train.checkpoint_globs = ["artifacts/checkpoints/step-*.ckpt"]
    database.upsert_seed_adapter(seed_key="pipeline_fixture", name=adapter["name"],
                                 manifest=manifest.model_dump(mode="json"))
    seed_repository_choices(service, manifest.slug, manifest=manifest)
    experiment = service.create_experiment(canonical_spec())
    service.submit_experiment(experiment["id"])
    run_id = experiment["runs"][0]["id"]
    run, stage, attempt = _mark_training_run_cancelled(database, run_id)
    root = tmp_path / "remote-run"
    monkeypatch.setattr(service, "_run_directory", lambda _: str(root))
    target, receipt, request = _receipt_fixture(
        root, run_id=run_id, plan=attempt["execution_snapshot_json"]["plan"],
        attempt_id=attempt["id"], job_id=str(attempt["slurm_job_id"]),
    )
    return service, cluster, run, stage, attempt, target, receipt


def test_manual_resume_recovers_unregistered_checkpoint_idempotently(cancelled_training):
    service, cluster, run, stage, attempt, target, receipt = cancelled_training
    pinned, error = pipeline_api._pinned_training_execution(run, stage, prefer_initial_attempt=True)
    assert error is None
    first = service._recover_manual_resume_checkpoint(run, stage, pinned, "sky2")
    second = service._recover_manual_resume_checkpoint(run, stage, pinned, "sky2")
    assert first["id"] == second["id"]
    assert first["is_resumable"] and not first["is_selected_for_inference"]
    assert first["produced_by_attempt_id"] == attempt["id"]
    assert first["sha256"] == receipt["sha256"]
    assert first["metadata_json"] == receipt
    assert len(service.database.get_run(run["id"])["checkpoints"]) == 1
    probe_count = sum("SKYNET_MANUAL_RESUME_CHECKPOINT" in cmd for cmd in cluster.run_commands)
    result = service.retry_run(run["id"], "resume", "sky2")
    assert result["status"] == "SUBMITTED"
    assert sum("SKYNET_MANUAL_RESUME_CHECKPOINT" in cmd for cmd in cluster.run_commands) == probe_count
    assert submitted_execution(cluster)["initial_checkpoint"] == str(target)
    assert service.database.get_run(run["id"])["attempts"][-1]["resume_checkpoint_id"] == first["id"]


def test_manual_resume_imports_before_submitting(cancelled_training):
    service, cluster, run, _, _, target, _ = cancelled_training
    assert service.database.get_run(run["id"])["checkpoints"] == []
    result = service.retry_run(run["id"], "resume", "sky2")
    refreshed = service.database.get_run(run["id"])
    assert result["status"] == "SUBMITTED"
    assert len(refreshed["checkpoints"]) == 1
    assert refreshed["attempts"][-1]["resume_checkpoint_id"] == refreshed["checkpoints"][0]["id"]
    assert submitted_execution(cluster)["initial_checkpoint"] == str(target)


def test_manual_resume_corruption_does_not_submit_or_replay(cancelled_training):
    service, cluster, run, _, _, target, _ = cancelled_training
    target.write_bytes(b"partial checkpoint")
    with pytest.raises(ValueError, match="Cannot verify manual resume checkpoint"):
        service.retry_run(run["id"], "resume", "sky2")
    refreshed = service.database.get_run(run["id"])
    assert refreshed["status"] == "CANCELLED"
    assert len(refreshed["attempts"]) == cluster.submit_count == 1
    assert refreshed["checkpoints"] == []


def test_manual_resume_unreadable_candidates_do_not_submit_or_replay(cancelled_training, monkeypatch):
    service, cluster, run, _, _, target, _ = cancelled_training
    (target.parents[2] / "checkpoints/latest.json").unlink()
    monkeypatch.setattr(pipeline_api, "_MANUAL_RESUME_CHECKPOINT_PROBE",
                        _scandir_failure(target.parent) + pipeline_api._MANUAL_RESUME_CHECKPOINT_PROBE)
    with pytest.raises(ValueError, match="Cannot verify manual resume checkpoint"):
        service.retry_run(run["id"], "resume", "sky2")
    refreshed = service.database.get_run(run["id"])
    assert refreshed["status"] == "CANCELLED"
    assert len(refreshed["attempts"]) == cluster.submit_count == 1
    assert refreshed["checkpoints"] == []
