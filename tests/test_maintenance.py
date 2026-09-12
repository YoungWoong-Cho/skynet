"""Exercise actual files and relational dependencies, including interrupted deletes."""

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import pytest

from skynet_app import storage_files
from skynet_app.database import Database
from skynet_app.maintenance import Maintenance


class LocalCluster:
    def resolve_gateway(self, value):
        return "test"

    def ssh(self, host, command, *, stdin=None, timeout=None):
        result = subprocess.run(
            shlex.split(command), input=stdin, text=True, capture_output=True
        )
        if result.returncode:
            raise OSError(result.stdout)
        return result.stdout


@pytest.fixture
def history(tmp_path):
    system = Database(tmp_path / "test.db")
    with system.transaction() as c:
        c.execute(
            "INSERT INTO workspaces(id,email) VALUES ('alice','alice@example.com'),('bob','bob@example.com')"
        )
        c.execute(
            "INSERT INTO workspace_storage(owner_id,work_root) VALUES (?,?)",
            ("alice", str(tmp_path / "cluster")),
        )
    db = system.for_workspace("alice")
    root = tmp_path / "cluster"
    project = db.create_project("Testing")
    experiment = db.create_experiment(
        project_id=project["id"], name="Retain original recordings", requested_spec={}
    )
    variant = db.create_variant(
        experiment["latest_revision"]["id"], name="one", parameters={}, resolved_spec={}
    )
    run = db.create_run(
        variant["id"],
        seed=1,
        adapter_name="hpt",
        adapter_version="1",
        run_directory="unset",
        status="SUCCEEDED",
    )
    run_dir = root / "jobs/runs" / run["id"]
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint.pt").write_bytes(b"weights")
    run = db.update_run(run["id"], run_directory=str(run_dir))
    train = db.create_stage(
        run["id"], stage_type="TRAIN", name="train", status="SUCCEEDED"
    )
    train_attempt = db.create_job_attempt(
        train["id"], status="SUCCEEDED", slurm_job_id="123"
    )
    checkpoint = db.create_checkpoint(
        run["id"],
        path=str(run_dir / "checkpoint.pt"),
        checkpoint_type="model",
        produced_by_attempt_id=train_attempt["id"],
    )
    stage = db.create_stage(
        run["id"], stage_type="EVALUATE", name="eval", status="SUCCEEDED"
    )
    attempt = db.create_job_attempt(stage["id"], status="SUCCEEDED", slurm_job_id="124")
    result_dir = root / "eval/runs" / stage["id"]
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text('{"success":true}')
    (result_dir / "videos").mkdir()
    (result_dir / "videos/one.mp4").write_bytes(b"video")
    evaluation = db.create_evaluation(
        run["id"],
        stage_id=stage["id"],
        checkpoint_id=checkpoint["id"],
        evaluator_adapter="hpt",
        evaluator_version="1",
        suite_name="cube",
        suite_version="1",
        tasks=["cube"],
        seeds=[1],
        episodes_per_task=1,
        status="SUCCEEDED",
        result_path=str(result_dir / "result.json"),
    )
    db.upsert_evaluation_episode(
        evaluation["id"],
        task="cube",
        seed=1,
        episode_index=0,
        status="SUCCEEDED",
        video_path=str(result_dir / "videos/one.mp4"),
    )
    db.create_artifact(
        run["id"],
        evaluation_id=evaluation["id"],
        stage_id=stage["id"],
        artifact_type="EVALUATION_RESULT",
        path=str(result_dir / "result.json"),
    )
    db.record_event(
        entity_type="job_attempt",
        entity_id=attempt["id"],
        event_type="COMMON_HYPERPARAMETERS_ENRICHED_V1",
        details={"provenance": "keep immutable until delete"},
    )
    raw = root / "datasets/raw/keep.pkl"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"original")
    (run_dir / "dataset-link").symlink_to(raw.parent, target_is_directory=True)
    service = Maintenance(db, LocalCluster(), local_capsules=tmp_path / "capsules")
    return service, db, system, experiment, run, evaluation, root


def test_evaluation_removes_own_attempt_inside_retained_run(history):
    service, db, _, _, run, evaluation, root = history
    attempt_dir = Path(run["run_directory"]) / "attempts/124"
    attempt_dir.mkdir(parents=True)
    receipt = attempt_dir / "job.sbatch"
    receipt.write_text("evaluation receipt")
    (attempt_dir / "stdout.log").symlink_to(root / "datasets/raw/keep.pkl")
    db.create_artifact(
        run["id"],
        evaluation_id=evaluation["id"],
        stage_id=evaluation["stage_id"],
        artifact_type="SBATCH",
        path=str(receipt),
    )
    refs = [(run["run_directory"], "runs", run["id"])]
    graph = {"evaluations": [evaluation]}
    assert not service._referenced(str(attempt_dir), refs, graph)
    assert service._referenced(run["run_directory"], refs, graph)
    assert service._referenced(
        str(attempt_dir), refs + [(str(receipt), "artifacts", "another")], graph
    )
    erase(service, "evaluation", evaluation["id"])
    assert not attempt_dir.exists()
    assert (Path(run["run_directory"]) / "checkpoint.pt").read_bytes() == b"weights"
    assert (root / "datasets/raw/keep.pkl").read_bytes() == b"original"


def erase(service, kind, identifier):
    plan = service.preview(kind, identifier)
    assert not plan["blockers"], plan["blockers"]
    return service.delete(kind, identifier, plan["token"])


def test_dependencies_then_complete_delete_and_idempotent_retry(history):
    service, db, _, experiment, run, evaluation, root = history
    assert service.preview("run", run["id"])["blockers"][0]["id"] == evaluation["id"]
    assert (
        service.preview("experiment", experiment["id"])["blockers"][0]["id"]
        == run["id"]
    )
    assert erase(service, "evaluation", evaluation["id"])["deleted"]
    assert not Path(evaluation["result_path"]).parent.exists()
    assert Path(run["run_directory"]).exists()
    assert db.get_evaluation(evaluation["id"]) is None
    assert service.delete("evaluation", evaluation["id"], "0" * 64)["already_deleted"]
    assert len(db.get_run(run["id"])["stages"]) == 1
    erase(service, "run", run["id"])
    erase(service, "experiment", experiment["id"])
    assert not Path(run["run_directory"]).exists()
    assert (root / "datasets/raw/keep.pkl").read_bytes() == b"original"
    with db.connection() as c:
        for table in (
            "runs",
            "experiments",
            "experiment_revisions",
            "variants",
            "job_attempts",
            "checkpoints",
            "artifacts",
            "events",
            "metrics",
            "evaluation_episodes",
            "maintenance_operations",
        ):
            assert c.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0, table


def test_stale_preview_does_not_remove_files(history):
    service, _, _, _, _, evaluation, _ = history
    plan = service.preview("evaluation", evaluation["id"])
    result = Path(evaluation["result_path"])
    result.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        service.delete("evaluation", evaluation["id"], plan["token"])
    assert result.read_text() == "changed"


def test_interruption_keeps_retry_record_and_blocks_reattachment(history, monkeypatch):
    service, db, _, _, run, evaluation, _ = history
    plan = service.preview("evaluation", evaluation["id"])
    original = service.remote

    def disconnect(operation, *args, **kwargs):
        result = original(operation, *args, **kwargs)
        if operation == "delete":
            raise OSError("Acknowledgment lost")
        return result

    monkeypatch.setattr(service, "remote", disconnect)
    with pytest.raises(OSError):
        service.delete("evaluation", evaluation["id"], plan["token"])
    assert db.get_evaluation(evaluation["id"])
    assert not Path(evaluation["result_path"]).exists()
    with pytest.raises(ValueError, match="being deleted"):
        db.update_evaluation(evaluation["id"], status="PENDING")
    # Other runs are not frozen by an incomplete deletion.
    db.update_run(run["id"], status="SUCCEEDED")
    monkeypatch.setattr(service, "remote", original)
    assert service.preview("evaluation", evaluation["id"])["retry"]
    erase(service, "evaluation", evaluation["id"])


def test_foreign_workspace_cannot_inspect_or_delete(history):
    service, _, system, _, run, _, _ = history
    other = Maintenance(system.for_workspace("bob"), service.cluster)
    with pytest.raises(KeyError):
        other.preview("run", run["id"])
    # A missing ID is idempotent only if there really is no such item.
    with pytest.raises(KeyError):
        other.delete("run", run["id"], "0" * 64)


def test_active_and_shared_checkpoint_dependencies_block(history):
    service, db, _, _, run, evaluation, _ = history
    db.update_evaluation(evaluation["id"], status="RUNNING")
    assert any(
        "active" in b["reason"]
        for b in service.preview("evaluation", evaluation["id"])["blockers"]
    )


def test_storage_scan_and_cleanup_preserve_registered_files(history):
    service, _, _, _, run, evaluation, root = history
    orphan = root / "jobs/runs/orphan"
    orphan.mkdir()
    (orphan / "tmp").write_bytes(b"stale")
    old = time.time() - 90000
    os.utime(orphan / "tmp", (old, old))
    os.utime(orphan, (old, old))
    (root / "services").mkdir()
    (root / "services/database").write_text("protected")
    report = service.inspect_storage()
    assert str(orphan) in {item["path"] for item in report["items"]}
    assert not any(item["path"] == run["run_directory"] for item in report["items"])
    service.clean_storage([str(orphan)], report["token"])
    assert not orphan.exists()
    assert Path(evaluation["result_path"]).exists()
    assert (root / "services/database").read_text() == "protected"


def test_batch_validates_every_path_before_deleting_and_rejects_symlinks(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    first = root / "first"
    first.write_bytes(b"keep")
    second = root / "second"
    second.write_bytes(b"keep")
    items = storage_files.execute(
        {
            "operation": "inspect",
            "root": str(root),
            "items": [{"path": str(p)} for p in (first, second)],
        }
    )["items"]
    second.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        storage_files.execute(
            {"operation": "delete", "root": str(root), "items": items}
        )
    assert first.exists()
    second.unlink()
    second.symlink_to(first)
    with pytest.raises(ValueError, match="Symbolic"):
        storage_files.execute(
            {
                "operation": "inspect",
                "root": str(root),
                "items": [{"path": str(second)}],
            }
        )


def test_central_evaluation_delete_scrubs_journal_and_removes_retired_bodies(
    history, monkeypatch
):
    from skynet_app.metadata_objects import _REMOTE, MetadataObjects
    from skynet_app.payload_migration import relocate
    from skynet_app.payload_store import PayloadStore
    from skynet_app.tracking_journal import TrackingJournal

    service, db, _, _, run, evaluation, root = history

    def exchange(self, request):
        result = subprocess.run(
            ["python3", "-c", _REMOTE],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout)

    monkeypatch.setattr(MetadataObjects, "_exchange", exchange)
    db.payload_store = PayloadStore(db)
    relocate(db)
    journal = TrackingJournal(db, run["id"]).file("wandb-spool.jsonl")
    event = {
        "sequence": 1,
        "operation": "log_batch",
        "payload": {
            "metrics": {"train/loss": 0.2, f"evaluation/{evaluation['id']}/reward": 1}
        },
    }
    journal.write_bytes(json.dumps(event).encode() + b"\n")
    plan = service.preview("evaluation", evaluation["id"])
    assert len(plan["journals"]) == 1
    old = [item["path"] for item in plan["journals"][0]["retired"]]
    service.delete("evaluation", evaluation["id"], plan["token"])
    assert evaluation["id"].encode() not in journal.read_bytes()
    assert b"train/loss" in journal.read_bytes()
    assert all(not Path(path).exists() for path in old)
    erase(service, "run", run["id"])
    with db.connection() as c:
        assert c.execute("SELECT count(*) FROM tracking_journals").fetchone()[0] == 0
        assert (
            c.execute("SELECT count(*) FROM metadata_payload_refs").fetchone()[0] == 0
        )
        assert c.execute("SELECT count(*) FROM metadata_payloads").fetchone()[0] == 0
