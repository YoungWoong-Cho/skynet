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
from skynet_app.slurm import source_cache_directory
from skynet_app.cluster_runtime import ClusterClient


def test_storage_reference_index_preserves_parent_child_and_sibling_boundaries():
    references = ['/work/keep', '/work/keep/a', '/work/keep/zzz', '/work/a/model', '/work/another']
    index = storage_files.PathReferences(references)
    candidates = ['/work', '/work/keep', '/work/keep/b', '/work/keep/zz', '/work/keep2',
                  '/work/a', '/work/a/model/file', '/work/abc', '/work/another2', '/other']
    for path in candidates:
        assert index.overlaps(path) == any(storage_files.overlaps(path, ref) for ref in references)


def test_large_storage_scan_returns_bounded_reviewable_batch(tmp_path):
    logs = tmp_path / 'logs'; logs.mkdir()
    for index in range(2001):
        (logs / str(index)).write_bytes(b'old')
    report = storage_files.execute({'operation': 'scan', 'root': str(tmp_path)})
    assert len(report['items']) == 2000 and report['truncated'] is True


def test_storage_scan_separates_pretrained_assets_from_shared_simulator(tmp_path):
    root = tmp_path / 'workspace'
    retained = root / 'assets/dexverse-v1/scene.usd'
    retained.parent.mkdir(parents=True); retained.write_text('keep')
    unused = root / 'assets/pretrained/vendor'
    unused.mkdir(parents=True); (unused / 'weights.pt').write_bytes(b'old weights')
    old = time.time() - 90000
    for path in [unused, unused / 'weights.pt']:
        os.utime(path, (old, old))
    report = storage_files.execute({'operation': 'scan', 'root': str(root), 'protected': [str(retained)]})
    assert [(item['path'], item['selectable']) for item in report['items']] == [(str(unused), True)]
    storage_files.execute({'operation': 'delete', 'root': str(root), 'items': report['items']})
    assert retained.read_text() == 'keep' and not unused.exists()


def test_pretrained_inspection_uses_same_safe_cleanup_contract(history, monkeypatch):
    service, _, _, _, _, evaluation, root = history
    assets = root / 'assets/pretrained/vendor'
    assets.mkdir(parents=True); (assets / 'weights.pt').write_bytes(b'unused')
    old = time.time() - 90000
    for path in (assets, assets / 'weights.pt'):
        os.utime(path, (old, old))
    logs = root / 'logs'; logs.mkdir(); (logs / 'unrelated').write_text('keep')
    report = service.inspect_storage(scope='pretrained')
    assert report['scope'] == 'pretrained'
    assert [item['path'] for item in report['items']] == [str(assets)]
    with pytest.raises(ValueError, match='Storage changed'):
        service.clean_storage([str(assets)], report['token'], scope='all')
    assert service.clean_storage([str(assets)], report['token'], scope='pretrained')['deleted'] == [str(assets)]
    assert (logs / 'unrelated').exists() and Path(evaluation['result_path']).exists()


class LocalCluster(ClusterClient):
    """Runs every remote program locally; the single gateway never needs probing."""

    def __init__(self):
        super().__init__(("test",))

    def resolve_gateway(self, value="auto"):
        return "test"

    def ssh(self, host, command, *, stdin=None, timeout=None):
        result = subprocess.run(
            shlex.split(command), input=stdin, text=True, capture_output=True
        )
        if result.returncode:
            raise OSError(result.stdout)
        return result.stdout


def build_history(tmp_path, *, offloaded=False):
    """A retained run with one evaluation; offloaded writes large bodies to cluster objects."""
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
    if offloaded:
        from skynet_app.payload_store import PayloadStore

        db.payload_store = PayloadStore(db)
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
    db.record_training_progress_sample(
        run["id"], train_attempt["id"], restart_count=0, completed=4, total=8,
        source_kind="log", evidence={"line": "step 4/8"},
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


@pytest.fixture
def history(tmp_path):
    return build_history(tmp_path)


@pytest.fixture
def offloaded_history(tmp_path, monkeypatch):
    """Bodies are offloaded from their first write, as every row in the central database is."""
    from skynet_app.metadata_objects import MetadataObjects
    from test_payload_store import local_exchange

    monkeypatch.setattr(MetadataObjects, "_exchange", local_exchange)
    return build_history(tmp_path, offloaded=True)


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
    # Bulk children enter the plan as identities plus the paths the file plan reads.
    with db.connection() as c:
        _, run_graph, _ = service._graph(c, "run", run["id"])
        _, evaluation_graph, _ = service._graph(c, "evaluation", evaluation["id"])
    assert [set(row) for row in run_graph["training_progress_samples"]] == [{"id"}]
    assert [set(row) for row in evaluation_graph["evaluation_episodes"]] == [{"id", "video_path", "raw_result_path"}]
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
            "training_progress_samples",
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


def checkpoint_draft(db, experiment, run):
    spec = {"resume_checkpoint": str(Path(run["run_directory"]) / "checkpoint.pt")}
    revision = db.create_experiment_revision(experiment["id"], spec)
    variant = db.create_variant(revision["id"], name="resume draft", parameters={}, resolved_spec=spec)
    return revision, variant


def test_unused_resume_draft_can_be_discarded_before_deleting_its_source(history):
    service, db, _, experiment, run, evaluation, _ = history
    erase(service, "evaluation", evaluation["id"])
    revision, variant = checkpoint_draft(db, experiment, run)
    blockers = service.preview("run", run["id"])["blockers"]
    assert [(b["kind"], b["id"]) for b in blockers] == [("draft-revision", revision["id"])]
    plan = service.preview("draft-revision", revision["id"])
    assert plan["counts"] == {"experiment_revisions": 1, "variants": 1}
    assert not plan["blockers"]
    service.delete("draft-revision", revision["id"], plan["token"])
    assert Path(run["run_directory"], "checkpoint.pt").read_bytes() == b"weights"
    assert db.get_experiment(experiment["id"])["latest_revision"]["id"] != revision["id"]
    with db.connection() as c:
        assert not c.execute("SELECT 1 FROM variants WHERE id=?", (variant["id"],)).fetchone()
    assert service.delete("draft-revision", revision["id"], "0" * 64)["already_deleted"]
    erase(service, "run", run["id"])
    erase(service, "experiment", experiment["id"])


def test_submitted_revision_cannot_be_discarded_as_a_draft(history):
    service, db, _, experiment, run, _, _ = history
    revision, _ = checkpoint_draft(db, experiment, run)
    db.claim_experiment_revision_submission(experiment["id"], revision["id"])
    plan = service.preview("draft-revision", revision["id"])
    assert "Submitted revisions" in plan["blockers"][0]["reason"]
    with pytest.raises(ValueError, match="dependencies"):
        service.delete("draft-revision", revision["id"], plan["token"])


def test_draft_submission_after_preview_blocks_deletion(history):
    service, db, _, experiment, run, _, _ = history
    revision, variant = checkpoint_draft(db, experiment, run)
    plan = service.preview("draft-revision", revision["id"])
    child = db.create_run(variant["id"], seed=1, adapter_name="hpt", adapter_version="1", status="PENDING", run_directory=str(Path(run["run_directory"]).parent / "child"))
    assert service.preview("draft-revision", revision["id"])["blockers"][0]["id"] == child["id"]
    with pytest.raises(ValueError, match="dependencies"):
        service.delete("draft-revision", revision["id"], plan["token"])
    assert db.get_run(child["id"]) is not None


def test_other_workspace_draft_reference_is_protected_without_disclosing_id(history):
    service, _, system, _, run, evaluation, _ = history
    erase(service, "evaluation", evaluation["id"])
    other = system.for_workspace("bob")
    project = other.create_project("Other")
    experiment = other.create_experiment(project_id=project["id"], name="Private draft", requested_spec={})
    revision, _ = checkpoint_draft(other, experiment, run)
    blockers = service.preview("run", run["id"])["blockers"]
    assert blockers and all(b["id"] is None for b in blockers)
    assert revision["id"] not in json.dumps(blockers)
    with pytest.raises(KeyError):
        service.preview("draft-revision", revision["id"])


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
    offloaded_history,
):
    from skynet_app.tracking_journal import TrackingJournal

    service, db, _, _, run, evaluation, root = offloaded_history
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


def test_remote_deletion_releases_sql_lock_and_blocks_new_references(history, monkeypatch):
    from skynet_app.db_backend import lock_key
    service, db, system, experiment, run, evaluation, root = history
    original = service.remote
    checked = []
    def remote(operation, root, items=None, **kwargs):
        if operation == 'delete':
            with db.connection() as c:
                assert c.execute('SELECT pg_try_advisory_xact_lock(?)', (lock_key('repository-write'),)).fetchone()[0]
            # Another run cannot gain a reference after validation and before unlink.
            parent = db.create_project('unrelated write during delete')
            checked.append(parent['id'])
            with pytest.raises(ValueError, match='being deleted'):
                db.create_artifact(run['id'], artifact_type='new consumer', path=items[0]['path'])
            from skynet_app.db_backend import INTEGRITY_ERRORS
            with pytest.raises(INTEGRITY_ERRORS):
                with db.transaction() as c:
                    c.execute('UPDATE runs SET run_directory=? WHERE id=?', (items[0]['path'], run['id']))
        return original(operation, root, items, **kwargs)
    monkeypatch.setattr(service, 'remote', remote)
    plan = service.preview('evaluation', evaluation['id'])
    assert service.delete('evaluation', evaluation['id'], plan['token'])['deleted']
    assert checked
    assert db.get_run(run['id']) is not None


def test_storage_scan_and_delete_do_not_hold_write_lock(history, monkeypatch):
    from skynet_app.db_backend import lock_key
    service, db, _, _, _, _, root = history
    orphan = root / 'jobs/runs/orphan'
    orphan.mkdir(parents=True)
    (orphan / 'old.pt').write_text('old')
    old = time.time() - 90000
    os.utime(orphan / 'old.pt', (old, old)); os.utime(orphan, (old, old))
    original = service.remote
    def remote(operation, root, items=None, **kwargs):
        with db.connection() as c:
            assert c.execute('SELECT pg_try_advisory_xact_lock(?)', (lock_key('repository-write'),)).fetchone()[0]
        return original(operation, root, items, **kwargs)
    monkeypatch.setattr(service, 'remote', remote)
    report = service.inspect_storage()
    result = service.clean_storage([str(orphan)], report['token'])
    assert str(orphan) in result['deleted']


def _age(path):
    old = time.time() - 90000
    for entry in (path, *path.rglob('*')):
        os.utime(entry, (old, old))


def _source_checkout(root, repository):
    """The checkout layout the sbatch preflight creates for a repository."""
    container = Path(source_cache_directory(str(root), repository))
    git = container / ('c' * 40) / '.git'
    git.mkdir(parents=True)
    (git / 'config').write_text(
        f'[core]\n\tbare = false\n[remote "origin"]\n\turl = {repository}\n'
        '\tfetch = +refs/heads/*:refs/remotes/origin/*\n'
    )
    return container


def test_dependencies_require_explicit_scope_and_exact_boundaries(history):
    service, db, _, _, _, _, root = history
    keep = root / 'envs/shared'; keep.mkdir(parents=True)
    operator_env = root / 'envs/obsolete'; operator_env.mkdir()
    manual_repo = root / 'repos/obsolete'; manual_repo.mkdir(parents=True)
    stale_checkout = _source_checkout(root, 'https://github.com/example/obsolete')
    retained_checkout = _source_checkout(root, 'https://github.com/example/retained')
    adapter_checkout = _source_checkout(root, 'https://github.com/example/adapter')
    forged = root / 'repos/forged-00000000'; (forged / 'c/.git').mkdir(parents=True)
    (forged / 'c/.git/config').write_text('[remote "origin"]\n\turl = https://github.com/example/forged\n')
    cache = root / '.cache/huggingface/hub/models--obsolete'; cache.mkdir(parents=True)
    service_dir = root / 'services'; service_dir.mkdir()
    outside = root.parent / 'outside'; outside.mkdir()
    linked = root / 'repos/linked'; linked.symlink_to(outside, target_is_directory=True)
    db.register_runtime_profile(name='shared', backend='existing', config={'environment_path': str(keep)})
    project = db.create_project('Sources')
    db.create_experiment(project_id=project['id'], name='Keeps its checkout',
                         requested_spec={'source': {'repository': 'https://github.com/example/retained'}})
    db.create_adapter(name='Adapter source', manifest={}, repository_url='https://github.com/example/adapter')
    for directory in (keep, operator_env, manual_repo, stale_checkout, retained_checkout,
                      adapter_checkout, forged, cache):
        (directory / 'data').write_text('payload')
        _age(directory)
    for path in (operator_env, manual_repo, stale_checkout, cache):
        with pytest.raises(ValueError, match='protected'):
            storage_files.checked_path(str(root), str(path))
    for path in (root/'envs', operator_env/'data', service_dir, root/'.cache', root/'.cache/huggingface/hub',
                 operator_env, manual_repo, forged):
        with pytest.raises(ValueError, match='protected'):
            storage_files.checked_path(str(root), str(path), scope='dependencies')
    report = service.inspect_storage(scope='dependencies')
    items = {item['path']: item for item in report['items']}
    # Retained experiments, active adapters and runtime profiles keep their dependencies.
    assert not {str(keep), str(retained_checkout), str(adapter_checkout)} & set(items)
    # Operator environments, manual clones, forged cache names and links stay visible but protected.
    for path in (operator_env, manual_repo, forged, linked):
        assert items[str(path)]['selectable'] is False
    targets = [str(p) for p in (stale_checkout, cache)]
    assert set(targets) == {path for path, item in items.items() if item['selectable']}
    assert set(service.clean_storage(targets, report['token'], scope='dependencies')['deleted']) == set(targets)
    assert all(not Path(path).exists() for path in targets)
    for path in (keep, operator_env, manual_repo, retained_checkout, adapter_checkout, forged):
        assert (path / 'data').exists()
    assert service_dir.exists() and outside.exists()


def test_dependency_new_reference_blocks_delete(history):
    service, db, _, _, _, _, root = history
    repository = 'https://github.com/example/late'
    checkout = _source_checkout(root, repository)
    (checkout / 'file').write_text('source')
    _age(checkout)
    report = service.inspect_storage(scope='dependencies')
    assert next(item for item in report['items'] if item['path'] == str(checkout))['selectable']
    project = db.create_project('Late consumer')
    db.create_experiment(project_id=project['id'], name='New consumer',
                         requested_spec={'source': {'repository': repository}})
    with pytest.raises(ValueError, match='Storage changed'):
        service.clean_storage([str(checkout)], report['token'], scope='dependencies')
    assert (checkout / 'file').read_text() == 'source'
