import json
import subprocess
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import skynet_app.pipeline_api as pipeline
from skynet_app.adapters import TrainingProgressContract, TrainingProgressJsonlSource
from skynet_app.adapters.dp_manifest import manifest
from skynet_app.cluster_runtime import ClusterClient
from skynet_app.database import Database
from skynet_app.experiments import ExperimentSpec
from skynet_app.tracking import WandBBridge, WandBSettings


def spec():
    return ExperimentSpec.model_validate(
        {
            "identity": {"project": "test", "experiment": "dp-progress"},
            "source": {
                "repository": "https://github.com/XPolicyLab/XPolicyLab",
                "revision": "a" * 40,
                "adapter": "xpolicylab-dp",
            },
            "native": {"config": {"epochs": 100}},
            "train": {"max_steps": 1000},
            "tracking": {
                "providers": [
                    {
                        "provider": "wandb",
                        "enabled": True,
                        "entity": "test",
                        "project": "dp-progress",
                    }
                ]
            },
        }
    ).model_dump(mode="json")


def epoch(index):
    return {
        "epoch": index,
        "global_step": (index + 1) * 634 - 1,
        "train_loss": 0.1,
        "val_loss": 0.05,
        "lr": 0.0001,
    }


def test_existing_log_contracts_keep_their_snapshot_shape():
    legacy = {
        "schema_version": "skynet.training-progress-source/v1",
        "unit": "step",
        "total_path": "train.max_steps",
        "source": {
            "kind": "log_regex", "stream": "stderr",
            "pattern": r"(?P<completed>\d+)/(?P<total>\d+)",
            "value_format": "integer", "elapsed_format": None,
            "tail_lines": 500, "poll_seconds": 5,
        },
    }
    assert TrainingProgressContract.model_validate(legacy).model_dump(mode="json") == legacy
    del legacy["source"]["kind"]
    assert TrainingProgressContract.model_validate(legacy).source.kind == "log_regex"


def test_dp_parser_reads_completed_epochs_and_ignores_partial_or_invalid_rows():
    content = (
        "\n".join(
            [
                json.dumps({"epoch": 0, "global_step": 0, "train_loss": 1.0}),
                json.dumps(epoch(0)),
                "malformed",
                "[]",
                json.dumps({**epoch(1), "train_loss": float("nan")}),
                json.dumps(epoch(100)),
                json.dumps({**epoch(2), "epoch": True}),
            ]
        )
        + "\n"
        + json.dumps(epoch(3))
    )
    records = pipeline.parse_declared_training_progress(
        content, manifest().train.progress, resolved_spec=spec()
    )
    assert [row["completed"] for row in records] == [1, 2]
    assert records[0]["total"] == 100  # The generic max_steps default is irrelevant.
    assert records[0]["metrics"]["validation/loss"] == 0.05
    assert "train/loss" not in records[1]["metrics"]


def test_epoch_eta_uses_completed_work_and_excludes_step_checkpoints():
    now = datetime.now(UTC)
    started = (now - timedelta(hours=1)).isoformat()
    summary = pipeline.training_progress_summary(
        {
            "status": "RUNNING",
            "adapter_name": "xpolicylab-dp",
            "resolved_spec_json": spec(),
        },
        attempts=[{"id": "attempt", "status": "RUNNING", "started_at": started}],
        checkpoints=[
            {
                "training_step": 999,
                "produced_by_attempt_id": "attempt",
                "created_at": now.isoformat(),
            }
        ],
        progress_samples=[
            {
                "attempt_id": "attempt",
                "completed": 20,
                "total": 100,
                "unit": "epoch",
                "recorded_at": now.isoformat(),
            }
        ],
        now=now,
    )
    assert summary["unit"] == "epoch"
    assert summary["completed"] == 20
    assert summary["fraction"] == 0.2
    assert summary["eta_seconds"] == 4 * 3600


def test_epoch_ingestion_and_wandb_publication_are_idempotent(tmp_path, monkeypatch):
    db = Database(tmp_path / "test.db")
    project = db.create_project("test")
    experiment = db.create_experiment(
        project_id=project["id"], name="progress", requested_spec=spec()
    )
    variant = db.create_variant(
        experiment["latest_revision"]["id"],
        name="test",
        parameters={},
        resolved_spec=spec(),
    )
    run = db.create_run(
        variant["id"],
        seed=42,
        adapter_name="xpolicylab-dp",
        adapter_version="1",
        run_directory="/cluster/run",
        status="RUNNING",
    )
    stage = db.create_stage(run["id"], stage_type="TRAIN", name="train")
    db.create_job_attempt(stage["id"], status="SUBMISSION_FAILED")
    attempt = db.create_job_attempt(
        stage["id"], status="RUNNING", gateway="test", started_at="2026-01-01T00:00:00Z"
    )
    calls = []
    content = "".join(json.dumps(epoch(i)) + "\n" for i in range(20))

    def read(path, gateway, **kwargs):
        calls.append((path, kwargs))
        return gateway, content

    service = pipeline.PipelineService.__new__(pipeline.PipelineService)
    service.database = db
    service.cluster = SimpleNamespace(read_log=read)
    monkeypatch.setattr(pipeline, "LOCAL_CAPSULE_ROOT", tmp_path / "capsules")
    monkeypatch.setattr(
        service, "_wandb_settings", lambda _: WandBSettings(auto_flush=False)
    )
    monkeypatch.setattr(service, "_native_tracking_provider_names", lambda _: set())
    monkeypatch.setattr(WandBBridge, "binding", lambda self, _: {"name": run["id"]})
    assert service._ingest_training_progress(db.get_run(run["id"])) == 20
    assert calls[0] == (
        "/cluster/run/artifacts/logs.json.txt",
        {"lines": 1000, "max_bytes": 1_000_000, "contains": '"val_loss"'},
    )
    service._training_progress_last_reads = {}
    assert service._ingest_training_progress(db.get_run(run["id"])) == 0
    samples = db.list_training_progress_samples(run["id"])
    assert len(samples) == 20
    assert all(
        row["attempt_id"] == attempt["id"] and row["unit"] == "epoch" for row in samples
    )
    assert service._publish_training_progress_tracking(run["id"]) == 20
    assert service._publish_training_progress_tracking(run["id"]) == 0
    spool = tmp_path / "capsules" / run["id"] / "wandb-spool.jsonl"
    events = [json.loads(line) for line in spool.read_text().splitlines()]
    assert len(events) == 20
    assert events[-1]["payload"]["step"] == 20
    assert events[-1]["payload"]["metrics"]["validation/loss"] == 0.05


def test_filtered_log_read_keeps_epoch_rows_before_tail_limit(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(epoch(0)) + "\n" + "{}\n" * 6000 + json.dumps(epoch(1)) + "\n"
    )
    client = ClusterClient()
    monkeypatch.setattr(client, "_remote_path", lambda value: value)
    monkeypatch.setattr(
        client,
        "run_with_fallback",
        lambda command, gateway, **kw: (
            gateway,
            subprocess.check_output(command, shell=True, text=True),
        ),
    )
    _, output = client.read_log(str(path), lines=2, contains='"val_loss"')
    assert [json.loads(line)["epoch"] for line in output.splitlines()] == [0, 1]


@pytest.mark.parametrize(
    "path", ["/etc/passwd", "../metrics", "artifacts/../../metrics", "a\\b"]
)
def test_progress_files_cannot_escape_run_directory(path):
    with pytest.raises(ValueError):
        TrainingProgressJsonlSource(
            path=path, completed_key="epoch", required_key="val_loss"
        )
