"""Durable metric batching and replay through the existing shared journals."""

from unittest.mock import patch

import pytest

from skynet_app.database import Database
from skynet_app.tracking import (
    MLflowBridge,
    TrackingSettings,
    WANDB_ENQUEUE_BATCH_SIZE,
    WandBBridge,
    WandBSettings,
)
from skynet_app.tracking_journal import TrackingJournal
from test_tracking import FakeWandBBridge


def metric_samples(count):
    return [
        {
            "metrics": {"loss": 1 / (index + 1)},
            "step": index,
            "timestamp_ms": 100000 + index * 500,
            "idempotency_key": f"training-progress:{index}",
        }
        for index in range(count)
    ]


def test_shared_journal_batches_once_and_replays_sample_identity_after_restart(tmp_path):
    database = Database(tmp_path / "skynet.db")
    journal = TrackingJournal(database, "batch-run")
    settings = WandBSettings(api_key="fake", entity="team", auto_flush=False)
    capsule = tmp_path / "capsule"
    bridge = FakeWandBBridge(capsule, settings, journal=journal)
    bridge.ensure_run(
        local_run_id="batch-run", entity="team", project="test", run_name="test", group="test"
    )
    assert bridge.drain_spool().remaining == 0
    samples = metric_samples(250)
    with patch.object(bridge, "_atomic_write", wraps=bridge._atomic_write) as writes:
        first = bridge.log_metrics_batch("batch-run", samples + [samples[0]])
    assert writes.call_count == 1
    assert first[0].event_id == first[-1].event_id
    assert len({result.event_id for result in first}) == 250
    assert all(result.queued for result in first)
    assert bridge.history_rows == []

    restarted = FakeWandBBridge(capsule, settings, journal=journal)
    # A retry reads the durable queue, rather than relying on process memory.
    with patch.object(restarted, "_atomic_write", wraps=restarted._atomic_write) as writes:
        duplicate = restarted.log_metrics_batch("batch-run", samples)
    assert writes.call_count == 0
    assert [result.event_id for result in duplicate] == [result.event_id for result in first[:-1]]
    restarted.remote_runs = bridge.remote_runs
    assert restarted.drain_spool(limit=100).remaining == 150
    assert restarted.drain_spool(limit=100).remaining == 50
    assert restarted.drain_spool(limit=100).remaining == 0
    assert [row["_step"] for row in restarted.history_rows] == list(range(250))
    assert [row["_timestamp"] for row in restarted.history_rows] == [
        sample["timestamp_ms"] / 1000 for sample in samples
    ]
    acknowledged = restarted.log_metrics_batch("batch-run", samples)
    assert all(result.delivered for result in acknowledged)
    assert len(restarted.history_rows) == 250


def test_large_batch_retries_after_partial_persistence_without_duplicates(tmp_path):
    settings = WandBSettings(api_key=None, auto_flush=False)
    bridge = WandBBridge(tmp_path, settings)
    samples = metric_samples(WANDB_ENQUEUE_BATCH_SIZE + 3)
    write = bridge._atomic_write
    calls = 0

    def fail_second_append(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated persistence interruption")
        return write(path, payload)

    with patch.object(bridge, "_atomic_write", side_effect=fail_second_append):
        with pytest.raises(OSError, match="persistence interruption"):
            bridge.log_metrics_batch("run", samples)
    assert bridge.pending_count() == WANDB_ENQUEUE_BATCH_SIZE
    persisted_ids = [event["id"] for event in bridge._events_unlocked()]
    restarted = WandBBridge(tmp_path, settings)
    with patch.object(restarted, "_atomic_write", wraps=restarted._atomic_write) as writes:
        retried = restarted.log_metrics_batch("run", samples)
    assert writes.call_count == 1
    assert [result.event_id for result in retried[:WANDB_ENQUEUE_BATCH_SIZE]] == persisted_ids
    assert restarted.pending_count() == len(samples)
    assert [result.sequence for result in retried] == list(range(1, len(samples) + 1))


def test_system_batch_preserves_its_clock_and_dedupes_separately_from_training(tmp_path):
    settings = WandBSettings(api_key="fake", auto_flush=False)
    bridge = FakeWandBBridge(tmp_path, settings)
    bridge.ensure_run(
        local_run_id="run", entity="team", project="test", run_name="test", group="test"
    )
    bridge.drain_spool()
    training = bridge.log_metrics_batch("run", [{
        "metrics": {"loss": 0.2}, "step": 77, "timestamp_ms": 100000,
        "idempotency_key": "same-key",
    }])
    samples = [{
        "metrics": {"gpu.0.gpu": 80 + index},
        "timestamp_ms": 101000 + index * 15000,
        "runtime_seconds": 1 + index * 15,
        "idempotency_key": "same-key" if index == 0 else f"gpu:{index}",
    } for index in range(3)]
    with patch.object(bridge, "_atomic_write", wraps=bridge._atomic_write) as writes:
        system = bridge.log_system_metrics_batch("run", samples + [samples[0]])
    assert writes.call_count == 1
    assert system[0].event_id != training[0].event_id
    assert system[0].event_id == system[-1].event_id
    bridge.drain_spool()
    assert [row["_runtime"] for row in bridge.system_rows] == [1, 16, 31]
    assert [row["_timestamp"] for row in bridge.system_rows] == [101, 116, 131]
    assert all("_step" not in row for row in bridge.system_rows)
    assert bridge.history_rows[0]["_step"] == 77


@pytest.mark.parametrize("provider", ["wandb", "mlflow"])
def test_terminal_events_are_idempotent_across_restart(tmp_path, provider):
    cls, settings = (
        (WandBBridge, WandBSettings(api_key=None, auto_flush=False))
        if provider == "wandb"
        else (MLflowBridge, TrackingSettings(tracking_uri=None, auto_flush=False))
    )

    def enqueue(bridge):
        return [
            bridge.log_metrics("run", {"final/loss": 0.1}, idempotency_key="final:attempt1:metrics"),
            bridge.log_artifact_link(
                "run", name="checkpoint", uri="/checkpoints/final.ckpt",
                idempotency_key="final:attempt1:checkpoint",
            ),
            bridge.finish_run("run", status="FINISHED", idempotency_key="final:attempt1:finished"),
        ]

    first = enqueue(cls(tmp_path, settings))
    restarted = cls(tmp_path, settings)
    second = enqueue(restarted)
    assert [result.event_id for result in first] == [result.event_id for result in second]
    assert restarted.pending_count() == 3
