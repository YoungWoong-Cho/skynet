"""Produced scalar metrics survive parsing, persisted enrichment, and W&B spooling."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Event
from types import SimpleNamespace

import pytest

import skynet_app.pipeline_api as pipeline
from skynet_app.adapters import builtin_adapter_manifests
from skynet_app.database import Database
from skynet_app.experiments import ExperimentSpec
from skynet_app.tracking import WandBBridge, WandBSettings


JSONL_ADAPTERS = [
    adapter for adapter in builtin_adapter_manifests()
    if adapter.train.progress and adapter.train.progress.source.kind == "jsonl"
]
LARGE_STEP = 2**60 + 123
BOOKKEEPING_METRICS = {
    "training/completed", "training/total", "training/progress",
    "training/restart_count", "training/elapsed_seconds",
}


def produced_row(epoch=0):
    return {
        "epoch": epoch,
        "global_step": LARGE_STEP,
        "train_loss": 0.4,
        "val_loss": 0.3,
        "best_val_loss": 0.25,
        "lr": 0.0002,
        "learning_rate": 0.0001,
        "grad_norm": 1.5,
        "train/grad_norm": 1.25,
        "timing/train_batch_s": 0.125,
        "Validation/IoU": 0.75,
        "ZeroMetric": 0,
        "NegativeMetric": -2,
        "metrics": {
            "Optimizer": {"LR": 0.0003},
            "Timing": {"forward_seconds": 0.0125, "backward_seconds": 0.025},
            "nested": {"deeper": {"Score": 0.875}},
            "_private": {"leak": 12},
            "nested_private": {"_leak": 13},
            "enabled": True,
            "string_value": "2.0",
            "list_value": [1, 2],
            "nan_value": float("nan"),
            "infinite_value": float("inf"),
        },
        "_private": 42,
        "_timestamp": 1234567890,
        "enabled": True,
        "disabled": False,
        "string_value": "0.5",
        "nothing": None,
        "list_value": [1, 2],
        "configuration": {"numeric_but_not_a_metric": 10},
        "nan_value": float("nan"),
        "infinite_value": float("inf"),
        "negative_infinite_value": -float("inf"),
    }


def expected_raw_metrics(epoch=0):
    return {
        "epoch": epoch,
        "global_step": LARGE_STEP,
        "train_loss": 0.4,
        "val_loss": 0.3,
        "best_val_loss": 0.25,
        "lr": 0.0002,
        "learning_rate": 0.0001,
        "grad_norm": 1.5,
        "train/grad_norm": 1.25,
        "timing/train_batch_s": 0.125,
        "Validation/IoU": 0.75,
        "ZeroMetric": 0,
        "NegativeMetric": -2,
        "Optimizer/LR": 0.0003,
        "Timing/forward_seconds": 0.0125,
        "Timing/backward_seconds": 0.025,
        "nested/deeper/Score": 0.875,
    }


def expected_metrics(adapter, epoch=0):
    expected = expected_raw_metrics(epoch)
    row = produced_row(epoch)
    expected.update({alias: row[key] for key, alias in adapter.train.progress.source.metrics.items()})
    return expected


def test_every_builtin_jsonl_adapter_participates():
    assert len(JSONL_ADAPTERS) == 5
    assert {adapter.slug for adapter in JSONL_ADAPTERS} == {
        "egoverse-act", "egoverse-hpt", "egoverse-pi", "xpolicylab-act", "xpolicylab-dp",
    }


@pytest.mark.parametrize("adapter", JSONL_ADAPTERS, ids=lambda adapter: adapter.slug)
def test_all_jsonl_adapters_forward_native_scalars_nested_metrics_and_aliases(adapter):
    records = pipeline.parse_declared_training_progress(
        json.dumps(produced_row()) + "\n",
        adapter.train.progress,
        resolved_spec={"native": {"config": {"epochs": 10}}},
    )
    assert len(records) == 1
    assert records[0]["completed"] == 1
    assert records[0]["total"] == 10
    assert records[0]["metrics"] == expected_metrics(adapter)
    # A float round-trip silently changes this valid global step.
    assert type(records[0]["metrics"]["global_step"]) is int
    assert records[0]["metrics"]["global_step"] == LARGE_STEP
    if "global_step" in adapter.train.progress.source.metrics:
        alias = adapter.train.progress.source.metrics["global_step"]
        assert type(records[0]["metrics"][alias]) is int
        assert records[0]["metrics"][alias] == LARGE_STEP


@pytest.mark.parametrize("adapter", JSONL_ADAPTERS, ids=lambda adapter: adapter.slug)
def test_invalid_or_missing_alias_values_do_not_hide_other_produced_metrics(adapter):
    row = {
        "epoch": 0,
        "val_loss": float("nan"),
        "lr": True,
        "global_step": "9007199254740993",
        "best_val_loss": 0.2,
        "GradNorm": 0.0,
        "metrics": {"timing": {"epoch_seconds": 1.25}},
    }
    records = pipeline.parse_declared_training_progress(
        json.dumps(row) + "\n", adapter.train.progress,
        resolved_spec={"native": {"config": {"epochs": 10}}},
    )
    assert len(records) == 1  # Required-key presence still defines a complete row.
    assert records[0]["metrics"] == {
        "epoch": 0, "best_val_loss": 0.2, "GradNorm": 0.0,
        "timing/epoch_seconds": 1.25,
    }


@pytest.mark.parametrize("reverse", [False, True], ids=["nested-first", "literal-first"])
def test_explicit_native_metric_names_win_collisions_in_either_field_order(reverse):
    nested = [("a", {"b": 1}), ("a/b", 2), ("native", {"key": 3})]
    items = [
        ("epoch", 0), ("val_loss", 0.3), ("train_loss", 0.4),
        ("metrics", dict(reversed(nested) if reverse else nested)),
        ("native/key", 7), ("train/loss", 9),
    ]
    row = dict(reversed(items) if reverse else items)
    adapter = next(adapter for adapter in JSONL_ADAPTERS if adapter.slug == "xpolicylab-dp")
    records = pipeline.parse_declared_training_progress(
        json.dumps(row) + "\n", adapter.train.progress,
        resolved_spec={"native": {"config": {"epochs": 10}}},
    )
    assert len(records) == 1
    metrics = records[0]["metrics"]
    assert metrics["a/b"] == 2, "Explicit nested key wins over the flattened a.b path"
    assert metrics["native/key"] == 7, "Top-level native key wins over the nested metrics path"
    assert metrics["train/loss"] == 9, "Native metric wins over the train_loss compatibility alias"
    assert metrics["train_loss"] == 0.4, "The distinct native source metric is also retained"


@pytest.fixture
def forwarding_service(tmp_path, monkeypatch):
    spec = ExperimentSpec.model_validate({
        "identity": {"project": "metric-test", "experiment": "all-scalars"},
        "source": {
            "repository": "https://github.com/XPolicyLab/XPolicyLab",
            "revision": "a" * 40, "adapter": "xpolicylab-dp",
        },
        "native": {"config": {"epochs": 10}},
        "tracking": {"providers": [{
            "provider": "wandb", "enabled": True,
            "entity": "synthetic-team", "project": "metric-test",
        }]},
    }).model_dump(mode="json")
    database = Database(tmp_path / "metrics.db")
    project = database.create_project("metric-test")
    experiment = database.create_experiment(project_id=project["id"], name="all-scalars", requested_spec=spec)
    variant = database.create_variant(experiment["latest_revision"]["id"], name="one", parameters={}, resolved_spec=spec)
    run = database.create_run(
        variant["id"], seed=0, adapter_name="xpolicylab-dp", adapter_version="1",
        run_directory="/synthetic/run", status="RUNNING",
    )
    stage = database.create_stage(run["id"], stage_type="TRAIN", name="train")
    attempt = database.create_job_attempt(
        stage["id"], status="RUNNING", gateway="synthetic-host", slurm_job_id="123",
        started_at="2026-01-01T00:00:00Z",
    )
    content = {"text": json.dumps(produced_row()) + "\n"}
    reads = []

    def read_log(path, gateway, **kwargs):
        reads.append((path, gateway, kwargs))
        return gateway, content["text"]

    settings = WandBSettings(api_key=None, entity="synthetic-team", auto_flush=False)
    monkeypatch.setattr(pipeline, "LOCAL_CAPSULE_ROOT", tmp_path / "capsules")
    monkeypatch.setattr(WandBBridge, "binding", lambda self, local_id: {"name": local_id})
    monkeypatch.setattr(
        "skynet_app.tracking.urllib.request.urlopen",
        lambda *args, **kwargs: pytest.fail("Metric regression must not contact the network"),
    )

    def new_service():
        service = pipeline.PipelineService.__new__(pipeline.PipelineService)
        service.database = database
        service.cluster = SimpleNamespace(read_log=read_log)
        service._wandb_settings = lambda provider: settings
        service._native_tracking_provider_names = lambda spec: set()
        service._tracking_failure = lambda *args, **kwargs: pytest.fail(f"Unexpected tracking failure: {args!r}")
        return service

    capsule = tmp_path / "capsules" / run["id"]
    return SimpleNamespace(
        database=database, run_id=run["id"], attempt_id=attempt["id"],
        service=new_service(), new_service=new_service, content=content, reads=reads,
        capsule=capsule, settings=settings,
        adapter=next(adapter for adapter in JSONL_ADAPTERS if adapter.slug == "xpolicylab-dp"),
    )


def spool_events(fixture):
    path = fixture.capsule / "wandb-spool.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_all_metrics_reach_durable_wandb_spool_once_across_repeat_polls(forwarding_service):
    fixture = forwarding_service
    service = fixture.service
    assert service._ingest_training_progress(fixture.database.get_run(fixture.run_id)) == 1
    sample = fixture.database.list_training_progress_samples(fixture.run_id)[0]
    assert sample["evidence_json"]["metrics"] == expected_metrics(fixture.adapter)
    assert service._publish_training_progress_tracking(fixture.run_id) == 1
    events = spool_events(fixture)
    assert len(events) == 1 and events[0]["operation"] == "log_metrics"
    payload = events[0]["payload"]
    assert payload["local_run_id"] == fixture.run_id
    assert payload["step"] == 1
    for name, value in expected_metrics(fixture.adapter).items():
        assert payload["metrics"][name] == value
    assert type(payload["metrics"]["global_step"]) is int
    assert payload["metrics"]["global_step"] == LARGE_STEP
    assert BOOKKEEPING_METRICS.isdisjoint(payload["metrics"])
    assert not any(name.startswith("_") for name in payload["metrics"])
    service = fixture.new_service()  # Deduplication survives lost in-memory state.
    assert service._ingest_training_progress(fixture.database.get_run(fixture.run_id)) == 0
    assert service._publish_training_progress_tracking(fixture.run_id) == 0
    assert len(fixture.reads) == 2
    assert spool_events(fixture) == events


def test_producer_collisions_cannot_restore_bookkeeping_counters_to_training_history(forwarding_service):
    fixture = forwarding_service
    fixture.content["text"] = json.dumps({
        **produced_row(), **dict.fromkeys(BOOKKEEPING_METRICS, 999),
        "training/custom_accuracy": 0.9,
    }) + "\n"
    assert fixture.service._ingest_training_progress(fixture.database.get_run(fixture.run_id)) == 1
    sample = fixture.database.list_training_progress_samples(fixture.run_id)[0]
    # UI progress is still derived from the declared contract, not native-name collisions.
    assert sample["completed"] == 1
    assert sample["total"] == 10
    assert sample["restart_count"] == 0
    assert fixture.service._publish_training_progress_tracking(fixture.run_id) == 1
    payload = spool_events(fixture)[0]["payload"]
    assert BOOKKEEPING_METRICS.isdisjoint(payload["metrics"])
    assert payload["metrics"]["training/global_step"] == LARGE_STEP
    assert payload["metrics"]["training/custom_accuracy"] == 0.9
    assert payload["metrics"]["train_loss"] == 0.4
    assert payload["step"] == 1
    assert fixture.service._publish_training_progress_tracking(fixture.run_id) == 0


def test_historical_sample_and_spool_gain_only_unsent_metrics_once(forwarding_service):
    fixture = forwarding_service
    recorded_at = "2026-01-02T03:04:05.678Z"
    legacy_aliases = {"train/loss": 0.4, "validation/loss": 0.3}
    old_sample = fixture.database.record_training_progress_sample(
        fixture.run_id, fixture.attempt_id,
        restart_count=0, completed=1, total=10, unit="epoch", source_kind="jsonl",
        evidence={
            "declaration_origin": "legacy-contract",
            "path": "/synthetic/run/artifacts/logs.json.txt", "stream": "file",
            "elapsed_seconds": None, "metrics": legacy_aliases,
        },
        recorded_at=recorded_at,
    )
    original_metrics = {
        "training/completed": 1, "training/restart_count": 0,
        "training/total": 10, "training/progress": 0.1, **legacy_aliases,
    }
    timestamp_ms = int(datetime.fromisoformat(recorded_at.replace("Z", "+00:00")).timestamp() * 1000)
    bridge = WandBBridge(fixture.capsule, fixture.settings)
    bridge.log_metrics(
        fixture.run_id, original_metrics, step=1, timestamp_ms=timestamp_ms,
        idempotency_key=f"training-progress:{old_sample['id']}",
    )
    old_events = spool_events(fixture)
    fixture.service._ingest_training_progress(fixture.database.get_run(fixture.run_id))
    samples = fixture.database.list_training_progress_samples(fixture.run_id)
    assert len(samples) == 1
    enriched = samples[0]
    assert enriched["id"] == old_sample["id"]
    assert enriched["recorded_at"] == old_sample["recorded_at"]
    assert enriched["created_at"] == old_sample["created_at"]
    assert enriched["evidence_json"]["declaration_origin"] == "legacy-contract"
    assert enriched["evidence_json"]["metrics"] == expected_metrics(fixture.adapter)
    assert fixture.service._publish_training_progress_tracking(fixture.run_id) == 1
    events = spool_events(fixture)
    assert events[:1] == old_events
    assert len(events) == 2
    delta = events[1]["payload"]
    expected_delta = {name: value for name, value in expected_metrics(fixture.adapter).items() if name not in original_metrics}
    assert delta["metrics"] == expected_delta
    assert delta["step"] == 1
    assert delta["timestamp_ms"] == timestamp_ms
    assert delta["local_run_id"] == fixture.run_id
    assert delta["idempotency_key"] != old_events[0]["payload"]["idempotency_key"]
    service = fixture.new_service()
    service._ingest_training_progress(fixture.database.get_run(fixture.run_id))
    assert service._publish_training_progress_tracking(fixture.run_id) == 0
    assert spool_events(fixture) == events


def test_native_wandb_is_excluded_unless_explicitly_requested(forwarding_service):
    fixture = forwarding_service
    fixture.service._ingest_training_progress(fixture.database.get_run(fixture.run_id))
    fixture.service._native_tracking_provider_names = lambda spec: {"wandb"}
    assert fixture.service._publish_training_progress_tracking(fixture.run_id) == 0
    assert spool_events(fixture) == []
    assert fixture.service._publish_training_progress_tracking(fixture.run_id, include_native=True) == 1
    assert len(spool_events(fixture)) == 1


def test_concurrent_enrichment_publishers_emit_overlapping_metric_names_only_once(forwarding_service, monkeypatch):
    fixture = forwarding_service
    fixture.service._ingest_training_progress(fixture.database.get_run(fixture.run_id))
    assert fixture.service._publish_training_progress_tracking(fixture.run_id) == 1
    sample = fixture.database.list_training_progress_samples(fixture.run_id)[0]
    original_events = spool_events(fixture)
    assert fixture.database.enrich_training_progress_metrics(sample["id"], {"extra/x": 1})

    first_before_append = Event()
    release_first = Event()
    second_started = Event()
    second_before_append = Event()
    real_log_metrics = WandBBridge.log_metrics

    def hold_first_append(bridge, local_run_id, metrics, **kwargs):
        if metrics == {"extra/x": 1}:
            first_before_append.set()
            assert release_first.wait(timeout=5), "First publisher was never released"
        else:
            second_before_append.set()
        return real_log_metrics(bridge, local_run_id, metrics, **kwargs)

    monkeypatch.setattr(WandBBridge, "log_metrics", hold_first_append)
    second_service = fixture.new_service()

    def publish_second():
        second_started.set()
        return second_service._publish_training_progress_tracking(fixture.run_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(fixture.service._publish_training_progress_tracking, fixture.run_id)
        try:
            assert first_before_append.wait(timeout=3)
            # A computed {x}, but its durable event does not exist yet. B sees a
            # newer database sample {x, y}; publication must wait for A's event.
            assert fixture.database.enrich_training_progress_metrics(sample["id"], {"extra/y": 2})
            second = pool.submit(publish_second)
            assert second_started.wait(timeout=3)
            overlapped = second_before_append.wait(timeout=0.2)
        finally:
            release_first.set()
        assert first.result(timeout=3) == 1
        assert second.result(timeout=3) == 1

    assert not overlapped, "Another service must not snapshot/append a delta before the first append finishes"
    events = spool_events(fixture)
    assert events[:1] == original_events
    assert len(events) == 3
    assert [event["payload"]["metrics"] for event in events[1:]] == [
        {"extra/x": 1}, {"extra/y": 2},
    ]
    for event in events[1:]:
        assert event["payload"]["step"] == original_events[0]["payload"]["step"]
        assert event["payload"]["timestamp_ms"] == original_events[0]["payload"]["timestamp_ms"]
    assert fixture.new_service()._publish_training_progress_tracking(fixture.run_id) == 0
    assert spool_events(fixture) == events
