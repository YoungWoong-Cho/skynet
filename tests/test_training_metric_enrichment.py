from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from skynet_app.database import Database


@pytest.fixture
def sample(tmp_path):
    database = Database(tmp_path / "metrics.db")
    project = database.create_project("metrics")
    experiment = database.create_experiment(
        project_id=project["id"], name="progress", requested_spec={}
    )
    variant = database.create_variant(
        experiment["latest_revision"]["id"], name="one", parameters={}, resolved_spec={}
    )
    run = database.create_run(
        variant["id"], seed=42, adapter_name="generic", adapter_version="1",
        run_directory="/fixture/run", status="RUNNING",
    )
    stage = database.create_stage(run["id"], stage_type="TRAIN", name="train")
    attempt = database.create_job_attempt(stage["id"], status="RUNNING")
    recorded = database.record_training_progress_sample(
        run["id"], attempt["id"], restart_count=2, completed=7, total=20,
        source_kind="jsonl", unit="epoch", recorded_at="2026-01-01T00:00:00Z",
        evidence={"path": "artifacts/logs.jsonl", "source": {"line": 8}, "loss": 0.75},
    )
    return database, recorded


def read_sample(database, recorded):
    samples = database.list_training_progress_samples(recorded["run_id"])
    assert len(samples) == 1
    return samples[0]


def test_enrichment_upgrades_old_sample_without_rewriting_observations(sample):
    database, recorded = sample
    metrics = {"train/loss": 0.5, "validation/loss": 0.25, "optimizer/updates": 7}

    assert database.enrich_training_progress_metrics(recorded["id"], metrics) is True
    expected = {**recorded, "evidence_json": {**recorded["evidence_json"], "metrics": metrics}}
    enriched = read_sample(database, recorded)
    assert enriched == expected
    assert type(enriched["evidence_json"]["metrics"]["optimizer/updates"]) is int
    assert database.enrich_training_progress_metrics(recorded["id"], metrics) is False
    assert read_sample(database, recorded) == expected


def test_enrichment_preserves_existing_metric_values_and_adds_only_missing_keys(sample):
    database, recorded = sample
    assert database.enrich_training_progress_metrics(recorded["id"], {"train/loss": 0.5})
    assert database.enrich_training_progress_metrics(
        recorded["id"], {"train/loss": 999, "validation/loss": 0.25}
    ) is True
    assert read_sample(database, recorded)["evidence_json"]["metrics"] == {
        "train/loss": 0.5, "validation/loss": 0.25,
    }
    before = read_sample(database, recorded)
    assert database.enrich_training_progress_metrics(recorded["id"], {"train/loss": 999}) is False
    assert read_sample(database, recorded) == before


def test_missing_sample_does_not_create_a_row(sample):
    database, recorded = sample
    assert database.enrich_training_progress_metrics("missing", {"train/loss": 0.5}) is False
    assert read_sample(database, recorded) == recorded


@pytest.mark.parametrize("invalid", [True, False, float("nan"), float("inf"), -float("inf"), "1", None])
def test_enrichment_excludes_nonfinite_and_nonnumeric_metrics(sample, invalid):
    database, recorded = sample
    assert database.enrich_training_progress_metrics(recorded["id"], {"invalid": invalid}) is False
    assert read_sample(database, recorded) == recorded
    assert database.enrich_training_progress_metrics(
        recorded["id"], {"invalid": invalid, "finite": 0, "negative": -0.25}
    ) is True
    assert read_sample(database, recorded)["evidence_json"]["metrics"] == {"finite": 0, "negative": -0.25}


def test_enrichment_ignores_nonstring_keys_and_empty_input(sample):
    database, recorded = sample
    assert database.enrich_training_progress_metrics(recorded["id"], {}) is False
    assert database.enrich_training_progress_metrics(recorded["id"], {1: 0.5}) is False
    assert read_sample(database, recorded) == recorded


def test_enrichment_preserves_large_integer_without_float_conversion(sample):
    database, recorded = sample
    value = 10 ** 400
    assert database.enrich_training_progress_metrics(recorded["id"], {"counter": value}) is True
    assert read_sample(database, recorded)["evidence_json"]["metrics"] == {"counter": value}


@pytest.mark.parametrize("keys", [("train/loss", "validation/loss"), ("train/loss", "train/loss")])
def test_concurrent_enrichment_preserves_metrics_across_database_instances(sample, tmp_path, keys):
    database, recorded = sample
    second = Database(tmp_path / "metrics.db")
    start = threading.Barrier(2)

    def enrich(instance, key):
        start.wait(timeout=3)
        return instance.enrich_training_progress_metrics(recorded["id"], {key: 0.5})

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(enrich, database, keys[0])
        other = pool.submit(enrich, second, keys[1])
        results = [first.result(timeout=5), other.result(timeout=5)]
    assert sum(results) == len(set(keys))
    expected = {**recorded, "evidence_json": {**recorded["evidence_json"], "metrics": dict.fromkeys(keys, 0.5)}}
    assert read_sample(database, recorded) == expected
