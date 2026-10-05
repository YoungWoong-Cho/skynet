"""A resumed counter must not be credited as work done since the new attempt began."""

import json

import pytest

from skynet_app.pipeline_api import training_progress_summary


START = "2026-09-20T10:33:22Z"
FIRST = "2026-09-20T10:38:22Z"
SECOND = "2026-09-20T10:43:22Z"


def run_spec():
    return {
        "status": "RUNNING",
        "resolved_spec_json": {
            "train": {"max_steps": 8000},
            "source": {"adapter_manifest": {"train": {"progress": {
                "schema_version": "skynet.training-progress-source/v1",
                "unit": "step",
                "starts_at_zero": True,
                "total_path": "train.max_steps",
                "source": {"kind": "jsonl", "path": "artifacts/logs.json.txt",
                           "completed_key": "global_step", "required_key": "train_loss"},
            }}}},
        },
    }


def attempt(number=2, **extra):
    return {
        "id": "current", "attempt_number": number, "status": "RUNNING",
        "started_at": START, "resume_checkpoint_id": None, "restart_count": 0,
        **extra,
    }


def sample(step, when, **extra):
    return {"attempt_id": "current", "completed": step, "unit": "step",
            "recorded_at": when, "restart_count": 0, **extra}


@pytest.mark.parametrize("include_previous", [False, True])
def test_auto_resume_null_checkpoint_id_waits_for_two_current_attempt_samples(include_previous):
    # Actual k1 topology: attempt2, scheduler restart_count0, no checkpoint row
    # in the API, although the remote wrapper restored its 4000-step checkpoint.
    attempts = ([{"id": "old", "attempt_number": 1, "status": "TIMEOUT"}]
                if include_previous else []) + [attempt()]
    result = training_progress_summary(run_spec(), attempts=attempts,
        progress_samples=[sample(4100, FIRST)], now=FIRST)
    assert result["completed"] == 4100
    assert result["eta_seconds"] is None
    assert result["eta_reason"] == "insufficient_progress_samples"


def test_auto_resume_rate_uses_counter_difference_and_ignores_old_attempt_rate():
    result = training_progress_summary(run_spec(),
        attempts=[{"id": "old", "attempt_number": 1, "status": "TIMEOUT"}, attempt()],
        progress_samples=[sample(4671, FIRST, attempt_id="old"),
                          sample(4100, FIRST), sample(4200, SECOND)], now=SECOND)
    assert result["completed"] == 4200
    assert result["eta_state"] == "estimating"
    assert result["eta_seconds"] == 11400  # 100 updates / 300s; 3800 updates remain.


@pytest.mark.parametrize("location", ["plan", "snapshot_spec", "provenance", "spec"])
@pytest.mark.parametrize("serialized", [False, True])
def test_first_visible_attempt_with_pinned_checkpoint_never_assumes_zero(location, serialized):
    run, current = run_spec(), attempt(number=1)
    snapshot = {}
    if location == "plan":
        snapshot = {"plan": {"native_config": {"initial_checkpoint": "/saved.ckpt"}}}
    elif location == "snapshot_spec":
        snapshot = {"resolved_spec": {"native": {"config": {"initial_checkpoint": "/saved.ckpt"}}}}
    elif location == "provenance":
        snapshot = {"migration_provenance": {"policy": "strict_pinned_resume",
                                             "checkpoint": {"path": "/saved.ckpt"}}}
    else:
        run["resolved_spec_json"]["native"] = {"config": {"initial_checkpoint": "/saved.ckpt"}}
    current["execution_snapshot_json"] = json.dumps(snapshot) if serialized else snapshot
    if serialized:
        run["resolved_spec_json"] = json.dumps(run["resolved_spec_json"])
    result = training_progress_summary(run, attempts=[current],
        progress_samples=[sample(4100, FIRST)], now=FIRST)
    assert result["eta_seconds"] is None
    assert result["completed"] == 4100


def test_first_fresh_attempt_retains_adapter_zero_baseline():
    result = training_progress_summary(run_spec(), attempts=[attempt(number=1)],
        progress_samples=[sample(100, FIRST)], now=FIRST)
    assert result["completed"] == 100
    assert result["eta_seconds"] == 23700  # 100 / 300s, 7900 remaining.


def test_known_checkpoint_step_keeps_its_explicit_resume_baseline():
    result = training_progress_summary(run_spec(),
        attempts=[attempt(resume_checkpoint_id="cp")],
        checkpoints=[{"id": "cp", "training_step": 4000,
                      "produced_by_attempt_id": "old", "created_at": "2026-09-20T10:30:00Z"}],
        progress_samples=[sample(4100, FIRST)], now=FIRST)
    assert result["eta_seconds"] == 11700  # 100 / 300s, 3900 remaining.


def test_unnumbered_prior_attempt_still_prevents_zero_baseline():
    result = training_progress_summary(run_spec(),
        attempts=[{"id": "old", "status": "FAILED"}, attempt(number=1)],
        progress_samples=[sample(4100, FIRST)], now=FIRST)
    assert result["eta_seconds"] is None


@pytest.mark.parametrize("first_batch", [[4001, 4200, 4600], [4600, 4200, 4001]])
def test_backfilled_poll_is_one_timing_observation_at_its_highest_step(first_batch):
    later = "2026-09-20T10:40:52Z"  # 150 seconds after FIRST.
    result = training_progress_summary(run_spec(), attempts=[attempt()],
        progress_samples=[*(sample(step, FIRST) for step in first_batch),
                          sample(4650, later)], now=later)
    assert result["completed"] == 4650
    assert result["eta_seconds"] == 10050  # 3350 remaining / (50 new steps /150s).


def test_backfill_without_a_later_poll_does_not_have_a_measured_rate():
    result = training_progress_summary(run_spec(), attempts=[attempt()],
        progress_samples=[sample(step, FIRST) for step in (4001, 4200, 4600)], now=SECOND)
    assert result["completed"] == 4600
    assert result["eta_seconds"] is None
    assert result["eta_reason"] == "insufficient_progress_samples"


def test_fresh_zero_baseline_still_measures_the_first_batched_observation():
    result = training_progress_summary(run_spec(), attempts=[attempt(number=1)],
        progress_samples=[sample(step, FIRST) for step in (1, 50, 100)], now=FIRST)
    assert result["eta_seconds"] == 23700


def test_counter_reset_between_polls_restarts_the_rate_anchor():
    third = "2026-09-20T10:48:22Z"
    result = training_progress_summary(run_spec(), attempts=[attempt()],
        progress_samples=[sample(4600, FIRST), sample(4001, SECOND),
                          sample(4100, SECOND), sample(4200, third)], now=third)
    assert result["completed"] == 4200
    assert result["eta_seconds"] == 11400  # reset batch is one observation at4100.
