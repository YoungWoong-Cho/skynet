from __future__ import annotations

from datetime import datetime, timezone

from skynet_app.pipeline_api import (
    evaluation_progress_summary,
    parse_declared_training_progress,
    training_progress_summary,
)
from skynet_app.adapters import TrainingProgressContract, TrainingProgressLogSource


NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def test_queued_evaluation_does_not_count_submission_time_as_execution():
    summary = evaluation_progress_summary(
        {"status": "PENDING", "started_at": "2026-01-01T10:00:00Z"},
        attempts=[{"attempt_number": 1, "status": "PENDING", "started_at": None}],
        now=NOW,
    )
    assert summary["elapsed_seconds"] is None


def test_training_eta_requires_recorded_progress_not_slurm_wall_time():
    summary = training_progress_summary(
        {"status": "RUNNING", "resolved_spec_json": {"train": {"max_steps": 100}}},
        attempts=[{
            "id": "attempt-1",
            "attempt_number": 1,
            "status": "RUNNING",
            "started_at": "2026-01-01T11:00:00Z",
            "time_limit_seconds": 60,
        }],
        now=NOW,
    )

    assert summary == {
        "completed": None,
        "total": 100,
        "unit": "step",
        "fraction": None,
        "observed_at": None,
        "elapsed_seconds": 3600,
        "eta_seconds": None,
        "eta_state": "waiting",
        "eta_reason": "waiting_for_progress",
    }


def test_training_eta_uses_resume_baseline_and_current_attempt_step():
    summary = training_progress_summary(
        {"status": "RUNNING", "resolved_spec_json": {"train": {"max_steps": 100}}},
        attempts=[{
            "id": "attempt-2",
            "attempt_number": 2,
            "status": "RUNNING",
            "resume_checkpoint_id": "checkpoint-40",
            "started_at": "2026-01-01T11:00:00Z",
        }],
        checkpoints=[{
            "id": "checkpoint-40",
            "produced_by_attempt_id": "attempt-1",
            "training_step": 40,
            "created_at": "2026-01-01T10:30:00Z",
        }],
        metrics=[{
            "name": "training_step",
            "scope": "TRAIN",
            "value": 70,
            "recorded_at": "2026-01-01T12:00:00Z",
        }],
        now=NOW,
    )

    assert summary["completed"] == 70
    assert summary["total"] == 100
    assert summary["fraction"] == 0.7
    assert summary["eta_seconds"] == 3600
    assert summary["eta_state"] == "estimating"
    assert summary["eta_reason"] == "current_attempt_progress_rate"


def test_training_retry_reset_does_not_reuse_prior_attempt_progress_rate():
    summary = training_progress_summary(
        {"status": "RUNNING", "resolved_spec_json": {"train": {"max_steps": 100}}},
        attempts=[
            {"id": "attempt-1", "attempt_number": 1, "status": "FAILED"},
            {
                "id": "attempt-2",
                "attempt_number": 2,
                "status": "RUNNING",
                "started_at": "2026-01-01T11:00:00Z",
            },
        ],
        checkpoints=[
            {
                "id": "old",
                "produced_by_attempt_id": "attempt-1",
                "training_step": 80,
                "created_at": "2026-01-01T10:00:00Z",
            },
            {
                "id": "new",
                "produced_by_attempt_id": "attempt-2",
                "training_step": 10,
                "created_at": "2026-01-01T11:30:00Z",
            },
        ],
        now=NOW,
    )

    assert summary["completed"] == 10
    assert summary["eta_seconds"] is None
    assert summary["eta_state"] == "unknown"
    assert summary["eta_reason"] == "insufficient_progress_samples"


def test_evaluation_eta_uses_only_current_attempt_completed_work():
    previous = [
        {
            "status": "SUCCEEDED",
            "completed_at": "2026-01-01T10:00:00Z",
        }
        for _ in range(40)
    ]
    current = [
        {
            "status": "SUCCEEDED",
            "completed_at": "2026-01-01T11:30:00Z",
        }
        for _ in range(10)
    ]
    summary = evaluation_progress_summary(
        {"status": "RUNNING", "progress_completed": 50, "progress_total": 100},
        attempts=[{
            "id": "attempt-2",
            "attempt_number": 2,
            "status": "RUNNING",
            "started_at": "2026-01-01T11:00:00Z",
        }],
        episodes=previous + current,
        now=NOW,
    )

    assert summary["completed"] == 50
    assert summary["fraction"] == 0.5
    assert summary["elapsed_seconds"] == 3600
    assert summary["eta_seconds"] == 18000
    assert summary["eta_state"] == "estimating"


def test_terminal_progress_states_do_not_project_remaining_time():
    failed = training_progress_summary(
        {"status": "FAILED", "resolved_spec_json": {"train": {"max_steps": 100}}},
        now=NOW,
    )
    completed = evaluation_progress_summary(
        {"status": "SUCCEEDED", "progress_completed": 8, "progress_total": 10},
        now=NOW,
    )

    assert failed["eta_seconds"] is None
    assert failed["eta_state"] == "not_applicable"
    assert completed["eta_seconds"] == 0
    assert completed["eta_state"] == "complete"


def test_adapter_declared_log_progress_normalizes_si_steps_and_elapsed_time():
    contract = TrainingProgressContract(
        source=TrainingProgressLogSource(
            pattern=(
                r"Progress on:\s*(?P<completed>[0-9.]+[kMGT]?)it/"
                r"(?P<total>[0-9.]+[kMGT]?)it.*elapsed:"
                r"(?P<elapsed>[0-9]+:[0-9]{2}:[0-9]{2})"
            ),
            value_format="decimal_si",
            elapsed_format="hms",
        )
    )

    records = parse_declared_training_progress(
        "\n".join([
            "18:00:00 [I] Progress on: 2.60kit/10.0kit rate:1.8s/it elapsed:1:27:12",
            "18:00:11 [I] Progress on: 2.61kit/10.0kit rate:1.8s/it elapsed:1:27:23",
        ]),
        contract,
    )

    assert records == [
        {"completed": 2600, "total": 10000, "elapsed_seconds": 5232},
        {"completed": 2610, "total": 10000, "elapsed_seconds": 5243},
    ]


def test_attempt_linked_progress_samples_exclude_previous_retry():
    summary = training_progress_summary(
        {"status": "RUNNING", "resolved_spec_json": {"train": {"max_steps": 100}}},
        attempts=[
            {"id": "attempt-1", "attempt_number": 1, "status": "FAILED"},
            {
                "id": "attempt-2",
                "attempt_number": 2,
                "status": "RUNNING",
                "started_at": "2026-01-01T11:00:00Z",
            },
        ],
        progress_samples=[
            {
                "attempt_id": "attempt-1",
                "completed": 90,
                "recorded_at": "2026-01-01T11:10:00Z",
            },
            {
                "attempt_id": "attempt-2",
                "completed": 10,
                "recorded_at": "2026-01-01T11:20:00Z",
            },
            {
                "attempt_id": "attempt-2",
                "completed": 20,
                "recorded_at": "2026-01-01T12:00:00Z",
            },
        ],
        now=NOW,
    )

    assert summary["completed"] == 20
    assert summary["eta_state"] == "estimating"


def test_openpi_progress_accepts_short_and_long_elapsed_clocks():
    from skynet_app.adapters import builtin_adapter_manifests

    contract = next(item for item in builtin_adapter_manifests() if item.slug == "openpi").train.progress
    records = parse_declared_training_progress("\n".join([
        "00:54:40 [I] Progress on: -/1 rate:- remaining:? elapsed:00:00 postfix:-",
        "00:56:24 [I] Progress on: 1.00it/1.00it rate:103.5s/it remaining:00:00 elapsed:01:43 postfix:-",
        "00:56:24 [I] Progress on: 1.00it/1.00it rate:103.5s/it remaining:00:00 elapsed:01:43 postfix:-",
        "18:00:11 [I] Progress on: 2.61kit/10.0kit rate:1.8s/it elapsed:1:27:23",
        "18:00:11 [I] Progress on: 2.61kit/10.0kit rate:1.8s/it elapsed:99:99",
    ]), contract)
    assert records == [
        {"completed": 1, "total": 1, "elapsed_seconds": 103},
        {"completed": 2610, "total": 10000, "elapsed_seconds": 5243},
    ]
