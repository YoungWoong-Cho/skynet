"""User-facing submission evidence, separate from scheduler control state."""

from collections.abc import Mapping, Sequence
from typing import Any

SUBMISSION_UNKNOWN_PREFIX = "Submission outcome unknown"
SUBMISSION_UNCONFIRMED = "SUBMISSION UNCONFIRMED"
SUBMISSION_UNCONFIRMED_DETAIL = (
    "Connection failed before Slurm acceptance could be confirmed. "
    "Recover the existing submission or cancel it."
)


def submission_unconfirmed(attempt: Mapping[str, Any]) -> bool:
    return (
        attempt.get("status") == "SUBMITTING"
        and not attempt.get("slurm_job_id")
        and str(attempt.get("slurm_reason") or "").startswith(SUBMISSION_UNKNOWN_PREFIX)
    )


def attach_attempt_display_status(attempt: dict[str, Any]) -> None:
    unknown = submission_unconfirmed(attempt)
    attempt["display_status"] = SUBMISSION_UNCONFIRMED if unknown else attempt.get("status")
    attempt["status_detail"] = SUBMISSION_UNCONFIRMED_DETAIL if unknown else None


def attach_job_display_status(job: dict[str, Any], attempts: Sequence[dict[str, Any]]) -> None:
    """Annotate a response without changing retry/cancellation eligibility."""
    for attempt in attempts:
        attach_attempt_display_status(attempt)
    latest = max(attempts, key=lambda item: (item.get("attempt_number") or 0, item.get("created_at") or ""), default={})
    unknown = job.get("status") == "SUBMITTING" and submission_unconfirmed(latest)
    job["display_status"] = SUBMISSION_UNCONFIRMED if unknown else job.get("status")
    job["status_detail"] = SUBMISSION_UNCONFIRMED_DETAIL if unknown else None
