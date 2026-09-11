import pytest

from skynet_app.job_status import attach_job_display_status, submission_unconfirmed


@pytest.mark.parametrize("state,job_id,reason,unknown", [
    ("SUBMITTING", None, "Submission outcome unknown: SSH operation timed out", True),
    ("SUBMITTING", None, None, False),
    ("SUBMITTING", "123", "Submission outcome unknown: old error", False),
    ("FAILED", None, "Submission outcome unknown: old error", False),
    ("CANCELLING", None, "Submission outcome unknown: old error", False),
])
def test_submission_evidence_preserves_control_state(state, job_id, reason, unknown):
    attempt = {"status": state, "slurm_job_id": job_id, "slurm_reason": reason}
    job = {"status": state}
    attach_job_display_status(job, [attempt])
    assert submission_unconfirmed(attempt) == unknown
    assert job["status"] == attempt["status"] == state
    assert job["display_status"] == attempt["display_status"] == ("SUBMISSION UNCONFIRMED" if unknown else state)
    assert bool(job["status_detail"]) == unknown


def test_latest_attempt_controls_warning_not_an_older_attempt():
    old = {"status": "SUBMITTING", "attempt_number": 1, "slurm_reason": "Submission outcome unknown"}
    latest = {"status": "SUBMITTING", "attempt_number": 2}
    job = {"status": "SUBMITTING"}
    attach_job_display_status(job, [latest, old])
    assert job["display_status"] == "SUBMITTING"
    assert old["display_status"] == "SUBMISSION UNCONFIRMED"
