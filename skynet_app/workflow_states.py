"""Workflow stage and attempt states shared by the pipeline service and the repository.

These are Skynet's own lifecycle states (a stage is CREATED before it is
SUBMITTING, PENDING_SLURM while the scheduler queues it, RETRY_PENDING between
automatic attempts); scheduler-reported job states live in preparation_states.
"""

# A stage whose work can still be stopped by a cancellation intent.
CANCELLABLE_STAGE_STATES = frozenset({
    "CREATED", "PENDING", "RETRY_PENDING", "SUBMITTING", "SUBMITTED",
    "PENDING_SLURM", "RUNNING", "REQUEUED",
})
# A stage that holds or is acquiring a scheduler job (the invariant repair's scope).
ACTIVE_STAGE_STATES = frozenset({"SUBMITTING", "SUBMITTED", "PENDING_SLURM", "RUNNING", "CANCELLING"})
# An attempt the scheduler still knows by its job id.
SLURM_BOUND_ATTEMPT_STATES = frozenset({"SUBMITTED", "PENDING", "RUNNING", "REQUEUED", "CANCELLING"})


def sql_list(states) -> str:
    """The states as a SQL IN-list literal, in a stable order."""
    return ", ".join(f"'{state}'" for state in sorted(states))
