"""Conservative scheduler states for observation and dataset preparation.

Slurm state flags can replace the base state in sacct output. Requeue/hold,
preemption and unknown states must retain an existing submission identity.
See https://slurm.schedmd.com/job_state_codes.html.
"""

TERMINAL_FAILURE_STATES = frozenset({
    'BOOT_FAIL', 'CANCELLED', 'DEADLINE', 'FAILED', 'NODE_FAIL',
    'OUT_OF_MEMORY', 'TIMEOUT',
})
# After these an attempt with automatic resume and remaining budget is resubmitted.
TRANSIENT_STATES = frozenset({'PREEMPTED', 'TIMEOUT', 'NODE_FAIL', 'BOOT_FAIL', 'REVOKED'})
RUNNING_STATES = frozenset({'RUNNING', 'COMPLETING', 'RESIZING', 'STAGE_OUT'})
WAITING_STATES = frozenset({
    'PENDING', 'CONFIGURING', 'SUSPENDED', 'STOPPED', 'POWER_UP_NODE',
    'EXPEDITING', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD', 'RESV_DEL_HOLD',
    'SPECIAL_EXIT', 'PREEMPTED', 'REVOKED',
})


def observed_state(state, previous):
    if state in RUNNING_STATES:
        return 'RUNNING'
    if state in WAITING_STATES:
        return 'PENDING'
    return previous if previous in {'PENDING', 'RUNNING'} else 'PENDING'
