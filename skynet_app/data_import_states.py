"""Lifecycle states of a data import row, shared by the service, the database and the UI.

SUBMITTING is the console handing the job to Slurm; the scheduled states follow
the Slurm job; FINALIZING publishes a completed job's result and CANCELLING awaits
a requested cancel; the terminal states never change again. This module stays free
of application imports so ``database`` can use it (``data_imports`` itself reaches
``database`` through the job-script helpers).
"""

DATA_IMPORT_TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})
# The import has a Slurm job that accounting may still describe; a cancel can be
# requested from any of these.
DATA_IMPORT_SCHEDULED_STATES = frozenset({"SUBMITTED", "PENDING", "RUNNING"})
# A terminal transition is in progress and may outlast what Slurm reports.
DATA_IMPORT_SETTLING_STATES = frozenset({"FINALIZING", "CANCELLING"})
# States a scheduler observation may write; a lagging observation never undoes settling.
DATA_IMPORT_OBSERVED_STATES = frozenset({"SUBMITTING"}) | DATA_IMPORT_SCHEDULED_STATES
# Imports with a Slurm job that reconcile still has to settle.
DATA_IMPORT_IN_FLIGHT_STATES = DATA_IMPORT_SCHEDULED_STATES | DATA_IMPORT_SETTLING_STATES
# Every state before a terminal one: while an import is here, an identical request is a duplicate.
DATA_IMPORT_ACTIVE_STATES = DATA_IMPORT_OBSERVED_STATES | DATA_IMPORT_SETTLING_STATES
