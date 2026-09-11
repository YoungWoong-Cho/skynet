# Training metrics

Skynet forwards every finite scalar in a declared JSONL training record. Native
top-level names are retained, nested `metrics` objects become slash-separated
names, and declared aliases remain available for existing dashboards. Booleans,
strings, arrays, private fields, NaN and infinity are not scalar chart data.
Integers remain integers in W&B, including optimizer step counters.

This shared path covers the native EgoVerse ACT, HPT, and PI adapters and XPolicyLab ACT/DP.
EgoVerse records losses, validation metrics, learning rates, gradient norms and
timings at epoch end. XPolicyLab's additional fields, such as `best_val_loss`,
are forwarded too. No per-model metric allowlist is required. Validation values
only refresh when validation executes; the logger can retain the last value
between validation passes.

OpenPI's declared native W&B integration continues to own its history stream.
Skynet avoids writing a second copy of those metrics. Custom and legacy adapters
without a declared metric source need a source or native tracking integration;
the server does not guess metric names from arbitrary console output. JSONL
source declarations use the same forwarding behavior regardless of adapter name.

## Internal progress counters

The following fields support Skynet's progress display, ETA and restart handling.
They are not forwarded to W&B or MLflow as training curves. Previously uploaded
history is retained; its old panels can be removed from the W&B workspace.

| Metric | Meaning |
| --- | --- |
| `training/completed` | Completed epochs or steps, according to the adapter's progress unit. |
| `training/total` | Configured target in the same unit. |
| `training/progress` | Completed divided by total, between 0 and 1 for valid progress. For example, 0.02 means 2%. |
| `training/restart_count` | Scheduler restarts of the same Slurm job/attempt, read from Slurm accounting. A new Skynet retry attempt is separate. |
| `training/elapsed_seconds` | Elapsed time reported by a declared progress source, when provided. |

Restart counts are refreshed before collecting progress. An unavailable scheduler
value does not reset a previously observed count. Reserved counter names are also
excluded from producer metrics so they cannot reappear through a name collision.

## Existing runs and delivery

On the next log read, existing progress samples acquire previously omitted metric
fields without changing their identity, observation time or original values.
Only new fields are appended to tracking history at the sample's original step
and timestamp. Durable spool records suppress duplicates across polls, offline
delivery and server restarts. Already published charts are retained.

The Skynet backend must be running to read cluster logs and upload central
metrics; the browser may be closed. Live reads are bounded by the source's
`tail_lines` and a one-megabyte response limit. Enrichment can recover retained
records within that window, but does not promise a full historical import of
older records outside it. Terminal logs are read once per server session.
