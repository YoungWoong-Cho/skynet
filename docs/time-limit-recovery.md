# Time-limit warnings and checkpoint recovery

Training allocations with automatic resume request a Slurm warning before their time limit. The batch shell catches that warning and leaves an attempt-scoped marker. The runtime wrapper requests normal termination of the trainer process group, waits for it, and records the latest available checkpoint as resumable. It never forwards raw USR1 to the trainer, and an interrupted trainer returning zero is not considered a completed training run.

The wrapper exits with code 124 and an interruption receipt bound to the run and Slurm job ID. Reconciliation recognizes a warning-triggered timeout only when both the exit code and receipt match. Ordinary failures and cancellations retain their original behavior. Automatic retries still require automatic resume and remaining attempt budget. The actual Slurm state and exit code remain recorded.

Trainers that handle TERM may save additional state during shutdown. Otherwise recovery uses their last completed periodic checkpoint; the wrapper cannot manufacture a checkpoint for a trainer that never saved one. A nonresponsive trainer remains subject to Slurm’s hard time limit.

Changes apply to newly submitted attempts. Existing submitted scripts are immutable; an already failed attempt is not silently restarted or relabeled.

## GPU allocation preflight

The controller occasionally starts a job that requested `--gres=gpu` without allocating one. Every generated GPU job script (training and evaluation attempts, collection sessions, Isaac and live XR jobs, observation rendering, the evaluator readiness smoke) runs the shared preflight from `skynet_app/gpu_preflight.py` before any work: when fewer GPUs are visible through `CUDA_VISIBLE_DEVICES`/`SLURM_JOB_GPUS`/`SLURM_STEP_GPUS` than the script requested, it exits with the dedicated code and, for attempts, leaves an interruption receipt with reason `gpu_not_allocated` next to the time-limit receipt. CPU-only scripts carry no preflight.

Reconciliation treats a matching receipt as the transient state `BOOT_FAIL` with the reason "The cluster started the job without the requested GPU", so the same automatic re-attempt rule applies: automatic resume on, attempt budget remaining, no cancellation requested. Unlike the time-limit code, an exit record carrying this code also qualifies, because the receipt is unambiguous. The other job families have no attempt budget; their status shows the same reason so the existing restart or re-request actions apply.

Verification:

```sh
python -m pytest tests/test_slurm_signals.py tests/test_time_limit_reconcile.py tests/test_slurm.py tests/test_gpu_preflight.py
```

These tests send real USR1 signals to a batch shell, exercise trainers with and without termination handlers, repeat the warning, resume from a saved checkpoint, and verify cancellation, ordinary failure and retry-budget handling without submitting cluster jobs.
