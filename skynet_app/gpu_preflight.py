"""GPU allocation preflight shared by every generated GPU job script.

The cluster controller occasionally starts a job that requested ``--gres=gpu``
without allocating one (the batch step then sees no ``SLURM_JOB_GPUS`` and no
``CUDA_VISIBLE_DEVICES``). Instead of letting the workload discover that minutes
later, every GPU job script runs this preflight first and exits with a dedicated
code; the services that watch those jobs recognise the code and report (or, for
training and evaluation attempts, retry) a cluster-side allocation failure
instead of a workload failure.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Outside sysexits (64-78), Slurm's own 124 time-limit convention and the
# 125+ shell range; the receipt below is what reconcile keys on, not the code.
GPU_MISSING_EXIT_CODE = 97
GPU_MISSING_REASON = "gpu_not_allocated"
GPU_MISSING_MESSAGE = "The cluster started the job without the requested GPU"
# Slurm's own name for a job that could not launch on its allocation; it is already
# a transient state, so reconcile's bounded re-attempt applies without a new status.
GPU_MISSING_STATE = "BOOT_FAIL"

# The same precedence as gpu_metrics.allocated_devices: Slurm's GRES plugin
# exports these for the batch step even under --export=NIL/NONE.
_VISIBLE_GPU_VARIABLES = ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS")


def gpu_preflight_lines(gpu_count: int, *, receipt_dir: str | None = None) -> list[str]:
    """Shell lines that stop a GPU job before any work when no GPU is visible.

    ``gpu_count`` is the count the script requests from Slurm (the same value
    given to ``CLUSTER.gres``); CPU-only scripts get no preflight. ``receipt_dir``
    is a shell expression for the directory of the attempt's interruption
    receipt (training and evaluation attempts), which reconcile reads to tell
    this exit from a workload failure. The lines assume ``set -e``.
    """
    if gpu_count <= 0:
        return []
    probe = " ".join(f'"${{{name}:-}}"' for name in _VISIBLE_GPU_VARIABLES)
    receipt = []
    if receipt_dir:
        receipt = [
            f'  mkdir -p {receipt_dir}',
            '  printf \'{"schema_version": 1, "job_id": "%s", "run_id": "%s", "reason": "%s", "exit_code": %d}\\n\' '
            f'"${{SLURM_JOB_ID:-}}" "${{SKYNET_RUN_ID:-}}" {GPU_MISSING_REASON} {GPU_MISSING_EXIT_CODE} '
            f'> {receipt_dir}/interruption.json',
        ]
    return [
        "# GPU preflight: the controller sometimes starts a GPU job without allocating one.",
        "skynet_visible_gpus=''",
        f"for skynet_gpu_list in {probe}; do",
        '  if [ -n "$skynet_gpu_list" ]; then skynet_visible_gpus="$skynet_gpu_list"; break; fi',
        "done",
        'skynet_visible_gpu_count=$(printf "%s" "$skynet_visible_gpus" | tr "," "\\n" | grep -c . || true)',
        f'if [ "$skynet_visible_gpu_count" -lt {gpu_count} ]; then',
        f'  echo "skynet: requested {gpu_count} GPU(s) but $skynet_visible_gpu_count visible '
        f'({GPU_MISSING_REASON}); exiting {GPU_MISSING_EXIT_CODE}" >&2',
        *receipt,
        f"  exit {GPU_MISSING_EXIT_CODE}",
        "fi",
    ]


def gpu_missing_exit(record: Mapping[str, Any] | None) -> bool:
    """Whether a Slurm status record ended with the preflight's exit code."""
    exit_code = str((record or {}).get("ExitCode") or "")
    return exit_code.split(":", 1)[0] == str(GPU_MISSING_EXIT_CODE)


def gpu_missing_receipt(receipt: Any) -> bool:
    """Whether an attempt's interruption receipt records the GPU preflight exit."""
    return (
        isinstance(receipt, Mapping)
        and receipt.get("reason") == GPU_MISSING_REASON
        and receipt.get("exit_code") == GPU_MISSING_EXIT_CODE
    )
