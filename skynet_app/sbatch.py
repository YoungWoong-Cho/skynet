"""The sbatch header and shell prelude every generated job script starts with.

Each job family keeps its own body (environment, capsule checks, workload);
the scheduler-facing part is assembled here so a directive is spelled once.
"""
from __future__ import annotations

from collections.abc import Iterable

from .cluster_config import QueueProfile

SHEBANG = "#!/usr/bin/env bash"


def sbatch_header(
    *,
    job_name: str,
    queue: QueueProfile,
    cpus: int,
    memory_gb: int,
    time_limit: str,
    output: str,
    error: str | None = None,
    gres: str | None = None,
    single_task: bool = False,
    chdir: str | None = None,
    extra: Iterable[str] = (),
) -> list[str]:
    """The ``#SBATCH`` lines of a job: name, queue, shape, time, paths, then ``extra``."""
    lines = [f"#SBATCH --job-name={job_name}", *queue.sbatch_directives()]
    if single_task:
        lines += ["#SBATCH --nodes=1", "#SBATCH --ntasks=1"]
    lines += [f"#SBATCH --cpus-per-task={cpus}", f"#SBATCH --mem={memory_gb}G"]
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    lines.append(f"#SBATCH --time={time_limit}")
    if chdir:
        lines.append(f"#SBATCH --chdir={chdir}")
    lines += [f"#SBATCH --output={output}", f"#SBATCH --error={error or output}", *extra]
    return lines


def shell_prelude(*, umask: str | None = "077", errtrace: bool = False) -> list[str]:
    """Fail-fast shell options; ``errtrace`` keeps traps active inside functions."""
    lines = ["set -Eeuo pipefail" if errtrace else "set -euo pipefail"]
    if umask:
        lines.append(f"umask {umask}")
    return lines


def cpu_thread_exports(cpus: int) -> str:
    """Pin the numeric libraries of a CPU-only job to its allocated cores."""
    return f"export OMP_NUM_THREADS={cpus} OPENBLAS_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus}"
