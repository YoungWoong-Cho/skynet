"""Isaac job template for simulator replay.

Submission, recovery and scheduler reads remain in ClusterClient.
"""

import re
import shlex

from skynet_app.cluster_runtime import validate_remote_path
from skynet_app.cluster_config import CLUSTER
from .gpu_preflight import gpu_preflight_lines
from .sbatch import SHEBANG, sbatch_header, shell_prelude


def compile_isaac_job(profile, root, name, argv, checks=(), after=(), *, resources=None):
    """One single-GPU Isaac job; ``resources`` defaults to the configured replay shape."""
    resources = resources or CLUSTER.defaults.background_jobs.isaac_replay
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("Invalid job name")
    for path in (root, profile["runtime"], profile["repository"]):
        validate_remote_path(path)
        if any(c.isspace() for c in path):
            raise ValueError("Isaac job paths cannot contain whitespace")
    # The partition must be a configured queue (which supplies the account) and
    # the GPU type a configured alias; both raise ValueError otherwise.
    gres = CLUSTER.gres(profile.get("gpu_type", CLUSTER.defaults.gpu_type), 1)
    return "\n".join(
        [
            SHEBANG,
            *sbatch_header(
                job_name=name, queue=CLUSTER.queue_for_partition(profile["partition"]),
                cpus=CLUSTER.defaults.cpus_per_gpu, memory_gb=resources.memory_gb, time_limit=resources.time_limit,
                output=f"{root}/stdout.log", error=f"{root}/stderr.log", gres=gres,
            ),
            *shell_prelude(umask="077"),
            *gpu_preflight_lines(1),
            *isaac_environment(profile, root),
            *checks,
            shlex.join(argv),
            *after,
            "",
        ]
    )


def isaac_environment(profile, root):
    """Shared bootstrap for pinned simulator jobs and observation workers."""
    runtime, repo = profile["runtime"], profile["repository"]
    for path in (root, runtime, repo):
        validate_remote_path(path)
    return [
            f"export TMPDIR={shlex.quote(root + '/tmp')}",
            'mkdir -p "$TMPDIR"',
            "export OMNI_KIT_ACCEPT_EULA=YES",
            "export PYTHONUNBUFFERED=1",
            f"export LD_LIBRARY_PATH={shlex.quote(runtime + '/lib')}:${{LD_LIBRARY_PATH:-}}",
            f"export PYTHONPATH={shlex.quote(repo + '/source/dexverse')}:${{PYTHONPATH:-}}",
            f"export PATH={shlex.quote(runtime + '/bin')}:$PATH",
            f"cd {shlex.quote(repo)}",
    ]
