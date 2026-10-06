"""Isaac job template for simulator replay.

Submission, recovery and scheduler reads remain in ClusterClient.
"""

import re
import shlex

from skynet_app.cluster_runtime import validate_remote_path
from skynet_app.cluster_config import CLUSTER


def compile_isaac_job(profile, root, name, argv, checks=(), after=()):
    for key in ("account", "partition"):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", profile[key]):
            raise ValueError(f"Invalid {key}")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("Invalid job name")
    for path in (root, profile["runtime"], profile["repository"]):
        validate_remote_path(path)
        if any(c.isspace() for c in path):
            raise ValueError("Isaac job paths cannot contain whitespace")
    runtime, repo = profile["runtime"], profile["repository"]
    gres = CLUSTER.gres(profile.get("gpu_type", CLUSTER.defaults.gpu_type), 1)
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name={name}",
            *CLUSTER.queue_for_partition(profile["partition"]).sbatch_directives(),
            f"#SBATCH --gres={gres}",
            f"#SBATCH --cpus-per-task={CLUSTER.defaults.cpus_per_gpu}",
            "#SBATCH --mem=48G",
            "#SBATCH --time=00:30:00",
            f"#SBATCH --output={root}/stdout.log",
            f"#SBATCH --error={root}/stderr.log",
            "set -euo pipefail",
            "umask 077",
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
