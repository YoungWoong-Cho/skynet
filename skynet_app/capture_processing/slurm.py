"""Shared Isaac job template extracted from the existing capture-cycle launcher.

Submission, recovery and scheduler reads remain in ClusterClient.
"""

import re
import shlex

from skynet_app.cluster_runtime import ClusterClient
from skynet_app.cluster_config import CLUSTER


def compile_isaac_job(profile, root, name, argv, checks=(), after=()):
    for key in ("account", "partition"):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", profile[key]):
            raise ValueError(f"Invalid {key}")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("Invalid job name")
    for path in (root, profile["runtime"], profile["repository"]):
        ClusterClient._remote_path(path)
        if any(c.isspace() for c in path):
            raise ValueError("Isaac job paths cannot contain whitespace")
    runtime, repo = profile["runtime"], profile["repository"]
    gpu_type = profile.get("gpu_type", "any")
    if gpu_type not in CLUSTER.gpu_aliases:
        raise ValueError(f"GPU type is not configured: {gpu_type}")
    gpu_alias = CLUSTER.gpu_aliases[gpu_type]
    gres = "gpu:1" if gpu_alias is None else f"gpu:{gpu_alias}:1"
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name={name}",
            f"#SBATCH --account={profile['account']}",
            f"#SBATCH --partition={profile['partition']}",
            f"#SBATCH --gres={gres}",
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --mem=48G",
            "#SBATCH --time=00:30:00",
            f"#SBATCH --output={root}/stdout.log",
            f"#SBATCH --error={root}/stderr.log",
            "set -euo pipefail",
            "umask 077",
            f"export TMPDIR={shlex.quote(root + '/tmp')}",
            'mkdir -p "$TMPDIR"',
            "export OMNI_KIT_ACCEPT_EULA=YES",
            "export PYTHONUNBUFFERED=1",
            f"export LD_LIBRARY_PATH={shlex.quote(runtime + '/lib')}:${{LD_LIBRARY_PATH:-}}",
            f"export PYTHONPATH={shlex.quote(repo + '/source/dexverse')}:${{PYTHONPATH:-}}",
            f"export PATH={shlex.quote(runtime + '/bin')}:$PATH",
            f"cd {shlex.quote(repo)}",
            *checks,
            shlex.join(argv),
            *after,
            "",
        ]
    )
