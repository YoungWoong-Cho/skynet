"""Every SSH connection takes its timeouts, keepalives and operation budgets from the cluster profile."""
import re
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from skynet_app import cluster_runtime, database_endpoint, live_xr_archive
from skynet_app.cluster_config import CLUSTER, SshOperationTimeouts, SshProfile

COMMAND = SshProfile(connect_timeout_seconds=123, server_alive_interval_seconds=45, server_alive_count_max=6)
TUNNEL = SshProfile(connect_timeout_seconds=234, server_alive_interval_seconds=56, server_alive_count_max=7)


def configure(monkeypatch):
    profile = CLUSTER.model_copy(update={"ssh": CLUSTER.ssh.model_copy(update={"command": COMMAND, "tunnel": TUNNEL})})
    for module in (cluster_runtime, database_endpoint, live_xr_archive):
        monkeypatch.setattr(module, "CLUSTER", profile)
    monkeypatch.setattr("skynet_app.cluster_config.CLUSTER", profile)
    return profile


def test_profile_options_are_the_ssh_arguments():
    assert COMMAND.options() == ["-o", "BatchMode=yes", "-o", "ConnectTimeout=123",
                                 "-o", "ServerAliveInterval=45", "-o", "ServerAliveCountMax=6"]
    assert CLUSTER.ssh.tunnel.connect_timeout_seconds >= CLUSTER.ssh.command.connect_timeout_seconds


def test_every_command_connection_uses_the_command_profile(monkeypatch):
    configure(monkeypatch)
    assert cluster_runtime.ssh_argv("sky9", "true") == ["ssh", "-T", *COMMAND.options(), "sky9", "true"]
    assert cluster_runtime.ssh_argv("sky9", "true", "-o", "LogLevel=ERROR") == \
        ["ssh", "-T", *COMMAND.options(), "-o", "LogLevel=ERROR", "sky9", "true"]
    assert live_xr_archive.LiveArchiveService._ssh_args("sky9", "print(1)") == \
        ["ssh", "-T", *COMMAND.options(), "sky9", "python3 -c 'print(1)'"]
    captured = []

    def run(argv, **kwargs):
        captured.append(argv)
        return subprocess.CompletedProcess(argv, 0, "ok\n", "")

    monkeypatch.setattr(cluster_runtime.subprocess, "run", run)
    assert cluster_runtime.ClusterClient(("sky9",)).ssh("sky9", "uptime") == "ok\n"
    assert captured == [["ssh", "-T", *COMMAND.options(), "sky9", "uptime"]]


def test_the_database_tunnel_uses_the_tunnel_profile(monkeypatch):
    configure(monkeypatch)
    endpoint = database_endpoint.SSHEndpoint({"ssh_host": "sky9", "remote_socket": "/test/socket", "user": "test"})
    assert endpoint.tunnel_argv(Path("/tmp/s")) == [
        "ssh", "-C", "-T", *TUNNEL.options(), "-o", "ExitOnForwardFailure=yes", "-o", "StreamLocalBindUnlink=no",
        "-L", "/tmp/s:/test/socket", "sky9", "cat >/dev/null",
    ]


def test_no_module_spells_its_own_ssh_timeouts():
    # The profile is the only place these options are written.
    offenders = [path.name for path in Path("skynet_app").glob("*.py")
                 if path.name != "cluster_config.py"
                 and re.search(r"ConnectTimeout=|ServerAliveInterval=|ServerAliveCountMax=|connect_timeout=\d", path.read_text())]
    assert offenders == []


def test_the_cluster_client_and_dashboard_take_their_operation_budgets_from_the_profile():
    assert cluster_runtime.TIMEOUTS is CLUSTER.ssh.operations
    for name in ("cluster_runtime.py", "main.py"):
        text = Path("skynet_app", name).read_text()
        assert not re.search(r"timeout=\d|timeout \d+s|\d+ days ago|SECONDS \+ \d", text), name


def test_an_operation_budget_must_outlast_the_remote_deadline_inside_it():
    operations = CLUSTER.ssh.operations
    assert operations.job_status_seconds > 2 * operations.slurm_tool_seconds
    for budget, deadline in (("submission_seconds", "submission_receipt_wait_seconds"),
                             ("validation_seconds", "validation_tool_seconds")):
        too_short = {**operations.model_dump(), budget: getattr(operations, deadline)}
        with pytest.raises(ValidationError, match=budget):
            SshOperationTimeouts.model_validate(too_short)
