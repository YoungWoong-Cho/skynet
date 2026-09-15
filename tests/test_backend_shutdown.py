"""Exercise terminal interruption in an isolated process group, never the app."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from skynet_app.metadata_objects import MetadataObjects


_FAKE_SSH = r"""
import json, os, pathlib, select, socket, sys, time
root = pathlib.Path(os.environ["SHUTDOWN_TEST_ROOT"])
if "-L" in sys.argv:
    address = sys.argv[sys.argv.index("-L") + 1].split(":", 1)[0]
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(address)
    # Like the real tunnel, exit on the owner's stdin EOF. A deadline also
    # bounds cleanup if the test itself fails.
    select.select([sys.stdin], [], [], 6)
    listener.close()
else:
    json.load(sys.stdin)
    (root / "metadata-ready").touch()
    time.sleep(0.2)
    print(json.dumps({"read": "completed"}))
"""

_OWNER = r"""
import json, os, pathlib, signal, threading, time
from types import SimpleNamespace
from skynet_app.database_endpoint import SSHEndpoint
from skynet_app.metadata_objects import MetadataObjects

root = pathlib.Path(os.environ["SHUTDOWN_TEST_ROOT"])
received = []
signal.signal(signal.SIGINT, lambda *args: received.append(True))
endpoint = SSHEndpoint({"ssh_host": "test", "remote_socket": "/test/socket", "user": "test"})
child = None
try:
    endpoint.connection_string()
    child = endpoint._process
    directory = endpoint._directory
    def interrupt():
        deadline = time.monotonic() + 3
        while not (root / "metadata-ready").exists():
            if time.monotonic() >= deadline:
                return
            time.sleep(0.005)
        os.killpg(os.getpgrp(), signal.SIGINT)
    sender = threading.Thread(target=interrupt)
    sender.start()
    objects = MetadataObjects(SimpleNamespace(url=None, endpoint_config={
        "ssh_host": "test", "object_store_root": str(root / "objects")}))
    result = objects._exchange({"operation": "get"})
    sender.join(timeout=4)
    assert received, "No terminal signal exercised"
    assert child.poll() is None, "Ctrl+C killed the DB tunnel before cleanup"
    assert result == {"read": "completed"}
finally:
    endpoint.close()
assert child.wait(timeout=2) == 0
assert not directory.exists()
print("Transfer survived Ctrl+C; DB tunnel closed by its owner")
"""


def test_ctrl_c_during_metadata_read_preserves_tunnel_until_owner_cleanup(tmp_path):
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(f"#!{sys.executable}\n" + _FAKE_SSH)
    fake_ssh.chmod(0o700)
    result = subprocess.run(
        [sys.executable, "-c", _OWNER],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "SHUTDOWN_TEST_ROOT": str(tmp_path),
            "PYTHONPATH": str(
                Path(sys.modules[MetadataObjects.__module__].__file__)
                .resolve()
                .parents[1]
            ),
        },
        start_new_session=True,
        capture_output=True,
        text=True,
        timeout=12,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Transfer survived Ctrl+C" in result.stdout


@pytest.mark.parametrize("code,reason", [(-2, "signal 2"), (255, "status 255")])
def test_metadata_transport_failure_reports_host_and_exit_reason(
    monkeypatch, code, reason
):
    objects = MetadataObjects.__new__(MetadataObjects)
    objects.host = "metadata-host"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=code, stdout="", stderr=""),
    )
    with pytest.raises(OSError, match=f"metadata-host; SSH.*{reason}"):
        objects._exchange({})
