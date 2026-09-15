from __future__ import annotations

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import skynet_app.cluster_runtime as cluster_runtime
from skynet_app.cluster_runtime import (
    ClusterClient,
    ClusterError,
    approved_operator_environment,
)


class RecordingClusterClient(ClusterClient):
    def __init__(self) -> None:
        super().__init__(("sky1", "sky2"))
        self.commands: list[tuple[str, str, str | None]] = []

    def resolve_gateway(self, gateway: str = "auto") -> str:
        return "sky2"

    def ssh(self, host, command, *, stdin=None, timeout=30):
        self.commands.append((host, command, stdin))
        return "4815162342\n"


def test_submission_archives_script_by_slurm_job_id() -> None:
    client = RecordingClusterClient()

    submission = client.submit_script("#!/bin/bash\ntrue\n", "run-123", "auto")

    assert submission.job_id == "4815162342"
    assert submission.gateway == "sky2"
    assert submission.script_path.endswith("/run-123/attempts/4815162342/job.sbatch")
    host, command, stdin = client.commands[0]
    assert host == "sky2"
    assert "sbatch --parsable" in command
    assert "--comment=skynet-" in command
    assert "/submissions/" in command
    assert ".claim" in command
    assert "nohup /bin/bash" in command
    assert "attempts/$job_id" in command
    assert 'cp -f ' in command
    assert stdin == "#!/bin/bash\ntrue\n"


def test_submission_explicitly_forwards_validated_environment() -> None:
    client = RecordingClusterClient()
    script = "#!/bin/bash\n#SBATCH --export=ALL\ntrue\n"

    client.submit_script(
        script,
        "run-123",
        forwarded_environment={"OMNI_KIT_ACCEPT_EULA": "YES"},
    )

    _, command, stdin = client.commands[0]
    assert "OMNI_KIT_ACCEPT_EULA=YES sbatch --parsable" in command
    assert "--export=ALL,OMNI_KIT_ACCEPT_EULA=YES" not in command
    assert "--export-file" not in command
    assert "setsid --fork" in command
    assert '</dev/null >/dev/null 2>&1' in command
    assert "OMNI_KIT_ACCEPT_EULA=YES" not in stdin


def test_nil_export_uses_file_without_get_user_environment() -> None:
    client = RecordingClusterClient()
    script = "#!/bin/bash\n#SBATCH --export=NIL\ntrue\n"

    client.submit_script(
        script,
        "run-123",
        forwarded_environment={"OMNI_KIT_ACCEPT_EULA": "YES"},
    )

    _, command, _ = client.commands[0]
    assert '--export-file="$export_file"' in command
    assert "--export=OMNI_KIT_ACCEPT_EULA" not in command


def test_operator_environment_requires_exact_explicit_acceptance() -> None:
    assert approved_operator_environment({"OMNI_KIT_ACCEPT_EULA": "YES"}) == {
        "OMNI_KIT_ACCEPT_EULA": "YES"
    }
    assert approved_operator_environment({"OMNI_KIT_ACCEPT_EULA": "yes"}) == {}
    assert approved_operator_environment({"OMNI_KIT_ACCEPT_EULA": "1"}) == {}
    assert approved_operator_environment({}) == {}
    assert approved_operator_environment(
        {"OMNI_KIT_ACCEPT_EULA": "YES"}, enabled=False
    ) == {}


def test_pinned_explicit_gateway_is_prompt_and_ambiguous_retry_submits_once(
    tmp_path: Path, monkeypatch
) -> None:
    work_root = tmp_path / "work"
    bin_root = tmp_path / "bin"
    bin_root.mkdir()
    counter = tmp_path / "sbatch-calls"
    sbatch = bin_root / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        "printf 'called\\n' >> \"$SKYNET_TEST_SBATCH_COUNTER\"\n"
        "sleep 0.3\n"
        "printf '3753915\\n'\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    for command_name in ("squeue", "sacct"):
        command = bin_root / command_name
        command.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        command.chmod(0o755)
    monkeypatch.setattr(cluster_runtime, "WORK_ROOT", str(work_root))
    monkeypatch.setattr(cluster_runtime, "SLURM_BIN", str(bin_root))
    monkeypatch.setenv("SKYNET_TEST_SBATCH_COUNTER", str(counter))

    class LocalAmbiguousClient(ClusterClient):
        def __init__(self) -> None:
            super().__init__(("sky2", "sky1"))
            self.timeout_once = True
            self.timeout_lock = threading.Lock()
            self.ambiguous = threading.Event()
            self.ssh_hosts: list[str] = []

        def resolve_gateway(self, gateway="auto"):
            return "sky1"

        def ssh(self, host, command, *, stdin=None, timeout=30):
            self.ssh_hosts.append(host)
            force_timeout = False
            if "nohup /bin/bash" in command:
                with self.timeout_lock:
                    if self.timeout_once:
                        self.timeout_once = False
                        force_timeout = True
            try:
                completed = subprocess.run(
                    ["/bin/bash", "-c", command],
                    input=stdin,
                    capture_output=True,
                    text=True,
                    timeout=0.08 if force_timeout else timeout,
                    check=False,
                    env=os.environ.copy(),
                )
            except subprocess.TimeoutExpired as error:
                self.ambiguous.set()
                raise ClusterError("local: SSH acknowledgement timed out") from error
            if completed.returncode != 0:
                raise ClusterError(
                    f"local: {(completed.stderr or completed.stdout).strip()}"
                )
            return completed.stdout

    client = LocalAmbiguousClient()
    script = (
        "#!/bin/bash\n"
        "#SBATCH --export=ALL\n"
        "#SBATCH --nodelist=randotron\n"
        "true\n"
    )
    started = time.monotonic()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            client.submit_script,
            script,
            "readiness-race",
            "sky2",
            submission_key="stable-readiness-request",
            forwarded_environment={"OMNI_KIT_ACCEPT_EULA": "YES"},
        )
        assert client.ambiguous.wait(timeout=2)
        second = pool.submit(
            client.submit_script,
            script,
            "readiness-race",
            "sky2",
            submission_key="stable-readiness-request",
            forwarded_environment={"OMNI_KIT_ACCEPT_EULA": "YES"},
        )
        submissions = [first.result(timeout=5), second.result(timeout=5)]

    assert [submission.job_id for submission in submissions] == [
        "3753915",
        "3753915",
    ]
    assert counter.read_text(encoding="utf-8").splitlines() == ["called"]
    assert time.monotonic() - started < 2
    assert set(client.ssh_hosts) == {"sky2"}

    client.ssh_hosts.clear()
    fresh_started = time.monotonic()
    fresh = client.submit_script(
        script,
        "readiness-pinned-fresh",
        "sky2",
        submission_key="stable-pinned-request",
        forwarded_environment={"OMNI_KIT_ACCEPT_EULA": "YES"},
    )

    assert fresh.job_id == "3753915"
    assert time.monotonic() - fresh_started < 2
    assert set(client.ssh_hosts) == {"sky2"}
    assert counter.read_text(encoding="utf-8").splitlines() == [
        "called",
        "called",
    ]


class LostAcknowledgementClient(ClusterClient):
    def __init__(self) -> None:
        super().__init__(("sky2",))
        self.calls = 0

    def resolve_gateway(self, gateway: str = "auto") -> str:
        return "sky2"

    def ssh(self, host, command, *, stdin=None, timeout=30):
        self.calls += 1
        if "sbatch --parsable" in command:
            raise ClusterError("sky2: SSH command failed with exit code 255")
        if "/submissions/" in command and "cat " in command:
            return "3745949\n"
        return ""


def test_submission_recovers_job_id_after_lost_ssh_acknowledgement() -> None:
    client = LostAcknowledgementClient()

    submission = client.submit_script(
        "#!/bin/bash\ntrue\n",
        "run-123",
        "sky2",
        submission_key="attempt-456",
    )

    assert submission.job_id == "3745949"
    assert submission.gateway == "sky2"
    assert submission.recovered is True


class FallbackClusterClient(ClusterClient):
    def ssh(self, host, command, *, stdin=None, timeout=30):
        if host == "sky1":
            raise ClusterError("sky1 unavailable")
        return "ready"


def test_read_operations_fall_back_to_second_gateway() -> None:
    client = FallbackClusterClient(("sky1", "sky2"))

    host, output = client.run_with_fallback("true")

    assert host == "sky2"
    assert output == "ready"


def test_fallback_reserves_operation_budget_for_each_gateway(monkeypatch) -> None:
    clock = [100.0]
    attempts: list[tuple[str, float]] = []

    class TimedOutFirstGatewayClient(ClusterClient):
        def ssh(self, host, command, *, stdin=None, timeout=30):
            attempts.append((host, timeout))
            if host == "sky1":
                clock[0] += timeout
                raise ClusterError("sky1: timed out")
            return "ready from sky2"

    monkeypatch.setattr(cluster_runtime.time, "monotonic", lambda: clock[0])
    client = TimedOutFirstGatewayClient(("sky1", "sky2"))

    host, output = client.run_with_fallback("true", timeout=20)

    assert host == "sky2"
    assert output == "ready from sky2"
    assert [attempt[0] for attempt in attempts] == ["sky1", "sky2"]
    assert attempts[0][1] == 10
    assert attempts[1][1] == 10


class AccountingClusterClient(ClusterClient):
    def __init__(self, output: str) -> None:
        super().__init__(("sky2",))
        self.output = output
        self.command = ""

    def run_with_fallback(self, command, gateway="auto", *, stdin=None, timeout=30):
        self.command = command
        return "sky2", self.output


def test_job_statuses_normalizes_cancelled_accounting_record() -> None:
    client = AccountingClusterClient(
        "3742328|CANCELLED by 3712043|0:0|None|claptrap|00:21:00|"
        "1788334844|1788336104|"
        "billing=48,gres/gpu=4|overcap|overcap\n"
    )

    host, statuses = client.job_statuses(["3742328"])

    assert host == "sky2"
    assert "State%64" in client.command
    assert "SLURM_TIME_FORMAT=%s" in client.command
    assert statuses["3742328"] == {
        "State": "CANCELLED",
        "ExitCode": "0:0",
        "Reason": "None",
        "NodeList": "claptrap",
        "Elapsed": "00:21:00",
        "Start": "1788334844",
        "End": "1788336104",
        "AllocTRES": "billing=48,gres/gpu=4",
        "Partition": "overcap",
        "Account": "overcap",
        "StateRaw": "CANCELLED by 3712043",
        "CancelledBy": "3712043",
    }


def test_job_statuses_parses_multiple_records_and_omits_missing_ids() -> None:
    client = AccountingClusterClient(
        "3742328|CANCELLED+|0:0|None|claptrap|00:21:00|"
        "1788334844|1788336104|gpu=4|overcap|overcap\n"
        "3742104|FAILED|2:0|None|flexo|00:00:33|"
        "1788326567|1788326600|gpu=1|rl2-lab|rl2-lab\n"
    )

    _, statuses = client.job_statuses(["3742328", "3742104", "9999999"])

    assert set(statuses) == {"3742328", "3742104"}
    assert statuses["3742328"]["State"] == "CANCELLED"
    assert statuses["3742328"]["StateRaw"] == "CANCELLED+"
    assert "CancelledBy" not in statuses["3742328"]
    assert statuses["3742104"]["State"] == "FAILED"
    assert statuses["3742104"]["ExitCode"] == "2:0"
    assert statuses["3742104"]["NodeList"] == "flexo"
    assert statuses["3742104"]["End"] == "1788326600"


def test_cancel_rejects_invalid_slurm_job_ids_before_ssh() -> None:
    client = RecordingClusterClient()

    for job_id in ("", "abc", "123,456", "123; touch /tmp/skynet-cancel"):
        try:
            client.cancel(job_id, "sky1")
        except ValueError as error:
            assert str(error) == "Invalid Slurm job ID"
        else:
            raise AssertionError(f"invalid Slurm job ID was accepted: {job_id!r}")

    assert client.commands == []


def test_validation_timeout_is_an_error_and_never_runs_the_job(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    timeout = fake_bin / "timeout"
    timeout.write_text("#!/bin/sh\nexit 124\n")
    timeout.chmod(0o755)
    monkeypatch.setattr(cluster_runtime, "SLURM_BIN", str(fake_bin))
    client = RecordingClusterClient()
    marker = tmp_path / "job-must-not-run"
    script = f"#!/bin/bash\ntouch {marker}\n"
    client.test_script(script, "sky1")
    _, command, _ = client.commands[-1]
    result = subprocess.run(["bash", "-c", command], input=script, text=True, capture_output=True)
    assert result.returncode == 124
    assert "No job was submitted" in result.stderr
    assert not marker.exists()


def test_pending_reasons_use_one_live_query_on_the_accounting_gateway(monkeypatch):
    client = AccountingClusterClient(
        "11|PENDING|0:0|None|None assigned|00:00:00|Unknown|Unknown||overcap|overcap\n"
        "12|PENDING|0:0|None|None assigned|00:00:00|Unknown|Unknown||overcap|overcap\n"
    )
    calls = []
    def live_queue(host, command, **kwargs):
        calls.append((host, command, kwargs))
        return "11|PENDING|QOSGrpGRES\n12|PENDING|(Priority)\n99|PENDING|Resources\n"
    monkeypatch.setattr(client, "ssh", live_queue)
    _, records = client.job_statuses(["11", "12"])
    assert len(calls) == 1
    assert calls[0][0] == "sky2"
    assert "-j 11,12" in calls[0][1]
    assert records["11"]["Reason"] == "QOSGrpGRES"
    assert records["12"]["Reason"] == "Priority"
    assert "99" not in records


def test_queue_reason_failure_is_visible_without_discarding_accounting(monkeypatch):
    client = AccountingClusterClient(
        "11|PENDING|0:0|None|None assigned|00:00:00|Unknown|Unknown||overcap|overcap\n"
    )
    def unavailable(*args, **kwargs):
        raise ClusterError("sky2: SSH operation timed out")
    monkeypatch.setattr(client, "ssh", unavailable)
    _, records = client.job_statuses(["11"])
    assert records["11"]["State"] == "PENDING"
    assert "Live queue reason unavailable" in records["11"]["Reason"]
    assert "timed out" in records["11"]["Reason"]


def test_unavailable_gpu_forecast_does_not_reject_a_valid_submission(tmp_path, monkeypatch):
    """The controller can accept GPU jobs while its will-run RPC is unavailable."""
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    timeout = fake_bin / 'timeout'
    monkeypatch.setattr(cluster_runtime, 'SLURM_BIN', str(fake_bin))
    marker = tmp_path / 'job-must-not-run'
    script = f'#!/bin/bash\ntouch {marker}\n'
    for message, expected in [
        ('allocation failure: Zero Bytes were transmitted or received', 0),
        ('sbatch: error: Invalid generic resource specification', 1),
    ]:
        timeout.write_text('#!/bin/sh\nprintf "%s\\n" "' + message + '" >&2\nexit 1\n')
        timeout.chmod(0o755)
        client = RecordingClusterClient()
        client.test_script(script, 'sky1')
        _, command, _ = client.commands[-1]
        result = subprocess.run(['bash', '-c', command], input=script, text=True, capture_output=True)
        assert result.returncode == expected
        assert not marker.exists()
        if expected == 0:
            assert 'submission will validate' in result.stdout


def test_cancelled_file_stream_interrupts_stalled_owned_ssh_process(monkeypatch):
    import sys

    original_popen = subprocess.Popen
    children = []

    def popen(*args, **kwargs):
        child = original_popen([sys.executable, "-c", "import sys,time; sys.stdout.buffer.write(b'ab'); sys.stdout.flush(); time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", popen)
    event = threading.Event()
    stream = ClusterClient(("host",)).stream_file_range(cluster_runtime.WORK_ROOT + "/video.mp4", "host", start=0, end=99, cancel_event=event)
    assert next(stream) == b"ab"
    received = []
    worker = threading.Thread(target=lambda: received.extend(stream))
    worker.start()
    start = time.monotonic()
    event.set()
    worker.join(2)
    assert not worker.is_alive()
    assert time.monotonic() - start < 1
    assert children[0].poll() is not None
    assert received == []


def test_file_stream_without_cancellation_preserves_original_contract(monkeypatch):
    import sys

    original_popen = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: original_popen(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'abcd')"], **kw))
    assert b"".join(ClusterClient(("host",)).stream_file_range(
        cluster_runtime.WORK_ROOT + "/video.mp4", "host", start=0, end=3)) == b"abcd"


def test_capsule_batch_transfers_exact_files_in_one_verified_connection(tmp_path, monkeypatch):
    root = tmp_path.resolve() / "work"
    monkeypatch.setattr(cluster_runtime, "WORK_ROOT", str(root))
    client = ClusterClient(("sky2",))
    calls = []
    def local(host, command, *, stdin=None, timeout=30):
        calls.append((host, command))
        result = subprocess.run(command, shell=True, input=stdin, text=True, capture_output=True)
        if result.returncode:
            raise ClusterError(result.stderr)
        return result.stdout
    monkeypatch.setattr(client, "ssh", local)
    files = {"worker/a.py": "print('한글')\n", "worker/space name.txt": "$(touch unexpected)\n", "request.json": "{}"}
    host, paths = client.write_capsule_files("batch-1", files, "sky2")
    assert host == "sky2" and len(calls) == 1
    assert {name: Path(path).read_text() for name, path in paths.items()} == files
    assert not list(root.rglob(".upload-*"))
    # Repeating a partial/uncertain upload is safe before job submission.
    client.write_capsule_files("batch-1", files, "sky2")
    assert {name: Path(path).read_text() for name, path in paths.items()} == files


def test_capsule_batch_rejects_invalid_paths_and_unverified_receipts(monkeypatch):
    import pytest
    client = ClusterClient(("sky2",))
    calls = []
    def ssh(*args, **kwargs):
        calls.append(args)
        return '{}'
    monkeypatch.setattr(client, "ssh", ssh)
    for files in ({"../outside": "bad"}, {"/absolute": "bad"}, {"a/b": "one", "a//b": "two"}):
        with pytest.raises(ValueError):
            client.write_capsule_files("batch-1", files, "sky2")
    assert calls == []
    with pytest.raises(ClusterError, match="verification failed"):
        client.write_capsule_files("batch-1", {"worker.py": "frozen"}, "sky2")


def test_capsule_batch_preserves_symlink_targets(tmp_path, monkeypatch):
    import pytest
    root = tmp_path.resolve() / "work"
    monkeypatch.setattr(cluster_runtime, "WORK_ROOT", str(root))
    destination = root / "jobs/runs/batch-1/worker"
    destination.mkdir(parents=True)
    protected = tmp_path.resolve() / "protected"
    protected.write_text("keep")
    (destination / "frozen.py").symlink_to(protected)
    client = ClusterClient(("sky2",))
    def local(host, command, *, stdin=None, timeout=30):
        result = subprocess.run(command, shell=True, input=stdin, text=True, capture_output=True)
        if result.returncode:
            raise ClusterError(result.stderr)
        return result.stdout
    monkeypatch.setattr(client, "ssh", local)
    with pytest.raises(ClusterError, match="symbolic links"):
        client.write_capsule_files("batch-1", {"worker/frozen.py": "replace"}, "sky2")
    assert protected.read_text() == "keep"
