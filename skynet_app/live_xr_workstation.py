"""Direct SSH execution of bounded live sessions, supervised by user systemd."""

import json
from pathlib import PurePosixPath
import re
import shlex
from types import SimpleNamespace

from .cluster_runtime import ClusterClient, ClusterError, SubmissionOutcomeUnknown


class LaunchRejected(ValueError):
    """The workstation definitively rejected a recorded launch attempt."""


def validate_profile(profile):
    memory = profile.get("memory_gb", 14)
    if type(memory) is not int or not 4 <= memory <= 128:
        raise ValueError("Workstation memory limit must be 4–128 GB")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", profile["gateway"]):
        raise ValueError("Workstation host must be an SSH alias or hostname")
    root = PurePosixPath(profile["work_root"])
    if not root.is_absolute() or ".." in root.parts or len(root.parts) < 4:
        raise ValueError("Set a dedicated absolute workstation workspace path")
    for key in ("repository", "runtime", "cloudxr_runtime"):
        path = PurePosixPath(profile[key])
        if ".." in path.parts or not path.is_relative_to(root) or path == root:
            raise ValueError(f"Workstation {key} must be inside its workspace")


class WorkstationClient(ClusterClient):
    def __init__(self, profile):
        validate_profile(profile)
        super().__init__([profile["gateway"]])
        self.profile = profile

    def run_root(self, identifier):
        if not re.fullmatch(r"[a-f0-9-]{36}", identifier):
            raise ValueError("Invalid live session identifier")
        return self.profile["work_root"] + "/sessions/" + identifier

    def _remote_path(self, path):
        candidate = PurePosixPath(path)
        root = PurePosixPath(self.profile["work_root"])
        if (
            ".." in candidate.parts
            or not candidate.is_relative_to(root)
            or candidate == root
        ):
            raise ValueError("Remote artifact must be inside the workstation workspace")
        return str(candidate)

    def write_capsule_file(self, identifier, name, content, gateway):
        if name not in {"runner.py", "request.json", "run.sh"}:
            raise ValueError("Unknown live session file")
        path = self.run_root(identifier) + "/" + name
        script = """import os,sys
from pathlib import Path
p=Path(sys.argv[1]); data=sys.stdin.read(); p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():
 if p.read_text()!=data: raise ValueError('Existing session file differs; refusing replacement')
else:
 with p.open('x') as f: f.write(data)
 p.chmod(0o600)
"""
        self.ssh(
            gateway,
            "python3 -c " + shlex.quote(script) + " " + shlex.quote(path),
            stdin=content,
        )

    @staticmethod
    def unit(identifier):
        return "skynet-live-" + identifier + ".service"

    def submit_script(self, script, identifier, gateway, **_):
        self.write_capsule_file(identifier, "run.sh", script, gateway)
        root, unit = self.run_root(identifier), self.unit(identifier)
        command = [
            "systemd-run",
            "--user",
            "--quiet",
            "--unit=" + unit,
            "--service-type=exec",
            "--remain-after-exit",
            "--property=RuntimeMaxSec="
            + str((self.profile["duration_minutes"] + (35 if self.profile.get("image_capture") else 5)) * 60),
            "--property=TimeoutStopSec=35",
            "--property=KillMode=mixed",
            "--property=Restart=no",
            "--property=UMask=0077",
            "--property=MemoryMax=" + str(self.profile.get("memory_gb", 14)) + "G",
            "--property=MemorySwapMax=1G",
            "--property=CPUQuota=400%",
            "--property=StandardOutput=append:" + root + "/stdout.log",
            "--property=StandardError=append:" + root + "/stderr.log",
            "--working-directory=" + root,
            "--setenv=SKYNET_LIVE_JOB_ID=" + unit,
            "--setenv=SKYNET_LIVE_SESSION_ID=" + identifier,
            "/usr/bin/flock",
            "--nonblock",
            "--no-fork",
            "--conflict-exit-code=75",
            self.profile["work_root"] + "/.gpu-session.lock",
            "/bin/bash",
            root + "/run.sh",
        ]
        launcher = """import json,subprocess,sys
from pathlib import Path
request=json.load(sys.stdin); root=Path(request['root'])
# Persist before starting. A lost acknowledgement must never launch twice.
try:
 with (root/'launch.attempt.json').open('x') as f: json.dump(request,f)
except FileExistsError:
 raise SystemExit('A launch was already attempted; recover the existing service')
r=subprocess.run(request['command'],capture_output=True,text=True,timeout=25)
result={'returncode':r.returncode,'error':r.stderr.strip()}
(root/'launch.result.json').write_text(json.dumps(result))
if r.returncode: raise SystemExit(result['error'])
print(request['unit'])
"""
        try:
            self.ssh(
                gateway,
                "python3 -c " + shlex.quote(launcher),
                stdin=json.dumps(dict(root=root, unit=unit, command=command)),
                timeout=35,
            )
        except ClusterError as exc:
            raise SubmissionOutcomeUnknown(
                "Workstation launch acknowledgement unavailable; refresh recovers this same service. "
                + str(exc)
            ) from exc
        return SimpleNamespace(job_id=unit)

    def recover_submission(self, identifier, _token, gateway):
        unit = self.unit(identifier)
        state = self.service_status(unit, gateway)
        if state["LoadState"] != "not-found":
            return SimpleNamespace(job_id=unit)
        root = self.run_root(identifier)
        result = self.ssh(
            gateway,
            "if test -f "
            + shlex.quote(root + "/launch.result.json")
            + "; then cat "
            + shlex.quote(root + "/launch.result.json")
            + "; fi",
        )
        if result:
            launch = json.loads(result)
            if launch["returncode"]:
                raise LaunchRejected(
                    "Workstation could not start the live service: " + launch["error"]
                )
        return None

    def service_status(self, unit, gateway):
        if not re.fullmatch(r"skynet-live-[a-f0-9-]{36}\.service", unit):
            raise ValueError("Invalid live service name")
        fields = "LoadState,ActiveState,SubState,Result,ExecMainStatus,MainPID"
        result = self.ssh(
            gateway,
            "systemctl --user show " + shlex.quote(unit) + " --property=" + fields,
        )
        return dict(line.split("=", 1) for line in result.splitlines() if "=" in line)

    def job_statuses(self, identifiers, gateway):
        statuses = {}
        for identifier in identifiers:
            status = self.service_status(identifier, gateway)
            if status.get("LoadState") == "not-found":
                # systemd may unload a successfully stopped transient unit.
                session_id = identifier.removeprefix("skynet-live-").removesuffix(
                    ".service"
                )
                path = self.run_root(session_id) + "/output/status.json"
                raw = self.ssh(
                    gateway,
                    "if test -f "
                    + shlex.quote(path)
                    + "; then head -c 1000000 "
                    + shlex.quote(path)
                    + "; fi",
                )
                control = json.loads(raw) if raw else {}
                clean = control.get("job_id") == identifier and control.get(
                    "state"
                ) in {"STOPPED", "CAPTURED", "TIMED_OUT"}
                state = "COMPLETED" if clean else "FAILED"
                status["Result"] = "worker-finished" if clean else "service-missing"
            elif status.get("ActiveState") == "failed":
                state = "FAILED"
                if status.get("ExecMainStatus") == "75":
                    status["Result"] = "Another Skynet GPU session is already running"
            elif status.get("SubState") == "exited":
                state = "COMPLETED" if status.get("ExecMainStatus") == "0" else "FAILED"
            elif status.get("ActiveState") == "inactive":
                state = "CANCELLED" if status.get("Result") == "success" else "FAILED"
            else:
                state = "RUNNING"
            statuses[identifier] = dict(status, State=state, backend="workstation")
        return gateway, statuses

    def cancel(self, identifier, gateway):
        # A completed transient service can disappear between refresh and Stop.
        if self.service_status(identifier, gateway).get("LoadState") == "not-found":
            return
        try:
            self.ssh(
                gateway, "systemctl --user stop --no-block " + shlex.quote(identifier)
            )
        except ClusterError:
            if self.service_status(identifier, gateway).get("LoadState") != "not-found":
                raise
