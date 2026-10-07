from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import select
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from .cluster_config import CLUSTER


if TYPE_CHECKING:
    from .workspace_storage import WorkspaceStorage


HOME_ROOT = CLUSTER.paths.home_root
WORK_ROOT = CLUSTER.paths.work_root
SLURM_BIN = CLUSTER.commands.slurm_bin
# How cluster work is routed when nothing chose a gateway: "auto" or one configured host.
DEFAULT_GATEWAY = CLUSTER.defaults.gateway
# Whole-operation budgets and the Slurm tool deadlines inside them, by purpose.
TIMEOUTS = CLUSTER.ssh.operations


def ssh_argv(host: str, command: str, *options: str) -> list[str]:
    """argv for one non-interactive SSH command with the configured connection policy."""
    return ["ssh", "-T", *CLUSTER.ssh.command.options(), *options, host, command]


# Slurm states in which a job still occupies the scheduler; every other state is final.
ACTIVE_STATES = frozenset({
    "PENDING",
    "CONFIGURING",
    "RUNNING",
    "COMPLETING",
    "REQUEUED",
    "RESIZING",
    "SUSPENDED",
})
# Queued again or not yet started: the controller's start time is only a forecast.
_UNSTARTED_STATES = frozenset({"PENDING", "REQUEUED"})
# The controller names conditions that accounting reports under these base states.
_CONTROLLER_STATES = {
    "REQUEUE_HOLD": "PENDING", "REQUEUE_FED": "PENDING", "RESV_DEL_HOLD": "PENDING",
    "SIGNALING": "RUNNING", "STAGE_OUT": "COMPLETING", "STOPPED": "SUSPENDED",
}
# Engineering choices of the client, not cluster policy.
_NODE_INVENTORY_TTL_SECONDS = 60  # How long one gateway's node list is reused.
_ACCOUNTING_LOOKBACK_DAYS = 7  # How far back accounting is searched for a lost submission's token.
_RECEIPT_POLL_SECONDS = 0.2  # How often the submitting shell looks for the detached worker's receipt.
_RECOVERY_RETRY_DELAYS = (0, 1, 2)  # Seconds before each recovery lookup after a lost sbatch response.
_STREAM_CLOSE_SECONDS = 10  # How long a finished file stream may take to close.
_PROCESS_EXIT_SECONDS = 2  # How long a terminated ssh process may take to exit.
_ACCOUNTING_FIELDS = (
    ("JobIDRaw", "JobIDRaw"),
    ("State", "State%64"),
    ("ExitCode", "ExitCode"),
    ("Reason", "Reason"),
    ("NodeList", "NodeList"),
    ("Elapsed", "Elapsed"),
    ("Start", "Start"),
    ("End", "End"),
    ("AllocTRES", "AllocTRES"),
    ("Partition", "Partition"),
    ("Account", "Account"),
    ("Restarts", "Restarts"),
)
# squeue -O names, in the order of the record keys they fill.
_CONTROLLER_FIELDS = (
    ("JobID", "JobID"),
    ("State", "State"),
    ("ExitCode", "exit_code"),
    ("Reason", "Reason"),
    ("NodeList", "NodeList"),
    ("Start", "StartTime"),
    ("End", "EndTime"),
    ("Restarts", "Restartcnt"),
)
_ACCOUNTING_MARKER = "__SKYNET_ACCOUNTING__"
_CONTROLLER_MARKER = "__SKYNET_CONTROLLER__"
ACCOUNTING_UNAVAILABLE_MARKER = "__SKYNET_ACCOUNTING_UNAVAILABLE__"


class ClusterError(RuntimeError):
    """A gateway or Slurm operation failed."""


class GatewayUnreachable(ClusterError):
    """SSH could not reach or log in to the gateway; the command did not run."""


class SubmissionOutcomeUnknown(ClusterError):
    """Slurm may have accepted a job whose SSH acknowledgement was lost."""


def validate_remote_path(path: str, roots: Iterable[str] = (WORK_ROOT,)) -> str:
    """Validate a cluster path against the caller's registered storage roots."""
    candidate = PurePosixPath(path)
    if not candidate.is_absolute() or not any(
        candidate == PurePosixPath(root) or PurePosixPath(root) in candidate.parents
        for root in roots
    ):
        raise ValueError("Remote path must be below a registered workspace root")
    if ".." in candidate.parts:
        raise ValueError("Remote path cannot contain '..'")
    return str(candidate)


def approved_operator_environment(
    environ: Mapping[str, str] | None = None,
    *,
    enabled: bool = True,
) -> dict[str, str]:
    """Return operator-controlled values that are safe to forward to Slurm."""

    source = os.environ if environ is None else environ
    if enabled and source.get("OMNI_KIT_ACCEPT_EULA") == "YES":
        return {"OMNI_KIT_ACCEPT_EULA": "YES"}
    return {}


def _validated_forwarded_environment(
    environment: Mapping[str, str] | None,
) -> dict[str, str]:
    validated: dict[str, str] = {}
    for name, value in sorted((environment or {}).items()):
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
            raise ValueError(f"Invalid forwarded environment variable: {name}")
        if not isinstance(value, str):
            raise ValueError(f"Forwarded environment value must be text: {name}")
        if any(character in value for character in ("\x00", "\n", "\r", ",")):
            raise ValueError(
                f"Forwarded environment value must be a safe single Slurm export value: {name}"
            )
        validated[name] = value
    return validated


def _sbatch_environment_options(
    script: str, environment: Mapping[str, str] | None
) -> tuple[str, str]:
    forwarded = _validated_forwarded_environment(environment)
    if not forwarded:
        return "", ""
    inherits_all = bool(
        re.search(
            r"^\s*#SBATCH\s+--export(?:=|\s+)ALL(?:,|\s|$)",
            script,
            flags=re.MULTILINE,
        )
    )
    assignments = " ".join(
        f"{name}={shlex.quote(value)}" for name, value in forwarded.items()
    )
    if inherits_all:
        # The existing script directive exports the prefixed values. Avoid
        # `--export=ALL,NAME=value`: Slurm treats a named export as an implicit
        # --get-user-env request, which can block a healthy submission for a minute.
        return assignments + " ", ""
    payload = b"\0".join(
        f"{name}={value}".encode("utf-8") for name, value in forwarded.items()
    ) + b"\0"
    return assignments + " ", base64.b64encode(payload).decode("ascii")


# The asterisk rule that frames a login notice above and below its marker line.
_BANNER_BORDER = re.compile(r"^\s*\*{20,}\s*$")


def _strip_login_banner(value: str) -> str:
    """Remove the configured gateway login notice without hiding command stderr."""

    markers = CLUSTER.ssh.login_banner_markers
    lines = value.splitlines()
    while True:
        notice = next(
            (
                index
                for index, line in enumerate(lines)
                if any(marker in line for marker in markers)
            ),
            None,
        )
        if notice is None:
            break
        start = next(
            (index for index in range(notice, -1, -1) if _BANNER_BORDER.match(lines[index])),
            notice,
        )
        end = next(
            (index for index in range(notice + 1, len(lines)) if _BANNER_BORDER.match(lines[index])),
            notice,
        )
        del lines[start : end + 1]
    return "\n".join(lines).strip()


@dataclass(frozen=True)
class JobStatusSnapshot:
    """What Slurm accounting and the live controller report for a batch of jobs.

    A job missing from ``statuses`` is gone from the scheduler only when
    ``controller_error`` is None: the controller answered and no longer lists it.
    """

    gateway: str
    statuses: dict[str, dict[str, str]]
    accounting_error: str | None = None
    controller_error: str | None = None


@dataclass(frozen=True)
class Submission:
    job_id: str
    raw_job_id: str
    gateway: str
    script_path: str
    run_directory: str
    recovered: bool = False


class ClusterClient:
    """Small synchronous SSH client with side-effect-safe gateway fallback."""

    def __init__(self, hosts: Sequence[str] | None = None) -> None:
        configured = hosts or tuple(CLUSTER.gateways)
        if not configured:
            raise ValueError("At least one SSH gateway is required")
        self.hosts = tuple(dict.fromkeys(configured))
        # Shared by the per-workspace copies: one refused login informs them all.
        self._unreachable_until: dict[str, float] = {}
        self.storage: WorkspaceStorage | None = None
        self._node_inventory_cache: dict[str, tuple[float, list[str]]] = {}

    def with_storage(self, storage: WorkspaceStorage) -> ClusterClient:
        client = copy.copy(self)
        client.storage = storage
        return client

    @property
    def work_root(self) -> str:
        return self.storage.require_root() if self.storage else WORK_ROOT

    def run_directory(self, run_id: str) -> str:
        run_id = self._run_id(run_id)
        return self.storage.run_directory(run_id) if self.storage else f"{WORK_ROOT}/jobs/runs/{run_id}"


    def candidates(self, gateway: str) -> tuple[str, ...]:
        """Hosts to try, in order.

        Every gateway reaches the same scheduler and shared storage, so a named
        gateway is tried first and the others remain as fallback.
        """
        if gateway == "auto":
            return self.hosts
        if gateway not in self.hosts:
            raise ValueError(f"Unknown SSH gateway: {gateway}")
        return (gateway, *(host for host in self.hosts if host != gateway))

    def gateway_for(self, recorded: str | None) -> str:
        """Routing for a gateway recorded earlier with an attempt, an archive or a preference.

        The record is provenance, not an address: a host that is no longer
        configured must not strand the work it once served.
        """
        return recorded if recorded == "auto" or recorded in self.hosts else DEFAULT_GATEWAY

    def ssh(
        self,
        host: str,
        command: str,
        *,
        stdin: str | None = None,
        timeout: float = TIMEOUTS.command_seconds,
    ) -> str:
        try:
            process = subprocess.run(
                ssh_argv(host, command),
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ClusterError(f"{host}: SSH operation timed out") from error
        if process.returncode != 0:
            stderr = _strip_login_banner(process.stderr or "")
            detail = (stderr or process.stdout or f"SSH command failed with exit code {process.returncode}").strip()
            # ssh itself exits 255 when it cannot connect or authenticate.
            failure = GatewayUnreachable if process.returncode == 255 else ClusterError
            raise failure(f"{host}: {detail}")
        return process.stdout

    def _unreachable(self, host: str) -> bool:
        return time.monotonic() < self._unreachable_until.get(host, 0.0)

    def run_with_fallback(
        self,
        command: str,
        gateway: str = "auto",
        *,
        stdin: str | None = None,
        timeout: float = TIMEOUTS.command_seconds,
        attempt_timeout: float | None = None,
    ) -> tuple[str, str]:
        """Run on the first gateway that answers.

        ``timeout`` bounds the whole operation. ``attempt_timeout`` instead allows
        every gateway that long, for work whose duration is its own.
        """
        route = self.gateway_for(gateway)
        candidates = self.candidates(route)
        if attempt_timeout is not None:
            timeout = attempt_timeout * len(candidates)
        if timeout <= 0:
            raise ValueError("SSH operation timeout must be positive")
        # A gateway that just refused its connection goes last and reserves no time.
        reachable = [host for host in candidates if not self._unreachable(host)]
        candidates = (*reachable, *(host for host in candidates if host not in reachable))
        deadline = time.monotonic() + timeout
        errors: list[str] = []
        for index, host in enumerate(candidates):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                errors.append(f"{host}: SSH operation budget exhausted before attempt")
                continue

            waiting = len(reachable) - index  # Reachable gateways still to be tried, this one included.
            if attempt_timeout is not None:
                allowed = min(attempt_timeout, remaining)
            elif waiting <= 1 or (route != "auto" and host == route):
                # A named gateway keeps the whole budget, as when it was the only
                # candidate; the others take over when it fails early.
                allowed = remaining
            else:
                # Divide the remaining operation budget between the reachable gateways
                # that still need an attempt. A slow gateway cannot starve the next one,
                # while a fast failure leaves the later gateways more time.
                allowed = remaining / waiting
            try:
                output = self.ssh(host, command, stdin=stdin, timeout=allowed)
            except GatewayUnreachable as error:
                self._unreachable_until[host] = time.monotonic() + TIMEOUTS.unreachable_gateway_seconds
                errors.append(str(error))
            except ClusterError as error:
                errors.append(str(error))
            else:
                self._unreachable_until.pop(host, None)
                return host, output
        raise ClusterError("; ".join(errors) or "No SSH gateway is available")

    def resolve_gateway(self, gateway: str = "auto") -> str:
        command = (
            f"export PATH={SLURM_BIN}:$PATH; "
            "command -v sbatch >/dev/null && command -v squeue >/dev/null && "
            "command -v sacct >/dev/null"
        )
        host, _ = self.run_with_fallback(command, gateway)
        return host

    def initialize_personal_workspace(self, root: str, owner_id: str, gateway: str = "auto") -> str:
        from .workspace_storage import validate_work_root

        root = validate_work_root(root)
        script = Path(__file__).with_name("workspace_storage_remote.py").read_text()
        command = f"python3 - {shlex.quote(root)} {shlex.quote(owner_id)}"
        host, output = self.run_with_fallback(command, gateway, stdin=script)
        try:
            result = json.loads(output)
            if result != {"work_root": root, "owner_id": owner_id, "ready": True}:
                raise ValueError("Unexpected initialization result")
        except (ValueError, TypeError) as error:
            raise ClusterError("Cluster storage initialization was not confirmed. Try again.") from error
        return host

    @staticmethod
    def _run_id(run_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
            raise ValueError("Invalid run ID")
        return run_id

    @staticmethod
    def _relative_path(relative_path: str) -> str:
        path = PurePosixPath(relative_path)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("Capsule paths must be relative and cannot contain '..'")
        return str(path)

    def _remote_path(self, path: str) -> str:
        roots = self.storage.allowed_roots() if self.storage else {WORK_ROOT}
        return validate_remote_path(path, roots)

    def write_capsule_file(
        self,
        run_id: str,
        relative_path: str,
        content: str,
        gateway: str = "auto",
    ) -> tuple[str, str]:
        run_id = self._run_id(run_id)
        relative_path = self._relative_path(relative_path)
        destination = f"{self.run_directory(run_id)}/{relative_path}"
        destination_q = shlex.quote(destination)
        parent_q = shlex.quote(str(PurePosixPath(destination).parent))
        command = (
            f"set -eu; umask 077; mkdir -p {parent_q}; "
            f"tmp=$(mktemp {parent_q}/.upload-XXXXXX); "
            "trap 'rm -f \"$tmp\"' EXIT; cat > \"$tmp\"; "
            f"mv \"$tmp\" {destination_q}; trap - EXIT"
        )
        host, _ = self.run_with_fallback(command, gateway, stdin=content)
        return host, destination

    def write_capsule_files(
        self,
        run_id: str,
        files: Mapping[str, str],
        gateway: str = "auto",
        *,
        immutable: bool = False,
    ) -> tuple[str, dict[str, str]]:
        """Upload a frozen capsule in one connection, verifying every file."""
        run_id = self._run_id(run_id)
        if not files:
            raise ValueError("A capsule upload must contain at least one file")
        contents = {}
        for name, content in files.items():
            relative = self._relative_path(name)
            if relative in contents:
                raise ValueError("Capsule paths must be unique")
            if not isinstance(content, str):
                raise ValueError("Capsule contents must be text")
            contents[relative] = content
        root = self.run_directory(run_id)
        expected = {
            name: hashlib.sha256(content.encode()).hexdigest()
            for name, content in contents.items()
        }
        script = r"""
import hashlib, json, os, pathlib, sys, tempfile
request = json.load(sys.stdin)
root = pathlib.Path(request["root"])
if not root.is_absolute():
    raise ValueError("Capsule root must be absolute")
files = []
for name, content in request["files"].items():
    relative = pathlib.PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Invalid capsule path")
    path = root / relative
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Capsule uploads cannot follow symbolic links")
    files.append((name, path, content.encode()))
receipts = {}
for name, path, content in files:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".upload-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if request.get("immutable"):
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != content:
                    raise ValueError("An immutable capsule file already exists with different content")
        else:
            os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    receipts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps(receipts))
"""
        host, output = self.run_with_fallback(
            shlex.join(["python3", "-c", script]), gateway,
            stdin=json.dumps({"root": root, "files": contents, "immutable": immutable}),
            timeout=TIMEOUTS.capsule_upload_seconds,
        )
        try:
            if json.loads(output) != expected:
                raise ValueError("Checksum receipt differs from the frozen capsule")
        except (ValueError, TypeError) as error:
            raise ClusterError("Cluster capsule upload verification failed") from error
        return host, {name: f"{root}/{name}" for name in contents}

    def remove_capsule_file(
        self,
        run_id: str,
        relative_path: str,
        gateway: str = "auto",
    ) -> tuple[str, str]:
        run_id = self._run_id(run_id)
        relative_path = self._relative_path(relative_path)
        destination = f"{self.run_directory(run_id)}/{relative_path}"
        host, _ = self.run_with_fallback(
            f"set -eu; rm -f -- {shlex.quote(destination)}",
            gateway,
            timeout=TIMEOUTS.short_command_seconds,
        )
        return host, destination

    def node_names(self, gateway: str = "auto") -> list[str]:
        cached = self._node_inventory_cache.get(gateway)
        if cached and time.monotonic() - cached[0] < _NODE_INVENTORY_TTL_SECONDS:
            return list(cached[1])
        _, output = self.run_with_fallback(
            f"export PATH={SLURM_BIN}:$PATH; sinfo -N -h -o '%N'", gateway,
            timeout=TIMEOUTS.short_command_seconds,
        )
        names = sorted(set(output.split()))
        if not names or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in names):
            raise ClusterError("Slurm returned an invalid node inventory")
        self._node_inventory_cache[gateway] = (time.monotonic(), names)
        return names

    def test_script(self, script: str, gateway: str = "auto") -> tuple[str, str]:
        deadline = TIMEOUTS.validation_tool_seconds
        command = (
            f"set -eu; export PATH={SLURM_BIN}:$PATH; umask 077; "
            "tmp=$(mktemp /tmp/skynet-test-XXXXXX.sbatch); "
            "trap 'rm -f \"$tmp\"' EXIT; cat > \"$tmp\"; bash -n \"$tmp\"; "
            "if command -v timeout >/dev/null 2>&1; then "
            f"set +e; output=$(timeout {deadline}s sbatch --test-only \"$tmp\" 2>&1); rc=$?; set -e; "
            f"if test $rc -eq 124; then printf '%s\\n' 'Slurm validation timed out after {deadline}s. No job was submitted; retry when the controller responds.' >&2; exit 124; "
            "elif test $rc -ne 0; then "
            "case \"$output\" in "
            "*'allocation failure: Zero Bytes were transmitted or received'*) "
            "printf '%s\\n' 'Slurm allocation forecast unavailable; submission will validate the request.' ;; "
            "*) printf '%s\\n' \"$output\" >&2; exit $rc ;; esac; "
            "else printf '%s\\n' \"$output\"; fi; "
            "else sbatch --test-only \"$tmp\"; fi"
        )
        return self.run_with_fallback(command, gateway, stdin=script, timeout=TIMEOUTS.validation_seconds)

    @classmethod
    def _submission_token(cls, submission_key: str) -> str:
        key = cls._run_id(submission_key)
        return f"skynet-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]}"

    @staticmethod
    def _submission_job_id(output: str) -> tuple[str, str]:
        value = output.strip()
        if not value:
            raise ClusterError("sbatch returned no job ID")
        raw_job_id = value.splitlines()[-1].strip()
        job_id = raw_job_id.split(";", 1)[0]
        if not re.fullmatch(r"\d+(?:_[0-9]+)?", job_id):
            raise ClusterError(f"Unexpected sbatch response: {raw_job_id}")
        return job_id, raw_job_id

    def recover_submission(
        self,
        run_id: str,
        submission_key: str,
        gateway: str = "auto",
    ) -> Submission | None:
        """Recover a job ID from an atomic receipt or its unique Slurm comment."""

        run_id = self._run_id(run_id)
        token = self._submission_token(submission_key)
        run_directory = self.run_directory(run_id)
        script_path = f"{run_directory}/job.sbatch"
        receipt_path = f"{run_directory}/submissions/{token}.jobid"
        host, receipt = self.run_with_fallback(
            f"if test -s {shlex.quote(receipt_path)}; then cat {shlex.quote(receipt_path)}; fi",
            gateway,
            timeout=TIMEOUTS.short_command_seconds,
        )
        job_id: str | None = None
        raw_job_id: str | None = None
        if receipt.strip():
            try:
                job_id, raw_job_id = self._submission_job_id(receipt)
            except ClusterError:
                job_id = raw_job_id = None

        if job_id is None:
            lookup = f'''set -eu
export PATH={SLURM_BIN}:$PATH
token={shlex.quote(token)}
since=$(date -d '{_ACCOUNTING_LOOKBACK_DAYS} days ago' +%Y-%m-%d 2>/dev/null || date +%Y-%m-%d)
LC_ALL=C squeue -h -u "$USER" -o '%i|%k'
LC_ALL=C timeout {TIMEOUTS.slurm_tool_seconds}s sacct -X -n -P -S "$since" -o JobIDRaw,Comment%128 \
  || printf '%s\n' {ACCOUNTING_UNAVAILABLE_MARKER}
'''
            host, output = self.run_with_fallback(lookup, gateway, timeout=TIMEOUTS.recovery_lookup_seconds)
            matches: dict[str, str] = {}
            for line in output.splitlines():
                candidate, separator, comment = line.partition("|")
                candidate = candidate.strip()
                if (
                    separator
                    and comment.strip() == token
                    and re.fullmatch(r"\d+(?:_[0-9]+)?", candidate)
                ):
                    matches[candidate] = candidate
            if len(matches) > 1:
                raise ClusterError(
                    f"Multiple Slurm jobs carry submission token {token}: "
                    + ", ".join(sorted(matches))
                )
            if not matches:
                if ACCOUNTING_UNAVAILABLE_MARKER in output:
                    # Only accounting remembers a job that already left the queue.
                    raise ClusterError(
                        f"{host}: Slurm accounting is unavailable; an accepted "
                        "submission cannot be ruled out"
                    )
                return None
            job_id = next(iter(matches))
            raw_job_id = job_id

        archived_script_path = f"{run_directory}/attempts/{job_id}/job.sbatch"
        try:
            self.ssh(
                host,
                f"set -eu; mkdir -p {shlex.quote(str(PurePosixPath(archived_script_path).parent))}; "
                f"test -f {shlex.quote(archived_script_path)} || "
                f"cp -f {shlex.quote(script_path)} {shlex.quote(archived_script_path)}",
                timeout=TIMEOUTS.short_command_seconds,
            )
        except ClusterError:
            archived_script_path = script_path
        return Submission(
            job_id,
            raw_job_id or job_id,
            host,
            archived_script_path,
            run_directory,
            recovered=True,
        )

    def submit_script(
        self,
        script: str,
        run_id: str,
        gateway: str = "auto",
        *,
        submission_key: str | None = None,
        forwarded_environment: Mapping[str, str] | None = None,
    ) -> Submission:
        """Submit once and recover an accepted job after a lost SSH response."""

        run_id = self._run_id(run_id)
        environment_prefix, export_file_payload = _sbatch_environment_options(
            script, forwarded_environment
        )
        export_option = (
            ' --export-file="$export_file"' if export_file_payload else ""
        )
        submission_key = submission_key or run_id
        token = self._submission_token(submission_key)
        gateway = self.gateway_for(gateway)
        # A named gateway is used as given, with no extra round trip, unless it
        # has just refused a connection; then a reachable one is resolved.
        host = (
            self.resolve_gateway(gateway)
            if gateway == "auto" or self._unreachable(gateway)
            else gateway
        )
        run_directory = self.run_directory(run_id)
        script_path = f"{run_directory}/job.sbatch"
        receipt_directory = f"{run_directory}/submissions"
        receipt_path = f"{receipt_directory}/{token}.jobid"
        error_path = f"{receipt_directory}/{token}.error"
        claim_directory = f"{receipt_directory}/{token}.claim"
        submitted_script_path = f"{receipt_directory}/{token}.sbatch"
        script_sha256 = hashlib.sha256(script.encode("utf-8")).hexdigest()
        request_digest = hashlib.sha256()
        request_digest.update(script.encode("utf-8"))
        request_digest.update(b"\0")
        request_digest.update(environment_prefix.encode("utf-8"))
        request_digest.update(export_file_payload.encode("ascii"))
        request_sha256 = request_digest.hexdigest()
        request_path = f"{receipt_directory}/{token}.request-sha256"
        submit_worker = f'''#!/bin/bash
set -u
umask 077
worker_path="$0"
export_file=
submit_error=
trap 'rm -f "$worker_path" "$export_file" "$submit_error"' EXIT
if ! mkdir {shlex.quote(claim_directory)} 2>/dev/null; then
  exit 0
fi
if test -n {shlex.quote(export_file_payload)}; then
  export_file=$(mktemp "${{TMPDIR:-/tmp}}/skynet-export-XXXXXX")
  printf '%s' {shlex.quote(export_file_payload)} | base64 --decode > "$export_file"
fi
submit_error=$(mktemp "${{TMPDIR:-/tmp}}/skynet-submit-error-XXXXXX")
set +e
raw_id=$({environment_prefix}sbatch --parsable{export_option} --comment={shlex.quote(token)} {shlex.quote(submitted_script_path)} 2>"$submit_error")
submit_rc=$?
set -e
if test "$submit_rc" -ne 0; then
  cat "$submit_error" > {shlex.quote(error_path)}
  printf '%s\n' '__SKYNET_SBATCH_REJECTED__' >> {shlex.quote(error_path)}
  exit "$submit_rc"
fi
job_id=${{raw_id%%;*}}
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  printf '%s\n' "Unexpected sbatch response: $raw_id" '__SKYNET_SBATCH_REJECTED__' > {shlex.quote(error_path)}
  exit 70
fi
printf '%s\n' "$raw_id" > {shlex.quote(receipt_path)}
cp -f {shlex.quote(submitted_script_path)} {shlex.quote(script_path)} || true
attempt_dir={shlex.quote(run_directory)}/attempts/$job_id
mkdir -p "$attempt_dir" || true
cp -f {shlex.quote(submitted_script_path)} "$attempt_dir/job.sbatch" || true
'''
        command = f'''set -eu
export PATH={SLURM_BIN}:$PATH
umask 077
mkdir -p {shlex.quote(run_directory)} {shlex.quote(receipt_directory)} {shlex.quote(str(PurePosixPath(run_directory).parents[2] / "logs"))} {shlex.quote(str(PurePosixPath(run_directory).parents[2] / "workspace"))}
if test -s {shlex.quote(receipt_path)}; then
  cat {shlex.quote(receipt_path)}
  exit 0
fi
upload_tmp=$(mktemp {shlex.quote(receipt_directory)}/.upload-XXXXXX.sbatch)
request_tmp=$(mktemp {shlex.quote(receipt_directory)}/.request-XXXXXX)
worker_tmp=$(mktemp {shlex.quote(receipt_directory)}/.worker-XXXXXX.sh)
trap 'rm -f "$upload_tmp" "$request_tmp" "$worker_tmp"' EXIT
cat > "$upload_tmp"
if test "$(sha256sum "$upload_tmp" | awk '{{print $1}}')" != {shlex.quote(script_sha256)}; then
  printf '%s\n' 'Uploaded submission script failed its local SHA-256 check' '__SKYNET_SUBMISSION_CONFLICT__' >&2
  exit 65
fi
if ! ln "$upload_tmp" {shlex.quote(submitted_script_path)} 2>/dev/null; then
  if test "$(sha256sum {shlex.quote(submitted_script_path)} | awk '{{print $1}}')" != {shlex.quote(script_sha256)}; then
    printf '%s\n' 'Submission key was reused with a different script' '__SKYNET_SUBMISSION_CONFLICT__' >&2
    exit 65
  fi
fi
printf '%s\n' {shlex.quote(request_sha256)} > "$request_tmp"
if ! ln "$request_tmp" {shlex.quote(request_path)} 2>/dev/null; then
  if test "$(cat {shlex.quote(request_path)})" != {shlex.quote(request_sha256)}; then
    printf '%s\n' 'Submission key was reused with different environment options' '__SKYNET_SUBMISSION_CONFLICT__' >&2
    exit 65
  fi
fi
printf '%s' {shlex.quote(submit_worker)} > "$worker_tmp"
chmod 700 "$worker_tmp"
if command -v setsid >/dev/null 2>&1; then
  nohup setsid --fork /bin/bash "$worker_tmp" </dev/null >/dev/null 2>&1 &
else
  nohup /bin/bash "$worker_tmp" </dev/null >/dev/null 2>&1 &
fi
worker_tmp=
deadline=$((SECONDS + {TIMEOUTS.submission_receipt_wait_seconds}))
while true; do
  if test -s {shlex.quote(receipt_path)}; then
    cat {shlex.quote(receipt_path)}
    exit 0
  fi
  if test -s {shlex.quote(error_path)}; then
    cat {shlex.quote(error_path)} >&2
    exit 70
  fi
  if test "$SECONDS" -ge "$deadline"; then
    printf '%s\n' '__SKYNET_SUBMISSION_PENDING__' >&2
    exit 75
  fi
  sleep {_RECEIPT_POLL_SECONDS}
done
'''
        try:
            output = self.ssh(host, command, stdin=script, timeout=TIMEOUTS.submission_seconds)
            job_id, raw_job_id = self._submission_job_id(output)
        except ClusterError as error:
            if "__SKYNET_SUBMISSION_CONFLICT__" in str(error):
                message = str(error).replace(
                    "__SKYNET_SUBMISSION_CONFLICT__", ""
                ).strip()
                raise ClusterError(message) from error
            for delay in _RECOVERY_RETRY_DELAYS:
                if delay:
                    time.sleep(delay)
                try:
                    recovered = self.recover_submission(
                        run_id, submission_key, host
                    )
                except ClusterError:
                    recovered = None
                if recovered is not None:
                    return recovered
            message = str(error).replace("__SKYNET_SBATCH_REJECTED__", "").strip()
            if "__SKYNET_SBATCH_REJECTED__" in str(error):
                raise ClusterError(message) from error
            raise SubmissionOutcomeUnknown(
                f"{message}; Slurm submission acknowledgement was lost and no "
                f"matching receipt or job token is currently visible"
            ) from error
        archived_script_path = f"{run_directory}/attempts/{job_id}/job.sbatch"
        return Submission(job_id, raw_job_id, host, archived_script_path, run_directory)

    def job_statuses(
        self, job_ids: Iterable[str], gateway: str = "auto"
    ) -> tuple[str, dict[str, dict[str, str]]]:
        snapshot = self.job_status_snapshot(job_ids, gateway)
        return snapshot.gateway, snapshot.statuses

    def job_status_snapshot(
        self, job_ids: Iterable[str], gateway: str = "auto"
    ) -> JobStatusSnapshot:
        """Ask accounting and the live controller in one connection.

        Accounting alone cannot be trusted with a job's fate: its daemon can be
        down or can drop updates, while the controller forgets a finished job
        after MinJobAge. A finished accounting record is used as is; otherwise
        the controller's record of the job is; an unfinished accounting record
        stands only when the controller could not be asked.
        """
        unique = tuple(dict.fromkeys(str(job_id) for job_id in job_ids if str(job_id)))
        if not unique:
            return JobStatusSnapshot(self.resolve_gateway(gateway), {})
        if any(not re.fullmatch(r"\d+(?:_[0-9]+)?", job_id) for job_id in unique):
            raise ValueError("Invalid Slurm job ID")
        jobs = shlex.quote(",".join(unique))
        accounting_format = ",".join(spec for _, spec in _ACCOUNTING_FIELDS)
        controller_format = ",".join(f"{spec}:|" for _, spec in _CONTROLLER_FIELDS)
        command = f'''export PATH={SLURM_BIN}:$PATH
accounting=$(LC_ALL=C SLURM_TIME_FORMAT=%s timeout {TIMEOUTS.slurm_tool_seconds}s sacct -X -n -P -j {jobs} -o {accounting_format} 2>&1)
accounting_rc=$?
controller=$(LC_ALL=C SLURM_TIME_FORMAT=%s timeout {TIMEOUTS.slurm_tool_seconds}s squeue -h --states=all -j {jobs} -O {shlex.quote(controller_format)} 2>&1)
controller_rc=$?
# Asked about a single job it has forgotten, squeue fails; that is an answer.
if test "$controller_rc" -ne 0; then
  case "$controller" in *'Invalid job id specified'*) controller_rc=0; controller= ;; esac
fi
if test "$accounting_rc" -ne 0 && test "$controller_rc" -ne 0; then
  printf '%s\\n' "$accounting" "$controller" >&2
  exit 1
fi
printf '%s %s\\n%s\\n' {_ACCOUNTING_MARKER} "$accounting_rc" "$accounting" {_CONTROLLER_MARKER} "$controller_rc" "$controller"
'''
        host, output = self.run_with_fallback(command, gateway, timeout=TIMEOUTS.job_status_seconds)
        accounting_text, separator, controller_text = output.partition(_CONTROLLER_MARKER)
        accounting_header, _, accounting_body = accounting_text.partition("\n")
        controller_header, _, controller_body = controller_text.partition("\n")
        if (not separator or not accounting_header.startswith(_ACCOUNTING_MARKER)):
            raise ClusterError(f"{host}: Slurm status response is invalid")
        accounting_ok = accounting_header.split()[-1] == "0"
        controller_ok = controller_header.strip() == "0"
        accounting = self._accounting_records(accounting_body) if accounting_ok else {}
        controller: dict[str, dict[str, str]] = {}
        if controller_ok:
            try:
                controller = self._controller_records(controller_body)
            except ValueError as error:
                # An unreadable row must not read as "the controller no longer lists the job".
                controller_ok, controller_body = False, str(error)
        statuses: dict[str, dict[str, str]] = {}
        for job_id in unique:
            recorded, listed = accounting.get(job_id), controller.get(job_id)
            if listed and listed["State"] in ACTIVE_STATES:
                # A requeued job is live again whatever accounting last recorded.
                statuses[job_id] = listed
            elif recorded and recorded["State"] not in ACTIVE_STATES:
                statuses[job_id] = recorded
            elif listed:
                statuses[job_id] = listed
            elif recorded and not controller_ok:
                statuses[job_id] = recorded
        return JobStatusSnapshot(
            host,
            statuses,
            accounting_error=None if accounting_ok else self._tool_error(accounting_body, "sacct"),
            controller_error=None if controller_ok else self._tool_error(controller_body, "squeue"),
        )

    @staticmethod
    def _tool_error(output: str, tool: str) -> str:
        detail = next((line.strip() for line in output.splitlines() if line.strip()), "")
        return (detail or f"{tool} did not answer")[:300]

    @staticmethod
    def _accounting_records(output: str) -> dict[str, dict[str, str]]:
        records: dict[str, dict[str, str]] = {}
        names = tuple(name for name, _ in _ACCOUNTING_FIELDS)
        for line in output.splitlines():
            values = line.rstrip("|").split("|")
            # Restart telemetry is optional: an absent accounting value must
            # remain unavailable rather than looking like a measured zero.
            if len(values) not in {len(names), len(names) - 1}:
                continue
            record = dict(zip(names[:len(values)], values, strict=True))
            restarts = record.pop("Restarts", "").strip()
            if re.fullmatch(r"[0-9]+", restarts):
                record["Restarts"] = str(int(restarts))
            job_id = record.pop("JobIDRaw")
            raw_state = record["State"].strip()
            state_match = re.match(r"[A-Za-z_]+", raw_state)
            record["StateRaw"] = raw_state
            record["State"] = state_match.group(0).upper() if state_match else "UNKNOWN"
            cancelled_by = re.match(r"CANCELLED\s+by\s+([^\s+]+)", raw_state, re.IGNORECASE)
            if cancelled_by:
                record["CancelledBy"] = cancelled_by.group(1)
            record["Source"] = "accounting"
            records[job_id] = record
        return records

    @staticmethod
    def _controller_records(output: str) -> dict[str, dict[str, str]]:
        records: dict[str, dict[str, str]] = {}
        names = tuple(name for name, _ in _CONTROLLER_FIELDS)
        for line in output.splitlines():
            if not line.strip():
                continue
            values = [value.strip() for value in line.rstrip().removesuffix("|").split("|")]
            record = dict(zip(names, values)) if len(values) == len(names) else {}
            job_id = record.pop("JobID", "")
            state_match = re.match(r"[A-Za-z_]+", record.get("State", ""))
            if not state_match or not re.fullmatch(r"\d+(?:_[0-9]+)?", job_id):
                raise ValueError(f"squeue printed an unexpected row: {line.strip()[:200]}")
            record["StateRaw"] = record["State"]
            state = state_match.group(0).upper()
            state = record["State"] = _CONTROLLER_STATES.get(state, state)
            restarts = record.pop("Restarts")
            if re.fullmatch(r"[0-9]+", restarts):
                record["Restarts"] = str(int(restarts))
            if record["NodeList"] in {"", "(null)", "N/A"}:
                record["NodeList"] = "None assigned"  # As accounting words it.
            # The controller forecasts a queued job's start and a running job's end.
            if state in _UNSTARTED_STATES:
                record.pop("Start")
            if state in ACTIVE_STATES or state in _UNSTARTED_STATES:
                record.pop("End")
            record["Source"] = "controller"
            records[job_id] = record
        return records

    def cancel(self, job_id: str, gateway: str = "auto") -> str:
        if not re.fullmatch(r"\d+(?:_[0-9]+)?", job_id):
            raise ValueError("Invalid Slurm job ID")
        host = self.resolve_gateway(gateway)
        self.ssh(host, f"export PATH={SLURM_BIN}:$PATH; scancel {shlex.quote(job_id)}")
        return host

    def read_file(
        self,
        path: str,
        gateway: str = "auto",
        *,
        max_bytes: int = 20_000_000,
    ) -> tuple[str, str]:
        path = self._remote_path(path)
        max_bytes = max(1024, min(max_bytes, 100_000_000))
        path_q = shlex.quote(path)
        command = (
            f"test -f {path_q} || exit 44; "
            f"size=$(stat -c %s {path_q}); "
            f"test \"$size\" -le {max_bytes} || {{ echo 'file exceeds read limit' >&2; exit 45; }}; "
            f"cat {path_q}"
        )
        return self.run_with_fallback(command, gateway)

    def read_optional_file(
        self,
        path: str,
        gateway: str = "auto",
        *,
        max_bytes: int = 20_000_000,
    ) -> tuple[str, str | None]:
        """Return None only when successful SSH confirms the file is absent.

        An explicit response header distinguishes absence from an empty file.
        Connection errors and failed reads retain normal ClusterError semantics.
        """
        path = self._remote_path(path)
        max_bytes = max(1024, min(max_bytes, 100_000_000))
        path_q = shlex.quote(path)
        command = (
            f"if test -f {path_q}; then "
            f"size=$(stat -c %s {path_q}) || exit 46; "
            f"test \"$size\" -le {max_bytes} || {{ echo 'file exceeds read limit' >&2; exit 45; }}; "
            "printf '%s\\n' SKYNET_FILE_PRESENT; "
            f"cat {path_q}; "
            "else printf '%s\\n' SKYNET_FILE_MISSING; fi"
        )
        host, output = self.run_with_fallback(command, gateway)
        header, separator, content = output.partition("\n")
        if separator and header == "SKYNET_FILE_PRESENT":
            return host, content
        if separator and header == "SKYNET_FILE_MISSING" and not content:
            return host, None
        raise ClusterError("Remote optional-file response is invalid")

    def file_size(self, path: str, gateway: str = "auto") -> tuple[str, int]:
        """Resolve a remote regular file and return its gateway and byte size."""

        path = self._remote_path(path)
        path_q = shlex.quote(path)
        host, output = self.run_with_fallback(
            f"test -f {path_q} || exit 44; stat -c %s {path_q}",
            gateway,
            timeout=TIMEOUTS.read_seconds,
        )
        try:
            size = int(output.strip())
        except ValueError as error:
            raise ClusterError(f"{host}: remote file returned an invalid size") from error
        if size < 0:
            raise ClusterError(f"{host}: remote file returned an invalid size")
        return host, size

    def stream_file_range(
        self,
        path: str,
        host: str,
        *,
        start: int,
        end: int,
        chunk_size: int = 1_048_576,
        cancel_event=None,
    ) -> Iterable[bytes]:
        """Stream an inclusive byte range from a previously resolved gateway."""

        path = self._remote_path(path)
        if host not in self.hosts:
            raise ValueError(f"Unknown SSH gateway: {host}")
        if start < 0 or end < start:
            raise ValueError("Invalid remote byte range")
        count = end - start + 1
        command = (
            "LC_ALL=C dd "
            f"if={shlex.quote(path)} iflag=skip_bytes,count_bytes "
            f"skip={start} count={count} status=none"
        )
        process = subprocess.Popen(
            ssh_argv(host, command),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert process.stdout is not None
        remaining = count
        completed = False
        try:
            while remaining:
                if cancel_event is not None:
                    if cancel_event.is_set():
                        return
                    if not select.select([process.stdout], [], [], 0.1)[0]:
                        continue
                    block = os.read(process.stdout.fileno(), min(chunk_size, remaining))
                else:
                    block = process.stdout.read(min(chunk_size, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block
            if cancel_event is not None:
                deadline = time.monotonic() + _STREAM_CLOSE_SECONDS
                while process.poll() is None:
                    if cancel_event.wait(0.1):
                        return
                    if time.monotonic() >= deadline:
                        raise ClusterError(f"{host}: remote video stream did not close")
            return_code = process.wait(timeout=_STREAM_CLOSE_SECONDS)
            completed = True
            if return_code != 0 or remaining:
                raise ClusterError(f"{host}: remote video stream ended unexpectedly")
        finally:
            process.stdout.close()
            if not completed and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=_PROCESS_EXIT_SECONDS)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=_PROCESS_EXIT_SECONDS)

    def read_log(
        self,
        path: str,
        gateway: str = "auto",
        *,
        lines: int = 500,
        max_bytes: int = 1_000_000,
        contains: str | None = None,
        execution_boundary: Mapping[str, object] | None = None,
    ) -> tuple[str, str]:
        path = self._remote_path(path)
        lines = max(1, min(lines, 5000))
        max_bytes = max(1024, min(max_bytes, 5_000_000))
        if execution_boundary is not None:
            # The same JSONL is reused on resume. Read only bytes written since
            # this execution's launch, even after a server or scheduler restart.
            boundary = dict(execution_boundary)
            boundary["boundary_path"] = self._remote_path(str(boundary["boundary_path"]))
            arguments = {**boundary, "path": path, "lines": lines,
                         "max_bytes": max_bytes, "contains": contains}
            script = Path(__file__).with_name("training_progress_log.py").read_text()
            command = "python3 -c " + shlex.quote(script) + " " + shlex.quote(json.dumps(arguments))
            return self.run_with_fallback(command, gateway, timeout=TIMEOUTS.read_seconds)
        reader = f"tail -n {lines} {shlex.quote(path)}"
        if contains is not None:
            if any(char in contains for char in ("\x00", "\n", "\r")):
                raise ValueError("log filter must be one line")
            reader = f"grep -F -- {shlex.quote(contains)} {shlex.quote(path)} | tail -n {lines}"
        command = (
            f"if test -f {shlex.quote(path)}; then "
            f"{reader} | tail -c {max_bytes}; "
            "else printf '%s\\n' SKYNET_LOG_NOT_READY >&2; exit 44; fi"
        )
        return self.run_with_fallback(command, gateway, timeout=TIMEOUTS.read_seconds)


__all__ = [
    "ACTIVE_STATES",
    "ClusterClient",
    "ClusterError",
    "DEFAULT_GATEWAY",
    "GatewayUnreachable",
    "HOME_ROOT",
    "JobStatusSnapshot",
    "SLURM_BIN",
    "Submission",
    "SubmissionOutcomeUnknown",
    "TIMEOUTS",
    "WORK_ROOT",
]
