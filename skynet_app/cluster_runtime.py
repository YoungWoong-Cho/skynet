from __future__ import annotations

import base64
import hashlib
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, Mapping, Sequence

from .cluster_config import CLUSTER


HOME_ROOT = CLUSTER.paths.home_root
WORK_ROOT = CLUSTER.paths.work_root
SLURM_BIN = CLUSTER.commands.slurm_bin


class ClusterError(RuntimeError):
    """A gateway or Slurm operation failed."""


class SubmissionOutcomeUnknown(ClusterError):
    """Slurm may have accepted a job whose SSH acknowledgement was lost."""


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


def _strip_known_ssh_banner(value: str) -> str:
    """Remove the Georgia Tech login notice without hiding command stderr."""

    lines = value.splitlines()
    while True:
        notice = next(
            (
                index
                for index, line in enumerate(lines)
                if "Georgia Institute of Technology - Terms of Use" in line
            ),
            None,
        )
        if notice is None:
            break
        border = re.compile(r"^\s*\*{20,}\s*$")
        start = next(
            (index for index in range(notice, -1, -1) if border.match(lines[index])),
            notice,
        )
        end = next(
            (index for index in range(notice + 1, len(lines)) if border.match(lines[index])),
            notice,
        )
        del lines[start : end + 1]
    return "\n".join(lines).strip()


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

    def candidates(self, gateway: str) -> tuple[str, ...]:
        if gateway == "auto":
            return self.hosts
        if gateway not in self.hosts:
            raise ValueError(f"Unknown SSH gateway: {gateway}")
        return (gateway,)

    def ssh(
        self,
        host: str,
        command: str,
        *,
        stdin: str | None = None,
        timeout: float = 30,
    ) -> str:
        try:
            process = subprocess.run(
                [
                    "ssh",
                    "-T",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=6",
                    "-o",
                    "ServerAliveInterval=5",
                    "-o",
                    "ServerAliveCountMax=1",
                    host,
                    command,
                ],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ClusterError(f"{host}: SSH operation timed out") from error
        if process.returncode != 0:
            stderr = _strip_known_ssh_banner(process.stderr or "")
            detail = (stderr or process.stdout or f"SSH command failed with exit code {process.returncode}").strip()
            raise ClusterError(f"{host}: {detail}")
        return process.stdout

    def run_with_fallback(
        self,
        command: str,
        gateway: str = "auto",
        *,
        stdin: str | None = None,
        timeout: float = 30,
    ) -> tuple[str, str]:
        if timeout <= 0:
            raise ValueError("SSH operation timeout must be positive")

        candidates = self.candidates(gateway)
        deadline = time.monotonic() + timeout
        errors: list[str] = []
        for index, host in enumerate(candidates):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                errors.append(f"{host}: SSH operation budget exhausted before attempt")
                continue

            # Divide the remaining operation budget between all gateways that
            # still need an attempt. A slow gateway cannot starve the next one,
            # while a fast failure leaves the later gateways more time.
            attempt_timeout = remaining / (len(candidates) - index)
            try:
                return host, self.ssh(host, command, stdin=stdin, timeout=attempt_timeout)
            except ClusterError as error:
                errors.append(str(error))
        raise ClusterError("; ".join(errors) or "No SSH gateway is available")

    def resolve_gateway(self, gateway: str = "auto") -> str:
        command = (
            f"export PATH={SLURM_BIN}:$PATH; "
            "command -v sbatch >/dev/null && command -v squeue >/dev/null && "
            "command -v sacct >/dev/null"
        )
        host, _ = self.run_with_fallback(command, gateway, timeout=15)
        return host

    def initialize_workspace(self, gateway: str = "auto") -> str:
        directories = (
            "workspace repos repos/shared envs datasets artifacts logs jobs eval/catalogs eval/datasets "
            "eval/assets eval/runs mlflow/db mlflow/artifacts .cache/uv "
            ".cache/huggingface .cache/torch"
        )
        command = (
            "set -eu; umask 077; "
            + " ".join(f"mkdir -p {shlex.quote(f'{WORK_ROOT}/{item}')} ;" for item in directories.split())
        )
        host, _ = self.run_with_fallback(command, gateway)
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

    @staticmethod
    def _remote_path(path: str) -> str:
        candidate = PurePosixPath(path)
        root = PurePosixPath(WORK_ROOT)
        if not candidate.is_absolute() or candidate != root and root not in candidate.parents:
            raise ValueError(f"Remote path must be below {WORK_ROOT}")
        if ".." in candidate.parts:
            raise ValueError("Remote path cannot contain '..'")
        return str(candidate)

    def write_capsule_file(
        self,
        run_id: str,
        relative_path: str,
        content: str,
        gateway: str = "auto",
    ) -> tuple[str, str]:
        run_id = self._run_id(run_id)
        relative_path = self._relative_path(relative_path)
        destination = f"{WORK_ROOT}/jobs/runs/{run_id}/{relative_path}"
        destination_q = shlex.quote(destination)
        parent_q = shlex.quote(str(PurePosixPath(destination).parent))
        command = (
            f"set -eu; umask 077; mkdir -p {parent_q}; "
            f"tmp=$(mktemp {parent_q}/.upload-XXXXXX); "
            "trap 'rm -f \"$tmp\"' EXIT; cat > \"$tmp\"; "
            f"mv \"$tmp\" {destination_q}; trap - EXIT"
        )
        host, _ = self.run_with_fallback(command, gateway, stdin=content, timeout=30)
        return host, destination

    def remove_capsule_file(
        self,
        run_id: str,
        relative_path: str,
        gateway: str = "auto",
    ) -> tuple[str, str]:
        run_id = self._run_id(run_id)
        relative_path = self._relative_path(relative_path)
        destination = f"{WORK_ROOT}/jobs/runs/{run_id}/{relative_path}"
        host, _ = self.run_with_fallback(
            f"set -eu; rm -f -- {shlex.quote(destination)}",
            gateway,
            timeout=15,
        )
        return host, destination

    def test_script(self, script: str, gateway: str = "auto") -> tuple[str, str]:
        command = (
            f"set -eu; export PATH={SLURM_BIN}:$PATH; umask 077; "
            "tmp=$(mktemp /tmp/skynet-test-XXXXXX.sbatch); "
            "trap 'rm -f \"$tmp\"' EXIT; cat > \"$tmp\"; bash -n \"$tmp\"; "
            "if command -v timeout >/dev/null 2>&1; then "
            "set +e; output=$(timeout 10s sbatch --test-only \"$tmp\" 2>&1); rc=$?; set -e; "
            "if test $rc -eq 124; then printf '%s\\n' 'Slurm validation timed out after 10s. No job was submitted; retry when the controller responds.' >&2; exit 124; "
            "elif test $rc -ne 0; then printf '%s\\n' \"$output\" >&2; exit $rc; "
            "else printf '%s\\n' \"$output\"; fi; "
            "else sbatch --test-only \"$tmp\"; fi"
        )
        return self.run_with_fallback(command, gateway, stdin=script, timeout=25)

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
        run_directory = f"{WORK_ROOT}/jobs/runs/{run_id}"
        script_path = f"{run_directory}/job.sbatch"
        receipt_path = f"{run_directory}/submissions/{token}.jobid"
        host, receipt = self.run_with_fallback(
            f"if test -s {shlex.quote(receipt_path)}; then cat {shlex.quote(receipt_path)}; fi",
            gateway,
            timeout=15,
        )
        job_id: str | None = None
        raw_job_id: str | None = None
        if receipt.strip():
            try:
                job_id, raw_job_id = self._submission_job_id(receipt)
            except ClusterError:
                job_id = raw_job_id = None

        if job_id is None:
            lookup = f'''export PATH={SLURM_BIN}:$PATH
token={shlex.quote(token)}
since=$(date -d '7 days ago' +%Y-%m-%d 2>/dev/null || date +%Y-%m-%d)
{{
  LC_ALL=C squeue -h -u "$USER" -o '%i|%k' 2>/dev/null || true
  LC_ALL=C sacct -X -n -P -S "$since" -o JobIDRaw,Comment%128 2>/dev/null || true
}}
'''
            host, output = self.run_with_fallback(lookup, gateway, timeout=20)
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
                timeout=15,
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
        host = self.resolve_gateway() if gateway == "auto" else self.candidates(gateway)[0]
        run_directory = f"{WORK_ROOT}/jobs/runs/{run_id}"
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
mkdir -p {shlex.quote(run_directory)} {shlex.quote(receipt_directory)} {shlex.quote(f"{WORK_ROOT}/logs")}
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
deadline=$((SECONDS + 30))
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
  sleep 0.2
done
'''
        try:
            output = self.ssh(host, command, stdin=script, timeout=45)
            job_id, raw_job_id = self._submission_job_id(output)
        except ClusterError as error:
            if "__SKYNET_SUBMISSION_CONFLICT__" in str(error):
                message = str(error).replace(
                    "__SKYNET_SUBMISSION_CONFLICT__", ""
                ).strip()
                raise ClusterError(message) from error
            for delay in (0, 1, 2):
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
        unique = tuple(dict.fromkeys(str(job_id) for job_id in job_ids if str(job_id)))
        if not unique:
            return self.resolve_gateway(gateway), {}
        if any(not re.fullmatch(r"\d+(?:_[0-9]+)?", job_id) for job_id in unique):
            raise ValueError("Invalid Slurm job ID")
        field_specs = (
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
        )
        fields = ",".join(spec for _, spec in field_specs)
        command = (
            f"export PATH={SLURM_BIN}:$PATH; "
            "LC_ALL=C SLURM_TIME_FORMAT=%s sacct -X -n -P "
            f"-j {shlex.quote(','.join(unique))} -o {fields}"
        )
        host, output = self.run_with_fallback(command, gateway, timeout=30)
        statuses: dict[str, dict[str, str]] = {}
        names = tuple(name for name, _ in field_specs)
        for line in output.splitlines():
            values = line.rstrip("|").split("|")
            if len(values) != len(names):
                continue
            record = dict(zip(names, values, strict=True))
            job_id = record.pop("JobIDRaw")
            raw_state = record["State"].strip()
            state_match = re.match(r"[A-Za-z_]+", raw_state)
            record["StateRaw"] = raw_state
            record["State"] = state_match.group(0).upper() if state_match else "UNKNOWN"
            cancelled_by = re.match(r"CANCELLED\s+by\s+([^\s+]+)", raw_state, re.IGNORECASE)
            if cancelled_by:
                record["CancelledBy"] = cancelled_by.group(1)
            statuses[job_id] = record
        pending = [job_id for job_id, record in statuses.items() if record["State"] == "PENDING"]
        if pending:
            # Accounting often reports Reason=None for queued jobs. Ask the
            # live scheduler once for the whole batch, on the same gateway.
            queue_command = (
                f"export PATH={SLURM_BIN}:$PATH; "
                "LC_ALL=C timeout 10s squeue -h "
                f"-j {shlex.quote(','.join(pending))} -o '%i|%T|%R'"
            )
            try:
                queue_output = self.ssh(host, queue_command, timeout=15)
            except ClusterError as error:
                for job_id in pending:
                    statuses[job_id]["Reason"] = f"Live queue reason unavailable: {error}"
            else:
                for line in queue_output.splitlines():
                    fields = line.split("|", 2)
                    if len(fields) != 3:
                        continue
                    job_id, state, reason = (value.strip() for value in fields)
                    if job_id in pending and state.upper() == "PENDING" and reason:
                        statuses[job_id]["Reason"] = reason.strip("()")
        return host, statuses

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
        return self.run_with_fallback(command, gateway, timeout=30)

    def file_size(self, path: str, gateway: str = "auto") -> tuple[str, int]:
        """Resolve a remote regular file and return its gateway and byte size."""

        path = self._remote_path(path)
        path_q = shlex.quote(path)
        host, output = self.run_with_fallback(
            f"test -f {path_q} || exit 44; stat -c %s {path_q}",
            gateway,
            timeout=20,
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
            [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=6",
                "-o",
                "ServerAliveInterval=5",
                "-o",
                "ServerAliveCountMax=1",
                host,
                command,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert process.stdout is not None
        remaining = count
        completed = False
        try:
            while remaining:
                block = process.stdout.read(min(chunk_size, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block
            return_code = process.wait(timeout=10)
            completed = True
            if return_code != 0 or remaining:
                raise ClusterError(f"{host}: remote video stream ended unexpectedly")
        finally:
            process.stdout.close()
            if not completed and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)

    def read_log(
        self,
        path: str,
        gateway: str = "auto",
        *,
        lines: int = 500,
        max_bytes: int = 1_000_000,
    ) -> tuple[str, str]:
        path = self._remote_path(path)
        lines = max(1, min(lines, 5000))
        max_bytes = max(1024, min(max_bytes, 5_000_000))
        command = (
            f"if test -f {shlex.quote(path)}; then "
            f"tail -n {lines} {shlex.quote(path)} | tail -c {max_bytes}; "
            "else printf '%s\\n' SKYNET_LOG_NOT_READY >&2; exit 44; fi"
        )
        return self.run_with_fallback(command, gateway, timeout=20)


__all__ = [
    "ClusterClient",
    "ClusterError",
    "HOME_ROOT",
    "SLURM_BIN",
    "Submission",
    "SubmissionOutcomeUnknown",
    "WORK_ROOT",
]
