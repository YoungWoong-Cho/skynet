"""Durable live DexVerse sessions on a cluster or a direct SSH workstation."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shlex
import threading
import time
from uuid import uuid4

from .cluster_runtime import (
    ClusterClient,
    ClusterError,
    SubmissionOutcomeUnknown,
    WORK_ROOT,
)
from .database import canonical_json, utc_now
from .capture_processing.dexverse_runner import TASK, ROBOT, REVISION
from .live_xr_workstation import LaunchRejected, WorkstationClient, validate_profile

ROOT = Path(__file__).resolve().parents[1]
EULA = "https://developer.download.nvidia.com/cloudxr/EULA/NVIDIA_CloudXR_GA_License_without_Data_Collection_25Feb2025.pdf"
TERMINAL = {"CAPTURED", "STOPPED", "TIMED_OUT", "FAILED"}
WORKER_STATES = TERMINAL | {
    "STARTING_SERVER",
    "STARTING_SIMULATION",
    "AWAITING_HEADSET",
    "STOPPING",
}


class LiveXRService:
    def __init__(self, database, cluster=None, root=ROOT):
        self.database, self.cluster, self.root = (
            database,
            cluster or ClusterClient(),
            Path(root),
        )
        self.lock = threading.RLock()
        self.active, self.refreshing = set(), set()
        self.refreshed = {}
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="live-xr")
        with database.transaction() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS live_xr_sessions (id TEXT PRIMARY KEY, payload_json TEXT NOT NULL)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS live_xr_consent (url TEXT PRIMARY KEY, accepted_at TEXT NOT NULL)"
            )

    def consent(self, accept=False):
        with self.database.transaction() as c:
            if accept:
                c.execute(
                    "INSERT OR IGNORE INTO live_xr_consent VALUES (?,?)",
                    (EULA, utc_now()),
                )
            row = c.execute(
                "SELECT accepted_at FROM live_xr_consent WHERE url=?", (EULA,)
            ).fetchone()
        return {
            "accepted": row is not None,
            "url": EULA,
            "accepted_at": row[0] if row else None,
        }

    def list(self):
        with self.database.connection() as c:
            jobs = [
                json.loads(row[0])
                for row in c.execute("SELECT payload_json FROM live_xr_sessions")
            ]
        return [
            self.public(j)
            for j in sorted(jobs, key=lambda j: j["created_at"], reverse=True)
        ]

    @staticmethod
    def public(job):
        return {k: v for k, v in job.items() if k not in {"worker", "script"}}

    def get(self, identifier):
        with self.database.connection() as c:
            row = c.execute(
                "SELECT payload_json FROM live_xr_sessions WHERE id=?", (identifier,)
            ).fetchone()
        if row is None:
            raise KeyError("Live session not found")
        return json.loads(row[0])

    def update(self, identifier, **changes):
        with self.lock, self.database.transaction() as c:
            job = self.get(identifier)
            job.update(changes, updated_at=utc_now())
            c.execute(
                "UPDATE live_xr_sessions SET payload_json=? WHERE id=?",
                (canonical_json(job), identifier),
            )
        return self.public(job)

    def profile(self):
        config = json.loads((self.root / "config/live_xr.json").read_text())
        pipelines = json.loads(
            (self.root / "config/capture_pipelines.json").read_text()
        )["pipelines"]
        profile = next(
            (p for p in pipelines if p["key"] == config["pipeline_key"]), None
        )
        if profile is None or (
            profile["task"],
            profile["robot"],
            profile["source_revision"],
        ) != (TASK, ROBOT, REVISION):
            raise ValueError(
                "Unsupported live task, robot or source revision; configure a matching live worker first"
            )
        profile = dict(profile, execution=config.get("execution", "slurm"))
        for key in (
            "cloudxr_runtime",
            "gpu_type",
            "duration_minutes",
            "gateway",
            "work_root",
            "repository",
            "runtime",
            "memory_gb",
        ):
            if key in config:
                profile[key] = config[key]
        if profile["execution"] == "workstation":
            validate_profile(profile)
            for key in ("account", "partition", "gpu_type", "asset_bundle"):
                profile.pop(key, None)
        elif profile["execution"] == "slurm":
            self.cluster.candidates(profile["gateway"])
            for key in ("repository", "runtime", "cloudxr_runtime"):
                if not self.cluster._remote_path(profile[key]).startswith(
                    WORK_ROOT + "/"
                ):
                    raise ValueError(
                        "Live runtime paths must be inside the configured cluster workspace"
                    )
            for key in ("account", "partition", "gpu_type"):
                if not re.fullmatch(r"[A-Za-z0-9_-]+", profile[key]):
                    raise ValueError(f"Invalid live profile {key}")
        else:
            raise ValueError("Unsupported live execution backend")
        if (
            type(profile["duration_minutes"]) is not int
            or not 5 <= profile["duration_minutes"] <= 60
        ):
            raise ValueError("Live session duration must be 5–60 minutes")
        profile.update(
            provider="dexverse-cloudxr",
            display_name="Live DexVerse · Shadow right hand · Pick up stick",
            description="Immersive hand-tracked manipulation in the DexVerse scene",
        )
        return profile

    def transport(self, job):
        if job["profile"].get("execution", "slurm") == "workstation":
            return WorkstationClient(job["profile"])
        return self.cluster

    def create(self, accepted_license=False):
        if not self.consent(accepted_license)["accepted"]:
            raise ValueError(
                "Accept the NVIDIA CloudXR license before starting a session"
            )
        profile = dict(
            self.profile(), cloudxr_eula_accepted=True, cloudxr_eula_url=EULA
        )
        worker = (self.root / "ops/xr/native_session.py").read_text()
        with self.lock, self.database.transaction() as c:
            for job in self.list():
                if job["state"] not in TERMINAL:
                    return job
            identifier = str(uuid4())
            job = dict(
                id=identifier,
                state="PREPARING",
                profile=profile,
                worker=worker,
                worker_sha256=hashlib.sha256(worker.encode()).hexdigest(),
                root=f"{profile['work_root']}/sessions/{identifier}"
                if profile["execution"] == "workstation"
                else f"{WORK_ROOT}/jobs/runs/{identifier}",
                gateway=profile["gateway"],
                created_at=utc_now(),
                updated_at=utc_now(),
                server_ready=False,
                detail="Checking the pinned CloudXR and DexVerse runtime",
            )
            c.execute(
                "INSERT INTO live_xr_sessions VALUES (?,?)",
                (identifier, canonical_json(job)),
            )
        self.dispatch(identifier)
        return self.public(job)

    def dispatch(self, identifier):
        with self.lock:
            if identifier in self.active:
                return
            self.active.add(identifier)
        self.executor.submit(self.prepare, identifier)

    def prepare(self, identifier):
        try:
            job = self.get(identifier)
            if job["state"] != "PREPARING":
                return
            p = job["profile"]
            transport = self.transport(job)
            checks = [
                f"test -x {shlex.quote(p['cloudxr_runtime'] + '/bin/cloudxr-service')}",
                f"test -x {shlex.quote(p['runtime'] + '/bin/python')}",
                f"test -f {shlex.quote(p['cloudxr_runtime'] + '/share/openxr/1/openxr_cloudxr.json')}",
                f"test -f {shlex.quote(p['repository'] + '/scripts/record_demos.py')}",
            ]
            if p.get("execution") == "workstation":
                checks += [
                    "systemctl --user show-environment >/dev/null",
                    f"test -f {shlex.quote(p['runtime'] + '/.skynet-runtime-ready.json')}",
                ]
            try:
                transport.ssh(job["gateway"], " && ".join(checks), timeout=15)
            except ClusterError as exc:
                raise ValueError(
                    "Live runtime is unavailable. Follow the live setup guide; no GPU job was submitted. "
                    + str(exc)
                ) from exc
            transport.write_capsule_file(
                identifier, "runner.py", job["worker"], job["gateway"]
            )
            transport.write_capsule_file(
                identifier, "request.json", canonical_json(p), job["gateway"]
            )
            script = self.compile(job)
            self.update(
                identifier,
                state="SUBMITTING",
                script=script,
                detail="Starting a workstation session"
                if p.get("execution") == "workstation"
                else "Submitting one GPU session",
            )
            submitted = transport.submit_script(
                script, identifier, job["gateway"], submission_key=identifier
            )
            self.update(
                identifier,
                state="STARTING_SERVER"
                if p.get("execution") == "workstation"
                else "PENDING",
                job_id=submitted.job_id,
                detail="Starting the live worker"
                if p.get("execution") == "workstation"
                else "Waiting for a GPU allocation",
                error=None,
            )
        except SubmissionOutcomeUnknown as exc:
            self.update(
                identifier,
                state="SUBMISSION_UNKNOWN",
                error=str(exc),
                server_ready=False,
            )
        except Exception as exc:
            self.update(identifier, state="FAILED", error=str(exc), server_ready=False)
        finally:
            with self.lock:
                self.active.discard(identifier)

    @staticmethod
    def compile(job):
        p, root = job["profile"], job["root"]
        minutes = p["duration_minutes"] + 5
        directives = (
            []
            if p.get("execution") == "workstation"
            else [
                f"#SBATCH --job-name=live-xr-{job['id'][:8]}",
                f"#SBATCH --account={p['account']}",
                f"#SBATCH --partition={p['partition']}",
                f"#SBATCH --gres=gpu:{p['gpu_type']}:1",
                "#SBATCH --cpus-per-task=4",
                "#SBATCH --mem=48G",
                f"#SBATCH --time={minutes // 60:02}:{minutes % 60:02}:00",
                f"#SBATCH --output={root}/stdout.log",
                f"#SBATCH --error={root}/stderr.log",
            ]
        )
        return "\n".join(
            [
                "#!/bin/bash",
                *directives,
                "set -euo pipefail",
                "umask 077",
                f"printf '%s  %s\\n' {job['worker_sha256']} {shlex.quote(root + '/runner.py')} | sha256sum --check --status",
                "exec "
                + shlex.join(
                    [
                        "python3",
                        root + "/runner.py",
                        "--config",
                        root + "/request.json",
                        "--output",
                        root + "/output",
                    ]
                ),
                "",
            ]
        )

    def refresh(self, identifier, force=False):
        with self.lock:
            job = self.get(identifier)
            if (
                (
                    job["state"] in TERMINAL
                    and (not job.get("job_id") or job.get("scheduler_final"))
                )
                or identifier in self.refreshing
                or (
                    not force
                    and time.monotonic() - self.refreshed.get(identifier, 0) < 10
                )
            ):
                return self.public(job)
            self.refreshed[identifier] = time.monotonic()
            self.refreshing.add(identifier)
        try:
            transport = self.transport(job)
            if job["state"] == "PREPARING":
                self.dispatch(identifier)
                return self.public(job)
            if job["state"] in {"SUBMITTING", "SUBMISSION_UNKNOWN"}:
                if identifier in self.active:
                    return self.public(job)
                receipt = transport.recover_submission(
                    identifier, identifier, job["gateway"]
                )
                if receipt is None:
                    return self.update(
                        identifier,
                        state="SUBMISSION_UNKNOWN",
                        error="Submission acknowledgement is unavailable. Refresh rechecks the original request; no duplicate session is started.",
                    )
                self.update(
                    identifier, state="PENDING", job_id=receipt.job_id, error=None
                )
                job = self.get(identifier)
            _, states = transport.job_statuses([job["job_id"]], job["gateway"])
            scheduler = states.get(job["job_id"])
            if not scheduler:
                raise ClusterError(
                    "The execution host has not returned this session's status yet"
                )
            script = """import json,sys
from pathlib import Path
p=Path(sys.argv[1]); value={}
if p.is_file():
 if p.stat().st_size>1000000: raise ValueError('Oversized live status record')
 value=json.loads(p.read_text())
print(json.dumps(value))
"""
            control = json.loads(
                transport.ssh(
                    job["gateway"],
                    "python3 - " + shlex.quote(job["root"] + "/output/status.json"),
                    stdin=script,
                    timeout=15,
                )
            )
            changes = dict(
                scheduler=scheduler,
                error=None,
                connection_check_failed=False,
                checked_at=utc_now(),
            )
            if control:
                if (
                    control.get("job_id") != job["job_id"]
                    or control.get("state") not in WORKER_STATES
                ):
                    raise ValueError(
                        "The live worker returned an invalid session status"
                    )
                changes.update(
                    {
                        k: control[k]
                        for k in (
                            "state",
                            "detail",
                            "error",
                            "address",
                            "host",
                            "server_ready",
                            "recordings",
                            "recording_summary",
                        )
                        if k in control
                    }
                )
            terminal_scheduler = scheduler["State"] not in {
                "RUNNING",
                "PENDING",
                "CONFIGURING",
                "COMPLETING",
                "SUSPENDED",
            }
            changes["scheduler_final"] = terminal_scheduler
            if terminal_scheduler:
                changes["server_ready"] = False
                if (
                    scheduler["State"] not in {"COMPLETED", "CANCELLED"}
                    or changes.get("state") not in TERMINAL
                ):
                    changes.update(
                        state="STOPPED"
                        if scheduler["State"] == "CANCELLED"
                        else "FAILED",
                        error=f"Live process ended ({scheduler['State']}, {scheduler.get('Result', 'see logs')}) without a completed capture status. Inspect logs.",
                    )
            elif changes.get("state") in TERMINAL:
                changes.update(
                    state="STOPPING",
                    server_ready=False,
                    detail="Worker finished; waiting for process cleanup to complete",
                )
            elif not control:
                changes.update(
                    state=scheduler["State"],
                    server_ready=False,
                    detail="Waiting for GPU: " + scheduler.get("Reason", "queued")
                    if scheduler["State"] == "PENDING"
                    else "Starting the live worker",
                )
            return self.update(identifier, **changes)
        except LaunchRejected as exc:
            return self.update(
                identifier,
                state="FAILED",
                scheduler_final=True,
                error=str(exc),
                server_ready=False,
            )
        except (ClusterError, ValueError, OSError) as exc:
            return self.update(
                identifier,
                error=str(exc),
                connection_check_failed=True,
                server_ready=False,
            )
        finally:
            with self.lock:
                self.refreshing.discard(identifier)

    def stop(self, identifier):
        job = self.get(identifier)
        transport = self.transport(job)
        if job["state"] in TERMINAL:
            return self.public(job)
        if not job.get("job_id"):
            raise ValueError(
                "Submission is still being resolved. Refresh before stopping this session."
            )
        # Ask the worker to stop cleanly; cancellation covers sessions still queued.
        if (
            job["state"] == "PENDING"
            or job["profile"].get("execution") == "workstation"
        ):
            transport.cancel(job["job_id"], job["gateway"])
        else:
            dest = job["root"] + "/output/stop.request"
            transport.ssh(
                job["gateway"],
                f"mkdir -p {shlex.quote(str(Path(dest).parent))} && touch {shlex.quote(dest)}",
                timeout=15,
            )
        return self.update(
            identifier,
            stop_requested=True,
            detail="Stop requested; waiting for process cleanup",
            server_ready=False,
        )

    def logs(self, identifier):
        job = self.get(identifier)
        parts = []
        for name in (
            "stderr.log",
            "stdout.log",
            "output/cloudxr.log",
            "output/simulation.log",
        ):
            path = shlex.quote(job["root"] + "/" + name)
            parts.append(f"if test -f {path}; then tail -c 16000 {path}; fi")
        return (
            self.transport(job).ssh(job["gateway"], "\n".join(parts), timeout=20)
            or job.get("error")
            or "No logs yet."
        )
