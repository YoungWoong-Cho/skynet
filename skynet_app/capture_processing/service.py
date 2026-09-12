"""Durable, idempotent processing jobs, separate from immutable raw captures."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
from uuid import uuid4

from skynet_app.cluster_runtime import (
    ClusterClient,
    ClusterError,
    SubmissionOutcomeUnknown,
    WORK_ROOT,
)
from skynet_app.database import canonical_json, utc_now
from skynet_app.cluster_config import CLUSTER
from .visionpro import convert
from skynet_app.simulation_hands import build as build_hand, upload as upload_hand
from .dexverse_runner import TASK, ROBOT, REVISION
from .slurm import compile_isaac_job

APP_ROOT = Path(__file__).resolve().parents[2]
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}
EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="capture-processing")


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def upload_capture(
    cluster,
    path,
    run_id,
    digest,
    gateway,
    *,
    relative_path="original.jsonl",
    timeout=120,
):
    """Bounded-memory upload; immutable destination and verified original bytes."""
    cluster.candidates(gateway)
    run_id = cluster._run_id(run_id)
    if not re.fullmatch("[a-f0-9]{64}", digest):
        raise ValueError("Invalid capture checksum")
    relative_path = cluster._relative_path(relative_path)
    destination = f"{WORK_ROOT}/jobs/runs/{run_id}/{relative_path}"
    parent = shlex.quote(str(Path(destination).parent))
    dest = shlex.quote(destination)
    command = (
        f"set -eu; umask 077; mkdir -p {parent}; tmp=$(mktemp {parent}/.capture-XXXXXX); "
        'trap \'rm -f "$tmp"\' EXIT; cat > "$tmp"; '
        f'printf "%s  %s\\n" {shlex.quote(digest)} "$tmp" | sha256sum --check --status; '
        f'if test -e {dest}; then cmp -s "$tmp" {dest}; else ln "$tmp" {dest}; fi'
    )
    with path.open("rb") as stream:
        try:
            result = subprocess.run(
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
                    gateway,
                    command,
                ],
                stdin=stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise ClusterError(
                "Capture upload timed out; no job was submitted"
            ) from error
    if result.returncode:
        raise ClusterError(
            "Capture upload or checksum verification failed; no job was submitted"
        )
    return destination


class ProcessingService:
    def __init__(self, captures, cluster=None, config_path=None):
        self.captures = captures
        self.database = captures.database
        self.cluster = cluster or ClusterClient()
        self.config_path = config_path or APP_ROOT / "config/capture_pipelines.json"
        self.lock = threading.RLock()
        self.refresh_times = {}
        self.refreshing = set()
        self.active = set()
        with self.database.transaction() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS capture_processing_jobs (
                id TEXT PRIMARY KEY, request_sha256 TEXT UNIQUE NOT NULL, capture_sha256 TEXT NOT NULL,
                state TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")

    def transport(self, job):
        if job.get("archive"):
            # Public jobs omit large manifests. Validate the persisted descriptor,
            # never reinterpret the frozen execution profile as its storage host.
            saved = self.get(job["id"], private=True)
            archive = saved.get("archive") or {}
            manifest, checksum = (
                archive.get("manifest"),
                archive.get("manifest_sha256", ""),
            )
            if (
                saved["state"] not in TERMINAL
                or archive.get("state") not in {"VERIFIED", "READY"}
                or not isinstance(manifest, dict)
                or manifest.get("schema") != "skynet.live-archive/v1"
                or manifest.get("session_id") != saved["id"]
                or not re.fullmatch(r"[a-f0-9]{64}", checksum)
                or hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
                != checksum
            ):
                raise ValueError("The saved processing archive has not been verified")
            expected = f"{CLUSTER.paths.datasets}/raw/dexverse-live/{saved['id']}/{checksum}/output"
            root = str(Path(expected).parent)
            if (
                archive.get("gateway") != "sky2"
                or archive.get("root") != expected
                or saved.get("root") != root
                or saved.get("gateway") != "sky2"
                or job.get("root") != root
                or job.get("gateway") != "sky2"
            ):
                raise ValueError(
                    "The saved processing archive location does not match its verified manifest"
                )
            return self.cluster
        if job["config"]["pipeline"].get("execution") == "workstation":
            from skynet_app.live_xr_workstation import WorkstationClient

            return WorkstationClient(job["config"]["pipeline"])
        return self.cluster

    def catalog(self):
        data = json.loads(self.config_path.read_text())
        return data["pipelines"]

    def list(self):
        with self.database.connection() as c:
            rows = c.execute(
                "SELECT * FROM capture_processing_jobs ORDER BY created_at DESC"
            ).fetchall()
        return [self.public(row) for row in rows]

    def public(self, row):
        item = dict(row)
        item.update(json.loads(item.pop("payload_json")))
        item.pop("script", None)
        item.pop("runner_source", None)
        item.pop("hand_bundle_path", None)
        if isinstance(item.get("archive"), dict):
            item["archive"] = {
                key: value
                for key, value in item["archive"].items()
                if key != "manifest"
            }
        return item

    def get(self, identifier, private=False):
        with self.database.connection() as c:
            row = c.execute(
                "SELECT * FROM capture_processing_jobs WHERE id=?", (identifier,)
            ).fetchone()
        if row is None:
            raise KeyError("Processing job not found")
        if not private:
            return self.public(row)
        item = dict(row)
        item.update(json.loads(item.pop("payload_json")))
        return item

    def update(self, identifier, **changes):
        with self.lock, self.database.transaction() as c:
            row = c.execute(
                "SELECT * FROM capture_processing_jobs WHERE id=?", (identifier,)
            ).fetchone()
            payload = json.loads(row["payload_json"])
            state = changes.pop("state", row["state"])
            payload.update(changes)
            c.execute(
                "UPDATE capture_processing_jobs SET state=?,payload_json=?,updated_at=? WHERE id=?",
                (state, canonical_json(payload), utc_now(), identifier),
            )

    def create(
        self,
        digest,
        *,
        pipeline_key="dexverse-shadow-right",
        seed=0,
        epochs=100,
        eval_episodes=1,
    ):
        if type(seed) is not int or not 0 <= seed < 2**31 - 20:
            raise ValueError("Seed must be an integer from 0 to 2147483627")
        if type(epochs) is not int or not 1 <= epochs <= 1000:
            raise ValueError("Training epochs must be 1–1000")
        if type(eval_episodes) is not int or not 1 <= eval_episodes <= 20:
            raise ValueError("Evaluation episodes must be 1–20")
        pipeline = next((p for p in self.catalog() if p["key"] == pipeline_key), None)
        if pipeline is None:
            raise ValueError("Unsupported capture-processing pipeline")
        if (
            pipeline["key"],
            pipeline["provider"],
            pipeline["hand"],
            pipeline["task"],
            pipeline["robot"],
            pipeline["source_revision"],
        ) != (
            "dexverse-shadow-right",
            "visionpro-local",
            "right",
            TASK,
            ROBOT,
            REVISION,
        ):
            raise ValueError(
                "Unsupported pipeline combination. Add a matching converter and worker implementation before publishing another task, robot or provider."
            )
        capture = next((p for p in self.captures.list() if p["sha256"] == digest), None)
        if capture is None:
            raise KeyError("Import this recording before processing it")
        if capture["provider"] != pipeline["provider"]:
            raise ValueError(
                "This pipeline does not support the selected recorder format"
            )
        runner = (Path(__file__).parent / "dexverse_runner.py").read_text()
        runner_sha = hashlib.sha256(runner.encode()).hexdigest()
        hand_path, hand_manifest = build_hand(pipeline["robot"])
        if hand_manifest["robot"] != pipeline["robot"]:
            raise ValueError("Prepared hand differs from the capture pipeline")
        params = {
            "pipeline": pipeline,
            "hand_bundle": {
                k: hand_manifest[k] for k in ("robot", "digest", "hand_asset")
            },
            "seed": seed,
            "epochs": epochs,
            "eval_episodes": eval_episodes,
            "runner_sha256": runner_sha,
            "converter_sha256": file_sha(Path(__file__).parent / "visionpro.py"),
            "capture_sha256": digest,
        }
        request_sha = hashlib.sha256(canonical_json(params).encode()).hexdigest()
        with self.lock, self.database.transaction() as c:
            existing = c.execute(
                "SELECT * FROM capture_processing_jobs WHERE request_sha256=?",
                (request_sha,),
            ).fetchone()
            if existing:
                return self.public(existing)
            identifier = str(uuid4())
            now = utc_now()
            payload = {
                "config": params,
                "capture_version_id": capture["version_id"],
                "name": capture["summary"]["header"]["task"],
                "gateway": pipeline["gateway"],
                "runner_source": runner,
                "hand_bundle_path": str(hand_path),
                "stages": {},
                "error": None,
            }
            c.execute(
                "INSERT INTO capture_processing_jobs VALUES (?,?,?,?,?,?,?)",
                (
                    identifier,
                    request_sha,
                    digest,
                    "PREPARING",
                    canonical_json(payload),
                    now,
                    now,
                ),
            )
        self.dispatch(identifier)
        return self.get(identifier)

    def dispatch(self, identifier):
        with self.lock:
            if identifier in self.active:
                return
            self.active.add(identifier)
        EXECUTOR.submit(self._prepare, identifier)

    def _prepare(self, identifier):
        try:
            job = self.get(identifier, private=True)
            if job["state"] != "PREPARING":
                return
            cfg = job["config"]
            profile = cfg["pipeline"]
            gateway = job["gateway"]
            path = self.captures.read(job["capture_sha256"])
            converted = convert(path, hand=profile["hand"], fps=60)
            if converted["source_sha256"] != job["capture_sha256"]:
                raise ValueError("Original recording checksum changed")
            root = f"{WORK_ROOT}/jobs/runs/{identifier}"
            repo = profile["repository"]
            runtime = profile["runtime"]
            rev = profile["source_revision"]
            if not re.fullmatch("[a-f0-9]{40}", rev):
                raise ValueError("A pinned DexVerse source revision is required")
            for value in (repo, runtime):
                self.cluster._remote_path(value)
                if not value.startswith(WORK_ROOT + "/"):
                    raise ValueError(
                        "Pipeline runtime must be inside the configured cluster workspace"
                    )
            precheck = f'test -x {shlex.quote(runtime + "/bin/python")} && test "$(cat {shlex.quote(repo + "/.skynet-source-revision")})" = {shlex.quote(rev)} && test -f {shlex.quote(repo + "/source/dexverse/dexverse/robot_agents/shadow/retarget/floating_shadow_right.urdf")}'
            try:
                self.transport(job).ssh(gateway, precheck, timeout=20)
            except ClusterError as error:
                raise ClusterError(
                    "DexVerse runtime is not configured on the selected host. Follow the saved-recording setup guide and retry. "
                    + str(error)
                ) from error
            self.update(
                identifier,
                preparation="Staging and verifying the original recording on sky2",
            )
            self.captures.stage(job["capture_sha256"], identifier, gateway)
            self.cluster.write_capsule_file(
                identifier, "tracking.json", canonical_json(converted), gateway
            )
            self.cluster.write_capsule_file(
                identifier, "runner.py", job["runner_source"], gateway
            )
            self.cluster.write_capsule_file(
                identifier, "request.json", canonical_json(cfg), gateway
            )
            if cfg.get("hand_bundle"):
                hand_root = upload_hand(
                    job["hand_bundle_path"], WORK_ROOT, self.transport(job), gateway
                )
                cfg["hand_bundle"]["root"] = hand_root
                self.update(identifier, config=cfg)
                job["config"] = cfg
                self.cluster.write_capsule_file(
                    identifier, "request.json", canonical_json(cfg), gateway
                )
            script = self.compile(job, root, converted)
            self.update(
                identifier,
                script=script,
                state="SUBMITTING",
                preparation="Submitting one GPU job",
                root=root,
            )
            submission = self.cluster.submit_script(
                script, identifier, gateway, submission_key=identifier
            )
            self.update(
                identifier,
                state="PENDING",
                job_id=submission.job_id,
                error=None,
                preparation=None,
            )
        except SubmissionOutcomeUnknown as error:
            self.update(identifier, state="SUBMISSION_UNKNOWN", error=str(error))
        except Exception as error:
            self.update(identifier, state="FAILED", error=str(error), preparation=None)
        finally:
            with self.lock:
                self.active.discard(identifier)

    @staticmethod
    def compile(job, root, converted):
        config = job["config"]
        profile = config["pipeline"]
        runtime = profile["runtime"]
        argv = [
            runtime + "/bin/python",
            root + "/runner.py",
            "--tracking",
            root + "/tracking.json",
            "--output",
            root + "/output",
            "--source-sha256",
            job["capture_sha256"],
            "--seed",
            str(config["seed"]),
            "--epochs",
            str(config["epochs"]),
            "--eval-episodes",
            str(config["eval_episodes"]),
            "--headless",
            "--enable_cameras",
            "--device",
            "cuda:0",
        ]
        if config.get("hand_bundle"):
            argv.extend(["--hand-bundle-root", config["hand_bundle"]["root"]])
        return compile_isaac_job(
            profile,
            root,
            f"capture-cycle-{job['id'][:8]}",
            argv,
            checks=[
                f'printf "%s  %s\\n" {config["runner_sha256"]} {shlex.quote(root + "/runner.py")} | sha256sum --check --status',
                f'printf "%s  %s\\n" {job["capture_sha256"]} {shlex.quote(root + "/original.jsonl")} | sha256sum --check --status',
                f'printf "%s  %s\\n" {hashlib.sha256(canonical_json(converted).encode()).hexdigest()} {shlex.quote(root + "/tracking.json")} | sha256sum --check --status',
            ],
            after=[f"test -s {shlex.quote(root + '/output/result.json')}"],
        )

    def refresh(self, identifier, force=False):
        with self.lock:
            job = self.get(identifier, private=True)
            if job["state"] in TERMINAL:
                return self.get(identifier)
            if identifier in self.refreshing or (
                not force
                and time.monotonic() - self.refresh_times.get(identifier, 0) < 10
            ):
                return self.get(identifier)
            self.refreshing.add(identifier)
            self.refresh_times[identifier] = time.monotonic()
        try:
            if job["state"] == "PREPARING":
                self.dispatch(identifier)
                return self.get(identifier)
            if job["state"] in ("SUBMITTING", "SUBMISSION_UNKNOWN"):
                if identifier in self.active:
                    return self.get(identifier)
                result = self.transport(job).recover_submission(
                    identifier, identifier, job["gateway"]
                )
                if result is None:
                    self.update(
                        identifier,
                        state="SUBMISSION_UNKNOWN",
                        error="Submission acknowledgement is unavailable. No second job has been submitted. Use Recover submission to recheck the original request.",
                    )
                    return self.get(identifier)
                self.update(
                    identifier, state="PENDING", job_id=result.job_id, error=None
                )
                job = self.get(identifier, private=True)
            if not job.get("job_id"):
                return self.get(identifier)
            _, states = self.transport(job).job_statuses(
                [job["job_id"]], job["gateway"]
            )
            status = states.get(job["job_id"])
            if not status:
                raise ClusterError(
                    "Scheduler has no status yet; the job has not been marked complete"
                )
            state = status["State"]
            # One bounded read of small control records; no expensive asset revalidation while polling.
            command = "python3 - " + shlex.quote(job["root"] + "/output")
            read = """import json,sys
from pathlib import Path
root=Path(sys.argv[1]); out={}
for name in ('progress.json','error.json','result.json'):
 p=root/name
 if p.is_file():
  if p.stat().st_size>2000000: raise ValueError('Control record exceeds limit')
  out[name]=json.loads(p.read_text())
print(json.dumps(out))
"""
            control = json.loads(
                self.transport(job).ssh(job["gateway"], command, stdin=read, timeout=20)
            )
            changes = {
                "scheduler": status,
                "stages": control.get("progress.json", {}).get(
                    "stages", job.get("stages", {})
                ),
                "refresh_error": None,
            }
            if state == "COMPLETED":
                result = control.get("result.json")
                if (
                    not result
                    or result.get("schema") != "skynet.dexverse-cycle/v1"
                    or result.get("status") != "SUCCEEDED"
                ):
                    raise ValueError(
                        "GPU job exited without a valid completed pipeline result"
                    )
                if result["dataset"]["source_sha256"] != job["capture_sha256"]:
                    raise ValueError("Pipeline result refers to a different recording")
                self.verify_artifacts(job, result)
                version = self.register_dataset(job, result)
                changes.update(
                    state="SUCCEEDED",
                    result=result,
                    dataset_version_id=version["id"],
                    error=None,
                )
            elif state in (
                "FAILED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "PREEMPTED",
                "CANCELLED",
                "BOOT_FAIL",
                "DEADLINE",
            ):
                error = (
                    control.get("error.json", {}).get("error")
                    or f"GPU job ended as {state}. Read the job log for details."
                )
                changes.update(
                    state="CANCELLED" if state == "CANCELLED" else "FAILED", error=error
                )
            else:
                changes["state"] = (
                    "RUNNING" if state in ("RUNNING", "COMPLETING") else "PENDING"
                )
            self.update(identifier, **changes)
        except ClusterError as error:
            self.update(identifier, refresh_error=str(error))
        except (ValueError, KeyError, TypeError) as error:
            self.update(identifier, state="FAILED", error=str(error))
        finally:
            with self.lock:
                self.refreshing.discard(identifier)
        return self.get(identifier)

    def verify_artifacts(self, job, result):
        artifacts = result.get("artifacts", {})
        dataset = result.get("dataset", {})
        profile = job.get("config", {}).get("pipeline", {})
        if (
            dataset.get("schema") != "skynet.dexverse-state-actions/v1"
            or dataset.get("task") != profile.get("task", TASK)
            or dataset.get("robot") != profile.get("robot", ROBOT)
        ):
            raise ValueError("Completed cycle has an unsupported dataset contract")
        if set(result.get("stages", {})) != {
            "simulation",
            "training",
            "evaluation",
        } or any(
            stage.get("status") != "SUCCEEDED" for stage in result["stages"].values()
        ):
            raise ValueError("Completed cycle contains an incomplete or failed stage")
        for required in (
            "dataset.hdf5",
            "dataset-manifest.json",
            "state-bc.pt",
            "replay.mp4",
            "evaluation-0.mp4",
        ):
            if required not in artifacts:
                raise ValueError(f"Completed cycle is missing {required}")
        if artifacts["dataset.hdf5"]["sha256"] != dataset.get("dataset_sha256"):
            raise ValueError("Dataset manifest and artifact checksums disagree")
        for name, item in artifacts.items():
            if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name) or name in (".", ".."):
                raise ValueError("Invalid result artifact name")
            if (
                not re.fullmatch("[a-f0-9]{64}", item.get("sha256", ""))
                or type(item.get("size_bytes")) is not int
                or item["size_bytes"] <= 0
            ):
                raise ValueError(f"Invalid artifact checksum or size for {name}")
        script = """import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); manifest=json.loads(sys.stdin.read())
for name,expected in manifest.items():
 path=root/name
 if path.stat().st_size!=expected['size_bytes']: raise ValueError('Artifact size mismatch: '+name)
 with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
 if digest!=expected['sha256']: raise ValueError('Artifact checksum mismatch: '+name)
print('verified')
"""
        # Python source and JSON are separate arguments; neither is shell-expanded.
        command = f"{shlex.quote(job['config']['pipeline']['runtime'] + '/bin/python')} -c {shlex.quote(script)} {shlex.quote(job['root'] + '/output')}"
        self.transport(job).ssh(
            job["gateway"], command, stdin=canonical_json(artifacts), timeout=45
        )

    def register_dataset(self, job, result):
        name = job["id"]
        metadata = result["dataset"]
        digest = metadata["dataset_sha256"]
        resources = self.database.list_data_resources(
            provider="collection", namespace="dexverse-offline", include_archived=True
        )
        resource = next((r for r in resources if r["name"] == name), None)
        if resource is None:
            resource = self.database.create_data_resource(
                category="dataset",
                provider="collection",
                namespace="dexverse-offline",
                name=name,
                kind="demonstrations",
                description=job["name"],
                metadata={"pipeline": "dexverse-shadow-right"},
            )
        versions = self.database.get_data_resource(resource["id"])["versions"]
        version = next((v for v in versions if v["revision"] == digest), None)
        if version is None:
            version = self.database.create_data_resource_version(
                resource["id"],
                revision=digest,
                format=metadata["schema"],
                path=job["root"] + "/output/dataset.hdf5",
                manifest_sha256=result["artifacts"]["dataset-manifest.json"]["sha256"],
                status="READY",
                size_bytes=result["artifacts"]["dataset.hdf5"]["size_bytes"],
                source_uri="capture-processing:" + job["id"],
                metadata={
                    **metadata,
                    "storage_location": "cluster",
                    "gateway": job["gateway"],
                    "training_compatibility": "Skynet state BC v1 only. VLA conversion requires a matching embodiment adapter.",
                },
            )
        with self.database.connection() as connection:
            derivation = connection.execute(
                "SELECT id FROM data_derivations WHERE output_version_id=?",
                (version["id"],),
            ).fetchone()
        if derivation is None:
            self.database.create_data_derivation(
                output_version_id=version["id"],
                inputs=[
                    {"version_id": job["capture_version_id"], "role": "raw_tracking"}
                ],
                converter_repository="skynet:capture_processing",
                converter_commit=job["config"]["runner_sha256"],
                converter_config=job["config"],
            )
        return version

    def retry(self, identifier):
        job = self.get(identifier, private=True)
        if job.get("archive"):
            raise ValueError(
                "This completed cycle has been archived. Create a new cycle to run it again; archived files are read-only."
            )
        if job["state"] not in ("FAILED", "SUBMISSION_UNKNOWN"):
            raise ValueError(
                "Only a failed preparation or uncertain submission can be recovered"
            )
        if job.get("job_id"):
            raise ValueError(
                "This GPU attempt has finished. Choose another seed to create a new immutable cycle."
            )
        if job.get("script"):
            # Reuse the durable submission token AND exact script. The cluster
            # receipt protocol deduplicates acceptance even after a lost response.
            result = self.cluster.submit_script(
                job["script"], identifier, job["gateway"], submission_key=identifier
            )
            self.update(identifier, state="PENDING", job_id=result.job_id, error=None)
        else:
            self.update(identifier, state="PREPARING", error=None)
            self.dispatch(identifier)
        return self.get(identifier)
