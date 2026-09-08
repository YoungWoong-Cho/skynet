"""Durable, deduplicated conversion of collected episodes; never starts training."""

from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import hashlib
import inspect
import json
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
from uuid import uuid4

from .database import canonical_json, utc_now
from .live_xr import TERMINAL
from .live_xr_review import ArrayUnpickler
from .cluster_runtime import (
    ClusterClient,
    ClusterError,
    SubmissionOutcomeUnknown,
    WORK_ROOT,
)
from .capture_processing.service import upload_capture
from .capture_processing.slurm import compile_isaac_job
from .simulation_hands import upload as upload_hand

FORMAT = "dexverse-demo-hdf5/v1"
PENDING = {
    "PREPARING",
    "SUBMITTING",
    "SUBMISSION_UNKNOWN",
    "QUEUED",
    "RUNNING",
    "DOWNLOADING",
}


def pinned_converter_digest(profile):
    upstream = subprocess.run(
        [
            "git",
            "-C",
            profile["local_source"],
            "show",
            profile["source_revision"]
            + ":scripts/demo_tools/create_demo_files_sequential.py",
        ],
        capture_output=True,
        check=True,
        timeout=20,
    ).stdout
    return hashlib.sha256(upstream).hexdigest()


class LiveConversionService:
    def __init__(self, reviews, root=None, cluster=None):
        self.reviews, self.live = reviews, reviews.live
        self.database = self.live.database
        self.cluster = cluster or ClusterClient()
        self.root = Path(root or self.live.root / "data/live-conversions")
        self.lock, self.active = threading.RLock(), set()
        self.stopping, self.monitor = threading.Event(), None
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="conversion"
        )
        with self.database.transaction() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS live_conversions (id TEXT PRIMARY KEY, payload_json TEXT NOT NULL)"
            )

    @cached_property
    def sources(self):
        return {
            **{
                name: (self.live.root / "ops/xr" / name).read_text()
                for name in ("convert_dataset.py", "wrist.py")
            },
            "arrays.py": "import pickle\nimport numpy as np\n"
            + inspect.getsource(ArrayUnpickler),
        }

    @cached_property
    def version(self):
        return hashlib.sha256(canonical_json(self.sources).encode()).hexdigest()

    def target(self):
        config = json.loads(
            (self.live.root / "config/live_conversion.json").read_text()
        )
        key = config["pipeline_key"]
        profiles = json.loads(
            (self.live.root / "config/capture_pipelines.json").read_text()
        )["pipelines"]
        profile = next((p for p in profiles if p["key"] == key), None)
        if profile is None:
            raise ValueError(
                "Configure an existing DexVerse Slurm pipeline for conversion"
            )
        self.cluster.candidates(profile["gateway"])
        for name in ("repository", "runtime", "asset_bundle"):
            self.cluster._remote_path(profile[name])
        return dict(
            profile, execution="slurm", work_root=WORK_ROOT, gpu_type=config["gpu_type"]
        )

    def profile(self, session):
        profile = self.target()
        original = session["profile"]
        if profile["source_revision"] != original["source_revision"]:
            raise ValueError(
                "The Slurm pipeline and recording use different DexVerse revisions"
            )
        for key in ("robot", "task", "hand", "hand_name", "task_name"):
            if key in original:
                profile[key] = original[key]
        if original.get("hand_bundle"):
            bundle = original["hand_bundle"]
            if not re.fullmatch(r"[a-f0-9]{64}", bundle["digest"]):
                raise ValueError("Recording has an invalid hand bundle identity")
            profile["hand_bundle"] = dict(
                bundle, root=f"{WORK_ROOT}/hands/{profile['robot']}/{bundle['digest']}"
            )
        return profile

    def start(self):
        if self.monitor and self.monitor.is_alive():
            return
        self.stopping.clear()

        def monitor():
            while not self.stopping.wait(5):
                for job in self.list():
                    if job["state"] in PENDING:
                        self.dispatch(job["id"])

        self.monitor = threading.Thread(
            target=monitor, name="conversion-status", daemon=True
        )
        self.monitor.start()

    def stop(self):
        self.stopping.set()
        if self.monitor:
            self.monitor.join(timeout=2)

    def list(self):
        with self.database.connection() as c:
            return sorted(
                [
                    json.loads(r[0])
                    for r in c.execute("SELECT payload_json FROM live_conversions")
                ],
                key=lambda j: j["created_at"],
                reverse=True,
            )

    def get(self, identifier):
        with self.database.connection() as c:
            row = c.execute(
                "SELECT payload_json FROM live_conversions WHERE id=?", (identifier,)
            ).fetchone()
        if row is None:
            raise KeyError("Conversion not found")
        return json.loads(row[0])

    def update(self, identifier, **changes):
        with self.lock, self.database.transaction() as c:
            value = dict(self.get(identifier), **changes, updated_at=utc_now())
            c.execute(
                "UPDATE live_conversions SET payload_json=? WHERE id=?",
                (canonical_json(value), identifier),
            )
        return value

    def create(self, session_id, name, indices=None):
        session = self.live.get(session_id)
        if session["state"] not in TERMINAL:
            raise ValueError("End collection before converting its saved recordings")
        files = session.get("recordings", [])
        if not files:
            raise ValueError("This session has no saved recordings")
        indices = list(range(len(files))) if indices is None else indices
        if (
            not indices
            or len(indices) > 1000
            or len(indices) != len(set(indices))
            or any(type(i) is not int or not 0 <= i < len(files) for i in indices)
        ):
            raise ValueError("Choose 1–1000 valid recordings, without duplicates")
        name = name.strip()
        if not name or len(name) > 100 or any(ord(c) < 32 for c in name):
            raise ValueError("Dataset name must be 1–100 characters")
        profile = self.profile(session)
        sources = []
        for i in sorted(indices):
            _, path = self.reviews.source(session_id, i)
            checksum = session.get("recording_checksums", {}).get(files[i])
            if not re.fullmatch(r"[a-f0-9]{64}", str(checksum or "")):
                raise ValueError(
                    "Recording has no saved checksum. Refresh its session before converting"
                )
            sources.append(dict(index=i, path=path, sha256=checksum))
        fingerprint = hashlib.sha256(
            canonical_json(
                dict(
                    session=session_id,
                    sources=sources,
                    converter=self.version,
                    profile=profile,
                    launcher=inspect.getsource(compile_isaac_job),
                )
            ).encode()
        ).hexdigest()
        with self.lock:
            existing = next(
                (
                    j
                    for j in self.list()
                    if j["fingerprint"] == fingerprint and j["state"] != "FAILED"
                ),
                None,
            )
            if existing:
                self.refresh(existing["id"])
                return self.get(existing["id"])
            identifier = str(uuid4())
            value = dict(
                id=identifier,
                session_id=session_id,
                name=name,
                state="PREPARING",
                detail="Preparing conversion…",
                fingerprint=fingerprint,
                converter_sha256=self.version,
                sources=sources,
                profile=profile,
                source_profile=session["profile"],
                source_gateway=session["gateway"],
                gateway=profile["gateway"],
                root=f"{WORK_ROOT}/jobs/runs/{identifier}",
                dataset_root=f"{WORK_ROOT}/datasets/derivatives/dexverse-live/{session_id}/{identifier}",
                created_at=utc_now(),
                indices=sorted(indices),
            )
            directory = self.root / identifier
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "worker-sources.json").write_text(canonical_json(self.sources))
            self.save_capsule(value, directory)
            with self.database.transaction() as c:
                c.execute(
                    "INSERT INTO live_conversions VALUES (?,?)",
                    (identifier, canonical_json(value)),
                )
            self.dispatch(identifier)
            return value

    def dispatch(self, identifier):
        with self.lock:
            if identifier in self.active:
                return
            self.active.add(identifier)
            self.executor.submit(self.work, identifier)

    def refresh(self, identifier):
        job = self.get(identifier)
        if (
            job["state"] == "FAILED"
            and job.get("job_id")
            and job.get("scheduler", {}).get("State") == "COMPLETED"
            and job.get("error")
            == "Slurm completed without a verified dataset. Open the conversion log."
        ):
            job = self.update(
                identifier,
                state="RUNNING",
                detail="Checking the completed dataset…",
                result_wait_since=time.time(),
                error=None,
            )
        if job["state"] in PENDING:
            self.dispatch(identifier)
        return job

    def save_capsule(self, job, directory):
        """Freeze the exact request and shared Slurm script before submission."""
        profile = job["profile"]
        request = dict(
            root=job["dataset_root"],
            profile=profile,
            sources=job["sources"],
            staged_sources=[
                dict(s, path=f"{job['root']}/recordings/{s['index']}.pkl")
                for s in job["sources"]
            ],
            session_id=job["session_id"],
            converter_sha256=job["converter_sha256"],
            upstream_converter_sha256=pinned_converter_digest(profile),
        )
        (directory / "request.json").write_text(canonical_json(request))
        files = {**self.sources, "request.json": canonical_json(request)}
        checks = [
            f'printf "%s  %s\\n" {hashlib.sha256(content.encode()).hexdigest()} {shlex.quote(job["root"] + "/" + name)} | sha256sum --check --status'
            for name, content in files.items()
        ]
        script = compile_isaac_job(
            profile,
            job["root"],
            "dexverse-convert-" + job["id"][:8],
            [
                profile["runtime"] + "/bin/python",
                job["root"] + "/convert_dataset.py",
                job["root"] + "/request.json",
            ],
            checks=checks,
            after=[
                shlex.join(
                    [
                        profile["runtime"] + "/bin/python",
                        "-c",
                        "import json,sys; s=json.load(open(sys.argv[1])); assert s.get('state') == 'READY', s",
                        job["dataset_root"] + "/status.json",
                    ]
                )
            ],
        )
        (directory / "job.sbatch").write_text(script)

    def local_recording(self, job, source):
        directory = self.reviews.directory(job["session_id"], source["index"])
        path = directory / "recording.pkl"
        if path.is_file():
            if not 0 < path.stat().st_size <= 100 * 1024 * 1024:
                raise ValueError("Recording is empty or exceeds 100 MB")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != source["sha256"]:
                raise ValueError(
                    "Cached recording checksum changed; original data was preserved"
                )
            return path
        # Download-only reuse of the existing review service. No GPU work runs
        # on the collection host; only the original bytes are retrieved.
        status = self.reviews.status(job["session_id"], source["index"])
        requested = job.get("requested_inputs", [])
        if status["state"] == "FAILED" and source["index"] in requested:
            raise ValueError(
                f"Could not copy recording {source['index'] + 1} from {job['source_gateway']}: "
                + status["error"]
            )
        if source["index"] not in requested:
            self.update(job["id"], requested_inputs=[*requested, source["index"]])
        self.reviews.create(job["session_id"], source["index"])
        return None

    def launch(self, job, transport):
        if not job.get("staged"):
            paths = []
            for source in job["sources"]:
                path = self.local_recording(job, source)
                if path is None:
                    self.update(
                        job["id"],
                        detail=f"Copying recording {source['index'] + 1} from {job['source_gateway']}…",
                        connection_error=None,
                    )
                    return False
                paths.append(path)
            profile = job["profile"]
            probe = """import json,sys
from pathlib import Path
p=json.load(sys.stdin); repo=Path(p['repository']); runtime=Path(p['runtime'])
print(json.dumps({'runtime':(runtime/'bin/python').is_file(), 'revision':(repo/'.skynet-source-revision').read_text().strip() if (repo/'.skynet-source-revision').is_file() else '', 'converter':(repo/'scripts/demo_tools/create_demo_files_sequential.py').is_file()}))
"""
            check = json.loads(
                transport.ssh(
                    job["gateway"],
                    "python3 -c " + shlex.quote(probe),
                    stdin=canonical_json(profile),
                    timeout=20,
                )
            )
            if (
                not check["runtime"]
                or not check["converter"]
                or check["revision"] != profile["source_revision"]
            ):
                raise ValueError(
                    "The configured Slurm DexVerse runtime/source is not ready. Use the existing DexVerse setup before retrying."
                )
            if profile.get("hand_bundle"):
                bundle = profile["hand_bundle"]
                local = (
                    self.live.root
                    / "data/simulation-hands"
                    / profile["robot"]
                    / bundle["digest"]
                )
                if not (local / "manifest.json").is_file():
                    raise ValueError(
                        "The recording's original hand bundle is missing locally; a different model cannot be substituted"
                    )
                if (
                    upload_hand(local, WORK_ROOT, transport, job["gateway"])
                    != bundle["root"]
                ):
                    raise ValueError(
                        "Uploaded hand bundle path differs from the saved request"
                    )
            for number, (source, path) in enumerate(zip(job["sources"], paths), 1):
                self.update(
                    job["id"],
                    detail=f"Uploading recording {number} of {len(paths)} to {job['gateway']}…",
                )
                upload_capture(
                    transport,
                    path,
                    job["id"],
                    source["sha256"],
                    job["gateway"],
                    relative_path=f"recordings/{source['index']}.pkl",
                )
            directory = self.root / job["id"]
            files = json.loads((directory / "worker-sources.json").read_text())
            if (
                hashlib.sha256(canonical_json(files).encode()).hexdigest()
                != job["converter_sha256"]
            ):
                raise ValueError("The saved converter source changed")
            files["request.json"] = (directory / "request.json").read_text()
            for name, content in files.items():
                transport.write_capsule_file(job["id"], name, content, job["gateway"])
            self.update(job["id"], staged=True, connection_error=None)
        script = (self.root / job["id"] / "job.sbatch").read_text()
        self.update(
            job["id"],
            state="SUBMITTING",
            detail=f"Submitting to Slurm via {job['gateway']}…",
        )
        try:
            submission = transport.submit_script(
                script, job["id"], job["gateway"], submission_key=job["id"]
            )
        except SubmissionOutcomeUnknown:
            raise
        except ClusterError as exc:
            # ClusterClient distinguishes a definitive sbatch rejection from
            # an uncertain acknowledgement; never invent a second protocol.
            raise ValueError("Slurm rejected conversion: " + str(exc)) from exc
        self.update(
            job["id"],
            state="QUEUED",
            detail="Waiting for a Slurm GPU allocation…",
            job_id=submission.job_id,
            launched=True,
            connection_error=None,
        )
        return True

    def work(self, identifier):
        try:
            job = self.get(identifier)
            if job["profile"].get("execution") != "slurm":
                raise ValueError(
                    "This older workstation conversion cannot restart. Retry as a new Slurm conversion."
                )
            transport = self.cluster
            if not job.get("job_id"):
                if not self.launch(job, transport):
                    return
                job = self.get(identifier)
            _, states = transport.job_statuses([job["job_id"]], job["gateway"])
            scheduler = states.get(job["job_id"])
            if not scheduler:
                raise ClusterError(
                    "Slurm has not reported this job's status yet; its submission is preserved"
                )
            reader = """import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if p.exists():
 if p.stat().st_size>2000000: raise ValueError('Conversion status exceeds limit')
 print(p.read_text())
else: print('{}')
"""
            result = json.loads(
                transport.ssh(
                    job["gateway"],
                    "python3 -c "
                    + shlex.quote(reader)
                    + " "
                    + shlex.quote(job["dataset_root"] + "/status.json"),
                    timeout=20,
                )
            )
            state = scheduler["State"].split()[0].rstrip("+")
            self.update(identifier, scheduler=scheduler, connection_error=None)
            if result.get("state") == "FAILED":
                self.update(
                    identifier,
                    state="FAILED",
                    error=result.get("error", "Conversion failed"),
                )
            elif state == "COMPLETED":
                if result.get("state") != "READY":
                    # The gateway's NFS attribute cache can lag the compute
                    # node after atomic result publication. Allow visibility to
                    # catch up; do not resubmit or register unverified output.
                    since = job.get("result_wait_since", time.time())
                    if time.time() - since >= 120:
                        raise ValueError(
                            "Slurm completed without a verified dataset. Open the conversion log."
                        )
                    self.update(
                        identifier,
                        state="RUNNING",
                        result_wait_since=since,
                        detail="Waiting for the completed dataset to appear on shared storage…",
                        error=None,
                    )
                    return
                self.update(
                    identifier,
                    state="DOWNLOADING",
                    detail="Verifying and registering dataset…",
                )
                self.finish(job, result, transport)
            elif state in {
                "FAILED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "PREEMPTED",
                "CANCELLED",
                "BOOT_FAIL",
                "DEADLINE",
            }:
                self.update(
                    identifier,
                    state="FAILED",
                    error=f"Slurm job {job['job_id']} ended as {state}. Open the conversion log.",
                )
            elif state in {"RUNNING", "COMPLETING"}:
                detail = (
                    "Finalizing dataset…"
                    if result.get("state") == "READY"
                    else result.get("detail", "Starting DexVerse on the allocated GPU…")
                )
                fields = {
                    k: result[k]
                    for k in ("completed", "total", "steps", "total_steps")
                    if k in result
                }
                self.update(identifier, state="RUNNING", detail=detail, **fields)
            else:
                reason = scheduler.get("Reason", "")
                detail = "Waiting for a Slurm GPU allocation" + (
                    ": " + reason
                    if reason and reason not in {"None", "Unknown", "(null)"}
                    else "…"
                )
                self.update(identifier, state="QUEUED", detail=detail)
        except SubmissionOutcomeUnknown as exc:
            self.update(
                identifier,
                state="SUBMISSION_UNKNOWN",
                detail="Checking the original Slurm submission…",
                connection_error=str(exc),
            )
        except ValueError as exc:
            self.update(
                identifier, state="FAILED", error=str(exc), connection_error=None
            )
        except Exception as exc:
            self.update(identifier, connection_error=str(exc))
        finally:
            with self.lock:
                self.active.discard(identifier)

    def download(self, job, name, expected, transport):
        directory = self.root / job["id"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        size = expected.get("size_bytes")
        digest = expected.get("sha256", "")
        if (
            not re.fullmatch("[a-f0-9]{64}", digest)
            or type(size) is not int
            or not 0 < size <= 1024**3
        ):
            raise ValueError("Invalid converted artifact size/checksum")
        if path.is_file() and path.stat().st_size == size:
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() == digest:
                    return path
        remote = job.get("dataset_root", job["root"]) + "/" + name
        _, actual_size = transport.file_size(remote, job["gateway"])
        if actual_size != size:
            raise ValueError("Converted artifact size changed")
        temp = path.with_suffix(".part")
        try:
            received, checksum = 0, hashlib.sha256()
            with temp.open("wb") as f:
                for chunk in transport.stream_file_range(
                    remote, job["gateway"], start=0, end=size - 1
                ):
                    received += len(chunk)
                    if received > size:
                        raise ValueError("Download exceeds the saved artifact size")
                    f.write(chunk)
                    checksum.update(chunk)
            if received != size or checksum.hexdigest() != digest:
                raise OSError("Incomplete dataset download; refresh to retry")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        return path

    def finish(self, job, result, transport):
        m = result["metadata"]
        if (
            m.get("format"),
            m.get("task"),
            m.get("robot"),
            m.get("converter_sha256"),
            m.get("sources"),
        ) != (
            FORMAT,
            job["profile"]["task"],
            job["profile"]["robot"],
            job["converter_sha256"],
            job["sources"],
        ):
            raise ValueError(
                "Converted dataset identity differs from the requested recordings"
            )
        if m.get("episodes", 0) < len(job["sources"]) or m.get("steps", 0) <= 0:
            raise ValueError("Converted dataset is incomplete")
        manifest = self.download(job, "manifest.json", result["manifest"], transport)
        if json.loads(manifest.read_text()) != m:
            raise ValueError("Dataset manifest differs from the conversion result")
        dataset = self.download(job, "dataset.hdf5", m["artifact"], transport)
        with dataset.open("rb") as stream:
            if stream.read(8) != b"\x89HDF\r\n\x1a\n":
                raise ValueError("Converted dataset is not an HDF5 file")
        version, bundle = self.register(job, m, result["manifest"]["sha256"])
        self.update(
            job["id"],
            state="READY",
            detail=f"{m['episodes']} episodes converted",
            metadata=m,
            version_id=version["id"],
            resource_id=version["resource_id"],
            bundle_id=bundle["id"],
            connection_error=None,
            error=None,
        )

    def register(self, job, metadata, manifest_sha):
        db = self.database
        resources = db.list_data_resources(
            provider="collection", namespace="dexverse-converted", include_archived=True
        )
        resource = next((r for r in resources if r["name"] == job["id"]), None)
        if resource is None:
            resource = db.create_data_resource(
                provider="collection",
                namespace="dexverse-converted",
                name=job["id"],
                kind="demonstrations",
                description=job["name"],
                metadata={"display_name": job["name"], "session_id": job["session_id"]},
            )
        digest = metadata["artifact"]["sha256"]
        versions = db.get_data_resource(resource["id"])["versions"]
        version = next((v for v in versions if v["revision"] == digest), None)
        if version is None:
            version = db.create_data_resource_version(
                resource["id"],
                revision=digest,
                format=FORMAT,
                path=job.get("dataset_root", job["root"]) + "/dataset.hdf5",
                manifest_sha256=manifest_sha,
                status="READY",
                size_bytes=metadata["artifact"]["size_bytes"],
                source_uri="live-collection:" + job["session_id"],
                metadata={
                    **metadata,
                    "display_name": job["name"],
                    "gateway": job["gateway"],
                    "storage_location": "cluster",
                    "conversion_id": job["id"],
                    "training_compatibility": "DexVerse state HDF5 trainer required. GR00T/OpenPI formats are unsupported.",
                },
            )
        if not db.get_data_resource_version(version["id"]).get("derivation_id"):
            originals = db.list_data_resources(
                provider="collection",
                namespace="dexverse-recorded",
                include_archived=True,
            )
            source = next(
                (r for r in originals if r["name"] == job["session_id"]), None
            )
            if source is None:
                source = db.create_data_resource(
                    provider="collection",
                    namespace="dexverse-recorded",
                    name=job["session_id"],
                    kind="raw_capture",
                    description="Original simulation recordings",
                    metadata={
                        "session_id": job["session_id"],
                        "display_name": job["name"] + " · originals",
                    },
                )
            source_digest = hashlib.sha256(
                canonical_json(job["sources"]).encode()
            ).hexdigest()
            source_versions = db.get_data_resource(source["id"])["versions"]
            source_version = next(
                (v for v in source_versions if v["revision"] == source_digest), None
            )
            if source_version is None:
                source_version = db.create_data_resource_version(
                    source["id"],
                    revision=source_digest,
                    format="dexverse-trajectory-manifest/v1",
                    path=job.get("dataset_root", job["root"]) + "/source-manifest.json",
                    manifest_sha256=source_digest,
                    size_bytes=len(canonical_json(job["sources"]).encode()),
                    metadata={
                        "sources": job["sources"],
                        "storage_location": "cluster",
                        "gateway": job["gateway"],
                    },
                )
            db.create_data_derivation(
                output_version_id=version["id"],
                inputs=[
                    dict(version_id=source_version["id"], role="native_demonstrations")
                ],
                converter_repository="skynet:ops/xr/convert_dataset.py",
                converter_commit=job["converter_sha256"],
                converter_config={
                    "source_revision": job["profile"]["source_revision"],
                    "task": job["profile"]["task"],
                    "robot": job["profile"]["robot"],
                    "replay": "saved_states",
                    "format": FORMAT,
                },
            )
        bundles = db.list_data_bundles(include_archived=True)
        bundle = next(
            (
                b
                for b in bundles
                if b.get("metadata", {}).get("conversion_id") == job["id"]
            ),
            None,
        )
        if bundle is None:
            bundle = db.create_data_bundle(
                name=job["name"],
                version=job["id"][:8],
                assignments=[dict(version_id=version["id"], role="training_data")],
                description="Converted DexVerse demonstrations",
                metadata={"conversion_id": job["id"], "gateway": job["gateway"]},
            )
        return version, bundle

    def artifact(self, identifier, name):
        job = self.get(identifier)
        if name not in {"dataset.hdf5", "manifest.json"}:
            raise KeyError("Dataset artifact not found")
        if job["state"] != "READY":
            raise ValueError("Dataset conversion is not complete")
        path = self.root / identifier / name
        if not path.is_file():
            self.update(
                identifier, state="DOWNLOADING", detail="Restoring dataset download…"
            )
            self.dispatch(identifier)
            raise ValueError("Dataset cache is being restored. Retry shortly.")
        return path

    def logs(self, identifier):
        job = self.get(identifier)
        if job["profile"].get("execution") != "slurm":
            raise ValueError(
                "Logs for the older workstation converter are unsupported here"
            )
        _, output = self.cluster.read_log(
            job["root"] + "/stdout.log", job["gateway"], max_bytes=20000
        )
        _, errors = self.cluster.read_log(
            job["root"] + "/stderr.log", job["gateway"], max_bytes=20000
        )
        return output + "\n" + errors
