"""Durable, deduplicated conversion of collected episodes; never starts training."""

from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import hashlib
import inspect
import json
from pathlib import Path
import re
import shlex
import threading
from uuid import uuid4

from .database import canonical_json, utc_now
from .live_xr import TERMINAL
from .live_xr_review import ArrayUnpickler

FORMAT = "dexverse-demo-hdf5/v1"
PENDING = {"PREPARING", "QUEUED", "RUNNING", "DOWNLOADING"}


class LiveConversionService:
    def __init__(self, reviews, root=None):
        self.reviews, self.live = reviews, reviews.live
        self.database = self.live.database
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
        if session["profile"].get("execution") != "workstation":
            raise ValueError(
                "Conversion of older cluster recordings is unsupported; select a workstation collection"
            )
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
                dict(session=session_id, sources=sources, converter=self.version)
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
                profile=session["profile"],
                gateway=session["gateway"],
                root=session["profile"]["work_root"] + "/conversions/" + identifier,
                created_at=utc_now(),
                indices=sorted(indices),
            )
            directory = self.root / identifier
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "worker-sources.json").write_text(canonical_json(self.sources))
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
        if job["state"] in PENDING:
            self.dispatch(identifier)
        return job

    @staticmethod
    def unit(identifier):
        return "skynet-convert-" + identifier + ".service"

    def launch(self, job, transport):
        q = shlex.quote
        p, root = job["profile"], job["root"]
        script = (
            "#!/bin/bash\nset -euo pipefail\ncd "
            + q(p["repository"])
            + "\n"
            + "export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1\n"
            + "export LD_LIBRARY_PATH="
            + q(p["runtime"] + "/lib")
            + "\n"
            + "export PYTHONPATH="
            + q(p["repository"] + "/source/dexverse")
            + "\n"
            + "export PATH="
            + q(p["runtime"] + "/bin")
            + ':"$PATH"\n'
            + "exec flock --wait 1800 --conflict-exit-code=75 "
            + q(p["work_root"] + "/.gpu-session.lock")
            + " timeout -k 10 3600 "
            + q(p["runtime"] + "/bin/python")
            + " "
            + q(root + "/convert_dataset.py")
            + " "
            + q(root + "/request.json")
            + "\n"
        )
        # A remote receipt is written before launch; reconnects recover this unit
        # instead of starting duplicate GPU work after an uncertain response.
        launcher = """import json,subprocess,sys
from pathlib import Path
v=json.load(sys.stdin); root=Path(v['root']); root.mkdir(parents=True,exist_ok=True)
for name,content in v['files'].items():
 p=root/name
 if p.exists() and p.read_text()!=content: raise ValueError('Conversion capsule changed: '+name)
 if not p.exists(): p.write_text(content)
receipt=root/'launch.json'
try:
 with receipt.open('x') as f: json.dump({'attempted':True},f)
except FileExistsError: pass
else:
 result=subprocess.run(v['argv'],capture_output=True,text=True,timeout=25)
 temp=receipt.with_suffix('.tmp'); temp.write_text(json.dumps({'returncode':result.returncode,'error':result.stderr.strip()})); temp.replace(receipt)
print(receipt.read_text())
"""
        worker_sources = json.loads(
            (self.root / job["id"] / "worker-sources.json").read_text()
        )
        if (
            hashlib.sha256(canonical_json(worker_sources).encode()).hexdigest()
            != job["converter_sha256"]
        ):
            raise ValueError("The saved converter source changed")
        request = {
            k: job[k]
            for k in ("root", "profile", "sources", "converter_sha256", "session_id")
        }
        files = dict(
            worker_sources,
            **{
                "request.json": canonical_json(request),
                "source-manifest.json": canonical_json(job["sources"]),
                "run.sh": script,
            },
        )
        command = [
            "systemd-run",
            "--user",
            "--quiet",
            "--unit=" + self.unit(job["id"]),
            "--service-type=exec",
            "--remain-after-exit",
            "--property=RuntimeMaxSec=5500",
            "--property=KillMode=control-group",
            "--property=MemoryMax=24G",
            "--property=MemorySwapMax=1G",
            "--property=CPUQuota=400%",
            "--property=UMask=0077",
            "--property=StandardOutput=append:" + root + "/conversion.log",
            "--property=StandardError=append:" + root + "/conversion.log",
            "/bin/bash",
            root + "/run.sh",
        ]
        receipt = json.loads(
            transport.ssh(
                job["gateway"],
                "python3 -c " + q(launcher),
                stdin=canonical_json(dict(root=root, files=files, argv=command)),
                timeout=40,
            )
        )
        if receipt.get("returncode"):
            raise ValueError(
                "Could not start conversion: "
                + receipt.get("error", "Unknown service error")
            )
        self.update(
            job["id"], state="QUEUED", detail="Waiting for the GPU…", launched=True
        )

    def work(self, identifier):
        try:
            job = self.get(identifier)
            transport = self.live.transport(job)
            if not job.get("launched"):
                self.launch(job, transport)
                job = self.get(identifier)
            reader = """import json,subprocess,sys
from pathlib import Path
p=Path(sys.argv[1]); state=json.loads(p.read_text()) if p.exists() else {'state':'QUEUED','detail':'Waiting for the GPU…'}
r=subprocess.run(['systemctl','--user','show',sys.argv[2],'--property=ActiveState,SubState,ExecMainStatus'],capture_output=True,text=True,timeout=10)
state['service']=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
print(json.dumps(state))
"""
            result = json.loads(
                transport.ssh(
                    job["gateway"],
                    "python3 -c "
                    + shlex.quote(reader)
                    + " "
                    + shlex.quote(job["root"] + "/status.json")
                    + " "
                    + shlex.quote(self.unit(identifier)),
                    timeout=20,
                )
            )
            system = result.pop("service")
            if result["state"] == "READY":
                self.update(
                    identifier,
                    state="DOWNLOADING",
                    detail="Verifying and registering dataset…",
                    connection_error=None,
                )
                self.finish(job, result, transport)
            elif result["state"] == "FAILED":
                self.update(
                    identifier,
                    state="FAILED",
                    error=result.get("error", "Conversion failed"),
                    connection_error=None,
                )
            elif (
                system.get("ActiveState") in {"failed", "inactive"}
                or system.get("SubState") == "exited"
            ):
                code = system.get("ExecMainStatus")
                message = (
                    "GPU remained busy for 30 minutes. End collection and retry."
                    if code == "75"
                    else "Conversion stopped before producing a complete dataset. Open its log and retry."
                )
                self.update(
                    identifier, state="FAILED", error=message, connection_error=None
                )
            else:
                fields = {
                    k: result[k]
                    for k in (
                        "state",
                        "detail",
                        "completed",
                        "total",
                        "steps",
                        "total_steps",
                    )
                    if k in result
                }
                if fields.get("state") not in {"RUNNING", "QUEUED"}:
                    raise ValueError("Unsupported conversion worker status")
                self.update(identifier, **fields, connection_error=None)
        except ValueError as exc:
            self.update(identifier, state="FAILED", error=str(exc))
        except Exception as exc:
            # A lost SSH response cannot prove the remote job failed. Retain its
            # identity and retry status/download on the next refresh.
            self.update(
                identifier, connection_error="Workstation unavailable: " + str(exc)
            )
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
        remote = job["root"] + "/" + name
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
                path=job["root"] + "/dataset.hdf5",
                manifest_sha256=manifest_sha,
                status="READY",
                size_bytes=metadata["artifact"]["size_bytes"],
                source_uri="live-collection:" + job["session_id"],
                metadata={
                    **metadata,
                    "display_name": job["name"],
                    "gateway": job["gateway"],
                    "storage_location": "workstation",
                    "conversion_id": job["id"],
                    "training_compatibility": "DexVerse state HDF5 trainer required. GR00T/OpenPI formats are unsupported. Transfer required before cluster training.",
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
                    path=job["root"] + "/source-manifest.json",
                    manifest_sha256=source_digest,
                    metadata={
                        "sources": job["sources"],
                        "storage_location": "workstation",
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
        return self.live.transport(job).ssh(
            job["gateway"],
            "tail -c 20000 " + shlex.quote(job["root"] + "/conversion.log"),
            timeout=15,
        )
