"""One preparation workflow for collection datasets and their verified locations.

The historic export API uses this same service. Immutable content versions and
lineage live in the data registry; mutable execution progress stays in the job.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import threading
from uuid import uuid4

from ops.datasets.artifacts import digest, verify, pack
from .database import canonical_json, utc_now
from . import dataset_cleanup
from .dataset_formats import RECIPES, XPL_COMMIT as COMMIT, XPL_REPOSITORY, catalog
from .live_xr import TERMINAL
from .live_xr_review import ArrayUnpickler
from .cluster_runtime import ClusterClient, ClusterError, WORK_ROOT
from .cluster_config import CLUSTER
from .capture_processing.service import upload_capture

FORMATS = list(RECIPES.values())


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class PolicyExportService:
    def __init__(self, reviews, root=None, cluster=None):
        self.reviews, self.live = reviews, reviews.live
        self.database = self.live.database
        self.root = Path(root or self.live.root / "data/policy-exports")
        self.cluster = cluster or ClusterClient()
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dataset-preparation"
        )
        self.processes, self.active = {}, set()
        self.stopping = False
        with self.database.transaction() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS policy_exports (id TEXT PRIMARY KEY, payload_json TEXT NOT NULL)"
            )

    def list(self):
        with self.database.connection() as c:
            return sorted(
                [
                    json.loads(r[0])
                    for r in c.execute("SELECT payload_json FROM policy_exports")
                ],
                key=lambda j: j["created_at"],
                reverse=True,
            )

    def get(self, identifier):
        with self.database.connection() as c:
            row = c.execute(
                "SELECT payload_json FROM policy_exports WHERE id=?", (identifier,)
            ).fetchone()
        if row is None:
            raise KeyError("Dataset preparation not found")
        return json.loads(row[0])

    def update(self, identifier, **changes):
        with self.lock, self.database.transaction() as c:
            job = self.get(identifier)
            job.update(changes, updated_at=utc_now())
            c.execute(
                "UPDATE policy_exports SET payload_json=? WHERE id=?",
                (canonical_json(job), identifier),
            )
        return job

    def start(self):
        for job in self.list():
            if job["state"] not in {"READY", "FAILED", "DELETE_FAILED"}:
                self.update(
                    job["id"],
                    state="FAILED",
                    error="Preparation interrupted by an app restart. Retry to continue from verified files.",
                )

    def stop(self):
        with self.lock:
            self.stopping = True
            for process in self.processes.values():
                if process.poll() is None:
                    process.terminate()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def sources(self, session, indices=None, *, require_images=True):
        if session["state"] not in TERMINAL:
            raise ValueError("End the collection session before preparing its dataset")
        recordings = session.get("recordings", [])
        if not 1 <= len(recordings) <= 1000:
            raise ValueError("The session has no saved demonstrations")
        indices = list(range(len(recordings))) if indices is None else indices
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(type(i) is not int or i < 0 or i >= len(recordings) for i in indices)
        ):
            raise ValueError("Select valid, unique recording numbers")
        result = []
        for index in sorted(indices):
            name = recordings[index]
            image = session.get("recording_images", {}).get(name)
            if not require_images and not image:
                self.reviews.source(session["id"], index)
                checksum = session.get("recording_checksums", {}).get(name, "")
                if not re.fullmatch(r"[a-f0-9]{64}", checksum):
                    raise ValueError("Recording checksum is missing")
                result.append(dict(session_id=session["id"], index=index, path=name, sha256=checksum, image_size_bytes=0))
                continue
            if not image:
                raise ValueError(
                    "This session has no completed training images. Wait for image preparation, or collect a new session with training images enabled."
                )
            self.reviews.source(session["id"], index)
            path = PurePosixPath(image["path"])
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.is_relative_to("recordings")
                or path.suffix != ".hdf5"
            ):
                raise ValueError("Invalid saved image path")
            checksum = session.get("recording_checksums", {}).get(name, "")
            if not all(
                isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v)
                for v in [checksum, image.get("sha256")]
            ):
                raise ValueError("Recording checksums are missing")
            if (
                type(image.get("size_bytes")) is not int
                or not 0 < image["size_bytes"] <= 4_000_000_000
            ):
                raise ValueError("Image recording exceeds its size limit")
            if image.get("source_sha256", checksum) != checksum:
                raise ValueError(
                    "Training images belong to a different original recording"
                )
            result.append(
                dict(
                    session_id=session["id"],
                    index=index,
                    path=name,
                    sha256=checksum,
                    image_path=str(path),
                    image_sha256=image["sha256"],
                    image_size_bytes=image["size_bytes"],
                )
            )
        return result

    def dataset(self, session, name=None, *, create=True):
        resources = self.database.list_data_resources(
            provider="collection", namespace="datasets", include_archived=True
        )
        existing = next((r for r in resources if r["name"] == session["id"]), None)
        if existing or not create:
            return existing
        label = name or session["profile"]["display_name"]
        return self.database.create_data_resource(
            provider="collection",
            namespace="datasets",
            name=session["id"],
            kind="demonstrations",
            description=label,
            metadata={
                "display_name": label,
                "session_id": session["id"],
                "managed_dataset": True,
            },
        )

    def register_source(self, resource, sources, split):
        revision = fingerprint({"sources": sources, "split": split})
        old = next(
            (
                v
                for v in self.database.get_data_resource(resource["id"])["versions"]
                if v["revision"] == revision and v["format"] == "skynet.episodes/v1"
            ),
            None,
        )
        if old:
            return old
        path = self.root / "manifests" / (revision + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        content = canonical_json({"sources": sources, "split": split})
        if not path.exists():
            path.write_text(content)
        return self.database.create_data_resource_version(
            resource["id"],
            revision=revision,
            format="skynet.episodes/v1",
            path=str(path),
            manifest_sha256=revision,
            status="RECORDED",
            metadata={
                "sources": sources,
                "split": split,
                "episodes": len(sources),
                "representation": "originals",
                "storage_location": "collection",
            },
        )

    @staticmethod
    def split(sources, validation_percent=20, seed=42):
        if (
            type(validation_percent) is not int
            or not 0 <= validation_percent <= 50
            or type(seed) is not int
            or not 0 <= seed < 2**31
        ):
            raise ValueError(
                "Validation must be 0–50 percent and the seed a non-negative integer"
            )
        count = (
            min(
                len(sources) - 1, max(1, round(len(sources) * validation_percent / 100))
            )
            if validation_percent and len(sources) > 1
            else 0
        )
        ranked = sorted(
            range(len(sources)), key=lambda i: fingerprint([seed, sources[i]["sha256"]])
        )
        validation = sorted(ranked[:count])
        return dict(
            train=[i for i in range(len(sources)) if i not in validation],
            validation=validation,
            seed=seed,
            validation_percent=validation_percent,
        )

    def options(self):
        sessions = []
        for session in self.live.list():
            resource = None
            try:
                sources = self.sources(session, require_images=False)
                reason = None
                with self.lock:
                    resource = self.dataset(session, create=False)
            except (ValueError, KeyError) as exc:
                sources, reason = [], str(exc)
            if session.get("recordings"):
                sessions.append(
                    dict(
                        id=session["id"],
                        name=session["profile"]["display_name"],
                        created_at=session["created_at"],
                        episodes=len(session["recordings"]),
                        eligible=reason is None
                        and not (resource or {}).get("archived_at"),
                        reason=reason
                        or (
                            "Dataset is archived"
                            if (resource or {}).get("archived_at")
                            else None
                        ),
                        images=len(session.get("recording_images") or {}),
                        resource_id=(resource or {}).get("id"),
                    )
                )
        policies = catalog(self.database)
        jobs = self.list()
        for job in jobs:
            policy = next((p for p in policies if p["id"] == job["format"]), {})
            job["training_setup"] = policy.get("training_setup")
            job["training_ready"] = bool(
                job.get("training_ready") and policy.get("trainable")
            )
            version = self.database.get_data_resource_version(job.get("version_id", ""))
            job["locations"] = version.get("locations", []) if version else []
            job["usage"] = (
                self.database.data_version_usage(version["manifest_sha256"])
                if version
                else []
            )
            if job.get("stage") == "CONVERTING":
                log = self.root / job["id"] / "export.log"
                if log.exists():
                    with log.open("rb") as stream:
                        stream.seek(max(0, log.stat().st_size - 1024))
                        for line in stream.read().decode(errors="replace").splitlines():
                            try:
                                progress = json.loads(line)
                                if "episodes_done" in progress:
                                    job["progress"] = progress
                            except ValueError:
                                pass
        return dict(
            formats=FORMATS,
            policies=policies,
            sessions=sessions,
            exports=jobs,
            targets=[
                dict(id="cluster", name="Training cluster"),
                dict(id="local", name="This computer"),
            ],
        )

    def create(
        self,
        session_id,
        format,
        name,
        resource_id=None,
        *,
        selections=None,
        target="local",
        validation_percent=20,
        seed=42,
        gateway="auto",
    ):
        if format not in RECIPES:
            raise ValueError("No converter is registered for this policy")
        if target not in {"local", "cluster"}:
            raise ValueError("Choose this computer or the training cluster")
        self.cluster.candidates(gateway)
        name = name.strip()
        if not name or len(name) > 100 or any(ord(c) < 32 for c in name):
            raise ValueError("Dataset name must contain 1–100 printable characters")
        if selections is None:
            selections = [dict(session_id=session_id, indices=None)]
        if (
            not selections
            or len(selections) > 25
            or len({s["session_id"] for s in selections}) != len(selections)
        ):
            raise ValueError("Select 1–25 unique collection sessions")
        sources = []
        for selection in sorted(selections, key=lambda s: s["session_id"]):
            sources.extend(
                self.sources(
                    self.live.get(selection["session_id"]), selection.get("indices"),
                    require_images="rgb" in RECIPES[format]["observations"]
                )
            )
        if (
            len(sources) > 1000
            or sum(s["image_size_bytes"] for s in sources) > 20_000_000_000
        ):
            raise ValueError("Select at most 1,000 episodes and 20 GB of source images")
        if len({s["sha256"] for s in sources}) != len(sources):
            raise ValueError(
                "The selection contains duplicate recordings; select each episode once"
            )
        split = self.split(sources, validation_percent, seed)
        if RECIPES[format]["trainable"] and not split["validation"]:
            raise ValueError(
                "DP training needs at least two episodes and a non-zero validation split"
            )
        source_root = self.live.root / "ops/datasets"
        worker = (source_root / "policy_export.py").read_text()
        arrays = "import pickle\nimport numpy as np\n" + inspect.getsource(
            ArrayUnpickler
        )
        provenance = json.loads(
            (source_root / "xpolicylab/provenance.json").read_text()
        )
        for file, checksum in provenance["files"].items():
            if digest(source_root / "xpolicylab/XPolicyLab" / file) != checksum:
                raise ValueError(
                    "Pinned policy converter files changed; review their provenance before preparing data"
                )
        runtime_lock = digest(self.live.root / "uv.lock")
        frozen_files = {
            "policy_export.py": worker,
            "arrays.py": arrays,
            "artifacts.py": (source_root / "artifacts.py").read_text(),
            "formats.json": canonical_json(
                {key: value["format"] for key, value in RECIPES.items()}
            ),
            "images.py": (self.live.root / "ops/xr/images.py").read_text(),
            "skynet_dp_training.py": (
                self.live.root / "skynet_app/adapters/dp_training.py"
            ).read_text(),
        }
        converter_sha = fingerprint([frozen_files, provenance, runtime_lock])
        identity = fingerprint(
            [sources, format, split, converter_sha, RECIPES[format]["contract"]]
        )
        with self.lock:
            if self.stopping:
                raise ValueError("The app is restarting; retry shortly")
            resource = (
                self.database.get_data_resource(resource_id)
                if resource_id
                else self.dataset(self.live.get(selections[0]["session_id"]), name)
            )
            if resource and any(j.get("resource_id") == resource["id"] and j["state"] == "DELETE_FAILED" for j in self.list()):
                raise ValueError("Finish dataset deletion before preparing it again")
            if (
                not resource
                or resource.get("archived_at")
                or resource.get("provider") != "collection"
            ):
                raise ValueError("Choose an active collection dataset")
            if resource.get("namespace") != "datasets":
                if (
                    resource.get("metadata", {}).get("session_id")
                    != selections[0]["session_id"]
                ):
                    raise ValueError(
                        "Choose a collection resource belonging to the selected session"
                    )
                resource = self.dataset(
                    self.live.get(selections[0]["session_id"]), name
                )
            metadata = {
                **resource.get("metadata", {}),
                "display_name": name,
                "managed_dataset": True,
            }
            self.database.update_data_resource(resource["id"], metadata=metadata)
            source_version = self.register_source(resource, sources, split)
            for job in self.list():
                if (
                    job.get("fingerprint") == identity
                    and job.get("resource_id") == resource["id"]
                    and job["state"] != "FAILED"
                ):
                    if job["state"] == "READY" and (
                        target == "cluster" and job.get("target") != "cluster"
                    ):
                        return self.retry(job["id"], target="cluster")
                    if (
                        job["state"] == "READY"
                        and not (self.root / job["id"] / "dataset.zip").is_file()
                    ):
                        return self.retry(job["id"], target=target)
                    return job
            identifier = str(uuid4())
            directory = self.root / identifier
            frozen = directory / "worker"
            shutil.copytree(source_root / "xpolicylab", frozen / "xpolicylab")
            for filename, content in frozen_files.items():
                (frozen / filename).write_text(content)
            job = dict(
                id=identifier,
                session_id=selections[0]["session_id"],
                selections=selections,
                resource_id=resource["id"],
                source_version_id=source_version["id"],
                source_revision=source_version["revision"],
                name=name,
                format=format,
                contract=RECIPES[format]["contract"],
                target=target,
                gateway=gateway,
                sources=sources,
                split=split,
                runtime_lock_sha256=runtime_lock,
                fingerprint=identity,
                converter_sha256=converter_sha,
                state="QUEUED",
                stage="QUEUED",
                detail="Waiting to prepare data",
                created_at=utc_now(),
                updated_at=utc_now(),
            )
            with self.database.transaction() as c:
                c.execute(
                    "INSERT INTO policy_exports VALUES (?,?)",
                    (identifier, canonical_json(job)),
                )
            self.dispatch(identifier)
            return job

    def retry(self, identifier, target=None):
        with self.lock:
            job = self.get(identifier)
            if job["state"] == "DELETE_FAILED":
                raise ValueError("Finish dataset deletion before preparing it again")
            resource = self.database.get_data_resource(job.get("resource_id", ""))
            if resource and resource.get("archived_at"):
                raise ValueError("Restore this dataset before preparing another copy")
            if identifier in self.active:
                return job
            if not job.get("source_version_id"):
                raise ValueError(
                    "This is an older export. Prepare a new version from its original recordings"
                )
            if target not in {None, "local", "cluster"}:
                raise ValueError("Unknown destination")
            job = self.update(
                identifier,
                state="QUEUED",
                target=target or job.get("target", "local"),
                error=None,
                detail="Retrying from verified files",
            )
            self.dispatch(identifier)
            return job

    def dispatch(self, identifier):
        with self.lock:
            if identifier in self.active or self.stopping:
                return
            self.active.add(identifier)
        self.executor.submit(self.prepare, identifier)

    def download(self, session, relative, target, expected, limit, expected_size=None):
        if target.is_file() and digest(target) == expected:
            return
        if self.stopping:
            raise ValueError("Preparation interrupted by app shutdown")
        remote = session["root"] + "/output/" + relative
        transport = self.live.transport(session)
        host, size = transport.file_size(remote, session["gateway"])
        if not 0 < size <= limit or (
            expected_size is not None and size != expected_size
        ):
            raise ValueError("Source recording size is invalid or changed")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".part")
        received = 0
        try:
            with temp.open("wb") as stream:
                for block in transport.stream_file_range(
                    remote, host, start=0, end=size - 1
                ):
                    if self.stopping:
                        raise ValueError("Preparation interrupted by app shutdown")
                    received += len(block)
                    if received > size:
                        raise ValueError("Source recording grew during download")
                    stream.write(block)
            if received != size or digest(temp) != expected:
                raise ValueError("Source recording checksum verification failed")
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)

    def prepare(self, identifier):
        directory = self.root / identifier
        try:
            job = self.get(identifier)
            manifest_path = directory / "output/manifest.json"
            manifest = None
            if manifest_path.exists() and job.get("manifest_sha256"):
                try:
                    manifest = verify(directory / "output", job["manifest_sha256"])
                except (ValueError, OSError):
                    pass
            if manifest is None:
                size = sum(s["image_size_bytes"] for s in job["sources"])
                if shutil.disk_usage(directory).free < size * 8 + 1_000_000_000:
                    raise ValueError(
                        "Not enough local disk space for this image preparation"
                    )
                self.update(
                    identifier,
                    state="RUNNING",
                    stage="FETCHING",
                    detail="Fetching and verifying original recordings",
                )
                sources = []
                for item in job["sources"]:
                    session = self.live.get(item.get("session_id", job["session_id"]))
                    recording = self.root / "sources" / (item["sha256"] + ".pkl")
                    images = self.root / "sources" / (item["image_sha256"] + ".hdf5") if item.get("image_sha256") else None
                    self.download(
                        session, item["path"], recording, item["sha256"], 100_000_000
                    )
                    if images is not None:
                        self.download(
                            session,
                            item["image_path"],
                            images,
                            item["image_sha256"],
                            4_000_000_000,
                            item["image_size_bytes"],
                        )
                    sources.append(
                        dict(item, recording=str(recording), images=str(images) if images else None)
                    )
                # This directory belongs only to this job. Failed partial output is never registered.
                if (directory / "output").exists():
                    shutil.rmtree(directory / "output")
                (directory / "dataset.zip.part").unlink(missing_ok=True)
                request = {
                    key: job[key]
                    for key in (
                        "format",
                        "split",
                        "contract",
                        "source_revision",
                        "converter_sha256",
                    )
                }
                request.update(output=str(directory / "output"), sources=sources)
                (directory / "request.json").write_text(canonical_json(request))
                self.update(
                    identifier,
                    stage="CONVERTING",
                    detail="Converting the selected observations and actions",
                )
                with (directory / "export.log").open("w") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            str(directory / "worker/policy_export.py"),
                            str(directory / "request.json"),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=dict(
                            os.environ,
                            PYTHONUNBUFFERED="1",
                            PYTHONDONTWRITEBYTECODE="1",
                        ),
                    )
                    with self.lock:
                        self.processes[identifier] = process
                    try:
                        result = process.wait(timeout=3600)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        raise ValueError("Dataset preparation exceeded one hour")
                if result:
                    raise ValueError(
                        "Dataset conversion failed: "
                        + (directory / "export.log").read_text()[-1800:]
                    )
                self.update(
                    identifier,
                    stage="VALIDATING",
                    detail="Verifying the prepared dataset",
                )
                manifest_sha = digest(manifest_path)
                manifest = verify(directory / "output", manifest_sha)
                self.update(identifier, manifest_sha256=manifest_sha)
            else:
                manifest_sha = job["manifest_sha256"]
            version = self.register(job, manifest, manifest_sha)
            if not (directory / "dataset.zip").is_file():
                pack(directory / "output")
            archive_sha = digest(directory / "dataset.zip")
            local = self.database.record_data_location(
                version["id"],
                kind="local",
                host="local",
                path=str(directory / "output"),
                manifest_sha256=manifest_sha,
            )
            self.update(
                identifier,
                version_id=version["id"],
                archive_sha256=archive_sha,
                size_bytes=version["size_bytes"],
                episodes=len(manifest["episodes"]),
                steps=manifest["steps"],
            )
            if job.get("target") == "cluster":
                self.update(
                    identifier,
                    stage="TRANSFERRING",
                    detail="Copying the verified dataset to the training cluster",
                )
                location = self.transfer(self.get(identifier))
                bundle = self.bundle(self.get(identifier), location)
                self.update(identifier, bundle_id=bundle["id"] if bundle else None)
            else:
                location = local
            can_train = (
                RECIPES[job["format"]]["trainable"] and location["kind"] == "cluster"
            )
            detail = (
                "Ready to use in a training experiment"
                if can_train
                else (
                    "Prepared on this computer"
                    if location["kind"] == "local"
                    else "Files prepared on the cluster; a training adapter is still required"
                )
            )
            self.update(
                identifier,
                state="READY",
                stage="READY",
                detail=detail,
                training_ready=can_train,
                error=None,
            )
        except Exception as exc:
            self.update(
                identifier,
                state="FAILED",
                error=str(exc),
                detail="Preparation failed; original recordings are preserved",
            )
        finally:
            with self.lock:
                self.processes.pop(identifier, None)
                self.active.discard(identifier)

    def register(self, job, manifest, manifest_sha):
        resource = self.database.get_data_resource(job["resource_id"])
        version = next(
            (v for v in resource["versions"] if v["revision"] == manifest_sha), None
        )
        if version is None:
            version = self.database.create_data_resource_version(
                resource["id"],
                revision=manifest_sha,
                format=manifest["format"],
                path=str(self.root / job["id"] / "output"),
                manifest_sha256=manifest_sha,
                status="LOCAL",
                size_bytes=sum(f["size_bytes"] for f in manifest["files"].values()),
                source_uri="collection:" + job["source_revision"],
                metadata={
                    **manifest,
                    "storage_location": "local",
                    "export_id": job["id"],
                    "source_version_id": job["source_version_id"],
                    "representation": job["format"],
                    "display_name": job["name"],
                },
            )
        if not self.database.get_data_resource_version(version["id"]).get(
            "derivation_id"
        ):
            self.database.create_data_derivation(
                output_version_id=version["id"],
                inputs=[
                    dict(
                        version_id=job["source_version_id"],
                        role="native_demonstrations",
                    )
                ],
                converter_repository=XPL_REPOSITORY,
                converter_commit=COMMIT,
                converter_config={
                    "recipe": job["format"],
                    "contract": job["contract"],
                    "converter_sha256": job["converter_sha256"],
                    "split": job["split"],
                },
                runtime_lock_sha256=job["runtime_lock_sha256"],
            )
        return version

    def transfer(self, job):
        failures = []
        for host in self.cluster.candidates(job.get("gateway", "auto")):
            try:
                return self._transfer_host(job, self.cluster.resolve_gateway(host))
            except ClusterError as exc:
                failures.append(str(exc))
        raise ClusterError("; ".join(failures))

    def _transfer_host(self, job, gateway):
        directory = self.root / job["id"]
        version = self.database.get_data_resource_version(job["version_id"])
        destination = f"{WORK_ROOT}/datasets/prepared/{version['manifest_sha256']}"
        script = (directory / "worker/artifacts.py").read_text()
        remote_script = f"{WORK_ROOT}/jobs/runs/{job['id']}/verify-dataset.py"
        self.cluster.write_capsule_file(job["id"], "verify-dataset.py", script, gateway)
        remote_archive = upload_capture(
            self.cluster,
            directory / "dataset.zip",
            job["id"],
            job["archive_sha256"],
            gateway,
            relative_path="dataset.zip",
            timeout=3600,
        )
        command = shlex.join(
            [
                "python3",
                remote_script,
                destination,
                version["manifest_sha256"],
                "--archive",
                remote_archive,
                "--archive-sha",
                job["archive_sha256"],
            ]
        )
        self.update(
            job["id"],
            stage="VERIFYING_COPY",
            detail="Verifying every file on the training cluster",
        )
        receipt = json.loads(self.cluster.ssh(gateway, command, timeout=3600))
        if (
            not receipt.get("verified")
            or receipt["manifest_sha256"] != version["manifest_sha256"]
        ):
            raise ValueError(
                "Training cluster verification did not match the prepared dataset"
            )
        if job["format"] in {"dp", "dp-state"}:
            self.update(
                job["id"],
                stage="CHECKING_LOADER",
                detail="Checking the DP data loader on the training cluster",
            )
            profile = CLUSTER.runtime_profiles["skynet-dp"]
            source = profile.source_prerequisites[0].path
            self.cluster.write_capsule_file(
                job["id"],
                "adapter-support/skynet_dp_training.py",
                (directory / "worker/skynet_dp_training.py").read_text(),
                gateway,
            )
            self.cluster.write_capsule_file(
                job["id"], "adapter-support/artifacts.py", script, gateway
            )
            command = shlex.join(
                [
                    str(profile.environment_path) + "/bin/python",
                    f"{WORK_ROOT}/jobs/runs/{job['id']}/adapter-support/skynet_dp_training.py",
                    "--repository",
                    str(source),
                    "--revision",
                    COMMIT,
                    "--dataset",
                    destination,
                    "--manifest-sha",
                    version["manifest_sha256"],
                    "--output",
                    f"{WORK_ROOT}/jobs/runs/{job['id']}/dataset-validation",
                    "--batch-size",
                    "1",
                    "--observation-mode",
                    "rgb" if job["format"] == "dp" else "state",
                    "--verify-only",
                ]
            )
            output = self.cluster.ssh(gateway, command, timeout=300)
            receipt = json.loads(output.strip().splitlines()[-1])
            if (
                receipt.get("manifest_sha256") != version["manifest_sha256"]
                or receipt.get("schema") not in {"skynet.dp-loader-validation/v1", "skynet.dp-loader-validation/v2"}
                or (receipt.get("schema") == "skynet.dp-loader-validation/v2" and receipt.get("observation_mode") != ("rgb" if job["format"] == "dp" else "state"))
            ):
                raise ValueError("DP loader verification returned a different dataset")
            self.update(job["id"], loader_validation=receipt)
        return self.database.record_data_location(
            version["id"],
            kind="cluster",
            host="skynet",
            path=destination,
            manifest_sha256=version["manifest_sha256"],
        )

    def bundle(self, job, location):
        if not RECIPES[job["format"]]["trainable"]:
            return None
        name, version = job["name"], job["fingerprint"][:16] + "-cluster"
        existing = next(
            (
                b
                for b in self.database.list_data_bundles()
                if b["name"] == name and b["version"] == version
            ),
            None,
        )
        return existing or self.database.create_data_bundle(
            name=name,
            version=version,
            description="Verified training data with pinned robot/camera mapping and split",
            assignments=[
                dict(
                    role="training_data",
                    version_id=job["version_id"],
                    config={"location_id": location["id"]},
                )
            ],
            metadata={
                "prepared_dataset": True,
                "adapter": RECIPES[job["format"]]["adapter"],
                "resource_id": job["resource_id"],
            },
        )

    def delete_dataset(self, resource_id, identifier=None):
        with self.lock:
            if any(
                j["id"] in self.active
                for j in self.list()
                if j.get("resource_id") == resource_id
            ):
                raise ValueError(
                    "Wait for dataset preparation to finish before deleting it"
                )
            return self.database.delete_prepared_dataset(
                resource_id, self._delete_dataset_copies, identifier=identifier
            )

    def _delete_dataset_copies(self, jobs, versions, locations):
        prepared = [v for v in versions if v["format"] != "skynet.episodes/v1"]
        job_ids = {j["id"] for j in jobs}
        for version in prepared:
            if json.loads(version["metadata_json"]).get("export_id") not in job_ids:
                raise ValueError("Dataset includes files not managed by preparation")
        remote_jobs = [
            j["id"]
            for j in jobs
            if j.get("version_id") and j.get("target") == "cluster"
        ]
        cluster_hashes = {
            v["manifest_sha256"]
            for v in prepared
            if any(
                j.get("version_id") == v["id"] and j["id"] in remote_jobs for j in jobs
            )
        }
        for location in locations:
            if location["kind"] == "cluster":
                expected = (
                    f"{WORK_ROOT}/datasets/prepared/{location['manifest_sha256']}"
                )
                if location["host"] != "skynet" or location["path"] != expected:
                    raise ValueError(
                        "Cluster copy is outside the prepared dataset directory"
                    )
                cluster_hashes.add(location["manifest_sha256"])
        if remote_jobs or cluster_hashes:
            payload = dict(
                root=str(WORK_ROOT),
                jobs=remote_jobs,
                prepared=sorted(cluster_hashes),
                cluster=True,
            )
            command = shlex.join(
                [
                    "python3",
                    "-c",
                    Path(dataset_cleanup.__file__).read_text(),
                    canonical_json(payload),
                ]
            )
            failures = []
            for host in self.cluster.candidates("auto"):
                try:
                    receipt = json.loads(
                        self.cluster.ssh(
                            self.cluster.resolve_gateway(host), command, timeout=60
                        )
                    )
                    if receipt.get("removed") is not True:
                        raise ValueError("Cluster did not confirm dataset deletion")
                    break
                except ClusterError as exc:
                    failures.append(str(exc))
            else:
                raise ClusterError("; ".join(failures))
        dataset_cleanup.cleanup(self.root, jobs=sorted(job_ids))
        # Source manifests and the source cache describe original recordings and
        # may be shared. They contain no converted training data.

    def remove_local_copy(self, identifier):
        with self.lock:
            job = self.get(identifier)
            if identifier in self.active or job["state"] != "READY":
                raise ValueError(
                    "Wait for preparation to finish before removing its local copy"
                )
            version = self.database.get_data_resource_version(job.get("version_id", ""))
            if not version:
                raise ValueError("This preparation has no registered version")
            if self.database.data_version_usage(version["manifest_sha256"]):
                raise ValueError(
                    "This dataset is pinned by an experiment; retain its copies for reproducibility"
                )
            cluster = next(
                (
                    l
                    for l in version["locations"]
                    if l["kind"] == "cluster" and l["status"] == "AVAILABLE"
                ),
                None,
            )
            if not cluster:
                raise ValueError(
                    "Keep at least one verified copy; transfer to the training cluster first"
                )
            gateway = self.cluster.resolve_gateway("auto")
            script = f"{WORK_ROOT}/jobs/runs/{identifier}/verify-dataset.py"
            self.cluster.ssh(
                gateway,
                shlex.join(
                    ["python3", script, cluster["path"], version["manifest_sha256"]]
                ),
                timeout=3600,
            )
            directory = self.root / identifier
            if (directory / "output").exists():
                shutil.rmtree(directory / "output")
            (directory / "dataset.zip").unlink(missing_ok=True)
            self.database.record_data_location(
                version["id"],
                kind="local",
                host="local",
                path=str(directory / "output"),
                manifest_sha256=version["manifest_sha256"],
                status="REMOVED",
            )
            return self.update(
                identifier,
                local_removed=True,
                detail="Verified dataset retained on the training cluster",
            )

    def artifact(self, identifier, name):
        job = self.get(identifier)
        if name not in {"dataset.zip", "manifest.json", "export.log"}:
            raise KeyError("Dataset file not found")
        if name != "export.log" and not job.get("version_id"):
            raise ValueError("Preparation is not complete")
        path = (
            self.root
            / identifier
            / ("output/manifest.json" if name == "manifest.json" else name)
        )
        if not path.is_file():
            raise KeyError(
                "Local copy is unavailable; use the verified cluster copy or prepare a local copy again"
            )
        return path
