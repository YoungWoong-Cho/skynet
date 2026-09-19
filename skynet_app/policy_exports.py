"""One preparation workflow for collection datasets and their verified locations.

The historic export API uses this same service. Immutable content versions and
lineage live in the data registry; mutable execution progress stays in the job.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import threading
from uuid import uuid4

from ops.datasets.artifacts import digest
from .recording_guard import guarded_recording
from .database import canonical_json, utc_now
from . import dataset_cleanup
from .dataset_formats import catalog, resolve_adapter

DATASET_FORMAT = "skynet.recording-dataset/v1"
from .live_xr import TERMINAL
from .live_xr_review import ArrayUnpickler
from .cluster_runtime import ClusterClient, ClusterError, WORK_ROOT
from .policy_exports_cluster import ClusterPolicyPreparation
from .observation_contracts import validate_requirements
from .observation_preparation import ObservationPreparation




def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def conversion_failure(returncode, output):
    """Describe worker termination without presenting progress as its cause."""
    progress = None
    diagnostics = []
    for line in output.splitlines():
        try:
            item = json.loads(line)
        except (ValueError, TypeError):
            item = None
        if isinstance(item, dict) and "episodes_done" in item:
            progress = f"{item['episodes_done']}/{item.get('episodes_total', '?')} episodes"
        elif line.strip() and not any(
            word in line for word in ("resource_tracker", "leaked semaphore", "warnings.warn(")
        ):
            diagnostics.append(line.strip())
    if returncode < 0:
        try:
            reason = f"signal {signal.Signals(-returncode).name}"
        except ValueError:
            reason = f"signal {-returncode}"
    else:
        reason = f"exit code {returncode}"
    message = f"Dataset conversion stopped ({reason})"
    if progress:
        message += f" after {progress}"
    if diagnostics:
        message += ": " + "\n".join(diagnostics[-8:])[-1200:]
    else:
        message += (
            ". The worker exited without an exception report; "
            "check the preparation and app logs for an interruption."
        )
    return message


class PolicyExportService(ClusterPolicyPreparation):
    def __init__(self, reviews, root=None, cluster=None):
        self.reviews, self.live = reviews, reviews.live
        self.database = self.live.database
        self.root = Path(root or self.live.root / "data/policy-exports")
        self.cluster = cluster or ClusterClient()
        self.observations = ObservationPreparation(self)
        self.lock = self.database.operation_lock("dataset-preparation")
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dataset-preparation"
        )
        self.active = set()
        self.stopping = False
        self.monitor_stop = threading.Event()
        self.monitor_thread = None
        self.monitor_interval = 2.0
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
        with self.lock:
            if self.stopping:
                return
            if self.monitor_thread is None:
                self.monitor_thread = threading.Thread(target=self._monitor, name="dataset-preparation-monitor", daemon=True)
                self.monitor_thread.start()
        for job in self.list():
            if job.get("format") == DATASET_FORMAT and job["state"] not in {"READY", "FAILED", "DELETE_FAILED"}:
                self.dispatch(job["id"])

    def _monitor(self):
        while not self.monitor_stop.wait(self.monitor_interval):
            try:
                self.observations.tick()
            except Exception:
                import logging
                logging.getLogger(__name__).exception("Observation monitor could not refresh cluster work")
            for job in self.list():
                if job.get("format") == DATASET_FORMAT and job["state"] not in {"READY", "FAILED", "DELETE_FAILED"}:
                    self.dispatch(job["id"])

    def stop(self):
        with self.lock:
            self.stopping = True
            self.monitor_stop.set()
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=5)
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
                self.reviews.source_for_session(session, index)
                checksum = session.get("recording_checksums", {}).get(name, "")
                if not re.fullmatch(r"[a-f0-9]{64}", checksum):
                    raise ValueError("Recording checksum is missing")
                result.append(dict(session_id=session["id"], index=index, path=name, sha256=checksum, image_size_bytes=0))
                continue
            if not image:
                raise ValueError(
                    "This session has no completed training images. Wait for image preparation, or collect a new session with training images enabled."
                )
            recording_path = PurePosixPath(name)
            if (recording_path.is_absolute() or ".." in recording_path.parts
                    or not recording_path.is_relative_to("recordings") or recording_path.suffix != ".pkl"):
                raise ValueError("Invalid saved recording path")
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

    def dataset(self, session, name=None, *, create=True, overfit_episode=None):
        identity = session["id"] if overfit_episode is None else f"{session['id']}:overfit:{overfit_episode}"
        existing = self.database.find_collection_dataset(session["id"], identity)
        if existing or not create:
            return existing
        label = name or session["profile"]["display_name"]
        return self.database.create_data_resource(
            category="dataset",
            provider="collection",
            namespace="datasets",
            source_key=identity,
            display_name=label,
            kind="demonstrations",
            description=label,
            metadata={
                "session_id": session["id"],
                "managed_dataset": True,
                **({"overfit_episode": overfit_episode} if overfit_episode is not None else {}),
                **({"recording_session_id": session["id"]}
                   if overfit_episode is not None and len(session.get("recordings", [])) == 1 else {}),
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
        content = canonical_json({"sources": sources, "split": split})
        from .metadata_objects import MetadataObjects
        path = MetadataObjects(self.database).put(content.encode())
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

    @staticmethod
    def recording_locations(session):
        archived = (session.get("archive") or {}).get("state") in {"VERIFIED", "CLEANUP_PENDING", "READY"}
        root = session.get("archive", {}).get("root") if archived else session.get("root")
        if not root:
            return []
        directories = set()
        for relative in session.get("recordings") or []:
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts:
                continue
            directories.add(str(PurePosixPath(root) / path.parent) if archived else str(PurePosixPath(root) / "output" / path.parent))
        return [dict(kind="remote", host="sky2" if archived else session.get("gateway") or session.get("profile", {}).get("gateway") or "Collection host", path=path) for path in sorted(directories)]

    def preparation_options(self, session_id, *, workspace_database=None):
        session = self.live.get(session_id)
        resource = self.database.find_collection_dataset(session_id)
        try:
            self.sources(session, require_images=False)
            reason = "Dataset is archived" if (resource or {}).get("archived_at") else None
        except (ValueError, KeyError) as exc:
            reason = str(exc)
        return dict(session=dict(id=session["id"], name=session["profile"]["display_name"],
                    episodes=len(session.get("recordings", [])), eligible=reason is None,
                    reason=reason, resource_id=(resource or {}).get("id")),
                    resource=resource, adapters=catalog(workspace_database or self.database),
                    output_format=DATASET_FORMAT)

    def options(self, *, workspace_database=None):
        resources = self.database.list_data_resources(
            provider="collection", namespace="datasets", include_archived=True)
        session_resources = {}
        for resource in resources:
            for identifier in (resource["source_key"], resource.get("metadata", {}).get("recording_session_id")):
                if identifier is not None:
                    session_resources.setdefault(identifier, resource)
        sessions = []
        for session in self.live.list(include_private=True):
            if not session.get("recordings"):
                continue
            resource = session_resources.get(session["id"])
            try:
                self.sources(session, require_images=False)
                reason = "Dataset is archived" if (resource or {}).get("archived_at") else None
            except (ValueError, KeyError) as exc:
                reason = str(exc)
            sessions.append(dict(id=session["id"], name=session["profile"]["display_name"],
                created_at=session["created_at"], episodes=len(session["recordings"]),
                eligible=reason is None, reason=reason, resource_id=(resource or {}).get("id"),
                locations=self.recording_locations(session)))
        jobs = self.job_overview(workspace_database=workspace_database)["exports"]
        resource_sessions = {s["resource_id"]: s["id"] for s in sessions if s["resource_id"]}
        for job in jobs:
            job["recording_session_id"] = resource_sessions.get(job.get("resource_id"))
        return dict(adapters=catalog(workspace_database or self.database), sessions=sessions,
                    exports=jobs, output_format=DATASET_FORMAT,
                    targets=[dict(id="cluster", name="Training cluster")])

    def job_overview(self, *, workspace_database=None):
        # Progress reads never schedule conversion or touch cluster files.
        jobs = [job for job in self.list() if job.get("format") == DATASET_FORMAT and not job.get("retired_at")]
        active_adapters = (workspace_database or self.database).active_adapter_ids(
            {job["adapter_id"] for job in jobs})
        versions = self.database.get_data_resource_versions(
            {job["version_id"] for job in jobs if job.get("version_id")})
        usage = self.database.data_version_usage_many(
            {version["manifest_sha256"] for version in versions.values()},
            workspace_id=getattr(workspace_database, "workspace_id", None))
        for job in jobs:
            version = versions.get(job.get("version_id"))
            job["locations"] = version.get("locations", []) if version else []
            job["usage"] = usage.get(version["manifest_sha256"], []) if version else []
            if job["adapter_id"] not in active_adapters or (version and version.get("retirement")):
                job["training_ready"] = False
                job["training_setup"] = None
        return {"output_format": DATASET_FORMAT, "exports": jobs}

    def _frozen_conversion_files(self, selection):
        root = self.live.root
        files = {
            "recording_prepare.py": (root / "ops/datasets/recording_prepare.py").read_text(),
            "recording_probe.py": (root / "ops/datasets/recording_probe.py").read_text(),
            "recording_dataset.py": (root / "skynet_app/adapters/recording_dataset.py").read_text(),
            "trajectory.py": (root / "skynet_app/trajectory.py").read_text(),
            "arrays.py": "import pickle\nimport numpy as np\n" + inspect.getsource(ArrayUnpickler),
            "recording_metadata.py": (root / "ops/xr/recording_metadata.py").read_text(),
            "scene_geometry.py": (root / "skynet_app/adapters/scene_geometry.py").read_text(),
            "observation_artifacts.py": (root / "skynet_app/adapters/observation_artifacts.py").read_text(),
            "observation_contracts.py": (root / "skynet_app/observation_contracts.py").read_text(),
        }
        for relative, content in selection.get("capsule_files", {}).items():
            name = PurePosixPath(relative)
            if name.is_absolute() or ".." in name.parts or name.parts[0] != "adapter-support":
                raise ValueError("Conversion support must be inside adapter-support")
            name = name.relative_to("adapter-support").as_posix()
            if name in files and files[name] != content:
                raise ValueError("Adapter support does not match the current recording schema; update its version")
            files[name] = content
        files["conversion-dependencies.json"] = canonical_json(selection.get("conversion_dependencies") or [])
        return files

    @guarded_recording
    def create(
        self,
        session_id,
        adapter_id,
        name,
        resource_id=None,
        *,
        adapter_version_id,
        adapter_data_preset=None,
        selections=None,
        target="cluster",
        validation_percent=20,
        seed=42,
        gateway="auto",
        overfit_episode=None,
    ):
        selection = resolve_adapter(self.database, adapter_id, adapter_version_id, adapter_data_preset)
        adapter = {key: selection[key] for key in (
            "adapter_id", "adapter_version_id", "adapter_version_number", "adapter_manifest_sha256",
            "adapter", "name", "adapter_data_preset")}
        requirements = {key: selection.get(key) for key in (
            "contract", "observations", "action_representation", "observation_requirements", "temporal", "preprocessing")}
        format = DATASET_FORMAT
        if target != "cluster":
            raise ValueError("Collection datasets are prepared and retained on sky2")
        self.cluster.candidates(gateway)
        name = name.strip()
        if not name or len(name) > 100 or any(ord(c) < 32 for c in name):
            raise ValueError("Dataset name must contain 1–100 printable characters")
        if overfit_episode is not None:
            if type(overfit_episode) is not int or overfit_episode < 0:
                raise ValueError("Select a valid recording number for single-episode training")
            if selections is not None or resource_id is not None:
                raise ValueError("Single-episode overfit creates its own dataset from the selected session")
            selections = [dict(session_id=session_id, indices=[overfit_episode])]
        if selections is None:
            selections = [dict(session_id=session_id, indices=None)]
        if (
            not selections
            or len(selections) > 25
            or len({s["session_id"] for s in selections}) != len(selections)
        ):
            raise ValueError("Select 1–25 unique collection sessions")
        sources = []
        sessions = {}
        for source_selection in sorted(selections, key=lambda s: s["session_id"]):
            sessions[source_selection["session_id"]] = self.live.get(source_selection["session_id"])
            sources.extend(
                self.sources(
                    sessions[source_selection["session_id"]], source_selection.get("indices"),
                    require_images=False
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
        minimum = selection.get("minimum_episodes", 1)
        if len(sources) < minimum:
            raise ValueError(f"This adapter requires at least {minimum} episodes")
        supported = set(selection.get("supported_robots") or [])
        if supported:
            unsupported = sorted({session["profile"].get("robot") for session in sessions.values()} - supported)
            if unsupported:
                raise ValueError("This adapter does not support these recording hands: " + ", ".join(str(value) for value in unsupported))
        observation_contract = validate_requirements(selection["observation_requirements"])
        observation_files = self.observations.frozen_files() if observation_contract["streams"] else {}
        split = self.split(sources, validation_percent, seed)
        if not split["validation"]:
            if selection.get("validation_required"):
                raise ValueError("This adapter requires a separate validation episode; choose a nonzero validation percentage")
            split["mode"] = "training_only"
        runtime_lock = digest(self.live.root / "uv.lock")
        frozen_files = self._frozen_conversion_files(selection)
        converter_sha = fingerprint([frozen_files, runtime_lock])
        identity = fingerprint([sources, adapter, split, converter_sha, requirements,
                                fingerprint(observation_files)])
        with self.lock:
            if self.stopping:
                raise ValueError("The app is restarting; retry shortly")
            resource = (
                self.database.get_data_resource(resource_id)
                if resource_id
                else self.dataset(sessions[selections[0]["session_id"]], name, overfit_episode=overfit_episode)
            )
            jobs = self.list()
            if resource and any(j.get("resource_id") == resource["id"] and j["state"] == "DELETE_FAILED" for j in jobs):
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
                    sessions[selections[0]["session_id"]], name
                )
            # Preparation adds immutable versions to an existing identity.
            # Use its current display label without changing the resource or
            # labels already captured in earlier conversion receipts.
            name = resource["display_name"]
            for job in jobs:
                if (
                    job.get("fingerprint") == identity
                    and job.get("resource_id") == resource["id"]
                    and job["state"] != "FAILED"
                ):
                    return job
            source_version = self.register_source(resource, sources, split)
            identifier = str(uuid4())
            directory = self.root / identifier
            frozen = directory / "worker"
            frozen.mkdir(parents=True)
            for filename, content in frozen_files.items():
                target_file = frozen / filename
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(content)
            observation_dir = directory / "observation-worker"
            observation_dir.mkdir()
            for filename, content in observation_files.items():
                (observation_dir / filename).write_text(content)
            job = dict(
                observation_contract=observation_contract,
                observation_worker_sha256=fingerprint(observation_files),
                id=identifier,
                session_id=selections[0]["session_id"],
                selections=selections,
                resource_id=resource["id"],
                source_version_id=source_version["id"],
                source_revision=source_version["revision"],
                name=name,
                format=format,
                contract=selection["contract"],
                adapter=adapter,
                adapter_id=selection["adapter_id"],
                adapter_version_id=selection["adapter_version_id"],
                adapter_data_preset=selection["adapter_data_preset"],
                requirements=requirements,
                training_setup=selection["training_setup"],
                loader_validation_spec=selection["loader_validation"],
                conversion_dependencies=selection.get("conversion_dependencies") or [],
                target=target,
                gateway="sky2",
                execution="cluster",
                sources=sources,
                split=split,
                split_mode=selection.get("split_mode", "episode"),
                runtime_lock_sha256=runtime_lock,
                fingerprint=identity,
                converter_sha256=converter_sha,
                state="QUEUED",
                stage="QUEUED",
                detail="Checking required observations before dataset conversion",
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
            if job.get("format") != DATASET_FORMAT:
                raise ValueError("Prepare this recording with a current adapter; the old conversion scheme is retired")
            if target not in {None, "cluster"}:
                raise ValueError("Collection datasets are prepared and retained on sky2")
            if (job.get("cluster_script") or job.get("preflight_script")) and job["state"] not in {"FAILED", "READY"}:
                self.dispatch(identifier)
                return job
            if job["state"] == "READY" and job.get("version_id"):
                return job
            self.observations.store.retry(identifier)
            job = self.update(
                identifier,
                state="QUEUED",
                stage="QUEUED",
                target="cluster",
                execution="cluster",
                preflight_script=None,
                preflight_job_id=None,
                preflight_sha256=None,
                cluster_script=None,
                cluster_job_id=None,
                cluster_root=None,
                training_ready=False,
                error=None,
                detail="Retrying from verified files",
                previous_cluster_root=job.get("previous_cluster_root") or job.get("cluster_root"),
            )
            self.dispatch(identifier)
            return job

    def dispatch(self, identifier):
        with self.lock:
            if identifier in self.active or self.stopping:
                return
            self.active.add(identifier)
            self.executor.submit(self.prepare, identifier)


    def prepare(self, identifier):
        try:
            self._prepare_cluster(identifier)
        finally:
            with self.lock:
                self.active.discard(identifier)


    def register(self, job, manifest, manifest_sha, *, remote_path=None):
        resource = self.database.get_data_resource(job["resource_id"])
        version = next(
            (v for v in resource["versions"] if v["revision"] == manifest_sha), None
        )
        if version is None:
            with self.database.transaction() as connection:
                version = self.database._insert_data_resource_version(
                    connection, resource["id"],
                    revision=manifest_sha,
                    format=manifest["format"],
                    path=remote_path or str(self.root / job["id"] / "output"),
                    manifest_sha256=manifest_sha,
                    status="ON_CLUSTER" if remote_path else "LOCAL",
                    size_bytes=len(canonical_json(manifest).encode()),
                    source_uri="collection:" + job["source_revision"],
                    metadata={
                        **manifest,
                        "storage_location": "cluster" if remote_path else "local",
                        "export_id": job["id"],
                        "source_version_id": job["source_version_id"],
                        "representation": job["format"],
                        "display_name": job["name"],
                    },
                )
                self.observations.store.bind_version(connection, job["id"], version["id"])
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
                converter_repository=job["training_setup"]["repository"],
                converter_commit=job["training_setup"]["revision"],
                converter_config={
                    "adapter": job["adapter"],
                    "contract": job["contract"],
                    "converter_sha256": job["converter_sha256"],
                    "split": job["split"],
                },
                runtime_lock_sha256=job["runtime_lock_sha256"],
            )
        return version

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
            resource = self.database.get_data_resource(resource_id)
            if resource is None:
                raise KeyError("Dataset not found")
            managed = (resource["provider"], resource["namespace"]) == ("collection", "datasets")
            # External registrations refer to source files this app does not own.
            # Their metadata is removable; their source files must be retained.
            cleanup = self._delete_dataset_copies if managed else lambda jobs, versions, locations: None
            return self.database.delete_prepared_dataset(resource_id, cleanup, identifier=identifier)

    def _delete_dataset_copies(self, jobs, versions, locations):
        prepared = [v for v in versions if v["format"] != "skynet.episodes/v1"]
        prepared_ids = {v["id"] for v in prepared}
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
            # Source manifests can live beside archived recordings on sky2.
            # They are shared provenance, not generated training copies.
            if location["version_id"] not in prepared_ids:
                continue
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
            configured = self.cluster.candidates("auto")
            recorded = [
                job.get("gateway") for job in jobs
                if job["id"] in remote_jobs and job.get("gateway") in configured
            ]
            # Use the confirmed preparation host first; retain configured
            # fallback without accepting arbitrary hosts from saved jobs.
            for host in dict.fromkeys([*recorded, *configured]):
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

    def artifact(self, identifier, name):
        self.get(identifier)
        if name != "export.log":
            raise KeyError("Prepared data is stored on the cluster")
        path = self.root / identifier / name
        if not path.is_file():
            raise KeyError("Log is stored on the cluster")
        return path
