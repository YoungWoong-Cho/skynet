"""Remote-only preparation and explicit streaming downloads for collection data."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
from uuid import uuid4

from .cluster_config import CLUSTER
from .sbatch import SHEBANG, cpu_thread_exports, sbatch_header, shell_prelude
from .cluster_runtime import DEFAULT_GATEWAY, ClusterError, SubmissionOutcomeUnknown, WORK_ROOT
from .database import canonical_json
from .training_contracts import RECORDING_DATASET_FORMAT as DATASET_FORMAT
from .remote_artifacts import RemoteArtifact
from .recording_preflight import RecordingPreflight
from .observation_preparation import ObservationsPending
from .preparation_states import TERMINAL_FAILURE_STATES, RUNNING_STATES, WAITING_STATES, observed_state


class ArchivePending(ValueError):
    pass


class ClusterPolicyPreparation(RecordingPreflight):
    def _archived_sources(self, job):
        archive = self.live.archive
        sources = []
        sessions = {}
        for item in job["sources"]:
            session_id = item.get("session_id", job["session_id"])
            if session_id not in sessions:
                sessions[session_id] = self.live.get(session_id)
            session = sessions[session_id]
            descriptor = session.get("archive") or {}
            if descriptor.get("state") not in {"VERIFIED", "CLEANUP_PENDING", "READY"}:
                archive.ensure(session_id)
                raise ArchivePending("Waiting for the collection archive to be verified on the training cluster")
            transport, _, recording = archive.resolve(session, item["path"])
            if transport is not archive.cluster:
                raise ValueError("Prepared collection sources must be stored on the training cluster")
            images = None
            if item.get("image_path"):
                image_transport, _, images = archive.resolve(session, item["image_path"])
                if image_transport is not archive.cluster:
                    raise ValueError("Prepared image sources must be stored on the training cluster")
            sources.append(dict(item, recording=recording, images=images,
                                session_profile=dict(session.get("profile") or {})))
        return sources

    def _loader_request(self, job, remote_worker, attempt_root):
        declaration = job["loader_validation_spec"]
        profile = CLUSTER.runtime_profiles[declaration["runtime_profile"]]
        setup = job["training_setup"]
        source = str(profile.source_prerequisites[0].path) if profile.source_prerequisites else setup.get("source_path")
        if not source:
            raise ValueError("Adapter validation has no pinned source checkout")
        replacements = {"{repository}": source, "{revision}": setup["revision"],
                        "{worker}": remote_worker, "{output}": attempt_root + "/dataset-validation"}
        argv = []
        for token in declaration["argv"]:
            for key, value in replacements.items():
                token = token.replace(key, str(value))
            argv.append(token)
        return dict(argv=[str(profile.environment_path) + "/bin/python",
                          remote_worker + "/" + declaration["script"], *argv],
                    schemas=declaration["schemas"], mode=declaration.get("mode", "rgb"))

    def _stage_cluster_attempt(self, job):
        sources = self._preflight_sources(job, self._archived_sources(job))
        sources = self.observations.ensure(job, sources)
        queue = CLUSTER.queue(CLUSTER.defaults.background_queue_policy)
        shape = CLUSTER.defaults.background_jobs.recording_preparation
        self.update(job["id"], state="STAGING", stage="STAGING", error=None,
                    cluster_partition=queue.partition, cluster_account=queue.account, cluster_cpus=shape.cpus_per_task,
                    detail="Preparing shared recording data on the training cluster")
        attempt_id = str(uuid4())
        relative = f"preparation/{attempt_id}"
        root = f"{WORK_ROOT}/jobs/runs/{job['id']}/{relative}"
        worker = root + "/worker"
        local_worker = self.root / job["id"] / "worker"
        files = {f"{relative}/worker/{path.relative_to(local_worker)}": path.read_text()
                 for path in sorted(local_worker.rglob("*"))
                 if path.is_file() and "__pycache__" not in path.parts}
        files[f"{relative}/worker/cluster_worker.py"] = Path(__file__).with_name("policy_export_worker.py").read_text()
        request = {key: job[key] for key in ("format", "adapter", "requirements", "split", "contract",
                                           "source_revision", "converter_sha256")}
        request.update(job_id=job["id"], attempt_id=attempt_id, sources=sources,
                       observation_requirements=job["observation_contract"],
                       observations=job["requirements"].get("observations") or [],
                       adapter_data_preset=job["adapter_data_preset"],
                       preprocessing=job["requirements"].get("preprocessing") or {},
                       action_representation=job["requirements"].get("action_representation"),
                       output=root + "/output", prepared_root=f"{WORK_ROOT}/datasets/prepared",
                       recording_root=f"{WORK_ROOT}/datasets/recordings",
                       receipt_path=root + "/result.json", loader=self._loader_request(job, worker, root))
        files[relative + "/request.json"] = canonical_json(request)
        self.cluster.write_capsule_files(job["id"], files, DEFAULT_GATEWAY)
        profile = CLUSTER.runtime_profile(CLUSTER.defaults.background_runtime_profile)
        interpreter = [str(profile.environment_path) + "/bin/python"]
        dependencies = job.get("conversion_dependencies") or []
        setup = []
        if dependencies:
            uv_bootstrap = shlex.quote(f"{CLUSTER.paths.uv_cache}/bootstrap-{CLUSTER.defaults.uv_version}/bin/uv")
            setup = ['UV_BIN=$(command -v uv || true)',
                     f'if [[ -z "$UV_BIN" && -x {uv_bootstrap} ]]; then UV_BIN={uv_bootstrap}; fi',
                     f'if [[ -z "$UV_BIN" && -x {shlex.quote(str(WORK_ROOT) + "/.local/bin/uv")} ]]; then UV_BIN={shlex.quote(str(WORK_ROOT) + "/.local/bin/uv")}; fi',
                     'if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then UV_BIN="$HOME/.local/bin/uv"; fi',
                     'test -n "$UV_BIN" || { echo "uv is required for recording preparation" >&2; exit 69; }']
            interpreter = ["run", "--no-project", "--python", interpreter[0]]
            for package in dependencies:
                interpreter.extend(["--with", package])
            interpreter.append("python")
        script = "\n".join([
            SHEBANG,
            *sbatch_header(job_name=f"prepare-{job['id'][:8]}", queue=queue, cpus=shape.cpus_per_task,
                           memory_gb=shape.memory_gb, time_limit=shape.time_limit, output=f"{root}/export.log"),
            *shell_prelude(umask="077"), "export CUDA_VISIBLE_DEVICES=",
            cpu_thread_exports(shape.cpus_per_task),
            "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1",
            f"export TMPDIR={shlex.quote(root + '/tmp')}", 'mkdir -p "$TMPDIR"',
            f"export UV_CACHE_DIR={shlex.quote(CLUSTER.paths.uv_cache)}", *setup,
            (('"$UV_BIN" ' if dependencies else "") + shlex.join([*interpreter, worker + "/cluster_worker.py", root + "/request.json"])), ""])
        return self.update(job["id"], execution="cluster", attempt_id=attempt_id,
                           cluster_root=root, cluster_script=script, cluster_job_id=None,
                           state="SUBMITTING", stage="SUBMITTING", error=None,
                           detail="Submitting shared recording preparation")

    def _prepare_cluster(self, identifier):
        job = self.get(identifier)
        try:
            if not job.get("cluster_script"):
                job = self._stage_cluster_attempt(job)
            if not job.get("cluster_job_id"):
                submission = self.cluster.submit_script(job["cluster_script"], identifier, DEFAULT_GATEWAY,
                    submission_key=f"{identifier}-preparation-{job['attempt_id']}")
                job = self.update(identifier, cluster_job_id=submission.job_id, gateway=submission.gateway,
                                  state="PENDING", stage="QUEUED",
                                  detail=f"Waiting for {CLUSTER.defaults.background_jobs.recording_preparation.cpus_per_task} CPU slots on the training cluster", error=None)
            _, statuses = self.cluster.job_statuses([job["cluster_job_id"]], DEFAULT_GATEWAY)
            status = statuses.get(job["cluster_job_id"])
            if not status:
                return
            state = status["State"]
            if state != "COMPLETED" and state not in TERMINAL_FAILURE_STATES:
                current = observed_state(state, job["state"])
                detail = (
                    "Preparing and validating data on the cluster" if state in RUNNING_STATES
                    else "Waiting for cluster CPU resources" + (f": {status['Reason']}" if status.get("Reason") else "")
                    if state in WAITING_STATES else f"Awaiting scheduler confirmation ({state})"
                )
                self.update(identifier, state=current, stage="CONVERTING" if current == "RUNNING" else "QUEUED",
                            error=None, detail=detail)
                return
            if state != "COMPLETED":
                message = f"Cluster preparation ended as {state}"
                try:
                    _, raw = self.cluster.read_file(job["cluster_root"] + "/result.json", DEFAULT_GATEWAY, max_bytes=100_000)
                    result = json.loads(raw)
                    if result.get("attempt_id") == job["attempt_id"]:
                        message = result.get("error") or message
                except (ClusterError, ValueError):
                    # Bootstrap failures happen before the worker can write a receipt.
                    try:
                        _, output = self.cluster.read_file(job["cluster_root"] + "/export.log", DEFAULT_GATEWAY, max_bytes=8000)
                        if output.strip():
                            message += ": " + output.strip()[-2000:]
                    except (ClusterError, ValueError):
                        pass
                raise ValueError(message)
            _, raw = self.cluster.read_file(job["cluster_root"] + "/result.json", DEFAULT_GATEWAY, max_bytes=10_000_000)
            result = json.loads(raw)
            self._publish_cluster_result(job, result)
        except ObservationsPending as exc:
            self.update(identifier, state="QUEUED", stage="OBSERVATIONS", detail=str(exc), error=None)
        except ArchivePending as exc:
            self.update(identifier, state="QUEUED", stage="ARCHIVING", detail=str(exc), error=None)
        except SubmissionOutcomeUnknown as exc:
            self.update(identifier, state="SUBMISSION_UNKNOWN", stage="SUBMITTING", error=str(exc),
                        detail="Checking whether the cluster accepted preparation; retries reuse the same submission")
        except ClusterError as exc:
            # An unavailable gateway cannot turn a possibly running job into a new attempt.
            self.update(identifier, error=str(exc), detail="Cluster status is temporarily unavailable; refresh to reconnect")
        except Exception as exc:
            self.update(identifier, state="FAILED", stage="FAILED", error=str(exc),
                        detail="Preparation failed; verified original recordings are preserved on the training cluster")

    def _publish_cluster_result(self, job, result):
        manifest_sha = result.get("manifest_sha256", "")
        expected_path = f"{WORK_ROOT}/datasets/prepared/{manifest_sha}"
        if (result.get("schema") != "skynet.cluster-preparation/v2" or result.get("job_id") != job["id"]
                or result.get("attempt_id") != job["attempt_id"] or result.get("verified") is not True
                or not re.fullmatch(r"[a-f0-9]{64}", manifest_sha) or result.get("path") != expected_path):
            raise ValueError("Cluster preparation receipt did not match this attempt")
        _, raw_manifest = self.cluster.read_file(expected_path + "/manifest.json", DEFAULT_GATEWAY, max_bytes=10_000_000)
        if hashlib.sha256(raw_manifest.encode()).hexdigest() != manifest_sha:
            raise ValueError("Prepared manifest changed after cluster verification")
        manifest = json.loads(raw_manifest)
        if (manifest != result.get("manifest") or manifest.get("format") != DATASET_FORMAT
                or manifest.get("adapter") != job["adapter"] or manifest.get("contract") != job["contract"]):
            raise ValueError("Cluster preparation returned a different adapter dataset")
        loader = self._loader_request(job, job["cluster_root"] + "/worker", job["cluster_root"])
        receipt = result.get("loader_validation") or {}
        if (receipt.get("schema") not in loader["schemas"] or receipt.get("manifest_sha256") != manifest_sha
                or receipt.get("observation_mode", "rgb") != loader["mode"]):
            raise ValueError("Cluster preparation did not validate its training reader")
        self.observations.store.register_shared(job, manifest.get("shared_artifacts", []))
        version = self.register(job, manifest, manifest_sha, remote_path=expected_path)
        self.database.record_data_location(version["id"], kind="cluster", host=CLUSTER.id,
                                            path=expected_path, manifest_sha256=manifest_sha)
        self.update(job["id"], version_id=version["id"], manifest_sha256=manifest_sha,
                    size_bytes=version["size_bytes"], episodes=len(manifest["episodes"]), steps=manifest["steps"],
                    loader_validation=receipt, bundle_id=None, state="READY", stage="READY",
                    error=None, training_ready=True, detail="Ready on the cluster; shared recording data verified")

    def status(self, identifier):
        job = self.get(identifier)
        if job["state"] not in {"READY", "FAILED", "DELETE_FAILED"}:
            self.dispatch(identifier)
        return job

    def remote_artifact(self, identifier, name):
        job = self.get(identifier)
        if name not in {"manifest.json", "export.log"}:
            raise KeyError("Dataset file not found")
        if name == "export.log":
            path = job.get("cluster_root", "") + "/export.log" if job.get("cluster_root") else None
        else:
            if not job.get("version_id"):
                raise ValueError("Preparation is not complete")
            version = self.database.get_data_resource_version(job["version_id"])
            location = next((v for v in version.get("locations", []) if v["kind"] == "cluster" and v["status"] == "AVAILABLE"), None)
            path = location["path"] + "/manifest.json" if location else None
        return RemoteArtifact(self.cluster, DEFAULT_GATEWAY, path) if path else None
