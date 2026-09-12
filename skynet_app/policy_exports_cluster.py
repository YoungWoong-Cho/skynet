"""Remote-only preparation and explicit streaming downloads for collection data."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
from uuid import uuid4

from .cluster_config import CLUSTER
from .cluster_runtime import ClusterError, SubmissionOutcomeUnknown, WORK_ROOT
from .database import canonical_json
from .dataset_formats import RECIPES, XPL_COMMIT
from .remote_artifacts import RemoteArtifact


class ArchivePending(ValueError):
    pass


def verified_local_manifest(output, expected):
    """A storage move may remove only files included in its verified manifest."""
    from ops.datasets.artifacts import verify
    if output.is_symlink():
        raise ValueError("Local dataset migration cannot follow symbolic links")
    manifest = verify(output, expected)
    paths = list(output.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Local dataset contains a symbolic link; preserved it")
    actual = {path.relative_to(output).as_posix() for path in paths if path.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Local dataset contains unlisted files; preserved all local files")
    return manifest


class ClusterPolicyPreparation:
    def _archived_sources(self, job):
        archive = getattr(self.live, "archive", None)
        if archive is None:
            raise ValueError("Collection archiving is unavailable; preparation requires a verified sky2 copy")
        sources = []
        for item in job["sources"]:
            session_id = item.get("session_id", job["session_id"])
            session = self.live.get(session_id)
            descriptor = session.get("archive") or {}
            if descriptor.get("state") not in {"VERIFIED", "CLEANUP_PENDING", "READY"}:
                archive.ensure(session_id)
                raise ArchivePending("Waiting for the collection archive to be verified on sky2")
            _, gateway, recording = archive.resolve(session, item["path"])
            if gateway != "sky2":
                raise ValueError("Prepared collection sources must be stored on sky2")
            images = None
            if item.get("image_path"):
                _, image_gateway, images = archive.resolve(session, item["image_path"])
                if image_gateway != "sky2":
                    raise ValueError("Prepared image sources must be stored on sky2")
            sources.append(dict(item, recording=recording, images=images))
        return sources

    def _loader_request(self, job, remote_worker, attempt_root):
        if not RECIPES[job["format"]]["trainable"]:
            return None
        kind = "egoverse" if job["format"] == "egoverse" else "act" if job["format"] == "act" else "dp"
        profile = CLUSTER.runtime_profiles["egoverse-native" if kind == "egoverse" else "skynet-dp"]
        mode = "rgb" if "rgb" in RECIPES[job["format"]]["observations"] else "state"
        script = "egoverse_runtime.py" if kind == "egoverse" else f"skynet_{kind}_training.py"
        return dict(
            argv=[str(profile.environment_path) + "/bin/python", f"{remote_worker}/{script}",
                  "--repository", str(profile.source_prerequisites[0].path),
                  "--revision", RECIPES[job["format"]].get("training_setup", {}).get("revision", XPL_COMMIT),
                  "--dataset", "{dataset}", "--manifest-sha", "{manifest_sha}",
                  "--output", f"{attempt_root}/dataset-validation", "--batch-size", "1", "--verify-only",
                  *(["--observation-mode", mode] if kind == "dp" else [])],
            schemas=[f"skynet.{kind}-loader-validation/v1", *(["skynet.dp-loader-validation/v2"] if kind == "dp" else [])],
            mode=mode,
        )

    def _stage_cluster_attempt(self, job):
        sources = self._archived_sources(job)
        queue = CLUSTER.queues["normal"]
        self.update(job["id"], state="STAGING", stage="STAGING", error=None,
                    cluster_partition=queue.partition, cluster_account=queue.account, cluster_cpus=4,
                    detail="Preparing CPU job submission to sky2")
        attempt_id = str(uuid4())
        relative = f"preparation/{attempt_id}"
        root = f"{WORK_ROOT}/jobs/runs/{job['id']}/{relative}"
        worker = root + "/worker"
        local_worker = self.root / job["id"] / "worker"
        for path in sorted(local_worker.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                self.cluster.write_capsule_file(job["id"], f"{relative}/worker/{path.relative_to(local_worker)}", path.read_text(), "sky2")
        self.cluster.write_capsule_file(job["id"], f"{relative}/worker/cluster_worker.py",
                                        Path(__file__).with_name("policy_export_worker.py").read_text(), "sky2")
        request = {key: job[key] for key in ("format", "split", "contract", "source_revision", "converter_sha256")}
        request.update(job_id=job["id"], attempt_id=attempt_id, sources=sources,
                       output=root + "/output", prepared_root=f"{WORK_ROOT}/datasets/prepared",
                       receipt_path=root + "/result.json", loader=self._loader_request(job, worker, root))
        if job.get("migrating_local_copy"):
            from .capture_processing.service import upload_capture
            output = self.root / job["id"] / "output"
            manifest = verified_local_manifest(output, job["manifest_sha256"])
            request.update(migration=True, expected_manifest_sha256=job["manifest_sha256"], sources=[], loader=None)
            for name, receipt in {**manifest["files"], "manifest.json": {"sha256": job["manifest_sha256"]}}.items():
                upload_capture(self.cluster, output / name, job["id"], receipt["sha256"], "sky2",
                               relative_path=f"{relative}/output/{name}", timeout=3600)
        self.cluster.write_capsule_file(job["id"], relative + "/request.json", canonical_json(request), "sky2")
        profile = CLUSTER.runtime_profiles["egoverse-native" if job["format"] == "egoverse" else "skynet-dp"]
        interpreter = [str(profile.environment_path) + "/bin/python"]
        dependencies = local_worker / "conversion-dependencies.json"
        uv_setup = []
        if dependencies.is_file():
            uv_bootstrap = shlex.quote(f"{CLUSTER.paths.uv_cache}/bootstrap-{CLUSTER.defaults.uv_version}/bin/uv")
            uv_setup = ['UV_BIN=$(command -v uv || true)',
                        f'if [[ -z "$UV_BIN" && -x {uv_bootstrap} ]]; then UV_BIN={uv_bootstrap}; fi',
                        f'if [[ -z "$UV_BIN" && -x {shlex.quote(str(WORK_ROOT) + "/.local/bin/uv")} ]]; then UV_BIN={shlex.quote(str(WORK_ROOT) + "/.local/bin/uv")}; fi',
                        'if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then UV_BIN="$HOME/.local/bin/uv"; fi',
                        'test -n "$UV_BIN" || { echo "uv is required for the pinned EgoVerse converter" >&2; exit 69; }']
            interpreter = ["run", "--no-project", "--python", interpreter[0]]
            for package in json.loads(dependencies.read_text()):
                interpreter.extend(["--with", package])
            interpreter.append("python")
        script = "\n".join([
            "#!/bin/bash", f"#SBATCH --job-name=prepare-{job['id'][:8]}",
            f"#SBATCH --account={queue.account}", f"#SBATCH --partition={queue.partition}",
            "#SBATCH --cpus-per-task=4", "#SBATCH --mem=32G", "#SBATCH --time=01:00:00",
            f"#SBATCH --output={root}/export.log", f"#SBATCH --error={root}/export.log",
            "set -euo pipefail", "umask 077", "export CUDA_VISIBLE_DEVICES=",
            "export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4",
            "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1",
            f"export TMPDIR={shlex.quote(root + '/tmp')}", 'mkdir -p "$TMPDIR"',
            f"export UV_CACHE_DIR={shlex.quote(CLUSTER.paths.uv_cache)}",
            *uv_setup, (('"$UV_BIN" ' if dependencies.is_file() else "") + shlex.join([*interpreter, worker + "/cluster_worker.py", root + "/request.json"])), "",
        ])
        # This exact script and token survive an accepted submission whose SSH reply was lost.
        return self.update(job["id"], execution="cluster", gateway="sky2", attempt_id=attempt_id,
                           cluster_root=root, cluster_script=script, cluster_job_id=None,
                           state="SUBMITTING", stage="SUBMITTING", error=None,
                           detail="Submitting CPU preparation on the training cluster")

    def _prepare_cluster(self, identifier):
        job = self.get(identifier)
        try:
            if job.get("migrating_local_copy") and not job.get("cluster_script") and self._migrate_existing_cluster_copy(job):
                return
            if not job.get("cluster_script"):
                job = self._stage_cluster_attempt(job)
            if not job.get("cluster_job_id"):
                submission = self.cluster.submit_script(job["cluster_script"], identifier, "sky2",
                    submission_key=f"{identifier}-preparation-{job['attempt_id']}")
                job = self.update(identifier, cluster_job_id=submission.job_id, state="PENDING", stage="QUEUED",
                                  detail="Waiting for 4 CPU slots on the training cluster", error=None)
            _, statuses = self.cluster.job_statuses([job["cluster_job_id"]], "sky2")
            status = statuses.get(job["cluster_job_id"])
            if not status:
                return
            state = status["State"]
            if state in {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED"}:
                self.update(identifier, state="RUNNING" if state in {"RUNNING", "COMPLETING"} else "PENDING",
                            stage="CONVERTING" if state in {"RUNNING", "COMPLETING"} else "QUEUED", error=None,
                            detail="Preparing and validating data on the cluster" if state in {"RUNNING", "COMPLETING"}
                            else "Waiting for cluster CPU resources" + (f": {status['Reason']}" if status.get("Reason") else ""))
                return
            if state != "COMPLETED":
                message = f"Cluster preparation ended as {state}"
                try:
                    _, raw = self.cluster.read_file(job["cluster_root"] + "/result.json", "sky2", max_bytes=100_000)
                    result = json.loads(raw)
                    if result.get("attempt_id") == job["attempt_id"]:
                        message = result.get("error") or message
                except (ClusterError, ValueError):
                    # Bootstrap failures happen before the worker can write a receipt.
                    try:
                        _, output = self.cluster.read_file(job["cluster_root"] + "/export.log", "sky2", max_bytes=8000)
                        if output.strip():
                            message += ": " + output.strip()[-2000:]
                    except (ClusterError, ValueError):
                        pass
                raise ValueError(message)
            _, raw = self.cluster.read_file(job["cluster_root"] + "/result.json", "sky2", max_bytes=10_000_000)
            result = json.loads(raw)
            self._publish_cluster_result(job, result)
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
                        detail="Preparation failed; verified original recordings are preserved on sky2")

    def _publish_cluster_result(self, job, result):
        manifest_sha = result.get("manifest_sha256", "")
        expected_path = f"{WORK_ROOT}/datasets/prepared/{manifest_sha}"
        if (result.get("schema") != "skynet.cluster-preparation/v1" or result.get("job_id") != job["id"]
                or result.get("attempt_id") != job["attempt_id"] or result.get("verified") is not True
                or not re.fullmatch(r"[a-f0-9]{64}", manifest_sha)
                or result.get("path") != expected_path
                or result.get("archive_path") != job["cluster_root"] + "/dataset.zip"
                or not re.fullmatch(r"[a-f0-9]{64}", result.get("archive_sha256", ""))):
            raise ValueError("Cluster preparation receipt did not match this attempt")
        if job.get("migrating_local_copy") and manifest_sha != job.get("manifest_sha256"):
            raise ValueError("Migration returned a different immutable dataset version")
        # Read the exact bounded manifest whose digest the worker verified; never trust a
        # JSON reserialization to reproduce the original manifest byte sequence.
        _, raw_manifest = self.cluster.read_file(expected_path + "/manifest.json", "sky2", max_bytes=10_000_000)
        if hashlib.sha256(raw_manifest.encode()).hexdigest() != manifest_sha:
            raise ValueError("Prepared manifest changed after cluster verification")
        manifest = json.loads(raw_manifest)
        if manifest != result.get("manifest") or manifest.get("format") != RECIPES[job["format"]]["format"]:
            raise ValueError("Cluster preparation returned a different data format")
        if RECIPES[job["format"]]["trainable"] and not job.get("migrating_local_copy"):
            loader = self._loader_request(job, job["cluster_root"] + "/worker", job["cluster_root"])
            receipt = result.get("loader_validation") or {}
            if (receipt.get("schema") not in loader["schemas"] or receipt.get("manifest_sha256") != manifest_sha
                    or receipt.get("observation_mode", "rgb") != loader["mode"]):
                raise ValueError("Cluster preparation did not validate its training loader")
        version = self.register(job, manifest, manifest_sha, remote_path=expected_path)
        location = self.database.record_data_location(version["id"], kind="cluster", host="skynet",
                                                      path=expected_path, manifest_sha256=manifest_sha)
        if job.get("migrating_local_copy") and (manifest_sha != job.get("manifest_sha256") or version["id"] != job.get("version_id")):
            raise ValueError("Migration returned a different immutable dataset version")
        job = self.update(job["id"], version_id=version["id"], manifest_sha256=manifest_sha,
                          archive_sha256=result["archive_sha256"], remote_archive=result["archive_path"],
                          size_bytes=version["size_bytes"], episodes=len(manifest["episodes"]), steps=manifest["steps"],
                          loader_validation=result.get("loader_validation") or job.get("loader_validation"))
        # The prepared result is selected directly by an experiment.
        if job.get("migrating_local_copy"):
            self._remove_migrated_local_payload(job)
        self.update(job["id"], bundle_id=None, state="READY", stage="READY",
                    error=None, local_removed=True, migrating_local_copy=False, training_ready=bool(RECIPES[job["format"]]["trainable"]),
                    detail="Ready on the cluster")

    def migrate_local_copy(self, identifier):
        """Queue a verified move of one registered local version, retaining its identity.

        Local payload removal happens only after the CPU job verifies and publishes
        its exact manifest on sky2. This is the operator's explicit migration entry.
        """
        with self.lock:
            job = self.get(identifier)
            if identifier in self.active:
                return job
            if not job.get("version_id") or not job.get("manifest_sha256"):
                raise ValueError("Only a registered prepared version can be migrated")
            if job.get("migrating_local_copy") and job["state"] != "FAILED":
                self.dispatch(identifier)
                return job
            output = self.root / identifier / "output"
            if not output.exists():
                if job.get("remote_archive"):
                    return job
                raise ValueError("The registered local dataset is unavailable")
            from ops.datasets.artifacts import digest
            archive = self.root / identifier / "dataset.zip"
            local_archive_sha = digest(archive) if archive.is_file() else None
            if local_archive_sha and job.get("archive_sha256") and local_archive_sha != job["archive_sha256"]:
                raise ValueError("Local ZIP differs from its recorded checksum; preserved it")
            job = self.update(identifier, migrating_local_copy=True, target="cluster", execution="cluster",
                              cluster_script=None, cluster_job_id=None, cluster_root=None,
                              migration_local_archive_sha256=local_archive_sha,
                              training_ready=False,
                              state="QUEUED", stage="QUEUED", error=None,
                              detail="Moving the exact prepared dataset to sky2 before removing its local copy")
            self.dispatch(identifier)
            return job

    def _migrate_existing_cluster_copy(self, job):
        """Retain an existing ZIP without scheduling a redundant repack job."""
        from ops.datasets.artifacts import digest
        from .capture_processing.service import upload_capture
        version = self.database.get_data_resource_version(job["version_id"])
        expected_path = f"{WORK_ROOT}/datasets/prepared/{job['manifest_sha256']}"
        location = next((v for v in version.get("locations", []) if v["kind"] == "cluster"
                         and v["status"] == "AVAILABLE" and v["host"] == "skynet" and v["path"] == expected_path), None)
        archive = self.root / job["id"] / "dataset.zip"
        if not location or not archive.is_file():
            return False
        self._archived_sources(job)
        manifest = verified_local_manifest(self.root / job["id"] / "output", job["manifest_sha256"])
        archive_sha = digest(archive)
        if archive_sha != job.get("migration_local_archive_sha256"):
            raise ValueError("Local ZIP changed during migration; preserved it")
        attempt_id = job.get("attempt_id") if job.get("cluster_root") else str(uuid4())
        relative = f"preparation/{attempt_id}"
        root = f"{WORK_ROOT}/jobs/runs/{job['id']}/{relative}"
        job = self.update(job["id"], attempt_id=attempt_id, cluster_root=root, stage="VERIFYING_COPY",
                          detail="Verifying the existing sky2 copy before removing local files")
        _, script = self.cluster.write_capsule_file(job["id"], relative + "/verify-dataset.py",
                            (self.root / job["id"] / "worker/artifacts.py").read_text(), "sky2")
        raw = self.cluster.ssh("sky2", shlex.join(["python3", script, expected_path, job["manifest_sha256"]]), timeout=3600)
        verification = json.loads(raw)
        if verification.get("verified") is not True or verification.get("manifest_sha256") != job["manifest_sha256"]:
            raise ValueError("The existing cluster copy did not pass verification")
        upload_capture(self.cluster, archive, job["id"], archive_sha, "sky2",
                       relative_path=relative + "/dataset.zip", timeout=3600)
        result = dict(schema="skynet.cluster-preparation/v1", job_id=job["id"], attempt_id=attempt_id,
                      verified=True, manifest=manifest, manifest_sha256=job["manifest_sha256"], path=expected_path,
                      archive_sha256=archive_sha, archive_path=root + "/dataset.zip",
                      loader_validation=job.get("loader_validation"))
        self._publish_cluster_result(job, result)
        return True

    def _remove_migrated_local_payload(self, job):
        import shutil
        from ops.datasets.artifacts import digest
        directory = self.root / job["id"]
        if any(path.is_symlink() for path in (directory, *directory.parents)):
            raise ValueError("Local migration cleanup cannot follow symbolic links")
        output = directory / "output"
        archive = directory / "dataset.zip"
        if archive.exists() and (archive.is_symlink() or digest(archive) != job.get("migration_local_archive_sha256")):
            raise ValueError("Local ZIP changed during migration; preserved it")
        if output.exists():
            verified_local_manifest(output, job["manifest_sha256"])
            shutil.rmtree(output)
        archive.unlink(missing_ok=True)
        self.database.record_data_location(job["version_id"], kind="local", host="local",
            path=str(output), manifest_sha256=job["manifest_sha256"], status="REMOVED")

    def status(self, identifier):
        job = self.get(identifier)
        if job["state"] not in {"READY", "FAILED", "DELETE_FAILED"}:
            self.dispatch(identifier)
        return job

    def remote_artifact(self, identifier, name):
        job = self.get(identifier)
        if name not in {"dataset.zip", "manifest.json", "export.log"}:
            raise KeyError("Dataset file not found")
        if not job.get("cluster_root") and not job.get("remote_archive"):
            return None
        if name == "export.log":
            path = job.get("cluster_root", "") + "/export.log" if job.get("cluster_root") else None
        elif not job.get("version_id"):
            raise ValueError("Preparation is not complete")
        elif name == "dataset.zip":
            path = job.get("remote_archive")
        else:
            version = self.database.get_data_resource_version(job["version_id"])
            location = next((v for v in version.get("locations", []) if v["kind"] == "cluster" and v["status"] == "AVAILABLE"), None)
            path = location["path"] + "/manifest.json" if location else None
        if not path:
            return None
        return RemoteArtifact(self.cluster, "sky2", path)
