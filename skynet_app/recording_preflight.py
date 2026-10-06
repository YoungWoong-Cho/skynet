"""Durable CPU preflight for an adapter's frozen recording inputs."""
import hashlib
import json
from pathlib import Path
import shlex
from uuid import uuid4

from .cluster_config import CLUSTER
from .cluster_runtime import DEFAULT_GATEWAY, WORK_ROOT, ClusterError
from .database import canonical_json
from .observation_preparation import ObservationsPending
from .preparation_states import TERMINAL_FAILURE_STATES


class RecordingPreflight:
    def _preflight_sources(self, job, sources):
        local_result = self.root / job["id"] / "source-preflight.json"
        if job.get("preflight_sha256"):
            raw = local_result.read_bytes()
            if hashlib.sha256(raw).hexdigest() != job["preflight_sha256"]:
                raise ValueError("Frozen source preflight changed; start a new conversion")
            result = json.loads(raw)
        else:
            if not job.get("preflight_script"):
                token = str(uuid4())
                relative = "preflight/" + token
                root = f"{WORK_ROOT}/jobs/runs/{job['id']}/{relative}"
                worker = self.root / job["id"] / "worker"
                files = {relative + "/worker/" + p.relative_to(worker).as_posix(): p.read_text()
                         for p in sorted(worker.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
                request = dict(job_id=job["id"], attempt_id=token, sources=sources,
                               requirements=job["requirements"], split=job["split"], receipt_path=root + "/result.json")
                files[relative + "/request.json"] = canonical_json(request)
                self.cluster.write_capsule_files(job["id"], files, DEFAULT_GATEWAY)
                queue = CLUSTER.queue(CLUSTER.defaults.background_queue_policy)
                python = str(CLUSTER.runtime_profile(CLUSTER.defaults.background_runtime_profile).environment_path) + "/bin/python"
                script = "\n".join([
                    "#!/bin/bash", f"#SBATCH --job-name=inspect-{job['id'][:8]}",
                    f"#SBATCH --account={queue.account}", f"#SBATCH --partition={queue.partition}",
                    "#SBATCH --cpus-per-task=2", "#SBATCH --mem=8G", "#SBATCH --time=00:30:00",
                    f"#SBATCH --output={root}/preflight.log", f"#SBATCH --error={root}/preflight.log",
                    "set -euo pipefail", "umask 077", "export CUDA_VISIBLE_DEVICES=",
                    "export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2",
                    "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1",
                    shlex.join([python, root + "/worker/recording_probe.py", root + "/request.json"]), ""])
                job = self.update(job["id"], preflight_token=token, preflight_root=root,
                                  preflight_script=script, preflight_job_id=None)
            if not job.get("preflight_job_id"):
                submission = self.cluster.submit_script(job["preflight_script"], job["id"], DEFAULT_GATEWAY,
                    submission_key=f"{job['id']}-preflight-{job['preflight_token']}")
                job = self.update(job["id"], preflight_job_id=submission.job_id)
            _, statuses = self.cluster.job_statuses([job["preflight_job_id"]], DEFAULT_GATEWAY)
            status = statuses.get(job["preflight_job_id"])
            if not status or status["State"] not in TERMINAL_FAILURE_STATES | {"COMPLETED"}:
                raise ObservationsPending("Checking recorded inputs and existing observations on cluster CPUs")
            if status["State"] in TERMINAL_FAILURE_STATES:
                message = f"Recorded input verification ended as {status['State']}"
                try:
                    _, raw = self.cluster.read_file(job["preflight_root"] + "/result.json", DEFAULT_GATEWAY, max_bytes=100_000)
                    failed = json.loads(raw)
                    if failed.get("job_id") == job["id"] and failed.get("attempt_id") == job["preflight_token"]:
                        message = failed.get("error") or message
                except (ClusterError, ValueError):
                    try:
                        _, log = self.cluster.read_file(job["preflight_root"] + "/preflight.log", DEFAULT_GATEWAY, max_bytes=8000)
                        if log.strip():
                            message += ": " + log.strip()[-2000:]
                    except (ClusterError, ValueError):
                        pass
                raise ValueError(message)
            _, raw = self.cluster.read_file(job["preflight_root"] + "/result.json", DEFAULT_GATEWAY, max_bytes=10_000_000)
            result = json.loads(raw)
            if (result.get("schema") != "skynet.recording-preflight/v1"
                    or result.get("job_id") != job["id"] or result.get("attempt_id") != job["preflight_token"]):
                raise ValueError("Source preflight returned a different request")
            if status["State"] != "COMPLETED" or result.get("verified") is not True:
                raise ValueError(result.get("error") or "Recorded input verification failed")
            raw = raw.encode()
            temporary = local_result.with_suffix(".part")
            temporary.write_bytes(raw)
            temporary.replace(local_result)
            self.update(job["id"], preflight_sha256=hashlib.sha256(raw).hexdigest())
        by_source = {item["source_sha256"]: item for item in result["sources"]}
        if len(by_source) != len(sources) or set(by_source) != {source["sha256"] for source in sources}:
            raise ValueError("Source preflight does not contain exactly the selected recordings")
        return [dict(source, shared_image_streams=by_source[source["sha256"]]["shared_image_streams"],
                     capture=by_source[source["sha256"]]["capture"])
                for source in sources]
