"""Use the existing Slurm submission protocol for archived recording replay."""

import hashlib
import json
import shlex

from .isaac_job import compile_isaac_job
from .cluster_runtime import WORK_ROOT, ClusterError, SubmissionOutcomeUnknown
from .cluster_config import CLUSTER
from .gpu_preflight import GPU_MISSING_MESSAGE, gpu_missing_exit
from .preparation_states import TERMINAL_FAILURE_STATES

TERMINAL = TERMINAL_FAILURE_STATES | {"COMPLETED", "PREEMPTED"}


def control_cluster(transport, job, generation, root, operation, **value):
    if root != f"{WORK_ROOT}/jobs/runs/{generation.token}":
        raise ValueError("Video capsule does not match its generation")
    # A cancellation cannot finish while a submission is still being made.
    with generation.control_lock:
        gateway = job["gateway"]
        if operation == "start" and (generation.cluster_job_id or generation.cluster_submission_started):
            operation = "status"  # Reconcile the immutable submission; never rewrite a running capsule.
        if operation == "start":
            if generation.cancel.is_set():
                return {"state": "CANCELLED"}
            profile = value["profile"]
            request = dict(value["request"], output=root + "/video.mp4")
            sources = dict(value["sources"], **{"request.json": json.dumps(request)})
            checks = []
            for name, content in sources.items():
                transport.write_capsule_file(generation.token, name, content, gateway)
                checks.append("printf '%s  %s\\n' " + hashlib.sha256(content.encode()).hexdigest()
                              + " " + shlex.quote(root + "/" + name) + " | sha256sum --check --status")
            script = compile_isaac_job(profile, root, "review-video-" + generation.token[:8],
                                       [profile["runtime"] + "/bin/python", root + "/render_recording.py", root + "/request.json"],
                                       checks=checks, resources=CLUSTER.defaults.background_jobs.review_video)
            if generation.cancel.is_set():
                return {"state": "CANCELLED"}
            generation.cluster_submission_started = True
            if generation.persist_submission:
                try:
                    generation.persist_submission()
                except Exception:
                    generation.cluster_submission_started = False
                    raise
            try:
                submission = transport.submit_script(script, generation.token, gateway, submission_key=generation.token)
            except SubmissionOutcomeUnknown:
                raise
            except ClusterError:
                generation.cluster_submission_started = False
                if generation.persist_submission:
                    generation.persist_submission()
                raise
            generation.cluster_job_id = submission.job_id
            generation.cluster_receipt_verified = True
            if generation.persist_submission:
                generation.persist_submission()
            return {"state": "STARTING", "phase": "queued", "job_id": submission.job_id}

        identifier = generation.cluster_job_id
        if identifier and not generation.cluster_receipt_verified:
            receipt = transport.recover_submission(generation.token, generation.token, gateway)
            if receipt is None or receipt.job_id != identifier:
                raise ValueError("The saved video job does not match its submission receipt")
            generation.cluster_receipt_verified = True
        if not identifier:
            if operation == "cancel" and not generation.cluster_submission_started:
                return {"state": "CANCELLED"}
            receipt = transport.recover_submission(generation.token, generation.token, gateway)
            if receipt is None:
                # Never acknowledge cleanup if an earlier submission is uncertain.
                raise ValueError("The video submission could not be reconciled. Retry cancellation to confirm its state.")
            identifier = generation.cluster_job_id = receipt.job_id
            generation.cluster_receipt_verified = True
        _, statuses = transport.job_statuses([identifier], gateway)
        scheduler = statuses.get(identifier, {})
        state = scheduler.get("State", "UNKNOWN")
        if operation == "cancel":
            if state not in TERMINAL:
                transport.cancel(identifier, gateway)
                _, statuses = transport.job_statuses([identifier], gateway)
                state = statuses.get(identifier, {}).get("State", "UNKNOWN")
            if state not in TERMINAL:
                raise ValueError("Waiting for Slurm to confirm video cancellation")
            return {"state": "CANCELLED"}
        program = "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(p.read_text() if p.is_file() and p.stat().st_size < 100000 else '{}')"
        metadata = json.loads(transport.run_with_fallback("python3 -c " + shlex.quote(program) + " " + shlex.quote(root + "/video.json"), gateway, attempt_timeout=20)[1])
        if state in TERMINAL:
            if state == "COMPLETED" and metadata.get("state") == "READY":
                if metadata.get("path") != root + "/video.mp4":
                    raise ValueError("Video artifact does not match its generation")
                return metadata
            return {"state": "FAILED", "error": metadata.get("error") or (
                GPU_MISSING_MESSAGE if gpu_missing_exit(scheduler) else f"Video job {identifier} ended with {state}")}
        if metadata.get("state") in {"RENDERING", "READY", "FAILED"}:
            return {"state": "PREPARING", "phase": "rendering" if metadata["state"] == "RENDERING" else "finalizing"}
        if state in {"RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"}:
            return {"state": "STARTING", "phase": "starting", "job_id": identifier}
        return {"state": "STARTING", "phase": "queued", "job_id": identifier}
