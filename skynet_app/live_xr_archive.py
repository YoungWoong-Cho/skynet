"""Verified workstation-to-cluster archives with no local payload files."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
from pathlib import PurePosixPath
import re
import shlex
import subprocess
import threading
import time
from uuid import UUID

from .cluster_config import CLUSTER
from .cluster_runtime import ClusterClient, ClusterError
from .database import canonical_json, utc_now
from .live_xr_archive_remote import archive_control

AVAILABLE_STATES = frozenset({"VERIFIED", "CLEANUP_PENDING", "READY"})
TERMINAL_STATES = frozenset({"CAPTURED", "STOPPED", "TIMED_OUT", "FAILED"})


def is_archived(job):
    return (job.get("archive") or {}).get("state") in AVAILABLE_STATES


class LiveArchiveService:
    def __init__(self, live, cluster=None, *, enabled=False, cleanup_enabled=False):
        self.live, self.cluster = live, cluster or live.cluster or ClusterClient()
        self.enabled, self.cleanup_enabled = enabled, cleanup_enabled
        self.busy = lambda identifier: False
        self.lock = threading.RLock()
        self.active, self.next_attempt = set(), {}
        self.stopping = threading.Event()
        self.monitor = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-archive")

    @staticmethod
    def _relative(path):
        candidate = PurePosixPath(path)
        if not path or candidate.is_absolute() or ".." in candidate.parts or str(candidate) != path or "\x00" in path:
            raise ValueError("Invalid session artifact path")
        return str(candidate)

    def _descriptor(self, job, *, require_available=True):
        archive = job.get("archive") or {}
        if require_available and not is_archived(job):
            raise ValueError("Session has no verified archive")
        manifest, checksum = archive.get("manifest"), archive.get("manifest_sha256", "")
        if (not isinstance(manifest, dict) or manifest.get("session_id") != job["id"]
                or manifest.get("schema") != "skynet.live-archive/v1"
                or not re.fullmatch(r"[a-f0-9]{64}", checksum)
                or hashlib.sha256(canonical_json(manifest).encode()).hexdigest() != checksum):
            raise ValueError("The saved archive manifest is invalid")
        expected = f"{CLUSTER.paths.datasets}/raw/dexverse-live/{job['id']}/{checksum}/output"
        if archive.get("gateway") != "sky2" or archive.get("root") != expected:
            raise ValueError("The saved archive storage location is invalid")
        self.cluster.candidates("sky2")
        self.cluster._remote_path(expected)
        return archive

    def resolve(self, job, relative_path):
        relative_path = self._relative(relative_path)
        if is_archived(job):
            archive = self._descriptor(job)
            if not any(item["path"] == "output/" + relative_path for item in archive["manifest"]["files"]):
                raise ValueError("This artifact is not part of the verified session archive")
            return self.cluster, archive["gateway"], archive["root"] + "/" + relative_path
        if (job.get("archive") or {}).get("source_removed"):
            raise ValueError("The archived session must be verified before its files can be used")
        return self.live.transport(job), job["gateway"], job["root"] + "/output/" + relative_path

    def derived_root(self, job):
        archive = self._descriptor(job)
        return f"{CLUSTER.paths.datasets}/raw/dexverse-live/{job['id']}/derived/{archive['manifest_sha256']}"

    def session_root(self, job):
        return str(PurePosixPath(self._descriptor(job)["root"]).parent)

    @staticmethod
    def is_archived(job):
        return is_archived(job)

    def _check(self):
        if self.stopping.is_set():
            raise RuntimeError("Archive interrupted by app shutdown; retry continues from verified files")

    @staticmethod
    def _source(job):
        identifier = job["id"]
        if str(UUID(identifier)) != identifier or job["profile"].get("execution") != "workstation":
            raise ValueError("Only owned workstation sessions can be archived")
        unlaunched = job.get("state") == "FAILED" and not job.get("job_id")
        if job.get("state") not in TERMINAL_STATES or not (job.get("scheduler_final") or unlaunched):
            raise ValueError("Wait for collection and image processing to finish before archiving")
        expected = job["profile"]["work_root"] + "/sessions/" + identifier
        if job["root"] != expected or (job.get("job_id") and job["job_id"] != f"skynet-live-{identifier}.service"):
            raise ValueError("The session execution identity is invalid")
        return dict(session_id=identifier, work_root=job["profile"]["work_root"],
                    profile_sha256=hashlib.sha256(canonical_json(job["profile"]).encode()).hexdigest(),
                    worker_sha256=job.get("worker_sha256"), allow_unlaunched=unlaunched,
                    allow_absent=not job.get("recordings"))

    @staticmethod
    def _program(mode="json"):
        read = "json.loads(sys.stdin.buffer.readline())" if mode == "receive" else "json.load(sys.stdin)"
        return (inspect.getsource(archive_control) + "\nimport json,sys\n"
                + f"_result=archive_control({read})\n"
                + "if _result is not None: print(json.dumps(_result))\n")

    @staticmethod
    def _ssh_args(gateway, program):
        return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=1",
                gateway, "python3 -c " + shlex.quote(program)]

    def _call(self, transport, gateway, operation, **request):
        self._check()
        transport.candidates(gateway)
        process = subprocess.Popen(self._ssh_args(gateway, self._program()), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        started = time.monotonic()
        payload = canonical_json(dict(request, operation=operation)).encode()
        output, errors, problems, threads = [], bytearray(), [], []

        def feed():
            try:
                # One writer owns stdin through EOF. Retrying communicate with
                # input=None after a timeout can leave a large input unfinished.
                process.stdin.write(payload)
                process.stdin.close()
            except (OSError, ValueError) as exc:
                problems.append(exc)

        def drain(stream, *, stderr=False):
            try:
                while chunk := stream.read(65536):
                    if stderr:
                        errors.extend(chunk)
                        del errors[:-1_000_000]
                    else:
                        output.append(chunk)
            except (OSError, ValueError) as exc:
                problems.append(exc)

        try:
            for function, args, kwargs in ((feed, (), {}), (drain, (process.stdout,), {}),
                                          (drain, (process.stderr,), {"stderr": True})):
                thread = threading.Thread(target=function, args=args, kwargs=kwargs, daemon=True)
                threads.append(thread)
                thread.start()
            while process.poll() is None:
                self._check()
                if time.monotonic() - started > 1800:
                    raise ClusterError("Archive verification timed out; source files were retained")
                if problems:
                    raise ClusterError("Archive request was interrupted; source files were retained")
                self.stopping.wait(0.1)
            for thread in threads:
                thread.join(timeout=2)
            self._check()
            if process.returncode or problems or any(thread.is_alive() for thread in threads):
                raise ClusterError(errors.decode(errors="replace").strip()[-2000:] or "Archive operation failed")
            return json.loads(b"".join(output))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            for thread in threads:
                thread.join(timeout=2)
            if not any(thread.is_alive() for thread in threads):
                for stream in (process.stdin, process.stdout, process.stderr):
                    stream.close()

    def _transfer(self, job, manifest, checksum):
        """Pipe remote tar bytes through bounded memory; never create a local archive."""
        self._check()
        transport = self.live.transport(job)
        transport.candidates(job["gateway"])
        self.cluster.candidates("sky2")
        source_request = dict(self._source(job), manifest=manifest, manifest_sha256=checksum, operation="stream")
        destination_request = dict(session_id=job["id"], datasets_root=CLUSTER.paths.datasets,
                                   manifest=manifest, manifest_sha256=checksum, operation="receive")
        source = subprocess.Popen(self._ssh_args(job["gateway"], self._program()),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        destination = None
        problems, errors, output = [], [], []

        def drain(stream, result):
            size = 0
            while chunk := stream.read(65536):
                if size < 1_000_000:
                    result.append(chunk[:1_000_000 - size])
                    size += len(chunk)

        def feed():
            try:
                source.stdin.write(canonical_json(source_request).encode())
                source.stdin.close()
            except (OSError, ValueError) as exc:
                problems.append(exc)

        def relay():
            try:
                destination.stdin.write(canonical_json(destination_request).encode() + b"\n")
                while chunk := source.stdout.read(1024 * 1024):
                    destination.stdin.write(chunk)
                destination.stdin.close()
            except (OSError, ValueError) as exc:
                problems.append(exc)

        threads = []
        try:
            destination = subprocess.Popen(self._ssh_args("sky2", self._program("receive")),
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for function, args in ((feed, ()), (relay, ()), (drain, (source.stderr, errors)),
                                   (drain, (destination.stderr, errors)), (drain, (destination.stdout, output))):
                thread = threading.Thread(target=function, args=args, daemon=True)
                threads.append(thread)
                thread.start()
            deadline = time.monotonic() + 7200
            while source.poll() is None or destination.poll() is None:
                self._check()
                if source.poll() not in {None, 0} or destination.poll() not in {None, 0} or problems:
                    raise ClusterError("Session transfer stopped before verification; source files were retained")
                if time.monotonic() >= deadline:
                    raise ClusterError("Session transfer exceeded two hours; source files were retained")
                self.stopping.wait(0.1)
            for thread in threads:
                thread.join(timeout=2)
            if source.returncode or destination.returncode or problems:
                raise ClusterError(b"".join(errors).decode(errors="replace")[-2000:] or "Session transfer failed")
            return json.loads(b"".join(output))
        finally:
            for process in (source, destination):
                if process is not None and process.poll() is None:
                    process.terminate()
            for process in (source, destination):
                if process is None:
                    continue
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            for thread in threads:
                thread.join(timeout=2)

    def _publish(self, identifier, **changes):
        job = self.live.get(identifier)
        archive = dict(job.get("archive") or {}, **changes, updated_at=utc_now())
        self.live.update(identifier, archive=archive)
        return archive

    def _verify(self, job, archive):
        try:
            result = self._call(self.cluster, "sky2", "verify", session_id=job["id"],
                                datasets_root=CLUSTER.paths.datasets, manifest=archive["manifest"],
                                manifest_sha256=archive["manifest_sha256"])
            if not result.get("verified") or result.get("manifest_sha256") != archive["manifest_sha256"] or result.get("root") != archive["root"]:
                raise ValueError("Archive verification returned a different session identity")
            return result
        except Exception as exc:
            self._publish(job["id"], state="VERIFY_FAILED", error=str(exc),
                          detail="Archive verification failed; source cleanup is paused")
            raise

    def archive(self, identifier, *, cleanup=False):
        job = self.live.get(identifier)
        try:
            if is_archived(job) or (job.get("archive") or {}).get("state") == "VERIFY_FAILED":
                archive = self._descriptor(job, require_available=False)
                self._verify(job, archive)
                self._publish(identifier, state="READY" if archive.get("source_removed") else "VERIFIED", error=None)
            else:
                source = self._source(job)
                if self.busy(identifier):
                    return self._publish(identifier, state="QUEUED", error=None,
                                         detail="Waiting for active recording review or video work to finish")
                self._publish(identifier, state="COPYING", gateway="sky2", error=None,
                              detail="Copying the complete session to the training cluster")
                if self.busy(identifier):
                    return self._publish(identifier, state="QUEUED", error=None,
                                         detail="Waiting for active recording review or video work to finish")
                report = self._call(self.live.transport(job), job["gateway"], "manifest", **source)
                if report.get("absent"):
                    return self._publish(identifier, state="EMPTY", source_removed=True, source_missing=True,
                                         error=None, detail="No session files remain on the workstation")
                manifest, checksum = report["manifest"], report["manifest_sha256"]
                if hashlib.sha256(canonical_json(manifest).encode()).hexdigest() != checksum:
                    raise ValueError("Source manifest checksum did not match")
                expected = f"{CLUSTER.paths.datasets}/raw/dexverse-live/{identifier}/{checksum}/output"
                archive = dict(gateway="sky2", root=expected, manifest=manifest, manifest_sha256=checksum)
                result = self._transfer(job, manifest, checksum)
                if not result.get("verified") or result.get("manifest_sha256") != checksum or result.get("root") != expected:
                    raise ValueError("Transferred archive verification did not match")
                self._check()
                # Publish the durable destination before any source deletion.
                self._publish(identifier, **archive, state="VERIFIED", source_removed=False,
                              verified_at=utc_now(), error=None, detail="Session verified on the training cluster")
            if cleanup:
                return self.cleanup_source(identifier)
            return self.live.get(identifier)["archive"]
        except Exception as exc:
            current = self.live.get(identifier)
            self._publish(identifier, state=(current.get("archive") or {}).get("state") if is_archived(current) or (current.get("archive") or {}).get("state") == "VERIFY_FAILED" else "FAILED",
                          error=str(exc), detail="Archive needs attention; cleanup will resume after verification")
            raise

    def cleanup_source(self, identifier):
        self._check()
        job = self.live.get(identifier)
        archive = self._descriptor(job)
        if archive.get("source_removed"):
            return archive
        self._publish(identifier, state="CLEANUP_PENDING", detail="Verified archive ready; waiting to remove the workstation copy")
        if self.busy(identifier):
            return self.live.get(identifier)["archive"]
        try:
            self._verify(job, archive)
            self._check()
            if self.busy(identifier):
                return self.live.get(identifier)["archive"]
            current = self.live.get(identifier)
            if self._descriptor(current)["manifest_sha256"] != archive["manifest_sha256"]:
                raise ValueError("Archive identity changed before cleanup")
            result = self._call(self.live.transport(job), job["gateway"], "delete", **self._source(job),
                                manifest=archive["manifest"], manifest_sha256=archive["manifest_sha256"])
            if not result.get("removed") or result.get("manifest_sha256") != archive["manifest_sha256"]:
                raise ValueError("Workstation cleanup acknowledgement did not match")
            return self._publish(identifier, state="READY", source_removed=True, removed_at=utc_now(),
                                 error=None, detail="Session stored on the training cluster; workstation copy removed")
        except Exception as exc:
            state = "VERIFY_FAILED" if (self.live.get(identifier).get("archive") or {}).get("state") == "VERIFY_FAILED" else "CLEANUP_PENDING"
            self._publish(identifier, state=state, error=str(exc),
                          detail="Verified archive retained; workstation cleanup needs attention")
            raise

    def ensure(self, identifier):
        job = self.live.get(identifier)
        if not is_archived(job):
            self.dispatch(identifier, cleanup=False)
        return self.live.get(identifier).get("archive") or {"state": "QUEUED"}

    def dispatch(self, identifier, *, cleanup=None):
        with self.lock:
            if not self.enabled or self.stopping.is_set() or identifier in self.active:
                return
            if time.monotonic() < self.next_attempt.get(identifier, 0):
                return
            self.active.add(identifier)
            self.executor.submit(self._run, identifier, self.cleanup_enabled if cleanup is None else cleanup)

    def _run(self, identifier, cleanup):
        try:
            self.archive(identifier, cleanup=cleanup)
        except Exception:
            pass  # archive() persists its actionable error and verified location.
        finally:
            with self.lock:
                self.active.discard(identifier)
                self.next_attempt[identifier] = time.monotonic() + 60

    def start(self, *, enabled=None, cleanup_enabled=None):
        if enabled is not None:
            self.enabled = enabled
        if cleanup_enabled is not None:
            self.cleanup_enabled = cleanup_enabled
        if not self.enabled or self.monitor is not None:
            return
        self.monitor = threading.Thread(target=self._monitor, name="live-archive-monitor", daemon=True)
        self.monitor.start()

    def _monitor(self):
        while not self.stopping.is_set():
            try:
                for job in self.live.list():
                    if self.stopping.is_set():
                        break
                    if job["profile"].get("execution") != "workstation":
                        continue
                    if job.get("state") not in TERMINAL_STATES or (job.get("job_id") and not job.get("scheduler_final")):
                        self.live.refresh(job["id"])
                        job = self.live.get(job["id"])
                    if (job.get("state") in TERMINAL_STATES and (job.get("scheduler_final") or not job.get("job_id"))
                            and (job.get("archive") or {}).get("state") not in {"READY", "EMPTY"}):
                        self.dispatch(job["id"])
            except Exception:
                pass  # Next pass reconciles persisted state after connection recovery.
            self.stopping.wait(10)

    def stop(self):
        self.stopping.set()
        if self.monitor is not None:
            self.monitor.join(timeout=2)
        self.executor.shutdown(wait=False, cancel_futures=True)
