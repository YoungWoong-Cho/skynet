"""CPU capsule entry point. Every payload path is on shared cluster storage."""

import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile

import hashlib
import fcntl
import shutil

from recording_dataset import verify_dataset


def run_loader(argv, log_path, timeout=300):
    # A file keeps diagnostics visible and avoids inherited output pipes keeping
    # the caller blocked after a multiprocessing loader exits.
    # multiprocessing creates AF_UNIX sockets below TMPDIR (108-byte Linux
    # pathname limit). Attempt paths contain UUIDs and are too long for those.
    # This node-local directory holds transient IPC only, never dataset copies.
    with tempfile.TemporaryDirectory(prefix="sk-load-", dir="/tmp") as ipc, log_path.open("w") as log:
        environment = {**os.environ, "TMPDIR": ipc, "TMP": ipc, "TEMP": ipc}
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True,
                                   env=environment)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise ValueError(f"Training loader validation timed out after {timeout}s; see loader.log") from exc
    text = log_path.read_text()
    print(text[-8000:], flush=True)
    if process.returncode:
        raise ValueError("Training loader validation failed: " + text[-4000:])
    return json.loads(text.strip().splitlines()[-1])


def run(request):
    from recording_prepare import prepare

    output = Path(request["output"])
    if output.exists():
        raise ValueError("Preparation attempt already has output; retry with a new attempt")
    prepare(request)
    manifest_sha = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    manifest = verify_dataset(output, manifest_sha, verify_files=True)
    destination = Path(request["prepared_root"]) / manifest_sha
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Both directories are on the shared dataset filesystem. Only a small manifest
    # is published; stream arrays already live in the recording's immutable store.
    lock_path = destination.parent / ("." + manifest_sha + ".publish.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            if destination.is_symlink():
                raise ValueError("Dataset publication cannot follow a symbolic link")
            verify_dataset(destination, manifest_sha, verify_files=False)
            shutil.rmtree(output)
        else:
            os.rename(output, destination)
    loader = request["loader"]
    argv = [value.replace("{dataset}", str(destination)).replace("{manifest_sha256}", manifest_sha)
            for value in loader["argv"]]
    receipt = run_loader(argv, Path(request["receipt_path"]).with_name("loader.log"))
    if (receipt.get("manifest_sha256") != manifest_sha
            or receipt.get("schema") not in loader["schemas"]
            or receipt.get("observation_mode", "rgb") != loader["mode"]):
        raise ValueError("Training reader verification returned a different dataset")
    return dict(schema="skynet.cluster-preparation/v2", job_id=request["job_id"],
                attempt_id=request["attempt_id"], verified=True, manifest=manifest,
                manifest_sha256=manifest_sha, path=str(destination), loader_validation=receipt)


def main(path):
    request = json.loads(Path(path).read_text())
    receipt_path = Path(request["receipt_path"])
    try:
        receipt = run(request)
        code = 0
    except Exception as exc:
        receipt = dict(schema="skynet.cluster-preparation/v2", job_id=request["job_id"],
                       attempt_id=request["attempt_id"], verified=False, error=str(exc))
        code = 1
    temporary = receipt_path.with_suffix(".part")
    temporary.write_text(json.dumps(receipt, allow_nan=False))
    os.replace(temporary, receipt_path)
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
