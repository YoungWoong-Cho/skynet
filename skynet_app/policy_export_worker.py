"""CPU capsule entry point. Every payload path is on shared cluster storage."""

import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile

from artifacts import digest, materialize, pack, verify


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
    from policy_export import export

    output = Path(request["output"])
    previous = Path(request["reuse_output"]) if request.get("reuse_output") else None
    if previous and (previous / "manifest.json").is_file() and (previous.parent / "dataset.zip").is_file():
        previous_manifest = verify(previous, digest(previous / "manifest.json"))
        for key in ("source_revision", "converter_sha256", "contract", "split"):
            if previous_manifest.get(key) != request.get(key):
                raise ValueError("Previous preparation does not match the pinned conversion inputs")
        expected_sources = [(s["sha256"], s.get("image_sha256")) for s in request["sources"]]
        if [(e["sha256"], e.get("image_sha256")) for e in previous_manifest["episodes"]] != expected_sources:
            raise ValueError("Previous preparation belongs to different recordings")
        output = previous
        print("Reusing verified conversion output; checking publication and loader", flush=True)
    else:
        # Each attempt builds new output. A published version is never overwritten.
        if output.exists() and not request.get("migration"):
            raise ValueError("Preparation attempt already has output; start a new attempt")
        for source in request.get("sources", []):
            for path_key, sha_key in (("recording", "sha256"), ("images", "image_sha256")):
                if source.get(path_key):
                    path = Path(source[path_key])
                    if path.is_symlink() or digest(path) != source[sha_key]:
                        raise ValueError("Archived source checksum verification failed")
        if request.get("migration"):
            verify(output, request["expected_manifest_sha256"])
            pack(output)
        else:
            export(request)
    manifest_sha = digest(output / "manifest.json")
    manifest = verify(output, manifest_sha)
    archive = output.parent / "dataset.zip"
    archive_sha = digest(archive)
    destination = Path(request["prepared_root"]) / manifest_sha
    materialize(archive, destination, manifest_sha, archive_sha)
    loader = request.get("loader")
    receipt = None
    if loader:
        argv = [value.replace("{dataset}", str(destination)).replace("{manifest_sha}", manifest_sha)
                for value in loader["argv"]]
        receipt = run_loader(argv, Path(request["receipt_path"]).with_name("loader.log"))
        if (receipt.get("manifest_sha256") != manifest_sha
                or receipt.get("schema") not in loader["schemas"]
                or receipt.get("observation_mode", "rgb") != loader["mode"]):
            raise ValueError("Training loader verification returned a different dataset")
    return dict(schema="skynet.cluster-preparation/v1", job_id=request["job_id"],
                attempt_id=request["attempt_id"], verified=True, manifest=manifest,
                manifest_sha256=manifest_sha, archive_sha256=archive_sha,
                path=str(destination), archive_path=str(archive), loader_validation=receipt)


def main(path):
    request = json.loads(Path(path).read_text())
    receipt_path = Path(request["receipt_path"])
    try:
        receipt = run(request)
        code = 0
    except Exception as exc:
        receipt = dict(schema="skynet.cluster-preparation/v1", job_id=request["job_id"],
                       attempt_id=request["attempt_id"], verified=False, error=str(exc))
        code = 1
    temporary = receipt_path.with_suffix(".part")
    temporary.write_text(json.dumps(receipt, allow_nan=False))
    os.replace(temporary, receipt_path)
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
