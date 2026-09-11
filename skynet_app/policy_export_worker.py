"""CPU capsule entry point. Every payload path is on shared cluster storage."""

import json
import os
from pathlib import Path
import subprocess
import sys

from artifacts import digest, materialize, pack, verify


def run(request):
    from policy_export import export

    output = Path(request["output"])
    # A retry has its own private attempt directory; never rewrite a published version.
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
        result = subprocess.run(argv, capture_output=True, text=True, timeout=300, check=False)
        if result.returncode:
            raise ValueError("Training loader validation failed: " + (result.stderr or result.stdout)[-4000:])
        receipt = json.loads(result.stdout.strip().splitlines()[-1])
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
