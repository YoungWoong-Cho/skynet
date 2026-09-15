from __future__ import annotations

import base64
import json
import re
import shlex
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Mapping

from .cluster_config import CLUSTER


UV_VERSION = "0.12.8"
UV_ARCHIVE_SHA256 = "2e2b37e9811e17675a9e70bed5e1a58fc8c0388be63d751d72cc735188c149ff"
HUGGINGFACE_HUB_VERSION = "1.29.0"


@dataclass(frozen=True)
class DataImportJob:
    script: str
    run_id: str
    job_name: str
    result_path: str


_IMPORT_PROGRAM = r'''from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime, timezone

from huggingface_hub import HfApi, RepoFile, hf_hub_download


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def inventory(root, output):
    records = []
    total = 0
    for current, directories, files in os.walk(root, followlinks=True):
        directories.sort()
        files.sort()
        current_path = Path(current)
        for name in files:
            path = current_path / name
            resolved = path.resolve(strict=True)
            size = resolved.stat().st_size
            relative = path.relative_to(root).as_posix()
            records.append((relative, sha256_file(resolved), size))
            total += size
    records.sort(key=lambda item: item[0])
    with output.open("w", encoding="utf-8") as stream:
        for relative, digest, size in records:
            stream.write(f"{digest}\t{size}\t{relative}\n")
    return len(records), total, sha256_file(output)


def validate_request(request):
    if request.get("provider") != "huggingface":
        raise ValueError("this importer only accepts the huggingface provider")
    revision = str(request.get("revision") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Hugging Face revision must be an exact 40-character commit SHA")
    subset = str(request.get("subset") or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", subset):
        raise ValueError("subset must be a repository-relative directory without globs")
    if any(part in {"", ".", ".."} for part in Path(subset).parts):
        raise ValueError("subset contains an unsafe path component")


def download_subset(request, work_root):
    repo_id = f"{request['namespace']}/{request['name']}"
    subset = request["subset"]
    revision = request["revision"]
    print(
        f"SKYNET_DATA_IMPORT_PHASE=listing repo={repo_id} revision={revision} subset={subset}",
        flush=True,
    )
    entries = list(
        HfApi().list_repo_tree(
            repo_id=repo_id,
            path_in_repo=subset,
            recursive=True,
            expand=True,
            revision=revision,
            repo_type="dataset",
        )
    )
    files = sorted(entry.path for entry in entries if isinstance(entry, RepoFile))
    if not files:
        raise RuntimeError(f"selected Hugging Face subset contains no files: {subset}")
    sizes = {entry.path: int(entry.size or 0) for entry in entries if isinstance(entry, RepoFile)}
    expected_bytes = sum(sizes.values())
    print(
        f"SKYNET_DATA_IMPORT_PHASE=downloading files={len(files)} expected_bytes={expected_bytes}",
        flush=True,
    )
    cache_dir = work_root / ".cache" / "huggingface" / "hub"

    def download(filename):
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir),
        )
        return filename, Path(cached)

    downloaded = {}
    with ThreadPoolExecutor(max_workers=int(request.get("cpus", 8))) as executor:
        futures = [executor.submit(download, filename) for filename in files]
        for completed, future in enumerate(as_completed(futures), start=1):
            filename, cached = future.result()
            downloaded[filename] = cached
            print(
                f"SKYNET_DATA_IMPORT_PROGRESS={completed}/{len(files)} bytes={sizes[filename]} file={filename}",
                flush=True,
            )

    first_filename = files[0]
    snapshot = downloaded[first_filename]
    for _ in Path(first_filename).parts:
        snapshot = snapshot.parent
    snapshot = snapshot.resolve(strict=True)
    for filename, cached in downloaded.items():
        if cached.relative_to(snapshot).as_posix() != filename:
            raise RuntimeError(f"Hugging Face cache returned an unexpected path for {filename}: {cached}")
    return snapshot


def existing_result(published, request):
    manifest_path = published / "manifest.json"
    ready_path = published / "READY"
    data_path = published / "data"
    if not (manifest_path.is_file() and ready_path.is_file() and data_path.exists()):
        raise RuntimeError(f"refusing to replace incomplete published path: {published}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source") or {}
    selection = manifest.get("selection") or {}
    if (
        source.get("namespace") != request["namespace"]
        or source.get("name") != request["name"]
        or source.get("revision") != request["revision"]
        or selection.get("subset") != request["subset"]
        or manifest.get("format") != request["format"]
    ):
        raise RuntimeError("published import path belongs to a different immutable request")
    manifest_sha256 = sha256_file(manifest_path)
    expected_ready = ready_path.read_text(encoding="utf-8").split()[0]
    if expected_ready != manifest_sha256:
        raise RuntimeError("published import READY digest does not match manifest.json")
    integrity = manifest.get("integrity") or {}
    return {
        "manifest_sha256": manifest_sha256,
        "inventory_sha256": integrity.get("inventory_sha256"),
        "file_count": integrity.get("file_count"),
        "size_bytes": integrity.get("payload_bytes"),
        "path": str(data_path),
        "manifest_path": str(manifest_path),
        "source_uri": source.get("pinned_uri"),
        "snapshot_path": (manifest.get("materialization") or {}).get("snapshot_path"),
    }


def main():
    request_path = Path(sys.argv[1]).resolve()
    result_path = Path(sys.argv[2]).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    validate_request(request)
    work_root = Path(request["work_root"]).resolve()
    datasets_root = work_root / "datasets"
    selection_key = hashlib.sha256(
        (request["subset"] + "\0" + request["format"]).encode("utf-8")
    ).hexdigest()[:12]
    selection_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", request["subset"]).strip("-")[:80]
    published = (
        datasets_root / "resources" / "huggingface" / request["namespace"] / request["name"]
        / "selections" / f"{selection_slug}-{selection_key}" / "revisions" / request["revision"]
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if published.exists():
        materialized = existing_result(published, request)
    else:
        staging = datasets_root / ".staging" / f"huggingface-{request['import_id']}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        snapshot = download_subset(request, work_root)
        subset_path = (snapshot / request["subset"]).resolve(strict=True)
        subset_path.relative_to(snapshot)
        if not subset_path.is_dir():
            raise RuntimeError(f"selected Hugging Face subset is not a directory: {request['subset']}")
        data_link = staging / "data"
        data_link.symlink_to(subset_path, target_is_directory=True)
        inventory_path = staging / "inventory.sha256"
        file_count, payload_bytes, inventory_sha256 = inventory(data_link, inventory_path)
        if file_count < 1:
            raise RuntimeError("selected Hugging Face subset contains no files")
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        pinned_uri = (
            f"https://huggingface.co/datasets/{request['namespace']}/{request['name']}"
            f"/tree/{request['revision']}/{request['subset']}"
        )
        manifest = {
            "schema_version": "dataset-resource/v1",
            "resource_id": (
                f"resource:huggingface/{request['namespace']}/{request['name']}"
                f"@{request['revision']}#subset={request['subset']}"
            ),
            "status": "ready",
            "kind": request["kind"],
            "format": request["format"],
            "created_at": created_at,
            "source": {
                "provider": "huggingface",
                "repository_type": "dataset",
                "namespace": request["namespace"],
                "name": request["name"],
                "revision": request["revision"],
                "url": f"https://huggingface.co/datasets/{request['namespace']}/{request['name']}",
                "pinned_uri": pinned_uri,
            },
            "selection": {
                "subset": request["subset"],
                "allow_pattern": f"{request['subset']}/**",
                "selection_sha256": selection_key,
            },
            "materialization": {
                "mode": "huggingface_cache_symlink",
                "snapshot_path": str(snapshot),
                "data_path": str(published / "data"),
            },
            "integrity": {
                "inventory_path": "inventory.sha256",
                "inventory_sha256": inventory_sha256,
                "file_count": file_count,
                "payload_bytes": payload_bytes,
            },
            "provenance": {
                "publisher": "skynet.data_imports/v1",
                "huggingface_hub_version": request["huggingface_hub_version"],
                "import_id": request["import_id"],
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_sha256 = sha256_file(manifest_path)
        (staging / "READY").write_text(f"{manifest_sha256}  manifest.json\n", encoding="utf-8")
        published.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(published)
        materialized = {
            "manifest_sha256": manifest_sha256,
            "inventory_sha256": inventory_sha256,
            "file_count": file_count,
            "size_bytes": payload_bytes,
            "path": str(published / "data"),
            "manifest_path": str(published / "manifest.json"),
            "source_uri": pinned_uri,
            "snapshot_path": str(snapshot),
        }
    result = {
        "schema_version": "skynet.data-import-result/v1",
        "import_id": request["import_id"],
        "resource_id": request["resource_id"],
        "provider": request["provider"],
        "namespace": request["namespace"],
        "name": request["name"],
        "kind": request["kind"],
        "revision": request["revision"],
        "version_revision": f"{request['revision']}#subset={request['subset']}",
        "subset": request["subset"],
        "format": request["format"],
        "role": request["role"],
        "bundle_name": request["bundle_name"],
        "bundle_version": request["bundle_version"],
        **materialized,
    }
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    print("SKYNET_DATA_IMPORT_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
'''


def build_huggingface_import_job(
    import_id: str,
    resource: Mapping[str, Any],
    request: Mapping[str, Any],
) -> DataImportJob:
    queue_name = str(request["queue"])
    queue = CLUSTER.queues.get(queue_name)
    if queue is None:
        raise ValueError(f"unknown import queue: {queue_name}")
    run_id = f"data-import-{import_id}"
    job_slug = re.sub(r"[^A-Za-z0-9_-]+", "-", str(resource["source_key"])).strip("-")[:30]
    job_name = f"hf-{job_slug}-{import_id[:8]}"
    run_directory = f"{CLUSTER.paths.jobs}/runs/{run_id}"
    result_path = f"{run_directory}/import-result.json"
    # Keep the v1 wire identity stable for existing imports and their receipts.
    payload = {
        "schema_version": "skynet.data-import-request/v1",
        "import_id": import_id,
        "resource_id": str(resource["id"]),
        "provider": str(resource["provider"]),
        "namespace": str(resource["namespace"]),
        "name": str(resource["source_key"]),
        "kind": str(resource["kind"]),
        "revision": str(request["revision"]),
        "subset": str(request["subset"]),
        "format": str(request["format"]),
        "role": str(request["role"]),
        "bundle_name": str(request["bundle_name"]),
        "bundle_version": str(request["bundle_version"]),
        "work_root": CLUSTER.paths.work_root,
        "huggingface_hub_version": HUGGINGFACE_HUB_VERSION,
        "cpus": int(request.get("cpus", 8)),
    }
    request_b64 = base64.b64encode(
        (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    ).decode("ascii")
    program_b64 = base64.b64encode(_IMPORT_PROGRAM.encode("utf-8")).decode("ascii")
    time_limit = str(request.get("time_limit") or ("04:00:00" if not queue.preemptible else "24:00:00"))
    cpus = int(request.get("cpus", 8))
    memory_gb = int(request.get("memory_gb", 32))
    script = dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH --job-name={job_name}
        #SBATCH --partition={queue.partition}
        #SBATCH --account={queue.account}
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task={cpus}
        #SBATCH --mem={memory_gb}G
        #SBATCH --time={time_limit}
        #SBATCH --requeue
        #SBATCH --chdir={CLUSTER.paths.workspace}
        #SBATCH --output={CLUSTER.paths.logs}/%x-%j.out
        #SBATCH --error={CLUSTER.paths.logs}/%x-%j.err

        set -euo pipefail
        umask 0027
        export HOME={shlex.quote(CLUSTER.paths.home_root)}
        export WORK_ROOT={shlex.quote(CLUSTER.paths.work_root)}
        export UV_CACHE_DIR={shlex.quote(CLUSTER.paths.uv_cache)}
        export HF_HOME={shlex.quote(CLUSTER.paths.huggingface_cache)}
        export XDG_CACHE_HOME="$WORK_ROOT/.cache"
        RUN_DIR={shlex.quote(run_directory)}
        RESULT_PATH={shlex.quote(result_path)}
        UV_VERSION={UV_VERSION}
        UV_ROOT="$WORK_ROOT/tools/uv/$UV_VERSION"
        UV="$UV_ROOT/uv"
        UV_ARCHIVE="$UV_ROOT/uv-x86_64-unknown-linux-gnu.tar.gz"
        PYTHON={shlex.quote(CLUSTER.paths.home_root + '/miniconda3/bin/python')}
        mkdir -p "$RUN_DIR" "$WORK_ROOT/logs" "$WORK_ROOT/tmp/$SLURM_JOB_ID" \
          "$WORK_ROOT/.cache/uv" "$WORK_ROOT/.cache/huggingface" "$UV_ROOT"
        export TMPDIR="$WORK_ROOT/tmp/$SLURM_JOB_ID"
        printf %s {shlex.quote(request_b64)} | base64 --decode > "$RUN_DIR/import-request.json"
        printf %s {shlex.quote(program_b64)} | base64 --decode > "$RUN_DIR/import-huggingface.py"
        if [[ ! -x "$UV" ]]; then
          curl --fail --location --retry 8 --retry-delay 5 --retry-all-errors \
            --output "$UV_ARCHIVE.part" \
            "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz"
          printf '%s  %s\\n' {UV_ARCHIVE_SHA256} "$UV_ARCHIVE.part" | sha256sum -c -
          mv "$UV_ARCHIVE.part" "$UV_ARCHIVE"
          tar -xzf "$UV_ARCHIVE" --strip-components=1 -C "$UV_ROOT" \
            uv-x86_64-unknown-linux-gnu/uv uv-x86_64-unknown-linux-gnu/uvx
        fi
        "$UV" --version | grep -F "uv $UV_VERSION"
        "$UV" run --python "$PYTHON" --with "huggingface-hub=={HUGGINGFACE_HUB_VERSION}" \
          python "$RUN_DIR/import-huggingface.py" "$RUN_DIR/import-request.json" "$RESULT_PATH"
        """
    )
    return DataImportJob(script=script, run_id=run_id, job_name=job_name, result_path=result_path)


__all__ = ["DataImportJob", "build_huggingface_import_job"]
