"""Offline execution fixture for real converter parity tests; never imported by the app."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ops.datasets.artifacts import digest, pack, verify
from skynet_app.dataset_formats import RECIPES
from skynet_app.database import canonical_json
from skynet_app.policy_exports import conversion_failure

def download(self, session, relative, target, expected, limit, expected_size=None):
    if target.is_file() and digest(target) == expected:
        return
    if self.stopping:
        raise ValueError("Preparation interrupted by app shutdown")
    remote = session["root"] + "/output/" + relative
    transport = self.live.transport(session)
    host, size = transport.file_size(remote, session["gateway"])
    if not 0 < size <= limit or (
        expected_size is not None and size != expected_size
    ):
        raise ValueError("Source recording size is invalid or changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".part")
    received = 0
    try:
        with temp.open("wb") as stream:
            for block in transport.stream_file_range(
                remote, host, start=0, end=size - 1
            ):
                if self.stopping:
                    raise ValueError("Preparation interrupted by app shutdown")
                received += len(block)
                if received > size:
                    raise ValueError("Source recording grew during download")
                stream.write(block)
        if received != size or digest(temp) != expected:
            raise ValueError("Source recording checksum verification failed")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def prepare(self, identifier):
    """Offline converter harness retained for format compatibility validation."""
    directory = self.root / identifier
    try:
        job = self.get(identifier)
        manifest_path = directory / "output/manifest.json"
        manifest = None
        if manifest_path.exists() and job.get("manifest_sha256"):
            try:
                manifest = verify(directory / "output", job["manifest_sha256"])
            except (ValueError, OSError):
                pass
        if manifest is None:
            size = sum(s["image_size_bytes"] for s in job["sources"])
            if shutil.disk_usage(directory).free < size * 8 + 1_000_000_000:
                raise ValueError(
                    "Not enough local disk space for this image preparation"
                )
            self.update(
                identifier,
                state="RUNNING",
                stage="FETCHING",
                detail="Fetching and verifying original recordings",
            )
            sources = []
            for item in job["sources"]:
                session = self.live.get(item.get("session_id", job["session_id"]))
                recording = self.root / "sources" / (item["sha256"] + ".pkl")
                images = self.root / "sources" / (item["image_sha256"] + ".hdf5") if item.get("image_sha256") else None
                self.download(
                    session, item["path"], recording, item["sha256"], 100_000_000
                )
                if images is not None:
                    self.download(
                        session,
                        item["image_path"],
                        images,
                        item["image_sha256"],
                        4_000_000_000,
                        item["image_size_bytes"],
                    )
                sources.append(
                    dict(item, recording=str(recording), images=str(images) if images else None)
                )
            # This directory belongs only to this job. Failed partial output is never registered.
            if (directory / "output").exists():
                shutil.rmtree(directory / "output")
            (directory / "dataset.zip.part").unlink(missing_ok=True)
            request = {
                key: job[key]
                for key in (
                    "format",
                    "split",
                    "contract",
                    "source_revision",
                    "converter_sha256",
                )
            }
            request.update(output=str(directory / "output"), sources=sources)
            (directory / "request.json").write_text(canonical_json(request))
            self.update(
                identifier,
                stage="CONVERTING",
                detail="Converting the selected observations and actions",
            )
            with (directory / "export.log").open("w") as log:
                interpreter = [sys.executable]
                dependencies = directory / "worker/conversion-dependencies.json"
                if dependencies.is_file():
                    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
                    if not Path(uv).is_file():
                        raise ValueError("Install uv to prepare this dataset format")
                    interpreter = [uv, "run", "--no-project", "--python", sys.executable]
                    for package in json.loads(dependencies.read_text()):
                        interpreter.extend(["--with", package])
                    interpreter.append("python")
                process = subprocess.Popen(
                    [
                        *interpreter,
                        str(directory / "worker/policy_export.py"),
                        str(directory / "request.json"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(
                        os.environ,
                        PYTHONUNBUFFERED="1",
                        PYTHONDONTWRITEBYTECODE="1",
                    ),
                )
                with self.lock:
                    self.processes[identifier] = process
                try:
                    result = process.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise ValueError("Dataset preparation exceeded one hour")
            if result:
                raise ValueError(conversion_failure(
                    result, (directory / "export.log").read_text()
                ))
            self.update(
                identifier,
                stage="VALIDATING",
                detail="Verifying the prepared dataset",
            )
            manifest_sha = digest(manifest_path)
            manifest = verify(directory / "output", manifest_sha)
            self.update(identifier, manifest_sha256=manifest_sha)
        else:
            manifest_sha = job["manifest_sha256"]
        version = self.register(job, manifest, manifest_sha)
        if not (directory / "dataset.zip").is_file():
            pack(directory / "output")
        archive_sha = digest(directory / "dataset.zip")
        local = self.database.record_data_location(
            version["id"],
            kind="local",
            host="local",
            path=str(directory / "output"),
            manifest_sha256=manifest_sha,
        )
        self.update(
            identifier,
            version_id=version["id"],
            archive_sha256=archive_sha,
            size_bytes=version["size_bytes"],
            episodes=len(manifest["episodes"]),
            steps=manifest["steps"],
        )
        if job.get("target") == "cluster":
            self.update(
                identifier,
                stage="TRANSFERRING",
                detail="Copying the verified dataset to the training cluster",
            )
            location = self.transfer(self.get(identifier))
            bundle = self.bundle(self.get(identifier), location)
            self.update(identifier, bundle_id=bundle["id"] if bundle else None)
        else:
            location = local
        can_train = (
            RECIPES[job["format"]]["trainable"] and location["kind"] == "cluster"
        )
        detail = (
            "Ready to use in a training experiment"
            if can_train
            else (
                "Prepared on this computer"
                if location["kind"] == "local"
                else "Files prepared on the cluster; a training adapter is still required"
            )
        )
        self.update(
            identifier,
            state="READY",
            stage="READY",
            detail=detail,
            training_ready=can_train,
            error=None,
        )
    except Exception as exc:
        self.update(
            identifier,
            state="FAILED",
            error=str(exc),
            detail="Preparation failed; original recordings are preserved",
        )
    finally:
        with self.lock:
            self.processes.pop(identifier, None)
            self.active.discard(identifier)
