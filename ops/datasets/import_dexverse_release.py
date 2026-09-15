"""Validate and split a pinned HF release into canonical recording archives.

Run beside the downloaded files on the cluster. The caller packages the existing
arrays.py restricted unpickler and trajectory.py; no simulator or DB is opened.
The JSON request supplies checked release entries and immutable session profiles.
"""

import argparse
import fcntl
import gc
import hashlib
import json
import os
import pickle
import re
import shutil
from pathlib import Path, PurePosixPath
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from arrays import ArrayUnpickler
from trajectory import validate_identity, validate_recorded_identity


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def session_id(revision, task):
    return str(
        uuid5(
            NAMESPACE_URL,
            f"https://huggingface.co/datasets/dexverse/DexVerse_release@{revision}/{task}",
        )
    )


def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or "\\" in value:
        raise ValueError("Invalid release path")
    return path


def safe_root(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected an absolute storage path")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Storage paths must not contain symbolic links")
    return path


def validate_numeric(value):
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("State keys must be text")
        for child in value.values():
            validate_numeric(child)
        return
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError("Recording state contains invalid numeric values")


def validate_episode(payload, episode):
    actions = np.asarray(episode.get("actions"))
    states = episode.get("states")
    if (
        actions.ndim != 2
        or not all(actions.shape)
        or actions.dtype.kind not in "iuf"
        or not np.isfinite(actions).all()
    ):
        raise ValueError("Invalid episode actions")
    if (
        episode.get("num_steps") != len(actions)
        or not isinstance(states, list)
        or len(states) != len(actions) + 1
    ):
        raise ValueError("Expected T actions and T+1 recorded states")
    if episode.get("success") is not True:
        raise ValueError("Release contains an unsuccessful episode")
    expected = 56 if payload["robot_type"] == "floating_shadow_bimanual" else 28
    if actions.shape[1] != expected:
        raise ValueError("Actions differ from the original Shadow hand layout")
    for state in states:
        if not isinstance(state, dict) or not state.get("articulation", {}).get(
            "robot"
        ):
            raise ValueError("Missing recorded robot scene state")
        validate_numeric(state)
    return len(actions)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def inventory(root, identifier):
    files, directories = [], []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Import archive must not contain links")
        name = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.append(name)
        elif path.is_file():
            files.append(
                {
                    "path": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_hash(path),
                }
            )
        else:
            raise ValueError("Unexpected archive file type")
    return {
        "schema": "skynet.live-archive/v1",
        "session_id": identifier,
        "directories": directories,
        "files": files,
    }


def import_task(request, entry):
    revision = request["revision"]
    identifier = session_id(revision, entry["task"])
    base = safe_root(request["datasets_root"]) / "raw/dexverse-live" / identifier
    base.mkdir(parents=True, exist_ok=True)
    with (base / ".import.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = base / "import.json"
        if receipt.exists():
            job = json.loads(receipt.read_text())
            archive = job["archive"]
            actual = inventory(Path(archive["root"]).parent, identifier)
            if actual != archive["manifest"]:
                raise ValueError(
                    "Existing imported recording changed; refusing to overwrite it"
                )
            return job
        stage = safe_root(base / ".importing")
        if stage.exists():
            shutil.rmtree(stage)  # Only the unpublished directory of this import ID.
        recordings_root = stage / "output/recordings/live"
        recordings_root.mkdir(parents=True)
        recordings, checksums, steps, sizes = [], {}, 0, 0
        profile = None
        for source in entry["files"]:
            relative = safe_relative(source["path"])
            if relative.parts[0] != "demonstrations" or relative.suffix != ".pkl":
                raise ValueError("Only release PKL files can be imported")
            path = safe_root(Path(request["download_root"]) / relative)
            if (
                path.stat().st_size != source["bytes"]
                or file_hash(path) != source["sha256"]
            ):
                raise ValueError(
                    "Downloaded file differs from the pinned release manifest"
                )
            with path.open("rb") as stream:
                payload = ArrayUnpickler(stream).load()
            validate_recorded_identity(payload, task=entry["task"])
            episodes = payload.get("episodes")
            if (
                not isinstance(episodes, list)
                or len(episodes) != source["episodes"]
                or payload.get("num_episodes") != len(episodes)
            ):
                raise ValueError("Release episode count differs from its manifest")
            robot = payload["robot_type"]
            if robot not in {
                "floating_shadow_right",
                "floating_shadow_left",
                "floating_shadow_bimanual",
            }:
                raise ValueError("Unsupported original recording hand")
            side = "both" if robot.endswith("bimanual") else robot.rsplit("_", 1)[1]
            candidate = dict(
                entry["profile"],
                recording_schema_version=payload["schema_version"],
                robot=robot,
                hand=side,
                hand_name="Shadow · "
                + ("both hands" if side == "both" else side + " hand"),
            )
            try:
                validate_identity(payload, task=entry["task"])
            except ValueError as error:
                candidate["replay_unavailable_reason"] = str(error)
            if profile is not None and profile != candidate:
                raise ValueError("A task has recordings from different hand layouts")
            profile = candidate
            metadata = {
                key: value for key, value in payload.items() if key != "episodes"
            }
            for episode in episodes:
                steps += validate_episode(payload, episode)
                name = f"recordings/live/episode-{len(recordings) + 1:06d}.pkl"
                destination = stage / "output" / name
                # Preserve arrays, goals, action layout, release provenance and reset ledger.
                single = dict(metadata, episodes=[episode], num_episodes=1)
                with destination.open("wb") as stream:
                    pickle.dump(single, stream, protocol=5)
                if not 0 < destination.stat().st_size <= 100 * 1024 * 1024:
                    raise ValueError(
                        "Split episode exceeds the review/conversion size limit"
                    )
                checksums[name] = file_hash(destination)
                sizes += destination.stat().st_size
                recordings.append(name)
            del payload, episodes, metadata, single, episode
            gc.collect()
        if len(recordings) != entry["episodes"] or profile is None:
            raise ValueError("Task episode count mismatch")
        provenance = {
            "provider": "huggingface",
            "repository": "dexverse/DexVerse_release",
            "revision": revision,
            "task": entry["task"],
            "files": entry["files"],
            "license": request["license"],
            "attribution": request["attribution"],
            "simulation_replay_verified": request["simulation_replay_verified"],
        }
        write_json(stage / "output/source-manifest.json", provenance)
        manifest = inventory(stage, identifier)
        checksum = hashlib.sha256(canonical(manifest).encode()).hexdigest()
        final = safe_root(base / checksum)
        if final.exists():
            if inventory(final, identifier) != manifest:
                raise ValueError("An existing archive differs from this import")
            shutil.rmtree(stage)
        else:
            stage.rename(final)
        (base / "manifests").mkdir(exist_ok=True)
        write_json(base / "manifests" / (checksum + ".json"), manifest)
        job = {
            "id": identifier,
            "state": "CAPTURED",
            "scheduler_final": True,
            "job_id": None,
            "profile": profile,
            "gateway": "sky2",
            "root": str(final),
            "created_at": request["created_at"],
            "updated_at": request["created_at"],
            "recordings": recordings,
            "recording_checksums": checksums,
            "recording_images": {},
            "recording_summary": {
                "episodes": len(recordings),
                "steps": steps,
                "size_bytes": sizes,
            },
            "import_source": provenance,
            "error": None,
            "archive": {
                "state": "READY",
                "gateway": "sky2",
                "root": str(final / "output"),
                "source_removed": True,
                "manifest": manifest,
                "manifest_sha256": checksum,
                "verified_at": request["created_at"],
            },
        }
        write_json(receipt, job)
        return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if not re.fullmatch(r"[a-f0-9]{40}", request["revision"]):
        raise ValueError("Pin the Hugging Face release to a full commit")
    for entry in request["tasks"]:
        job = import_task(request, entry)
        print(
            canonical(
                {
                    "id": job["id"],
                    "task": entry["task"],
                    "episodes": len(job["recordings"]),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
