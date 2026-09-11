"""Standard-library-only archive worker, shipped to either storage host."""


def archive_control(value):
    import fcntl
    import hashlib
    import json
    import os
    from pathlib import Path, PurePosixPath
    import re
    import shutil
    import stat
    import subprocess
    import sys
    import tarfile
    from uuid import UUID

    def canonical(item):
        return json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def digest(item):
        return hashlib.sha256(canonical(item).encode()).hexdigest()

    identifier = value["session_id"]
    if str(UUID(identifier)) != identifier:
        raise ValueError("Invalid archive session identity")

    def safe(path):
        path = Path(path)
        if not path.is_absolute() or ".." in path.parts or len(path.parts) < 4:
            raise ValueError("Invalid archive workspace path")
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError("Archive paths cannot contain symbolic links")
        return path

    def relative(name):
        path = PurePosixPath(name)
        if (not isinstance(name, str) or not name or path.is_absolute() or ".." in path.parts
                or str(path) != name or "\x00" in name):
            raise ValueError("Invalid relative archive path")
        return name

    def file_hash(path):
        with path.open("rb") as stream:
            result = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                result.update(chunk)
        return result.hexdigest()

    def sync_directory(path):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def write_json(path, item):
        safe(path.parent).mkdir(parents=True, exist_ok=True, mode=0o700)
        safe(path)
        temporary = safe(path.with_suffix(path.suffix + ".tmp"))
        with temporary.open("w") as stream:
            stream.write(canonical(item))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        sync_directory(path.parent)

    def inventory(root):
        safe(root)
        if not root.is_dir() or root.stat().st_uid != os.getuid():
            raise ValueError("Session directory is missing or is owned by another user")
        files, directories = [], []
        for parent, names, leaves in os.walk(root, followlinks=False):
            for name in sorted(names):
                path = Path(parent) / name
                info = path.lstat()
                if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                    raise ValueError("Archive contains a linked or foreign directory")
                directories.append(relative(path.relative_to(root).as_posix()))
            for name in sorted(leaves):
                path = Path(parent) / name
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                    raise ValueError("Archive accepts only owned regular files")
                files.append(dict(path=relative(path.relative_to(root).as_posix()),
                                  size_bytes=info.st_size, sha256=file_hash(path)))
                if len(files) > 100000:
                    raise ValueError("Session exceeds the archive file limit")
        return dict(schema="skynet.live-archive/v1", session_id=identifier,
                    directories=sorted(directories), files=sorted(files, key=lambda entry: entry["path"]))

    def manifest_request():
        manifest = value["manifest"]
        checksum = value["manifest_sha256"]
        if (manifest.get("schema") != "skynet.live-archive/v1" or manifest.get("session_id") != identifier
                or not re.fullmatch(r"[a-f0-9]{64}", str(checksum)) or digest(manifest) != checksum):
            raise ValueError("Archive manifest identity does not match")
        entries = {}
        for item in manifest["files"]:
            name = relative(item["path"])
            if (name in entries or type(item["size_bytes"]) is not int or item["size_bytes"] < 0
                    or not re.fullmatch(r"[a-f0-9]{64}", str(item["sha256"]))):
                raise ValueError("Invalid archive file manifest")
            entries[name] = item
        directories = [relative(name) for name in manifest["directories"]]
        if len(set(directories)) != len(directories) or set(directories) & entries.keys():
            raise ValueError("Invalid archive directory manifest")
        return manifest, checksum, entries

    def service_idle(unit, marker):
        result = subprocess.run(["systemctl", "--user", "show", unit,
            "--property=LoadState,ActiveState,SubState,Environment,MainPID,ControlGroup"],
            capture_output=True, text=True, timeout=15)
        fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        if fields.get("LoadState") == "not-found":
            return
        if result.returncode or not fields.get("LoadState") or marker not in fields.get("Environment", "").split():
            raise ValueError("Could not verify collection process ownership")
        if (fields.get("MainPID") != "0" or
                fields.get("ActiveState") not in {"inactive", "failed"} and fields.get("SubState") != "exited"):
            raise ValueError("Collection or replay is still running; archive waits for it to finish")
        group = fields.get("ControlGroup", "")
        if not group:
            return
        if not group.startswith("/") or ".." in Path(group).parts:
            raise ValueError("Invalid collection control group")
        for hierarchy in ("/sys/fs/cgroup", "/sys/fs/cgroup/systemd"):
            directory = Path(hierarchy + group)
            if not directory.exists():
                continue
            events = directory / "cgroup.events"
            if events.exists():
                fields = dict(line.split() for line in events.read_text().splitlines())
                if fields.get("populated") != "0":
                    raise ValueError("Collection descendants are still running")
            else:
                groups = list(directory.rglob("cgroup.procs"))
                if not groups or any(path.read_text().strip() for path in groups):
                    raise ValueError("Collection descendants are still running")
            return

    def source_idle(root):
        unit = f"skynet-live-{identifier}.service"
        service_idle(unit, f"SKYNET_LIVE_SESSION_ID={identifier}")
        profile_path = safe(root / "request.json")
        if profile_path.is_file():
            if digest(json.loads(profile_path.read_text())) != value["profile_sha256"]:
                raise ValueError("The session execution profile no longer matches")
        elif not value.get("allow_unlaunched"):
            raise ValueError("The session execution profile is missing")
        worker = safe(root / "runner.py")
        if worker.exists() and value.get("worker_sha256") and file_hash(worker) != value["worker_sha256"]:
            raise ValueError("The session worker no longer matches its submitted source")
        status_path = safe(root / "output/status.json")
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            if status.get("job_id") != unit or status.get("state") not in {"CAPTURED", "STOPPED", "TIMED_OUT", "FAILED"}:
                raise ValueError("The collection has not published a final owned status")
        elif not value.get("allow_unlaunched"):
            raise ValueError("The collection has not published a final owned status")
        for path in root.glob("output/review-videos/**/owner.json"):
            owner = json.loads(safe(path).read_text())
            token = owner.get("generation", "")
            unit = f"skynet-video-{token}.service"
            if not re.fullmatch(r"[a-f0-9]{32}", token) or owner.get("unit") != unit or path.parent.name != token:
                raise ValueError("Invalid replay ownership record")
            service_idle(unit, f"SKYNET_VIDEO_GENERATION={token}")

    operation = value["operation"]
    if operation in {"manifest", "stream", "delete"}:
        workspace = safe(value["work_root"])
        root = safe(workspace / "sessions" / identifier)
        lock_directory = safe(workspace / ".archive-locks")
    else:
        dataset_root = safe(value["datasets_root"])
        namespace = value.get("namespace", "dexverse-live")
        if namespace not in {"dexverse-live", "operator-cache"}:
            raise ValueError("Unknown archive storage namespace")
        base = safe(dataset_root / "raw" / namespace / identifier)
        lock_directory = base
    lock_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = safe(lock_directory / (identifier + ".lock"))
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if operation == "manifest":
            if not root.exists() and value.get("allow_absent"):
                service_idle(f"skynet-live-{identifier}.service", f"SKYNET_LIVE_SESSION_ID={identifier}")
                return {"absent": True}
            source_idle(root)
            manifest = inventory(root)
            return dict(manifest=manifest, manifest_sha256=digest(manifest))
        manifest, checksum, entries = manifest_request()
        if operation == "stream":
            source_idle(root)
            if inventory(root) != manifest:
                raise ValueError("The source session changed before transfer")
            # Never hold a source lock while waiting on a destination lock.
            # Concurrent identical transfers could otherwise deadlock across
            # the two hosts. Cleanup re-verifies the complete source tree.
            fcntl.flock(lock, fcntl.LOCK_UN)
            with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
                for name, item in entries.items():
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = item["size_bytes"], 0o600
                    with safe(root / name).open("rb") as stream:
                        archive.addfile(info, stream)
            return None
        if operation == "delete":
            quarantine = safe(workspace / "sessions" / f".archived-{identifier}-{checksum}")
            receipt = safe(lock_directory / (identifier + ".deleting.json"))
            if quarantine.exists():
                if root.exists() or not receipt.is_file() or json.loads(receipt.read_text()) != {"manifest_sha256": checksum}:
                    raise ValueError("Interrupted archive cleanup ownership is unclear")
                remaining = inventory(quarantine)
                if (not set(remaining["directories"]).issubset(manifest["directories"])
                        or any(entries.get(item["path"]) != item for item in remaining["files"])):
                    raise ValueError("Unverified data appeared during archive cleanup")
            elif root.exists():
                source_idle(root)
                if inventory(root) != manifest:
                    raise ValueError("The source session changed; its files were retained")
                write_json(receipt, {"manifest_sha256": checksum})
                root.rename(quarantine)
                sync_directory(root.parent)
            if quarantine.exists():
                if not shutil.rmtree.avoids_symlink_attacks:
                    raise ValueError("Safe session directory cleanup is unavailable")
                shutil.rmtree(quarantine)
                sync_directory(quarantine.parent)
            receipt.unlink(missing_ok=True)
            return dict(removed=True, manifest_sha256=checksum)
        final = safe(base / checksum)
        if operation == "verify":
            if inventory(final) != manifest:
                raise ValueError("The archived session is incomplete or changed")
            return dict(verified=True, manifest_sha256=checksum, root=str(final / "output"))
        if operation != "receive":
            raise ValueError("Unknown archive operation")
        complete = final.exists()
        stage = final if complete else safe(base / (".incoming-" + checksum))
        if complete and inventory(final) != manifest:
            raise ValueError("Existing archive differs; refusing replacement")
        stage.mkdir(mode=0o700, exist_ok=True)
        for name in manifest["directories"]:
            safe(stage / name).mkdir(mode=0o700, parents=True, exist_ok=True)
        parts = safe(base / ".parts" / checksum)
        if not complete:
            parts.mkdir(mode=0o700, parents=True, exist_ok=True)
            needed = sum(item["size_bytes"] for item in entries.values()
                         if not (stage / item["path"]).is_file())
            if shutil.disk_usage(base).free < needed + 100_000_000:
                raise ValueError("Not enough space on the archive destination")
        seen = set()
        with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as archive:
            for member in archive:
                name = relative(member.name)
                item = entries.get(name)
                if not item or name in seen or not member.isfile() or member.size != item["size_bytes"]:
                    raise ValueError("Transfer contains an unexpected archive entry")
                seen.add(name)
                target = safe(stage / name)
                retained = target.is_file() and target.stat().st_size == item["size_bytes"] and file_hash(target) == item["sha256"]
                temporary = safe(parts / hashlib.sha256(name.encode()).hexdigest())
                output = None if retained or complete else temporary.open("wb")
                calculated, received = hashlib.sha256(), 0
                try:
                    source = archive.extractfile(member)
                    while chunk := source.read(1024 * 1024):
                        calculated.update(chunk)
                        received += len(chunk)
                        if output:
                            output.write(chunk)
                    if received != item["size_bytes"] or calculated.hexdigest() != item["sha256"]:
                        raise ValueError("Transferred file checksum does not match")
                    if output:
                        output.flush()
                        os.fsync(output.fileno())
                finally:
                    if output:
                        output.close()
                if not retained and not complete:
                    temporary.replace(target)
                    sync_directory(target.parent)
        if seen != entries.keys() or inventory(stage) != manifest:
            raise ValueError("Transfer did not preserve the complete session")
        if not complete:
            for directory in sorted((stage / name for name in manifest["directories"]), key=lambda path: len(path.parts), reverse=True):
                sync_directory(directory)
            sync_directory(stage)
            stage.rename(final)
            sync_directory(base)
            shutil.rmtree(parts)
        write_json(safe(base / "manifests" / (checksum + ".json")), manifest)
        return dict(verified=True, manifest_sha256=checksum, root=str(final / "output"))
