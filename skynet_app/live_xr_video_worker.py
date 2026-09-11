"""Remote supervisor for one video generation, separate from collection services."""


def control(value):
    # This function is shipped verbatim to the workstation's system Python.
    import fcntl
    import json
    import os
    from pathlib import Path
    import re
    import subprocess

    token = value["generation"]
    if not re.fullmatch(r"[a-f0-9]{32}", token):
        raise ValueError("Invalid video generation")
    root = Path(value["root"])
    if not root.is_absolute() or ".." in root.parts or root.name != token or root.parent.name != "attempts":
        raise ValueError("Invalid video generation directory")
    unit = f"skynet-video-{token}.service"
    marker = f"SKYNET_VIDEO_GENERATION={token}"
    root.mkdir(parents=True, exist_ok=True)

    def run(args, timeout=15):
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)

    def status():
        result = run(["systemctl", "--user", "show", unit,
                      "--property=LoadState,ActiveState,SubState,ExecMainStatus,Environment,MainPID,ControlGroup"])
        fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        if fields.get("LoadState") == "not-found":
            return fields
        if result.returncode or not fields.get("LoadState"):
            raise RuntimeError(result.stderr.strip() or "Could not inspect the video service")
        if marker not in fields.get("Environment", "").split():
            raise RuntimeError("Video service ownership does not match; refusing to stop it")
        return fields

    def stopped(fields):
        if fields.get("LoadState") == "not-found":
            return True
        if fields.get("ActiveState") not in {"inactive", "failed"} or fields.get("MainPID") != "0":
            return False
        group = fields.get("ControlGroup", "")
        if not group:
            return True
        if not group.startswith("/") or ".." in Path(group).parts:
            raise RuntimeError("Invalid video control group")
        # A failed service can still contain an uninterruptible descendant.
        # Confirm the whole owned cgroup is empty, on both cgroup v2 and v1.
        for hierarchy in ("/sys/fs/cgroup", "/sys/fs/cgroup/systemd"):
            directory = Path(hierarchy + group)
            if not directory.exists():
                continue
            events = directory / "cgroup.events"
            if events.exists():
                values = dict(line.split() for line in events.read_text().splitlines())
                return values.get("populated") == "0"
            processes = list(directory.rglob("cgroup.procs"))
            return bool(processes) and all(not path.read_text().strip() for path in processes)
        return True

    def stop():
        fields = status()
        if not stopped(fields):
            result = run(["systemctl", "--user", "stop", unit], timeout=20)
            fields = status()
            if result.returncode or not stopped(fields):
                raise RuntimeError(result.stderr.strip() or "The video service has not stopped yet")
        return fields

    def active_result(metadata):
        # systemd starts flock before the renderer acquires the GPU. Only the
        # renderer's own status file proves that the protected work started.
        if metadata.get("state") == "RENDERING":
            result = {"state": "PREPARING", "phase": "rendering"}
            frame, total = metadata.get("frame"), metadata.get("total")
            if type(frame) is int and type(total) is int and 0 <= frame <= total and total > 0:
                result.update(frame=frame, total=total)
            return result
        if metadata.get("state") in {"READY", "FAILED"}:
            return {"state": "PREPARING", "phase": "finalizing"}
        return {"state": "STARTING"}

    def waiting_result():
        result = {"state": "WAITING_GPU"}
        # The flock exit status is authoritative. Owner information is only a
        # read-only explanation; /proc may be restricted or change mid-read.
        try:
            request = json.loads((root / "request.json").read_text())
            profile = request.get("profile") or value.get("profile") or {}
            lock_path = Path(profile["work_root"]) / ".gpu-session.lock"
            info = lock_path.stat()
            identity = (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
            for line in Path("/proc/locks").read_text().splitlines():
                fields = line.split()
                if len(fields) < 6 or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
                    continue
                device = fields[5].split(":")
                if len(device) != 3:
                    continue
                held = (int(device[0], 16), int(device[1], 16), int(device[2]))
                if held != identity or not fields[4].isdigit() or int(fields[4]) <= 0:
                    continue
                pid = int(fields[4])
                busy_owner = {"pid": pid, "kind": "process"}
                try:
                    groups = Path(f"/proc/{pid}/cgroup").read_text()
                    match = re.search(r"/(skynet-(live-[a-f0-9-]{36}|video-[a-f0-9]{32})\.service)(?:/|\s|$)", groups)
                    if match:
                        busy_owner.update(unit=match[1], kind="collection" if match[2].startswith("live-") else "video")
                except OSError:
                    pass
                result["busy_owner"] = busy_owner
                break
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            pass
        return result

    # Cancel and launch share only this local filesystem lock. A cancellation
    # marker written before a delayed launch prevents that launch altogether.
    with (root / "control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        owner = root / "owner.json"
        identity = {"generation": token, "unit": unit}
        if owner.exists():
            if json.loads(owner.read_text()) != identity:
                raise RuntimeError("Video directory ownership does not match")
        else:
            temporary = root / "owner.tmp"
            temporary.write_text(json.dumps(identity))
            temporary.replace(owner)
        if value["operation"] == "cancel":
            (root / "cancelled").touch()
            stop()
            # These files belong exclusively to this generation; never touch
            # the capture source, another replay, or the collection service.
            for path in root.glob("*.partial.mp4"):
                path.unlink(missing_ok=True)
            for name in ("video.mp4", "video.json"):
                (root / name).unlink(missing_ok=True)
            return {"state": "CANCELLED"}
        if (root / "cancelled").exists():
            return {"state": "CANCELLED"}
        fields = status()
        metadata_path = root / "video.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if fields.get("SubState") == "exited":
            stop()  # Reap remaining descendants before publishing a completed video.
        elif not stopped(fields):
            return active_result(metadata)
        if metadata.get("state") == "READY":
            if metadata.get("path") != str(root / "video.mp4"):
                raise RuntimeError("Rendered video path does not match its generation")
            cache = root.parent.parent / "video.json"
            temporary = cache.with_name(f"video.{token}.tmp")
            temporary.write_text(json.dumps(metadata))
            temporary.replace(cache)
            return metadata
        if value["operation"] == "status":
            if fields.get("ExecMainStatus") == "75":
                return waiting_result()
            error = metadata.get("error")
            log = root / "render.log"
            if not error and log.exists():
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 1500))
                    error = stream.read().decode(errors="replace")
            return {"state": "FAILED", "error": error or "Video service exited without a completed video"}
        # Preserve the previous remote-cache behavior, including completed
        # videos from older releases, without launching a new GPU process.
        base = root.parent.parent
        cache = base / "video.json"
        if cache.exists():
            cached = json.loads(cache.read_text())
            path = Path(cached.get("path", ""))
            legacy = path == base / "video.mp4"
            owned = (path.name == "video.mp4" and path.parent.parent == base / "attempts"
                     and re.fullmatch(r"[a-f0-9]{32}", path.parent.name)
                     and not (path.parent / "cancelled").exists())
            if cached.get("state") == "READY" and (legacy or owned) and path.is_file():
                return cached
        profile = value["profile"]
        for name, content in value["sources"].items():
            if name not in {"render_recording.py", "video.py", "wrist.py", "arrays.py"}:
                raise ValueError("Invalid video worker source")
            path = root / name
            if path.exists() and path.read_text() != content:
                raise ValueError("Video worker differs from its immutable source")
            path.write_text(content)
        request = dict(value["request"], output=str(root / "video.mp4"))
        (root / "request.json").write_text(json.dumps(request))
        # A stopped attempt may have left progress behind, even if systemd
        # already garbage collected its unit. A new launch has no GPU yet.
        metadata_path.unlink(missing_ok=True)
        # A failed nonblocking flock can be retried on the same owned unit.
        if fields.get("LoadState") != "not-found":
            stop()
            run(["systemctl", "--user", "reset-failed", unit])
            # Transient units may remain loaded until systemd garbage collects
            # them. Restart an existing owned unit instead of redefining it.
            if status().get("LoadState") != "not-found":
                result = run(["systemctl", "--user", "start", unit], timeout=25)
                if not result.returncode:
                    return {"state": "STARTING"}
                if status().get("LoadState") != "not-found":
                    raise RuntimeError(result.stderr.strip() or "Could not restart the video service")
        runtime = profile["runtime"]
        args = ["systemd-run", "--user", "--quiet", f"--unit={unit}",
                "--service-type=exec", "--remain-after-exit",
                "--property=RuntimeMaxSec=600", "--property=TimeoutStopSec=5",
                "--property=KillMode=control-group", "--property=Restart=no",
                "--property=UMask=0077", f"--working-directory={profile['repository']}",
                f"--property=StandardOutput=append:{root / 'render.log'}",
                f"--property=StandardError=append:{root / 'render.log'}",
                f"--setenv={marker}", "--setenv=OMNI_KIT_ACCEPT_EULA=YES",
                "--setenv=PYTHONUNBUFFERED=1", f"--setenv=LD_LIBRARY_PATH={runtime}/lib",
                f"--setenv=PYTHONPATH={profile['repository']}/source/dexverse",
                f"--setenv=PATH={runtime}/bin:{os.environ.get('PATH', '/usr/bin:/bin')}",
                "/usr/bin/flock", "--nonblock", "--no-fork", "--conflict-exit-code=75",
                profile["work_root"] + "/.gpu-session.lock", runtime + "/bin/python",
                str(root / "render_recording.py"), str(root / "request.json")]
        result = run(args, timeout=25)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Could not start the video service")
        return {"state": "STARTING"}
