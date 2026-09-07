"""Run CloudXR and DexVerse inside a managed workstation or Slurm session."""

from __future__ import annotations
import argparse
import json
import ipaddress
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time


def inspect_recording(
    path, task="Dexverse-PickUpStick-v0", robot="floating_shadow_right"
):
    """Validate a trusted pickle written by this session's pinned recorder."""
    import pickle
    import numpy as np

    if not 0 < path.stat().st_size <= 1024**3:
        raise ValueError(
            "Native demonstration is empty or exceeds the 1 GiB validation limit"
        )
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "dexverse_trajectory"
        or payload.get("schema_version") != 3
        or payload.get("task") != task
        or payload.get("robot_type") != robot
    ):
        raise ValueError("Unsupported native demonstration format, task or robot")
    episodes = payload.get("episodes")
    if (
        not isinstance(episodes, list)
        or not episodes
        or payload.get("num_episodes") != len(episodes)
    ):
        raise ValueError("Native recording contains no complete demonstration")
    steps = 0
    for episode in episodes:
        actions = np.asarray(episode.get("actions"))
        if (
            actions.ndim != 2
            or not all(actions.shape)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Native demonstration has empty or invalid action values")
        if episode.get("num_steps") != len(actions):
            raise ValueError(
                "Native demonstration action count does not match its step count"
            )
        states = episode.get("states")
        if (
            not isinstance(states, list)
            or len(states) != len(actions) + 1
            or not all(isinstance(s, dict) and s for s in states)
        ):
            raise ValueError(
                "Native demonstration must have one initial state and one state per action"
            )
        if episode.get("success") is not True:
            raise ValueError(
                "Native demonstration did not satisfy the task success condition"
            )
        steps += len(actions)
    return {"episodes": len(episodes), "steps": steps, "success": True}


def simulation_failure(path, fallback):
    """Surface a bounded, actionable setup error rather than only an exit code."""
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 16000))
            tail = stream.read().decode(errors="replace")
    except OSError:
        return fallback + "; inspect the simulation log"
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", line).strip() for line in tail.splitlines()]
    hand_errors = [
        line.partition("SKYNET_HAND_ERROR: ")[2]
        for line in lines
        if "SKYNET_HAND_ERROR: " in line
    ]
    errors = hand_errors or [
        line
        for line in lines
        if re.search(
            r"(?:\b[A-Za-z]+(?:Error|Exception):|Failed to create (?:environment|teleop device):)",
            line,
        )
    ]
    return (
        fallback + ": " + errors[-1][:1000]
        if errors
        else fallback + "; inspect the simulation log"
    )


def server_address():
    """Select the node's actual default-route address, never a previous job's IP."""
    result = subprocess.run(
        ["ip", "-j", "-4", "route", "get", "1.1.1.1"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    # This inspects the routing table; it does not send a packet to that address.
    address = json.loads(result.stdout)[0]["prefsrc"]
    if ipaddress.ip_address(address).is_loopback:
        raise ValueError("GPU node has no reachable network address")
    return address


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect-recording", type=Path)
    parser.add_argument("--task", default="Dexverse-PickUpStick-v0")
    parser.add_argument("--robot", default="floating_shadow_right")
    args = parser.parse_args()
    if args.inspect_recording:
        print(
            json.dumps(inspect_recording(args.inspect_recording, args.task, args.robot))
        )
        return
    if not args.config or not args.output:
        parser.error("--config and --output are required to run a live session")
    cfg = json.loads(args.config.read_text())
    if not cfg.get("cloudxr_eula_accepted"):
        raise ValueError("Accept the NVIDIA CloudXR license before starting a session")
    execution = cfg.get("execution", "slurm")
    if execution == "workstation":
        identifier = os.environ.get("SKYNET_LIVE_SESSION_ID", "")
        job_id = os.environ.get("SKYNET_LIVE_JOB_ID", "")
        if (
            not re.fullmatch(r"[a-f0-9-]{36}", identifier)
            or job_id != "skynet-live-" + identifier + ".service"
        ):
            raise ValueError(
                "Workstation sessions must run through their managed live service"
            )
    elif execution == "slurm" and os.environ.get("SLURM_JOB_ID"):
        identifier = job_id = os.environ["SLURM_JOB_ID"]
    else:
        raise ValueError(
            "Live XR requires an allocated Slurm job or managed workstation session"
        )
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cxr = Path(cfg["cloudxr_runtime"])
    repo = Path(cfg["repository"])
    runtime = Path(cfg["runtime"])
    if (repo / ".skynet-source-revision").read_text().strip() != cfg["source_revision"]:
        raise ValueError("The DexVerse source revision does not match this session")
    run = Path("/tmp") / ("skynet-cloudxr-" + identifier)
    run.mkdir(mode=0o700)
    started = time.time()
    children = []
    stopping = False
    status = {
        "schema": "skynet.live-xr/v1",
        "job_id": job_id,
        "execution": execution,
        "host": socket.getfqdn(),
        "address": server_address(),
        "task": cfg["task"],
        "robot": cfg["robot"],
        "started_at": started,
    }

    pending_result = None

    def publish(state, **details):
        status.update(state=state, updated_at=time.time(), **details)
        tmp = root / "status.tmp"
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(root / "status.json")
        print(json.dumps(status), flush=True)

    def update(state, **details):
        nonlocal pending_result
        if state in ("CAPTURED", "STOPPED", "TIMED_OUT", "FAILED"):
            details["server_ready"] = False
            if state == "FAILED":
                details.setdefault("failed_stage", status.get("startup_stage"))
            pending_result = (state, details)
            publish(
                "STOPPING",
                server_ready=False,
                detail="Closing live processes and saving logs",
                **{
                    key: details[key]
                    for key in ("error", "failed_stage")
                    if key in details
                },
            )
        else:
            publish(state, **details)

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        update(
            "STARTING_SERVER", startup_stage="stream", detail="Starting CloudXR 5.0.1"
        )
        # Holding an input pipe avoids Monado epoll errors on /dev/null.
        with (root / "cloudxr.log").open("w") as stream:
            server = subprocess.Popen(
                [str(cxr / "bin/cloudxr-service")],
                stdin=subprocess.PIPE,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=dict(
                    os.environ,
                    ACCEPT_EULA="Y",
                    XDG_RUNTIME_DIR=str(run),
                    LD_LIBRARY_PATH=str(cxr / "lib"),
                ),
            )
        children.append(server)
        deadline = time.monotonic() + 45
        while (
            not (run / "runtime_started").exists()
            or not (run / "ipc_cloudxr").is_socket()
        ):
            if stopping or (root / "stop.request").exists():
                raise InterruptedError("Session stopped during server startup")
            if server.poll() is not None:
                raise RuntimeError(
                    "CloudXR exited before readiness; inspect the CloudXR log"
                )
            if time.monotonic() > deadline:
                raise RuntimeError("CloudXR did not become ready within 45 seconds")
            time.sleep(0.3)
        # Do not export the server's bundled libraries into Isaac Sim.
        env = dict(
            os.environ,
            OMNI_KIT_ACCEPT_EULA="YES",
            PYTHONUNBUFFERED="1",
            XDG_RUNTIME_DIR=str(run),
            XR_RUNTIME_JSON=str(cxr / "share/openxr/1/openxr_cloudxr.json"),
            LD_LIBRARY_PATH=str(runtime / "lib"),
            PYTHONPATH=str(repo / "source/dexverse"),
            TMPDIR=str(run),
            DEXVERSE_DATA_DIR=str(root / "recordings"),
            PATH=str(runtime / "bin") + ":" + os.environ["PATH"],
        )
        recorder = repo / "scripts/record_demos.py"
        if cfg.get("hand_bundle"):
            bundle = cfg["hand_bundle"]
            if bundle["robot"] != cfg["robot"]:
                raise ValueError("Selected robot differs from the prepared hand")
            recorder = Path(bundle["root"]) / "record.py"
            env["SKYNET_DEXVERSE_RECORDER"] = str(repo / "scripts/record_demos.py")
        argv = [
            str(runtime / "bin/python"),
            str(recorder),
            "--task",
            cfg["task"],
            "--robot_type",
            cfg["robot"],
            "--teleop_device",
            "handtracking",
            "--headless",
            "--device",
            "cuda:0",
            "--num_demos",
            "1",
            "--dataset_dir",
            "live",
            "--teleop_retargeter",
            "relative",
            "--retargeting_scheme",
            cfg.get("hand_bundle", {}).get("retargeting_scheme", "dexpilot"),
        ]
        with (root / "simulation.log").open("w") as stream:
            sim = subprocess.Popen(
                argv,
                cwd=repo,
                env=env,
                stdin=subprocess.PIPE,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        children.append(sim)
        update(
            "STARTING_SIMULATION",
            startup_stage="simulation",
            stream_ready_at=time.time(),
            server_ready=True,
            detail="CloudXR is ready; loading the DexVerse scene",
        )
        deadline = time.monotonic() + cfg.get("duration_minutes", 30) * 60
        while not stopping and time.monotonic() < deadline:
            if (root / "stop.request").exists():
                stopping = True
                break
            if server.poll() is not None:
                raise RuntimeError("CloudXR stopped during the live session")
            if sim.poll() is not None:
                if sim.returncode:
                    raise RuntimeError(
                        simulation_failure(
                            root / "simulation.log",
                            f"DexVerse exited with code {sim.returncode}",
                        )
                    )
                files = list((root / "recordings").rglob("*.pkl"))
                if not files:
                    raise RuntimeError(
                        simulation_failure(
                            root / "simulation.log",
                            "DexVerse exited without a demonstration file",
                        )
                    )
                summaries = []
                for path in files:
                    result = subprocess.run(
                        [
                            str(runtime / "bin/python"),
                            str(Path(__file__).resolve()),
                            "--inspect-recording",
                            str(path),
                            "--task",
                            cfg["task"],
                            "--robot",
                            cfg["robot"],
                        ],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if result.returncode:
                        raise ValueError(
                            "Native demonstration validation failed: "
                            + result.stderr.strip()[-2000:]
                        )
                    summaries.append(json.loads(result.stdout))
                update(
                    "CAPTURED",
                    startup_stage="ready",
                    scene_ready_at=status.get("scene_ready_at", time.time()),
                    detail="A successful demonstration was saved and validated. Open Review recording to inspect and download it.",
                    recordings=[str(p.relative_to(root)) for p in files],
                    recording_summary={
                        "episodes": sum(s["episodes"] for s in summaries),
                        "steps": sum(s["steps"] for s in summaries),
                    },
                )
                return
            with (root / "simulation.log").open("rb") as log:
                log.seek(max(0, (root / "simulation.log").stat().st_size - 16000))
                tail = log.read().decode(errors="replace")
            if "Teleop Device:" in tail and status["state"] == "STARTING_SIMULATION":
                update(
                    "AWAITING_HEADSET",
                    startup_stage="ready",
                    scene_ready_at=time.time(),
                    detail="Scene loaded. Connect the headset, then choose Play to calibrate and record.",
                )
            time.sleep(1)
        update(
            "STOPPED" if stopping else "TIMED_OUT",
            detail="Live session stopped; any saved recordings remain available",
        )
    except InterruptedError as exc:
        update("STOPPED", error=str(exc))
    except Exception as exc:
        update("FAILED", detail=str(exc), server_ready=False, error=str(exc))
        raise
    finally:
        try:
            for child in reversed(children):
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)
                if child.stdin:
                    child.stdin.close()
            for log in Path("/tmp").glob("cxr_*.log"):
                if log.stat().st_uid == os.getuid() and log.stat().st_mtime >= started:
                    (root / log.name).write_bytes(log.read_bytes())
            # Isaac creates nested logs in TMPDIR. Remove only this newly-created
            # private job directory, after both child processes have exited.
            shutil.rmtree(run)
        except Exception as exc:
            publish(
                "FAILED",
                server_ready=False,
                detail="Session cleanup failed",
                error=str(exc),
            )
            raise
        if pending_result:
            state, details = pending_result
            publish(state, **details)


if __name__ == "__main__":
    main()
