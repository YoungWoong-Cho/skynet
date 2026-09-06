"""Run CloudXR and the pinned DexVerse recorder as ordinary Slurm processes."""

from __future__ import annotations
import argparse
import json
import ipaddress
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time


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
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if not cfg.get("cloudxr_eula_accepted"):
        raise ValueError("Accept the NVIDIA CloudXR license before starting a session")
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Live XR must run inside an allocated Slurm GPU job")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cxr = Path(cfg["cloudxr_runtime"])
    repo = Path(cfg["repository"])
    runtime = Path(cfg["runtime"])
    if (repo / ".skynet-source-revision").read_text().strip() != cfg["source_revision"]:
        raise ValueError("The DexVerse source revision does not match this session")
    run = Path("/tmp") / ("skynet-cloudxr-" + os.environ["SLURM_JOB_ID"])
    run.mkdir(mode=0o700)
    started = time.time()
    children = []
    stopping = False
    status = {
        "schema": "skynet.live-xr/v1",
        "job_id": os.environ["SLURM_JOB_ID"],
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
            pending_result = (state, details)
            publish(
                "STOPPING",
                server_ready=False,
                detail="Closing live processes and saving logs",
            )
        else:
            publish(state, **details)

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        update("STARTING_SERVER", detail="Starting CloudXR 5.0.1")
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
            if stopping:
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
        argv = [
            str(runtime / "bin/python"),
            str(repo / "scripts/record_demos.py"),
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
            "dexpilot",
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
                        f"DexVerse exited with code {sim.returncode}; inspect the simulation log"
                    )
                files = list((root / "recordings").rglob("*.pkl"))
                if not files:
                    text = (root / "simulation.log").read_text(errors="replace")
                    errors = [
                        line
                        for line in text.splitlines()
                        if "Failed to create environment:" in line
                    ]
                    raise RuntimeError(
                        errors[-1]
                        if errors
                        else "DexVerse exited without a demonstration file"
                    )
                update(
                    "CAPTURED",
                    detail="A native demonstration was saved; dataset validation is still required",
                    recordings=[str(p.relative_to(root)) for p in files],
                )
                return
            with (root / "simulation.log").open("rb") as log:
                log.seek(max(0, (root / "simulation.log").stat().st_size - 16000))
                tail = log.read().decode(errors="replace")
            if "Teleop Device:" in tail and status["state"] == "STARTING_SIMULATION":
                update(
                    "AWAITING_HEADSET",
                    detail="Scene loaded. Connect the headset, then use START to calibrate and record.",
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
