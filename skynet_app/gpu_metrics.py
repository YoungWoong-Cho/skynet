"""Dependency-free GPU sampler, also frozen into each training capsule."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time

FIELDS = (
    "index",
    "uuid",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
)
INTERVAL_SECONDS = 15


def allocated_devices(environ=None):
    env = os.environ if environ is None else environ
    # Slurm GRES indices can differ from NVIDIA's visible indices (for example
    # GRES 2,3,6,7 becomes CUDA 0,1,2,3 on this cluster). Prefer the CUDA mapping.
    for key in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        if key in env:
            values = [v.strip() for v in env[key].split(",") if v.strip()]
            if not all(re.fullmatch(r"\d+|GPU-[\w-]+|MIG-[\w/-]+", v) for v in values):
                return []
            return values
    # Never report other tenants' GPUs when the allocation is unknown.
    return []


def parse_devices(text):
    devices = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if len(row) != len(FIELDS) or not row[0].strip().isdigit():
            continue
        raw = dict(zip(FIELDS, (value.strip() for value in row)))

        def number(key):
            try:
                value = float(raw[key])
                return value if math.isfinite(value) and value >= 0 else None
            except ValueError:
                return None

        metrics = {
            "gpu": number("utilization.gpu"),
            "memory": number("utilization.memory"),
            "temp": number("temperature.gpu"),
            "powerWatts": number("power.draw"),
            "smClock": number("clocks.sm"),
            "memoryClock": number("clocks.mem"),
        }
        used, total = number("memory.used"), number("memory.total")
        power, limit = number("power.draw"), number("power.limit")
        if used is not None:
            metrics["memoryAllocatedBytes"] = used * 1024**2
        if total:
            metrics["memoryTotalBytes"] = total * 1024**2
            if used is not None:
                metrics["memoryAllocated"] = used / total * 100
        if power is not None and limit:
            metrics["powerPercent"] = power / limit * 100
        devices.append(
            {
                "index": int(raw["index"]),
                "uuid": raw["uuid"],
                "name": raw["name"],
                "metrics": {
                    key: value for key, value in metrics.items() if value is not None
                },
            }
        )
    return devices


def sample(environ=None):
    env = os.environ if environ is None else environ
    ids = allocated_devices(env)
    if not ids:
        return None
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + ",".join(ids),
            "--query-gpu=" + ",".join(FIELDS),
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    devices = parse_devices(result.stdout)
    if not devices:
        return None
    return {
        "schema_version": 1,
        "timestamp_ms": int(time.time() * 1000),
        "job_id": env.get("SLURM_JOB_ID", ""),
        "node": socket.gethostname(),
        "node_rank": int(env.get("SLURM_NODEID", "0")),
        "devices": devices,
    }


class GPUSampler:
    """One sampler per allocation/node; failures never stop the policy trainer."""

    def __init__(self, directory, interval=INTERVAL_SECONDS):
        self.directory = Path(directory)
        self.interval = interval
        self.stopped = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(
            target=self.run, daemon=True, name="skynet-gpu-statistics"
        )
        self.thread.start()
        return self

    def stop(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=6)

    def run(self, once=False):
        lock = None
        warned = False
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            node = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname())
            lock = (self.directory / (node + ".lock")).open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            while not self.stopped.is_set():
                try:
                    record = sample()
                    if record is not None:
                        with (self.directory / (node + ".jsonl")).open("a") as stream:
                            stream.write(json.dumps(record, allow_nan=False) + "\n")
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    if not warned:
                        print(
                            f"GPU statistics unavailable: {type(error).__name__}",
                            file=sys.stderr,
                        )
                        warned = True
                if once or self.stopped.wait(self.interval):
                    break
        except OSError as error:
            print(
                f"GPU statistics unavailable: {type(error).__name__}", file=sys.stderr
            )
        finally:
            if lock is not None:
                lock.close()


def read_records(directory, cursors, limit=1_000_000):
    """Read complete records incrementally, retaining a partial last line."""
    records, next_cursors, latest, oldest = [], dict(cursors), 0.0, None
    backlog = False
    remaining = limit
    for path in sorted(Path(directory).glob("*.jsonl")):
        if path.is_symlink():
            continue
        info = path.stat()
        latest = max(latest, info.st_mtime)
        oldest = min(oldest, info.st_mtime) if oldest is not None else info.st_mtime
        offset = max(0, int(cursors.get(path.name, 0)))
        if offset > info.st_size:
            offset = 0
        if remaining <= 0:
            backlog = backlog or offset < info.st_size
            continue
        with path.open("rb") as stream:
            stream.seek(offset)
            content = stream.read(remaining)
        complete = content[: content.rfind(b"\n") + 1]
        for line in complete.splitlines():
            try:
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)
            except (ValueError, UnicodeDecodeError):
                continue
        next_cursors[path.name] = offset + len(complete)
        # A partially written record is retried on the next poll; a full page
        # means more complete history may remain and should be drained now.
        backlog = backlog or offset + len(content) < info.st_size
        remaining -= len(complete)
    return {
        "records": records,
        "cursors": next_cursors,
        "latest": latest,
        "oldest": oldest or 0,
        "backlog": backlog,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--read-dir")
    parser.add_argument("--cursors", default="{}")
    args = parser.parse_args()
    if args.read_dir:
        print(json.dumps(read_records(args.read_dir, json.loads(args.cursors))))
    elif args.output:
        GPUSampler(args.output).run(once=args.once)
    else:
        parser.error("Choose --output or --read-dir")


if __name__ == "__main__":
    main()
