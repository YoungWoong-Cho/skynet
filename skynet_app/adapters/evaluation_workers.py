"""Coordinate isolated GPU rollout workers and publish one canonical result.

The adapter worker consumes explicit (seed, episode) assignments. Each Slurm
step owns one GPU; no simulator shares a CUDA context with another worker.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading

from xpolicy_runtime import write_json


def episode_key(episode):
    return episode["task"], episode["seed"], episode["episode_index"]


def expected_keys(context):
    return [
        (task, seed, index)
        for task in context["tasks"]
        for seed in context["seeds"]
        for index in range(context["episodes_per_task"])
    ]


def worker_contexts(context):
    """Stable disjoint assignments keep completed episodes reusable after preemption."""
    total = expected_keys(context)
    parallelism = context["parallelism"]
    if not 1 <= parallelism <= min(8, len(total)) or len(total) != len(set(total)):
        raise ValueError(
            "Choose distinct tasks/seeds and no more workers than episodes"
        )
    root = Path(context["result_path"]).parent
    units = []
    for task in context["tasks"]:
        pairs = [
            [seed, index]
            for seed in context["seeds"]
            for index in range(context["episodes_per_task"])
        ]
        for lane in range(min(parallelism, len(pairs))):
            unit = dict(context)
            index = len(units)
            directory = root / "workers" / str(index)
            unit.update(
                tasks=[task],
                parallelism=1,
                episode_assignments=pairs[lane::parallelism],
                worker_index=index,
                result_path=str(directory / "result.json"),
                progress_path=str(directory / "progress.jsonl"),
                video_path=str(root / "videos" / str(index)),
            )
            units.append(unit)
    return units


def observed_results(context, units):
    """Read atomic results and durable episode ledgers; reject foreign/duplicate data."""
    from dexverse_evaluation import observed_episodes, identity

    observed = {}
    for unit in units:
        rows = observed_episodes(unit["progress_path"], identity(unit))
        allowed = {(unit["tasks"][0], s, i) for s, i in unit["episode_assignments"]}
        if not set(rows) <= allowed or set(rows) & set(observed):
            raise ValueError("Worker returned unexpected or duplicate episodes")
        observed.update(rows)
    return observed


def completed_results(context, units):
    return {
        key: ep
        for key, ep in observed_results(context, units).items()
        if ep["status"] == "SUCCEEDED"
    }


def publish_progress(context, observed):
    path = Path(context["progress_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    total = len(expected_keys(context))
    with temporary.open("w") as stream:
        for ep in observed.values():
            stream.write(
                json.dumps(
                    dict(
                        kind="episode_observed",
                        total=total,
                        episode=ep,
                        recorded_at=datetime.now(timezone.utc).isoformat(),
                    ),
                    allow_nan=False,
                )
                + "\n"
            )
    temporary.replace(path)


def merged_result(context, observed):
    expected = expected_keys(context)
    if set(observed) != set(expected):
        raise ValueError("Evaluation workers did not return every requested episode")
    episodes = [observed[key] for key in expected]
    aggregates = []
    for task in [None, *context["tasks"]]:
        values = [
            float(ep["success"])
            for ep in episodes
            if task is None or ep["task"] == task
        ]
        aggregates.append(
            dict(
                metric="success_rate",
                unit="fraction",
                mean=statistics.fmean(values),
                std=statistics.pstdev(values),
                sample_count=len(values),
                task=task,
            )
        )
    return dict(
        schema_version=1,
        run_id=context["run_id"],
        checkpoint={k: context["checkpoint"][k] for k in ("path", "sha256")},
        evaluator=context["evaluator"],
        environment=dict(
            suite=context["suite"]["name"], version=context["suite"]["version"]
        ),
        aggregate=aggregates,
        episodes=episodes,
        raw_metrics_path=context["progress_path"],
        artifacts=[ep["video_path"] for ep in episodes],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--source-dir", required=True)
    args = parser.parse_args()
    context = json.loads(Path(args.context).read_text())
    units = worker_contexts(context)
    processes, lock, stopped = set(), threading.Lock(), threading.Event()

    def terminate(signum=None, frame=None):
        stopped.set()
        with lock:
            for process in list(processes):
                if process.poll() is None:
                    process.terminate()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, terminate)

    def run(unit):
        if stopped.is_set():
            return
        directory = Path(unit["result_path"]).parent
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "context.json"
        write_json(path, unit)
        resources = context["worker_resources"]
        command = [
            sys.executable,
            str(Path(__file__).with_name("recorded_policy_evaluation.py" if context.get("compatibility", {}).get("policy_loader") else "xpolicy_evaluation.py")),
            "--context",
            str(path),
            "--source-dir",
            args.source_dir,
        ]
        if context["parallelism"] > 1:
            if not os.environ.get("SLURM_JOB_ID"):
                raise ValueError("Parallel GPU evaluation requires a Slurm allocation")
            command = [
                "srun",
                "--exclusive",
                "--exact",
                "--nodes=1",
                "--ntasks=1",
                "--gres=gpu:"
                + resources["gpu"].get("type", resources["gpu"].get("gpu_type", "a40"))
                + ":1",
                "--cpus-per-task=" + str(resources["cpus_per_task"]),
                "--mem=" + str(resources["memory_gb"]) + "G",
                *command,
            ]
        print(
            json.dumps(
                dict(
                    event="worker_started",
                    worker=unit["worker_index"],
                    task=unit["tasks"][0],
                    assignments=unit["episode_assignments"],
                )
            ),
            flush=True,
        )
        with (
            (directory / "stdout.log").open("a") as stdout,
            (directory / "stderr.log").open("a") as stderr,
        ):
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            with lock:
                processes.add(process)
                if stopped.is_set():
                    process.terminate()
            try:
                code = process.wait()
            finally:
                with lock:
                    processes.discard(process)
        if code:
            print(
                (directory / "stderr.log").read_text()[-12000:],
                file=sys.stderr,
                flush=True,
            )
            raise RuntimeError(
                f"Rollout worker {unit['worker_index']} exited with status {code}"
            )
        # A zero exit alone is not evidence that Kit published its result.
        result = json.loads(Path(unit["result_path"]).read_text())
        expected = {(unit["tasks"][0], s, i) for s, i in unit["episode_assignments"]}
        if {episode_key(ep) for ep in result["episodes"]} != expected:
            raise ValueError("Rollout worker result does not match its assignments")
        print(
            json.dumps(dict(event="worker_finished", worker=unit["worker_index"])),
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=context["parallelism"]) as pool:
        pending = {pool.submit(run, unit) for unit in units}
        failure = None
        while pending:
            done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    future.result()
                except BaseException as exc:
                    failure = failure or exc
                    terminate()
            publish_progress(context, observed_results(context, units))
        if failure:
            raise failure
        if stopped.is_set():
            raise RuntimeError(
                "Evaluation was interrupted; completed episodes are preserved"
            )
    observed = completed_results(context, units)
    publish_progress(context, observed)
    write_json(context["result_path"], merged_result(context, observed))


if __name__ == "__main__":
    main()
