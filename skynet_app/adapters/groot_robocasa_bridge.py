from __future__ import annotations

GROOT_ROBOCASA_BRIDGE_SOURCE = r'''#!/usr/bin/env python3
"""Pinned GR00T N1.6 GR1 RoboCasa evaluator.

This capsule owns only GR00T/RoboCasa-specific translation. Slurm placement,
checkpoint pinning, retries, and the canonical evaluation result contract are
owned by the surrounding Skynet pipeline.
"""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from typing import Any


SCHEMA = "skynet.evaluation-context/v1"
SUITE = "groot_gr1_tabletop"
ENVIRONMENT = "mujoco"
EMBODIMENT = "GR1"
CATALOG_HASH = "f4e3061c8d95f6433fd08596aee7e8adbac705641f3eea77b6e42c2390c210c8"
ROOT_REVISION = "9b37aa1ce69c73c6d165233fa88128283bba4508"
CLIENT_RELATIVE = Path("gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python")
ROLLOUT_RELATIVE = Path("gr00t/eval/rollout_policy.py")
SERVER_RELATIVE = Path("gr00t/eval/run_gr00t_server.py")
RESULT_PATTERN = re.compile(r"results:\s*(\(.*\))\s*success rate:\s*([0-9.]+)", re.DOTALL)
VIDEO_DIRECTORY_PATTERN = re.compile(r"^Video saved to:\s*(/.+?)\s*$", re.MULTILINE)
POLICY_PROFILE = "gr1"
POLICY_MODALITY_KEYS = {
    "video": ["ego_view"],
    "state": ["left_arm", "right_arm", "left_hand", "right_hand"],
    "action": ["left_arm", "right_arm", "left_hand", "right_hand"],
    "language": ["annotation.human.action.task_description"],
}
SIMULATOR_VIDEO_KEY = "video.ego_view_pad_res256_freq20"
SIMULATOR_CROP_VIDEO_KEY = "video.ego_view_bg_crop_pad_res256_freq20"
SIMULATOR_LANGUAGE_KEY = "annotation.human.coarse_action"

CLIENT_RUNNER_SOURCE = r"""import os
import random
import runpy
import sys

import numpy as np
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.sim import env_utils
from robocasa.utils.gym_utils.gymnasium_groot import GrootRoboCasaEnv


SOURCE_VIDEO = "video.ego_view_pad_res256_freq20"
SOURCE_CROP_VIDEO = "video.ego_view_bg_crop_pad_res256_freq20"
TARGET_VIDEO = "video.ego_view"
SOURCE_LANGUAGE = "annotation.human.coarse_action"
TARGET_LANGUAGE = "annotation.human.action.task_description"
EXPECTED_OBSERVATIONS = {
    TARGET_VIDEO,
    "state.left_arm",
    "state.right_arm",
    "state.left_hand",
    "state.right_hand",
    TARGET_LANGUAGE,
}
EXPECTED_ACTIONS = {
    "action.left_arm",
    "action.right_arm",
    "action.left_hand",
    "action.right_hand",
}
ARMS_ONLY_ENV_PREFIX = "robocasa_gr1_arms_only_fourier_hands"

original_init = GrootRoboCasaEnv.__init__
original_observation = GrootRoboCasaEnv.get_groot_observation
original_embodiment_resolver = env_utils.get_embodiment_tag_from_env_name


def adapted_embodiment_resolver(env_name):
    prefix = env_name.split("/", 1)[0]
    if prefix == ARMS_ONLY_ENV_PREFIX:
        return EmbodimentTag.GR1
    return original_embodiment_resolver(env_name)


def adapted_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    spaces = self.observation_space.spaces
    actual_observations = set(spaces)
    simulator_contract = {
        SOURCE_VIDEO,
        SOURCE_CROP_VIDEO,
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        SOURCE_LANGUAGE,
    }
    if actual_observations != simulator_contract:
        raise RuntimeError(
            "RoboCasa arms-only observation contract changed: "
            f"expected {sorted(simulator_contract)}, received {sorted(actual_observations)}"
        )
    spaces[TARGET_VIDEO] = spaces.pop(SOURCE_VIDEO)
    spaces.pop(SOURCE_CROP_VIDEO)
    spaces[TARGET_LANGUAGE] = spaces.pop(SOURCE_LANGUAGE)
    if set(spaces) != EXPECTED_OBSERVATIONS:
        raise RuntimeError("RoboCasa observation adapter produced an invalid policy contract")
    if set(self.action_space.spaces) != EXPECTED_ACTIONS:
        raise RuntimeError(
            "RoboCasa arms-only action contract changed: "
            f"expected {sorted(EXPECTED_ACTIONS)}, received {sorted(self.action_space.spaces)}"
        )


def adapted_observation(self, raw_observation):
    observation = original_observation(self, raw_observation)
    actual = set(observation)
    required = {
        SOURCE_VIDEO,
        SOURCE_CROP_VIDEO,
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        SOURCE_LANGUAGE,
    }
    if actual != required:
        raise RuntimeError(
            "RoboCasa arms-only observation values changed: "
            f"expected {sorted(required)}, received {sorted(actual)}"
        )
    observation[TARGET_VIDEO] = observation.pop(SOURCE_VIDEO)
    observation.pop(SOURCE_CROP_VIDEO)
    observation.pop(SOURCE_LANGUAGE)
    language = raw_observation.get("language")
    if not isinstance(language, str) or not language.strip():
        raise RuntimeError("RoboCasa did not provide a non-empty task instruction")
    observation[TARGET_LANGUAGE] = language
    if set(observation) != EXPECTED_OBSERVATIONS:
        raise RuntimeError("RoboCasa observation values do not match the checkpoint contract")
    return observation


GrootRoboCasaEnv.__init__ = adapted_init
GrootRoboCasaEnv.get_groot_observation = adapted_observation
env_utils.get_embodiment_tag_from_env_name = adapted_embodiment_resolver

seed = int(os.environ["SKYNET_EVAL_SEED"])
random.seed(seed)
np.random.seed(seed)
target = sys.argv[1]
sys.argv = [target, *sys.argv[2:]]
runpy.run_path(target, run_name="__main__")
"""


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(mapping: dict[str, Any], key: str, where: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "" or value == []:
        fail(f"Missing required {where}.{key}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Cannot read evaluation context {path}: {exc}")
    if not isinstance(value, dict):
        fail("Evaluation context must be a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return cleaned[-120:] or "task"


def allocate_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(process: subprocess.Popen[str], port: int, stderr_path: Path) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            detail = stderr_path.read_text(encoding="utf-8", errors="replace")[-8000:]
            fail(f"GR00T policy server exited with code {return_code}:\n{detail}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    fail("GR00T policy server did not become ready within 15 minutes")


def parse_rollout(stdout: str) -> tuple[bool, float | None, int | None]:
    match = RESULT_PATTERN.search(stdout)
    if match is None:
        fail("Official GR00T rollout output did not contain its documented results tuple")
    try:
        result = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError) as exc:
        fail(f"Could not parse official GR00T rollout result: {exc}")
    if not isinstance(result, tuple) or len(result) != 3:
        fail("Official GR00T rollout returned an unexpected result shape")
    successes = result[1]
    info = result[2]
    if not isinstance(successes, list) or len(successes) != 1:
        fail(f"Expected exactly one episode result, received {successes!r}")
    if not isinstance(info, dict):
        fail("Official GR00T rollout episode information is not a dictionary")
    rewards = info.get("episode_rewards")
    lengths = info.get("episode_lengths")
    reward = float(rewards[0]) if isinstance(rewards, list) and len(rewards) == 1 else None
    length = int(lengths[0]) if isinstance(lengths, list) and len(lengths) == 1 else None
    return bool(successes[0]), reward, length


def validate_checkpoint_contract(checkpoint_path: Path) -> None:
    processor_path = checkpoint_path / "processor_config.json"
    try:
        processor = json.loads(processor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Cannot read checkpoint processor contract {processor_path}: {exc}")
    try:
        profile = processor["processor_kwargs"]["modality_configs"][POLICY_PROFILE]
        actual = {
            modality: list(profile[modality]["modality_keys"])
            for modality in POLICY_MODALITY_KEYS
        }
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"Checkpoint does not declare a complete {POLICY_PROFILE} modality contract: {exc}")
    if actual != POLICY_MODALITY_KEYS:
        fail(
            "Checkpoint modality contract is incompatible with this RoboCasa arms-only suite: "
            f"expected {POLICY_MODALITY_KEYS}, received {actual}"
        )


def library_environment(base: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    candidates: list[Path] = []
    declared = env.get("SKYNET_SIM_LIBRARY_PATH", "").strip()
    if declared:
        candidates.extend(Path(item) for item in declared.split(":") if item)
    work_root = env.get("WORK_ROOT", "").strip()
    if work_root:
        candidates.extend(
            [
                Path(work_root) / "envs/system-libs/libglu/usr/lib/x86_64-linux-gnu",
                Path(work_root) / "envs/system-libs/libglu/usr/lib64",
                Path(work_root) / "envs/system-libs/libglu/root/usr/lib/x86_64-linux-gnu",
                Path(work_root) / "envs/system-libs/libglu/root/usr/lib64",
            ]
        )
    existing = [path for path in candidates if path.is_dir()]
    inherited = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join([*(str(path) for path in existing), inherited]).strip(":")
    has_glu = any((path / name).exists() for path in existing for name in ("libGLU.so.1", "libGLU.so.0"))
    if not has_glu and shutil.which("ldconfig"):
        probe = subprocess.run(["ldconfig", "-p"], text=True, capture_output=True, check=False)
        has_glu = "libGLU.so" in probe.stdout
    if not has_glu:
        fail(
            "RoboCasa requires libGLU, but no usable library was found. Install it in the "
            "evaluator runtime or set SKYNET_SIM_LIBRARY_PATH explicitly."
        )
    env.update(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "NVIDIA_DRIVER_CAPABILITIES": env.get("NVIDIA_DRIVER_CAPABILITIES", "all"),
        }
    )
    return env


def discover_videos(stdout: str, search_roots: list[Path], started_ns: int) -> list[Path]:
    found: list[Path] = []
    declared_roots = [Path(match.group(1).strip()) for match in VIDEO_DIRECTORY_PATTERN.finditer(stdout)]
    for root in [*declared_roots, *search_roots]:
        if not root.exists():
            continue
        for path in root.rglob("*.mp4"):
            try:
                if path.stat().st_mtime_ns >= started_ns - 2_000_000_000:
                    found.append(path)
            except OSError:
                continue
    return sorted(set(found), key=lambda path: path.stat().st_mtime_ns)


def load_completed(path: Path, identity: str) -> dict[tuple[str, int, int], dict[str, Any]]:
    completed: dict[tuple[str, int, int], dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"Progress ledger line {line_number} is invalid JSON: {exc}")
        if row.get("identity") != identity:
            fail("Existing progress ledger belongs to a different pinned evaluation")
        if row.get("record_type") != "episode" or row.get("status") != "SUCCEEDED":
            continue
        key = (str(row["task"]), int(row["seed"]), int(row["episode_index"]))
        completed[key] = row["episode"]
    return completed


def terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--source-dir", required=True)
    args = parser.parse_args()

    context_path = Path(args.context).resolve()
    source_dir = Path(args.source_dir).resolve()
    context = load_json(context_path)
    if context.get("schema_version") != SCHEMA:
        fail(f"Expected context schema {SCHEMA}, received {context.get('schema_version')!r}")
    if context.get("environment") != ENVIRONMENT:
        fail(f"This evaluator supports {ENVIRONMENT}, not {context.get('environment')!r}")

    suite = require(context, "suite", "evaluation")
    if not isinstance(suite, dict) or suite.get("name") != SUITE:
        fail(f"This evaluator supports only the exact suite {SUITE}")
    suite_config = require(suite, "config", "evaluation.suite")
    if not isinstance(suite_config, dict):
        fail("evaluation.suite.config must be an object")
    if suite_config.get("task_catalog_complete") is not True:
        fail("GR00T RoboCasa requires a complete immutable task catalog")
    if suite_config.get("task_catalog_sha256") != CATALOG_HASH:
        fail("GR00T RoboCasa task catalog hash does not match the pinned upstream catalog")
    provenance = suite_config.get("task_catalog_provenance", {})
    if provenance.get("revision") != ROOT_REVISION:
        fail("GR00T RoboCasa suite provenance does not match the pinned N1.6 source revision")

    policy = require(context, "policy", "evaluation")
    if not isinstance(policy, dict) or policy.get("adapter") != "groot":
        fail("GR00T RoboCasa evaluation requires a pinned groot policy adapter")
    source = require(policy, "source", "evaluation.policy")
    if not isinstance(source, dict) or source.get("revision") != ROOT_REVISION:
        fail("The selected policy checkout is not the pinned GR00T N1.6 revision for this suite")
    native = policy.get("native_config", {})
    if native.get("embodiment_tag") != EMBODIMENT:
        fail(f"GR00T RoboCasa requires embodiment_tag={EMBODIMENT}")

    checkpoint = require(context, "checkpoint", "evaluation")
    checkpoint_path = Path(require(checkpoint, "path", "evaluation.checkpoint")).resolve()
    checkpoint_sha = str(require(checkpoint, "sha256", "evaluation.checkpoint"))
    if not checkpoint_path.exists():
        fail(f"Pinned checkpoint does not exist: {checkpoint_path}")
    if not source_dir.is_dir():
        fail(f"Pinned source checkout does not exist: {source_dir}")
    validate_checkpoint_contract(checkpoint_path)

    catalog_tasks = suite_config.get("tasks")
    requested_tasks = require(context, "tasks", "evaluation")
    if not isinstance(catalog_tasks, list) or not all(isinstance(item, str) for item in catalog_tasks):
        fail("Pinned suite has an invalid task catalog")
    if not isinstance(requested_tasks, list) or not all(isinstance(item, str) for item in requested_tasks):
        fail("evaluation.tasks must be a non-empty list of exact task IDs")
    unknown = [task for task in requested_tasks if task not in catalog_tasks]
    if unknown:
        fail(f"Requested tasks are absent from the pinned catalog: {unknown}")

    episodes_per_task = int(require(context, "episodes_per_task", "evaluation"))
    seeds = require(context, "seeds", "evaluation")
    if episodes_per_task < 1 or not isinstance(seeds, list) or not seeds:
        fail("episodes_per_task and seeds must define at least one rollout")
    seeds = [int(seed) for seed in seeds]

    result_path = Path(require(context, "result_path", "evaluation")).resolve()
    progress_path = Path(require(context, "progress_path", "evaluation")).resolve()
    video_path = Path(require(context, "video_path", "evaluation")).resolve()
    run_root = result_path.parent
    log_root = run_root / "logs" / f"adapter-attempt-{os.environ.get('SLURM_JOB_ID', 'local')}"
    log_root.mkdir(parents=True, exist_ok=True)
    video_path.mkdir(parents=True, exist_ok=True)

    client_python = source_dir / CLIENT_RELATIVE
    rollout_script = source_dir / ROLLOUT_RELATIVE
    server_script = source_dir / SERVER_RELATIVE
    for required_path in (client_python, rollout_script, server_script):
        if not required_path.exists():
            fail(f"Pinned GR00T evaluator runtime is incomplete: {required_path}")

    identity_document = {
        "run_id": context["run_id"],
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": policy.get("manifest_sha256"),
        "suite_version": suite.get("version"),
        "suite_catalog_sha256": suite_config.get("task_catalog_sha256"),
        "tasks": requested_tasks,
        "episodes_per_task": episodes_per_task,
        "seeds": seeds,
    }
    identity = hashlib.sha256(
        json.dumps(identity_document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    completed = load_completed(progress_path, identity)

    env = library_environment(os.environ)
    port = allocate_port()
    server_stdout_path = log_root / "policy-server.stdout.log"
    server_stderr_path = log_root / "policy-server.stderr.log"
    server_stdout = server_stdout_path.open("w", encoding="utf-8")
    server_stderr = server_stderr_path.open("w", encoding="utf-8")
    server_command = [
        sys.executable,
        str(server_script),
        "--model-path",
        str(checkpoint_path),
        "--embodiment-tag",
        EMBODIMENT,
        "--use-sim-policy-wrapper",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    server = subprocess.Popen(
        server_command,
        cwd=source_dir,
        env=env,
        text=True,
        stdout=server_stdout,
        stderr=server_stderr,
        start_new_session=True,
    )

    try:
        wait_for_server(server, port, server_stderr_path)
        for task in requested_tasks:
            for seed in seeds:
                for episode_index in range(episodes_per_task):
                    key = (task, seed, episode_index)
                    if key in completed:
                        continue
                    episode_seed = seed * 1_000_003 + episode_index
                    slug = f"{safe_name(task)}-seed-{seed}-episode-{episode_index:04d}"
                    stdout_path = log_root / f"{slug}.stdout.log"
                    stderr_path = log_root / f"{slug}.stderr.log"
                    scratch = run_root / "runtime" / slug
                    scratch.mkdir(parents=True, exist_ok=True)
                    child_env = dict(env)
                    child_env.update(
                        {
                            "PYTHONHASHSEED": str(episode_seed),
                            "SKYNET_EVAL_SEED": str(episode_seed),
                            "TMPDIR": str(scratch),
                        }
                    )
                    client_runner = scratch / "skynet-robocasa-client.py"
                    client_runner.write_text(CLIENT_RUNNER_SOURCE, encoding="utf-8")
                    command = [
                        str(client_python),
                        str(client_runner),
                        str(rollout_script),
                        "--n_episodes=1",
                        "--policy_client_host=127.0.0.1",
                        f"--policy_client_port={port}",
                        "--max_episode_steps=720",
                        f"--env_name={task}",
                        "--n_action_steps=8",
                        "--n_envs=1",
                    ]
                    started_ns = time.time_ns()
                    process = subprocess.run(
                        command,
                        cwd=source_dir,
                        env=child_env,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    stdout_path.write_text(process.stdout, encoding="utf-8")
                    stderr_path.write_text(process.stderr, encoding="utf-8")
                    if process.returncode != 0:
                        fail(
                            f"RoboCasa rollout failed for {task}, seed {seed}, episode {episode_index}; "
                            f"see {stderr_path}"
                        )
                    success, reward, length = parse_rollout(process.stdout)
                    discovered = discover_videos(process.stdout, [scratch], started_ns)
                    if not discovered:
                        fail(
                            f"RoboCasa rollout produced no MP4 for {task}, seed {seed}, episode {episode_index}"
                        )
                    copied: list[str] = []
                    for video_index, source_video in enumerate(discovered):
                        suffix = "" if video_index == 0 else f"-{video_index:02d}"
                        destination = video_path / f"{slug}{suffix}.mp4"
                        shutil.copy2(source_video, destination)
                        copied.append(str(destination))
                    episode = {
                        "task": task,
                        "seed": seed,
                        "episode_index": episode_index,
                        "success": success,
                        "reward": reward,
                        "episode_length": length,
                        "status": "SUCCEEDED",
                        "metrics": {"effective_seed": float(episode_seed)},
                        "video_path": copied[0],
                        "failure_reason": None,
                    }
                    append_jsonl(
                        progress_path,
                        {
                            "record_type": "episode",
                            "identity": identity,
                            "status": "SUCCEEDED",
                            "task": task,
                            "seed": seed,
                            "episode_index": episode_index,
                            "episode": episode,
                            "videos": copied,
                        },
                    )
                    completed[key] = episode
    finally:
        terminate(server)
        server_stdout.close()
        server_stderr.close()

    expected = len(requested_tasks) * len(seeds) * episodes_per_task
    episodes = [completed[(task, seed, index)] for task in requested_tasks for seed in seeds for index in range(episodes_per_task)]
    if len(episodes) != expected:
        fail(f"Evaluation ledger is incomplete: expected {expected} episodes, found {len(episodes)}")

    aggregate: list[dict[str, Any]] = []
    for task in requested_tasks:
        values = [1.0 if row["success"] else 0.0 for row in episodes if row["task"] == task]
        aggregate.append(
            {
                "metric": "success_rate",
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "sample_count": len(values),
                "task": task,
                "unit": "fraction",
            }
        )
    all_values = [1.0 if row["success"] else 0.0 for row in episodes]
    aggregate.insert(
        0,
        {
            "metric": "success_rate",
            "mean": statistics.fmean(all_values),
            "std": statistics.pstdev(all_values) if len(all_values) > 1 else 0.0,
            "sample_count": len(all_values),
            "task": None,
            "unit": "fraction",
        },
    )
    artifacts = sorted(
        {
            str(server_stdout_path),
            str(server_stderr_path),
            *(str(path) for path in video_path.glob("*.mp4")),
            *(str(path) for path in log_root.glob("*.log")),
        }
    )
    result = {
        "schema_version": 1,
        "run_id": context["run_id"],
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "evaluator": {"adapter": "mujoco", "version": "1"},
        "environment": {"suite": SUITE, "version": suite["version"]},
        "aggregate": aggregate,
        "episodes": episodes,
        "raw_metrics_path": str(progress_path),
        "artifacts": artifacts,
    }
    write_json(result_path, result)
    print(json.dumps({"result_path": str(result_path), "episodes": expected, "success_rate": aggregate[0]["mean"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"GR00T RoboCasa evaluation failed: {exc}", file=sys.stderr)
        raise
'''
