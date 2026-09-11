from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from typing import Any


_SUITE_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}

_SUITE_TASK_SHA256 = {
    "libero_spatial": "a4d00263fa730821095e894f4824ea28d6b2fc5b5f3b0b892123b835e74aea2c",
    "libero_object": "6bad932e9362f7d880e9139a0fac9384b64722211754ead26ff4aca8391574b0",
    "libero_goal": "583173aab3839ebb148bb9931743c7fedebc26ea6f2ce040f8abf714e4f654e4",
    "libero_10": "b77efb0074cb2c65907a1ca705115ffc3cce2177947ffed5056984c9093ea05d",
    "libero_90": "aeefdcac929d4d56f1476683a95a0427aa18c654390a16887a223da9c4f645ad",
}

_LEDGER_SCHEMA = "skynet.openpi-libero-progress/v2"
_RESUME_GRANULARITY = "task_seed_group"

# This adapter-owned proxy intentionally delegates the rollout to the pinned
# upstream OpenPI client. It limits task selection and composes replay frames;
# the upstream loop still owns policy inference and simulator stepping.
_FILTERED_CLIENT_SOURCE = r'''from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import logging
from pathlib import Path
import sys


EXPECTED_ARGS = {
    "host",
    "port",
    "resize_size",
    "replan_steps",
    "task_suite_name",
    "num_steps_wait",
    "num_trials_per_task",
    "video_out_path",
    "seed",
}


def include_policy_views(module):
    """Replace only the upstream replay frame; policy observations stay intact."""
    from evaluation_video import compose_camera_views

    source = inspect.getsource(module.eval_libero)
    tree = ast.parse(source)
    replacements = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "replay_images" and node.func.attr == "append"
                and len(node.args) == 1 and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "img"):
            node.args[0] = ast.parse(
                '_skynet_camera_views({"agentview_image": img, '
                '"robot0_eye_in_hand_image": wrist_img})', mode="eval"
            ).body
            replacements += 1
    if replacements != 1:
        raise RuntimeError("Pinned OpenPI replay camera interface changed")
    module._skynet_camera_views = compose_camera_views
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(module.eval_libero), "exec"), module.__dict__)


def load_upstream(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Pinned OpenPI LIBERO client is missing: {path}")
    spec = importlib.util.spec_from_file_location("_skynet_pinned_openpi_libero", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pinned OpenPI LIBERO client: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not dataclasses.is_dataclass(getattr(module, "Args", None)):
        raise RuntimeError("Pinned OpenPI LIBERO client Args interface is missing")
    fields = {item.name for item in dataclasses.fields(module.Args)}
    if fields != EXPECTED_ARGS:
        raise RuntimeError(
            "Pinned OpenPI LIBERO client Args interface changed: "
            f"expected {sorted(EXPECTED_ARGS)}, found {sorted(fields)}"
        )
    if not callable(getattr(module, "eval_libero", None)):
        raise RuntimeError("Pinned OpenPI LIBERO client eval_libero interface is missing")
    benchmark_module = getattr(module, "benchmark", None)
    if benchmark_module is None or not callable(getattr(benchmark_module, "get_benchmark_dict", None)):
        raise RuntimeError("Pinned OpenPI LIBERO benchmark registry interface is missing")
    return module


def inspect_suite(module, suite_name: str, expected_sha256: str):
    registry = module.benchmark.get_benchmark_dict()
    if not isinstance(registry, dict) or suite_name not in registry or not callable(registry[suite_name]):
        raise RuntimeError(f"Pinned LIBERO registry does not contain suite {suite_name!r}")
    suite = registry[suite_name]()
    for attribute in ("n_tasks", "get_task", "get_task_init_states"):
        if not hasattr(suite, attribute):
            raise RuntimeError(f"Pinned LIBERO suite interface is missing {attribute!r}")
    if not isinstance(suite.n_tasks, int) or suite.n_tasks < 1:
        raise RuntimeError(f"Pinned LIBERO suite has invalid n_tasks: {suite.n_tasks!r}")
    task_ids = []
    for index in range(suite.n_tasks):
        task = suite.get_task(index)
        task_id = getattr(task, "name", None)
        if not isinstance(task_id, str) or not task_id or task_id in task_ids:
            raise RuntimeError(f"Pinned LIBERO suite returned invalid task name at index {index}: {task_id!r}")
        task_ids.append(task_id)
    digest = hashlib.sha256(
        json.dumps(task_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(
            f"Pinned LIBERO task catalog mismatch for {suite_name}: "
            f"expected {expected_sha256}, found {digest}"
        )
    return registry, suite, task_ids


class FilteredSuite:
    def __init__(self, upstream, task_ids, selected_ids):
        index_by_id = {task_id: index for index, task_id in enumerate(task_ids)}
        unknown = [task_id for task_id in selected_ids if task_id not in index_by_id]
        if unknown:
            raise ValueError(f"Unknown pinned LIBERO task IDs: {unknown}")
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("Selected LIBERO task IDs must be unique")
        self._upstream = upstream
        self._task_ids = list(selected_ids)
        self._indices = [index_by_id[task_id] for task_id in selected_ids]
        self.n_tasks = len(self._indices)

    def _index(self, index):
        if not isinstance(index, int) or index < 0 or index >= self.n_tasks:
            raise IndexError(index)
        return self._indices[index]

    def get_task(self, index):
        print(f"SKYNET_TASK_ID: {self._task_ids[index]}", flush=True)
        return self._upstream.get_task(self._index(index))

    def get_task_init_states(self, index):
        return self._upstream.get_task_init_states(self._index(index))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-main", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--expected-catalog-sha256", required=True)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--video-path", default="data/libero/videos")
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()

    module = load_upstream(Path(arguments.source_main).resolve())
    registry, upstream_suite, catalog = inspect_suite(
        module, arguments.suite, arguments.expected_catalog_sha256
    )
    if arguments.list_tasks:
        print("SKYNET_TASK_CATALOG_JSON=" + json.dumps(catalog, separators=(",", ":")))
        return 0

    include_policy_views(module)
    selected = list(arguments.task_id) if arguments.task_id else list(catalog)
    filtered = FilteredSuite(upstream_suite, catalog, selected)
    original_get_benchmark_dict = module.benchmark.get_benchmark_dict

    def filtered_registry():
        current = original_get_benchmark_dict()
        if not isinstance(current, dict) or arguments.suite not in current:
            raise RuntimeError("Pinned LIBERO benchmark registry changed during evaluation")
        result = dict(current)
        result[arguments.suite] = lambda: filtered
        return result

    module.benchmark.get_benchmark_dict = filtered_registry
    if len(selected) != 1:
        raise RuntimeError("The pinned OpenPI LIBERO runner requires exactly one selected task")

    video_state = {"path": None, "count": 0}
    original_mimwrite = module.imageio.mimwrite

    def write_unique_video(uri, *args, **kwargs):
        path = Path(uri)
        index = int(video_state["count"])
        unique_path = path.with_name(f"{path.stem}-episode-{index:06d}{path.suffix}")
        result = original_mimwrite(unique_path, *args, **kwargs)
        video_state["path"] = str(unique_path.resolve())
        video_state["count"] = index + 1
        return result

    class EpisodeResultHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.count = 0

        def emit(self, record):
            match = __import__("re").fullmatch(r"Success:\s*(True|False)", record.getMessage())
            if not match:
                return
            index = self.count
            if index >= arguments.trials:
                raise RuntimeError("Pinned OpenPI emitted more episode results than requested")
            video_path = video_state.get("path")
            if not video_path:
                raise RuntimeError("Pinned OpenPI emitted an episode result without a video")
            payload = {
                "task": selected[0],
                "seed": arguments.seed,
                "episode_index": index,
                "success": match.group(1) == "True",
                "reward": None,
                "episode_length": None,
                "status": "SUCCEEDED",
                "metrics": {},
                "video_path": video_path,
                "failure_reason": None,
            }
            print("SKYNET_EPISODE_JSON=" + json.dumps(payload, separators=(",", ":")), flush=True)
            self.count += 1

    logging.basicConfig(level=logging.INFO, force=True)
    episode_handler = EpisodeResultHandler()
    logging.getLogger().addHandler(episode_handler)
    module.imageio.mimwrite = write_unique_video
    upstream_args = module.Args(
        host=arguments.host,
        port=arguments.port,
        task_suite_name=arguments.suite,
        num_trials_per_task=arguments.trials,
        video_out_path=arguments.video_path,
        seed=arguments.seed,
    )
    try:
        module.eval_libero(upstream_args)
    finally:
        module.imageio.mimwrite = original_mimwrite
        logging.getLogger().removeHandler(episode_handler)
    if episode_handler.count != arguments.trials:
        raise RuntimeError(
            f"Pinned OpenPI emitted {episode_handler.count} episode results; "
            f"expected {arguments.trials}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _allocate_attempt_directory(root: Path) -> tuple[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1_000_000):
        attempt_id = f"attempt-{index:06d}"
        path = root / attempt_id
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return attempt_id, path
    raise RuntimeError(f"OpenPI LIBERO exhausted attempt IDs under {root}")


def _validated_episode(
    value: Any,
    *,
    task: str,
    seed: int,
    episodes_per_task: int,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    record = dict(value)
    if record.get("task") != task or type(record.get("seed")) is not int:
        raise RuntimeError(f"{label} task/seed does not match its ledger group")
    if record["seed"] != seed:
        raise RuntimeError(f"{label} task/seed does not match its ledger group")
    episode_index = record.get("episode_index")
    if (
        type(episode_index) is not int
        or episode_index < 0
        or episode_index >= episodes_per_task
    ):
        raise RuntimeError(f"{label} has an invalid episode_index")
    if type(record.get("success")) is not bool or record.get("status") != "SUCCEEDED":
        raise RuntimeError(f"{label} is not a completed rollout episode")
    if not isinstance(record.get("metrics"), dict):
        raise RuntimeError(f"{label} metrics must be a JSON object")
    return record


def _load_progress_ledger(
    path: Path,
    *,
    identity: dict[str, Any],
    catalog_tasks: list[str],
    groups: list[tuple[str, int]],
    episodes_per_task: int,
) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], str]:
    identity_sha256 = _canonical_sha256(identity)
    catalog_sha256 = _canonical_sha256(catalog_tasks)
    total = len(groups) * episodes_per_task
    if not path.exists():
        _append_jsonl(
            path,
            {
                "schema_version": 1,
                "ledger_schema": _LEDGER_SCHEMA,
                "kind": "context",
                "identity": identity,
                "identity_sha256": identity_sha256,
                "catalog_tasks": catalog_tasks,
                "catalog_sha256": catalog_sha256,
                "resume_granularity": _RESUME_GRANULARITY,
                "completed": 0,
                "total": total,
                "recorded_at": _timestamp(),
            },
        )
        return {}, identity_sha256

    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"Cannot read OpenPI LIBERO progress ledger: {path}") from error
    if not payload or not payload.endswith("\n"):
        raise RuntimeError(f"OpenPI LIBERO progress ledger is empty or truncated: {path}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            record = json.loads(line)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"OpenPI LIBERO progress ledger has invalid JSON at line {line_number}: {path}"
            ) from error
        if not isinstance(record, dict):
            raise RuntimeError(
                f"OpenPI LIBERO progress ledger line {line_number} is not an object: {path}"
            )
        records.append(record)

    header = records[0]
    if (
        header.get("schema_version") != 1
        or header.get("ledger_schema") != _LEDGER_SCHEMA
        or header.get("kind") != "context"
        or header.get("identity") != identity
        or header.get("identity_sha256") != identity_sha256
        or header.get("catalog_tasks") != catalog_tasks
        or header.get("catalog_sha256") != catalog_sha256
        or header.get("resume_granularity") != _RESUME_GRANULARITY
        or header.get("completed") != 0
        or header.get("total") != total
    ):
        raise RuntimeError(
            "OpenPI LIBERO progress ledger context/catalog does not match this evaluation"
        )

    expected_index = {group: index for index, group in enumerate(groups)}
    if len(expected_index) != len(groups):
        raise RuntimeError("OpenPI LIBERO task+seed groups must be unique")
    completed_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    started: set[tuple[str, tuple[str, int]]] = set()
    observed: dict[tuple[str, tuple[str, int]], list[dict[str, Any]]] = {}
    for line_number, record in enumerate(records[1:], start=2):
        if (
            record.get("schema_version") != 1
            or record.get("ledger_schema") != _LEDGER_SCHEMA
            or record.get("identity_sha256") != identity_sha256
            or record.get("resume_granularity") != _RESUME_GRANULARITY
            or record.get("total") != total
        ):
            raise RuntimeError(
                f"OpenPI LIBERO progress ledger metadata mismatch at line {line_number}"
            )
        kind = record.get("kind")
        if kind not in {"group_started", "episode_observed", "group_completed"}:
            raise RuntimeError(
                f"OpenPI LIBERO progress ledger has unknown record kind at line {line_number}"
            )
        attempt_id = record.get("attempt_id")
        if not isinstance(attempt_id, str) or not re.fullmatch(r"attempt-[0-9]{6}", attempt_id):
            raise RuntimeError(
                f"OpenPI LIBERO progress ledger has invalid attempt ID at line {line_number}"
            )
        group_value = record.get("group")
        if not isinstance(group_value, dict):
            raise RuntimeError(
                f"OpenPI LIBERO progress ledger has invalid group at line {line_number}"
            )
        task = group_value.get("task")
        seed = group_value.get("seed")
        index = group_value.get("index")
        if not isinstance(task, str) or type(seed) is not int or type(index) is not int:
            raise RuntimeError(
                f"OpenPI LIBERO progress ledger has invalid group identity at line {line_number}"
            )
        group = (task, seed)
        if expected_index.get(group) != index:
            raise RuntimeError(
                f"OpenPI LIBERO progress ledger group does not match the canonical plan at line {line_number}"
            )
        completed_value = record.get("completed")
        if (
            type(completed_value) is not int
            or completed_value < 0
            or completed_value > total
        ):
            raise RuntimeError(
                f"OpenPI LIBERO progress ledger has invalid completion count at line {line_number}"
            )
        attempt_group = (attempt_id, group)
        if kind == "group_started":
            if group in completed_groups or attempt_group in started:
                raise RuntimeError(
                    f"OpenPI LIBERO progress ledger has an invalid group start at line {line_number}"
                )
            if completed_value != len(completed_groups) * episodes_per_task:
                raise RuntimeError(
                    f"OpenPI LIBERO progress ledger group start count is inconsistent at line {line_number}"
                )
            started.add(attempt_group)
            observed[attempt_group] = []
            continue
        if attempt_group not in started or group in completed_groups:
            raise RuntimeError(
                f"OpenPI LIBERO progress ledger record has no valid group start at line {line_number}"
            )
        if kind == "episode_observed":
            episode = _validated_episode(
                record.get("episode"),
                task=task,
                seed=seed,
                episodes_per_task=episodes_per_task,
                label=f"ledger line {line_number} episode",
            )
            prior = observed[attempt_group]
            if episode["episode_index"] != len(prior):
                raise RuntimeError(
                    f"OpenPI LIBERO progress ledger episode order is invalid at line {line_number}"
                )
            prior.append(episode)
            if completed_value != len(completed_groups) * episodes_per_task + len(prior):
                raise RuntimeError(
                    f"OpenPI LIBERO progress ledger episode count is inconsistent at line {line_number}"
                )
            continue

        episodes_value = record.get("episodes")
        if not isinstance(episodes_value, list) or len(episodes_value) != episodes_per_task:
            raise RuntimeError(
                f"OpenPI LIBERO progress ledger group commit is incomplete at line {line_number}"
            )
        episodes = [
            _validated_episode(
                episode,
                task=task,
                seed=seed,
                episodes_per_task=episodes_per_task,
                label=f"ledger line {line_number} committed episode",
            )
            for episode in episodes_value
        ]
        if [episode["episode_index"] for episode in episodes] != list(
            range(episodes_per_task)
        ):
            raise RuntimeError(
                f"OpenPI LIBERO progress ledger group commit order is invalid at line {line_number}"
            )
        if episodes != observed[attempt_group]:
            raise RuntimeError(
                f"OpenPI LIBERO progress ledger group commit does not match observations at line {line_number}"
            )
        if record.get("group_sha256") != _canonical_sha256(episodes):
            raise RuntimeError(
                f"OpenPI LIBERO progress ledger group digest mismatch at line {line_number}"
            )
        next_group = next(
            (candidate for candidate in groups if candidate not in completed_groups),
            None,
        )
        if group != next_group:
            raise RuntimeError(
                f"OpenPI LIBERO progress ledger commits groups out of order at line {line_number}"
            )
        if completed_value != (len(completed_groups) + 1) * episodes_per_task:
            raise RuntimeError(
                f"OpenPI LIBERO progress ledger group count is inconsistent at line {line_number}"
            )
        completed_groups[group] = episodes
    return completed_groups, identity_sha256


def _run_logged(argv: list[str], *, cwd: Path, environment: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(argv) + "\n")
        stream.flush()
        process = subprocess.run(
            argv,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(f"command failed with exit code {process.returncode}: {argv[0]}")


def _configure_libero_environment(
    source_dir: Path,
    result_path: Path,
    environment: dict[str, str],
) -> None:
    """Create a non-interactive, evaluation-pinned LIBERO path configuration."""

    benchmark_root = source_dir / "third_party/libero/libero/libero"
    paths = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": result_path.parents[2] / "datasets/libero",
        "assets": benchmark_root / "assets",
    }
    missing = [str(path) for key, path in paths.items() if key != "datasets" and not path.exists()]
    if missing:
        raise RuntimeError(
            "Pinned LIBERO benchmark paths are missing; initialize repository submodules: "
            + ", ".join(missing)
        )
    paths["datasets"].mkdir(parents=True, exist_ok=True)
    config_root = result_path.parent / "runtime/libero"
    config_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        config_root / "config.yaml",
        {key: str(path.resolve()) for key, path in paths.items()},
    )
    environment["LIBERO_CONFIG_PATH"] = str(config_root.resolve())


def _client_environment(
    source_dir: Path, run_dir: Path, environment: dict[str, str], setup_log: Path
) -> Path:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("OpenPI LIBERO evaluation requires uv on PATH")
    requirements = [
        source_dir / "examples/libero/requirements.txt",
        source_dir / "third_party/libero/requirements.txt",
    ]
    missing = [str(path) for path in requirements if not path.is_file()]
    if missing:
        raise RuntimeError(
            "OpenPI LIBERO requirements are missing; initialize repository submodules: "
            + ", ".join(missing)
        )
    digest = hashlib.sha256()
    for path in requirements:
        digest.update(path.read_bytes())
    digest.update(b"openpi-libero-client-v1")
    expected = digest.hexdigest()
    venv = run_dir / "adapter-support/libero-client-venv"
    python = venv / "bin/python"
    marker = venv / ".skynet-ready.json"
    if marker.is_file() and python.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")).get("sha256") == expected:
                return python
        except (OSError, ValueError, TypeError):
            pass

    if venv.exists():
        shutil.rmtree(venv)
    _run_logged(
        [uv, "venv", "--python", "3.8", str(venv)],
        cwd=source_dir,
        environment=environment,
        log=setup_log,
    )
    _run_logged(
        [
            uv,
            "pip",
            "sync",
            "--python",
            str(python),
            str(requirements[0]),
            str(requirements[1]),
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu113",
            "--index-strategy",
            "unsafe-best-match",
        ],
        cwd=source_dir,
        environment=environment,
        log=setup_log,
    )
    _run_logged(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "-e",
            str(source_dir / "packages/openpi-client"),
            "-e",
            str(source_dir / "third_party/libero"),
        ],
        cwd=source_dir,
        environment=environment,
        log=setup_log,
    )
    _atomic_json(marker, {"sha256": expected, "python": str(python)})
    return python


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_server(process: subprocess.Popen[Any], port: int, timeout: float = 600.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"OpenPI policy server exited with code {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"OpenPI policy server did not listen on port {port} within {timeout:g}s")


def _discover_suite_tasks(
    *,
    client_python: Path,
    client_script: Path,
    source_dir: Path,
    environment: dict[str, str],
    suite_name: str,
    expected_catalog_sha256: str,
    log_path: Path,
) -> list[str]:
    argv = [
        str(client_python),
        str(client_script),
        "--source-main",
        str(source_dir / "examples/libero/main.py"),
        "--suite",
        suite_name,
        "--expected-catalog-sha256",
        expected_catalog_sha256,
        "--list-tasks",
    ]
    process = subprocess.run(
        argv,
        cwd=source_dir,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("$ " + " ".join(argv) + "\n" + process.stdout, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(
            f"OpenPI LIBERO catalog inspection exited with code {process.returncode}; see {log_path}"
        )
    marker = "SKYNET_TASK_CATALOG_JSON="
    lines = [line for line in process.stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError(f"OpenPI LIBERO catalog inspection returned no unique catalog; see {log_path}")
    try:
        tasks = json.loads(lines[0][len(marker) :])
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"OpenPI LIBERO catalog inspection returned invalid JSON; see {log_path}") from error
    if not isinstance(tasks, list) or not all(isinstance(task, str) and task for task in tasks):
        raise RuntimeError(f"OpenPI LIBERO catalog inspection returned invalid task IDs; see {log_path}")
    return tasks


def _run_client(
    *,
    client_python: Path,
    client_script: Path,
    source_dir: Path,
    environment: dict[str, str],
    port: int,
    suite_name: str,
    expected_catalog_sha256: str,
    selected_task: str,
    seed: int,
    episodes_per_task: int,
    video_dir: Path,
    log_path: Path,
    progress_path: Path,
    identity_sha256: str,
    attempt_id: str,
    group_index: int,
    completed: int,
    total: int,
) -> tuple[list[dict[str, Any]], int]:
    argv = [
        str(client_python),
        str(client_script),
        "--source-main",
        str(source_dir / "examples/libero/main.py"),
        "--suite",
        suite_name,
        "--expected-catalog-sha256",
        expected_catalog_sha256,
        "--task-id",
        selected_task,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--trials",
        str(episodes_per_task),
        "--video-path",
        str(video_dir),
        "--seed",
        str(seed),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    current_task: str | None = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(argv) + "\n")
        log.flush()
        process = subprocess.Popen(
            argv,
            cwd=source_dir,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            task_match = re.search(r"(?:^|\s)SKYNET_TASK_ID:\s*(\S+)\s*$", line)
            if task_match:
                current_task = task_match.group(1).strip()
                if current_task != selected_task:
                    process.terminate()
                    raise RuntimeError(
                        f"OpenPI LIBERO client emitted task {current_task!r}; expected {selected_task!r}"
                    )
            episode_marker = "SKYNET_EPISODE_JSON="
            marker_index = line.find(episode_marker)
            if marker_index >= 0:
                try:
                    payload = json.loads(line[marker_index + len(episode_marker) :].strip())
                except (TypeError, ValueError) as error:
                    process.terminate()
                    raise RuntimeError("OpenPI LIBERO client emitted invalid episode JSON") from error
                episode_index = len(episodes)
                if episode_index >= episodes_per_task:
                    process.terminate()
                    raise RuntimeError(
                        f"OpenPI LIBERO client emitted more than {episodes_per_task} episodes "
                        f"for {selected_task!r}"
                    )
                if not isinstance(payload, dict):
                    process.terminate()
                    raise RuntimeError("OpenPI LIBERO episode marker must contain an object")
                if payload.get("task") != selected_task or payload.get("seed") != seed:
                    process.terminate()
                    raise RuntimeError("OpenPI LIBERO episode marker task or seed does not match the request")
                if payload.get("episode_index") != episode_index or not isinstance(payload.get("success"), bool):
                    process.terminate()
                    raise RuntimeError("OpenPI LIBERO episode marker index or success value is invalid")
                video_path = Path(str(payload.get("video_path") or "")).resolve()
                if video_dir.resolve() not in video_path.parents or not video_path.is_file():
                    process.terminate()
                    raise RuntimeError("OpenPI LIBERO episode marker references an invalid video path")
                record = {
                    "task": selected_task,
                    "seed": seed,
                    "episode_index": episode_index,
                    "success": payload["success"],
                    "reward": payload.get("reward"),
                    "episode_length": payload.get("episode_length"),
                    "status": "SUCCEEDED",
                    "metrics": dict(payload.get("metrics") or {}),
                    "video_path": str(video_path),
                    "failure_reason": payload.get("failure_reason"),
                }
                episodes.append(record)
                completed += 1
                _append_jsonl(
                    progress_path,
                    {
                        "schema_version": 1,
                        "ledger_schema": _LEDGER_SCHEMA,
                        "kind": "episode_observed",
                        "identity_sha256": identity_sha256,
                        "resume_granularity": _RESUME_GRANULARITY,
                        "attempt_id": attempt_id,
                        "group": {
                            "index": group_index,
                            "task": selected_task,
                            "seed": seed,
                        },
                        "completed": completed,
                        "total": total,
                        "episode": record,
                        "recorded_at": _timestamp(),
                    },
                )
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"OpenPI LIBERO client exited with code {return_code}; see {log_path}")
    return episodes, completed


def _metrics(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    groups: list[tuple[str | None, list[dict[str, Any]]]] = [(None, episodes)]
    for task in sorted({str(item["task"]) for item in episodes}):
        groups.append((task, [item for item in episodes if item["task"] == task]))
    for task, records in groups:
        values = [float(bool(item["success"])) for item in records if item["success"] is not None]
        if not values:
            continue
        metrics.append(
            {
                "metric": "success_rate",
                "unit": "fraction",
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "sample_count": len(values),
                "task": task,
            }
        )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    arguments = parser.parse_args()
    context_path = Path(arguments.context).resolve()
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if context.get("schema_version") != "skynet.evaluation-context/v1":
        raise ValueError("OpenPI evaluation context schema is unsupported")
    if os.environ.get("SKYNET_EVAL_RESUME_GRANULARITY") != _RESUME_GRANULARITY:
        raise RuntimeError(
            "OpenPI evaluation resume granularity is missing or does not match the adapter"
        )

    source_dir = Path(os.environ["SKYNET_SOURCE_DIR"]).resolve()
    run_dir = Path(os.environ["SKYNET_RUN_DIR"]).resolve()
    checkpoint = dict(context["checkpoint"])
    suite = dict(context["suite"])
    policy = dict(context["policy"])
    native_config = dict(policy.get("native_config") or {})
    policy_config = str(native_config.get("config_name") or "").strip()
    if not policy_config:
        raise ValueError("OpenPI evaluation requires policy.native_config.config_name")
    checkpoint_path = Path(str(checkpoint["path"])).resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"OpenPI checkpoint directory does not exist: {checkpoint_path}")

    suite_name = str(suite["name"])
    if suite_name not in _SUITE_TASK_COUNTS:
        raise ValueError(f"OpenPI LIBERO bridge does not support suite: {suite_name}")
    requested_tasks = [str(item) for item in context.get("tasks") or []]
    if len(set(requested_tasks)) != len(requested_tasks):
        raise ValueError("OpenPI LIBERO selected task IDs must be unique")
    suite_task_count = _SUITE_TASK_COUNTS[suite_name]
    expected_catalog_sha256 = _SUITE_TASK_SHA256[suite_name]
    seeds = [int(seed) for seed in context.get("seeds") or []]
    if not seeds:
        raise ValueError("OpenPI LIBERO evaluation requires at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("OpenPI LIBERO evaluation seeds must be unique")
    episodes_per_task = int(context["episodes_per_task"])
    if episodes_per_task < 1 or episodes_per_task > 50:
        raise ValueError("OpenPI LIBERO episodes_per_task must be between 1 and 50")

    result_path = Path(
        os.environ.get("SKYNET_EVAL_RESULT_PATH") or str(context["result_path"])
    ).resolve()
    progress_path = Path(
        os.environ.get("SKYNET_EVAL_PROGRESS_PATH") or str(context["progress_path"])
    ).resolve()
    video_base = Path(
        str(context.get("video_path") or result_path.parent / "videos")
    ).resolve()
    attempt_id, attempt_root = _allocate_attempt_directory(result_path.parent / "attempts")
    logs = result_path.parent / "logs" / attempt_id
    video_root = video_base / attempt_id
    logs.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        attempt_root / "attempt.json",
        {"attempt_id": attempt_id, "status": "INITIALIZING", "started_at": _timestamp()},
    )

    environment = dict(os.environ)
    environment["MUJOCO_GL"] = environment.get("MUJOCO_GL", "egl")
    environment["PYOPENGL_PLATFORM"] = environment.get("PYOPENGL_PLATFORM", "egl")
    environment["MUJOCO_EGL_DEVICE_ID"] = environment.get("MUJOCO_EGL_DEVICE_ID", "0")
    python_path = str(source_dir / "third_party/libero")
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path
    _configure_libero_environment(source_dir, result_path, environment)

    client_python = _client_environment(
        source_dir, run_dir, environment, logs / "client-setup.log"
    )
    client_script = run_dir / "adapter-support/openpi-libero-filtered-client.py"
    client_script.parent.mkdir(parents=True, exist_ok=True)
    client_script.write_text(_FILTERED_CLIENT_SOURCE, encoding="utf-8")
    catalog_tasks = _discover_suite_tasks(
        client_python=client_python,
        client_script=client_script,
        source_dir=source_dir,
        environment=environment,
        suite_name=suite_name,
        expected_catalog_sha256=expected_catalog_sha256,
        log_path=logs / "client-catalog.log",
    )
    if len(catalog_tasks) != suite_task_count:
        raise RuntimeError(
            f"Pinned LIBERO suite {suite_name} returned {len(catalog_tasks)} tasks; "
            f"expected {suite_task_count}"
        )
    declared_catalog_sha256 = str(suite.get("task_catalog_sha256") or "")
    if declared_catalog_sha256 != expected_catalog_sha256:
        raise RuntimeError(
            f"Evaluation context catalog digest mismatch for {suite_name}: "
            f"expected {expected_catalog_sha256}, found {declared_catalog_sha256 or '<missing>'}"
        )
    if suite.get("task_catalog_complete") is not True or suite.get("tasks") != catalog_tasks:
        raise RuntimeError(
            f"Evaluation context task catalog does not match pinned LIBERO suite {suite_name}"
        )
    unknown_tasks = [task for task in requested_tasks if task not in catalog_tasks]
    if unknown_tasks:
        raise ValueError(f"Unknown pinned LIBERO task IDs for {suite_name}: {unknown_tasks}")
    requested_set = set(requested_tasks)
    selected_tasks = (
        [task for task in catalog_tasks if task in requested_set]
        if requested_tasks
        else catalog_tasks
    )
    checkpoint_sha256 = str(checkpoint.get("sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
        raise ValueError("OpenPI evaluation requires a valid checkpoint SHA-256")
    groups = [(task, seed) for seed in seeds for task in selected_tasks]
    total = len(groups) * episodes_per_task
    identity = {
        "schema_version": 1,
        "canonical_context_sha256": _canonical_sha256(context),
        "run_id": str(context["run_id"]),
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha256},
        "suite": {
            "name": suite_name,
            "version": str(suite["version"]),
            "catalog_sha256": expected_catalog_sha256,
        },
        "selected_tasks": selected_tasks,
        "seeds": seeds,
        "episodes_per_task": episodes_per_task,
        "policy": {
            "adapter": str(policy.get("adapter") or ""),
            "adapter_version": policy.get("adapter_version"),
            "manifest_sha256": str(policy.get("manifest_sha256") or ""),
            "config_name": policy_config,
            "source": dict(policy.get("source") or {}),
        },
        "result_path": str(result_path),
        "progress_path": str(progress_path),
        "resume_granularity": _RESUME_GRANULARITY,
    }
    completed_groups, identity_sha256 = _load_progress_ledger(
        progress_path,
        identity=identity,
        catalog_tasks=catalog_tasks,
        groups=groups,
        episodes_per_task=episodes_per_task,
    )
    _atomic_json(
        attempt_root / "attempt.json",
        {
            "attempt_id": attempt_id,
            "status": "RUNNING",
            "started_at": _timestamp(),
            "identity_sha256": identity_sha256,
            "resume_granularity": _RESUME_GRANULARITY,
            "already_completed_groups": len(completed_groups),
            "total_groups": len(groups),
        },
    )
    completed = len(completed_groups) * episodes_per_task
    pending_groups = [group for group in groups if group not in completed_groups]
    if pending_groups:
        port = _free_port()
        server_stdout = (logs / "policy-server.stdout.log").open("w", encoding="utf-8")
        server_stderr = (logs / "policy-server.stderr.log").open("w", encoding="utf-8")
        server = subprocess.Popen(
            [
                sys.executable,
                "scripts/serve_policy.py",
                "--env",
                "LIBERO",
                "--port",
                str(port),
                "policy:checkpoint",
                "--policy.config",
                policy_config,
                "--policy.dir",
                str(checkpoint_path),
            ],
            cwd=source_dir,
            env=environment,
            stdout=server_stdout,
            stderr=server_stderr,
            text=True,
        )

        def stop_server(*_: Any) -> None:
            if server.poll() is None:
                server.terminate()

        previous_term = signal.signal(signal.SIGTERM, stop_server)
        previous_int = signal.signal(signal.SIGINT, stop_server)
        try:
            _wait_for_server(server, port)
            for group_index, (task, seed) in enumerate(groups):
                if (task, seed) in completed_groups:
                    continue
                task_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", task).strip("-")
                task_slug = f"{task_slug}-{hashlib.sha256(task.encode('utf-8')).hexdigest()[:8]}"
                group_value = {"index": group_index, "task": task, "seed": seed}
                _append_jsonl(
                    progress_path,
                    {
                        "schema_version": 1,
                        "ledger_schema": _LEDGER_SCHEMA,
                        "kind": "group_started",
                        "identity_sha256": identity_sha256,
                        "resume_granularity": _RESUME_GRANULARITY,
                        "attempt_id": attempt_id,
                        "group": group_value,
                        "completed": completed,
                        "total": total,
                        "recorded_at": _timestamp(),
                    },
                )
                records, observed_completed = _run_client(
                    client_python=client_python,
                    client_script=client_script,
                    source_dir=source_dir,
                    environment=environment,
                    port=port,
                    suite_name=suite_name,
                    expected_catalog_sha256=expected_catalog_sha256,
                    selected_task=task,
                    seed=seed,
                    episodes_per_task=episodes_per_task,
                    video_dir=video_root / f"seed-{seed}" / task_slug,
                    log_path=logs / f"client-seed-{seed}-{task_slug}.log",
                    progress_path=progress_path,
                    identity_sha256=identity_sha256,
                    attempt_id=attempt_id,
                    group_index=group_index,
                    completed=completed,
                    total=total,
                )
                if len(records) != episodes_per_task or observed_completed != completed + episodes_per_task:
                    raise RuntimeError(
                        f"OpenPI LIBERO produced {len(records)} parsed episodes for "
                        f"{task!r}/seed-{seed}; expected {episodes_per_task}"
                    )
                completed = observed_completed
                _append_jsonl(
                    progress_path,
                    {
                        "schema_version": 1,
                        "ledger_schema": _LEDGER_SCHEMA,
                        "kind": "group_completed",
                        "identity_sha256": identity_sha256,
                        "resume_granularity": _RESUME_GRANULARITY,
                        "attempt_id": attempt_id,
                        "group": group_value,
                        "episodes": records,
                        "group_sha256": _canonical_sha256(records),
                        "completed": completed,
                        "total": total,
                        "recorded_at": _timestamp(),
                    },
                )
                completed_groups[(task, seed)] = records
        finally:
            stop_server()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)
            server_stdout.close()
            server_stderr.close()

    if set(completed_groups) != set(groups):
        raise RuntimeError("OpenPI LIBERO evaluation ended without all task+seed groups committed")
    episodes = [
        episode
        for group in groups
        for episode in completed_groups[group]
    ]
    if len(episodes) != total:
        raise RuntimeError(f"OpenPI LIBERO committed {len(episodes)} episodes; expected {total}")

    raw_metrics = result_path.parent / "raw-episodes.json"
    _atomic_json(raw_metrics, episodes)
    artifacts = sorted({
        str(path)
        for root in (result_path.parent / "logs", video_base, result_path.parent / "attempts")
        for path in root.rglob("*")
        if path.is_file()
    })
    evaluator = dict(context.get("evaluator") or {})
    result = {
        "schema_version": 1,
        "run_id": str(context["run_id"]),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": str(checkpoint["sha256"]),
        },
        "evaluator": {
            "adapter": str(evaluator.get("adapter") or "libero"),
            "version": str(evaluator.get("version") or "1"),
        },
        "environment": {
            "suite": suite_name,
            "version": str(suite["version"]),
            "task": None,
        },
        "aggregate": _metrics(episodes),
        "episodes": episodes,
        "raw_metrics_path": str(raw_metrics),
        "artifacts": artifacts,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(result_path, result)
    attempt = json.loads((attempt_root / "attempt.json").read_text(encoding="utf-8"))
    attempt.update({"status": "SUCCEEDED", "finished_at": _timestamp()})
    _atomic_json(attempt_root / "attempt.json", attempt)
    return 0


OPENPI_LIBERO_BRIDGE_SOURCE = Path(__file__).read_text(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
