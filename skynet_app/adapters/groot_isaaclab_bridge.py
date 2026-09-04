from __future__ import annotations

"""Pinned dual-runtime GR00T N1.6 / Isaac Lab evaluation capsule.

The outer process runs in the policy's pinned GR00T UV environment. It starts
the official GR00T policy server and forwards policy calls over an authenticated
stdlib-only local connection. The simulator process runs through the exact
Isaac conda interpreter snapshotted by the evaluator runtime profile.
"""

import argparse
import hashlib
import inspect
import json
import os
from multiprocessing.connection import Client, Listener
from pathlib import Path
import re
import secrets
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from typing import Any


SCHEMA = "skynet.evaluation-context/v1"
SUITE = "groot_gr1_isaaclab_evaltasks"
ENVIRONMENT = "isaac_sim"
EMBODIMENT = "GR1"
GROOT_REVISION = "9b37aa1ce69c73c6d165233fa88128283bba4508"
EVALTASKS_REPOSITORY = "https://github.com/isaac-sim/IsaacLabEvalTasks"
EVALTASKS_REVISION = "460f2878bdcb4db2d21913db789174fb316b73e2"
TASK_CATALOG_HASH = "bb7a5f1f7146307530c2be63f9779d47d2a16332931e637569492c361c41043e"
ISAAC_KIT_ARGS = "--/rtx/verifyDriverVersion/enabled=false"
OMNI_WARP_VERSION = "1.8.1"
OMNI_WARP_SOURCE_REVISION = "ad1092b23dc6dae502d4741e2e0ed34e6bcc8dc1"
OMNI_WARP_WHEEL_SHA256 = "0fee51dae3ff053ad6b982f14ff980730ce09f5a51820e6a4c9cd27264142580"
OMNI_WARP_BINARY_SHA256 = "59d5e9998a03d1ee12a1df61227dfb38ae811194ae6882f0a0cdfe786d61425a"
OMNI_WARP_OVERLAY_SCHEMA = "skynet.omni-warp-overlay/v1"
OMNI_WARP_OVERLAY_RELATIVE = Path(".skynet/compat/omni-warp-1.8.1")
EXPECTED_RUNTIME_VERSIONS = {
    "python": "3.11",
    "isaac_sim": "5.0.0",
    "isaac_lab": "2.2.0",
}
EXPECTED_MODALITIES = {
    "video": {"keys": ["ego_view"], "delta_indices": [0]},
    "state": {
        "keys": ["left_arm", "right_arm", "left_hand", "right_hand"],
        "delta_indices": [0],
    },
    "action": {
        "keys": ["left_arm", "right_arm", "left_hand", "right_hand"],
        "delta_indices": list(range(16)),
    },
    "language": {
        "keys": ["annotation.human.action.task_description"],
        "delta_indices": [0],
    },
}
TASK_DEFINITIONS = {
    "nutpouring": {
        "environment_id": "Isaac-NutPour-GR1T2-ClosedLoop-v0",
        "instruction": (
            "Pick up the beaker and tilt it to pour out 1 metallic nut into the bowl. "
            "Pick up the bowl and place it on the metallic measuring scale."
        ),
        "rollout_length": 30,
        "feedback_actions": 16,
        "simulation_device": "cuda:0",
    },
    "pipesorting": {
        "environment_id": "Isaac-ExhaustPipe-GR1T2-ClosedLoop-v0",
        "instruction": "Pick up the blue pipe and place it into the blue bin.",
        "rollout_length": 20,
        "feedback_actions": 16,
        "simulation_device": "cuda:0",
    },
}


def _launch_isaac_app(
    app_launcher,
    *,
    device,
    kit_args,
    phase_marker,
    stage_api_loader,
):
    """Launch pinned Isaac Sim and ensure its explicit post-launch USD stage."""

    phase_marker("app_launcher_constructor_started")
    launcher = app_launcher(
        headless=True,
        enable_cameras=True,
        num_envs=1,
        device=device,
        kit_args=kit_args,
    )

    phase_marker("app_launcher_constructor_complete")
    simulation_app = launcher.app

    # AppLauncher intentionally starts without a stage. Create it only after
    # Kit returns, through the public USD context used by pinned EvalTasks.
    omni_usd = stage_api_loader()
    usd_context = omni_usd.get_context()
    if usd_context is None:
        raise RuntimeError("Isaac Sim started without an omni.usd context")
    if usd_context.get_stage() is None:
        phase_marker("usd_stage_creation_started")
        usd_context.new_stage()
    if usd_context.get_stage() is None:
        raise RuntimeError("Isaac Sim did not create the required USD stage")
    phase_marker("usd_stage_ready")
    return launcher, simulation_app


ISAAC_CLIENT_SOURCE = r'''#!/usr/bin/env python3
"""Isaac-only side of the pinned GR1-T2 closed-loop evaluator."""

import argparse
import json
from multiprocessing.connection import Client
from pathlib import Path
import random

import imageio.v2 as imageio
import numpy as np

import isaacsim  # noqa: F401 - bootstrap Isaac Sim before Isaac Lab imports
import pinocchio  # noqa: F401 - force Isaac Lab's build before AppLauncher
from isaaclab.app import AppLauncher


POLICY_JOINTS = {
    "left_arm": [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_pitch_joint",
        "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    ],
    "right_arm": [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_pitch_joint",
        "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    ],
    "left_hand": [
        "L_pinky_proximal_joint", "L_ring_proximal_joint",
        "L_middle_proximal_joint", "L_index_proximal_joint",
        "L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint",
    ],
    "right_hand": [
        "R_pinky_proximal_joint", "R_ring_proximal_joint",
        "R_middle_proximal_joint", "R_index_proximal_joint",
        "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
    ],
}

# Exact indices from scripts/config/gr1/state_joint_space.yaml at the pinned
# IsaacLabEvalTasks revision.
STATE_INDEX = {
    "left_shoulder_pitch_joint": 12, "right_shoulder_pitch_joint": 13,
    "left_shoulder_roll_joint": 17, "right_shoulder_roll_joint": 18,
    "left_shoulder_yaw_joint": 22, "right_shoulder_yaw_joint": 23,
    "left_elbow_pitch_joint": 24, "right_elbow_pitch_joint": 25,
    "left_wrist_yaw_joint": 26, "right_wrist_yaw_joint": 27,
    "left_wrist_roll_joint": 28, "right_wrist_roll_joint": 29,
    "left_wrist_pitch_joint": 30, "right_wrist_pitch_joint": 31,
    "L_index_proximal_joint": 32, "L_middle_proximal_joint": 33,
    "L_pinky_proximal_joint": 34, "L_ring_proximal_joint": 35,
    "L_thumb_proximal_yaw_joint": 36, "R_index_proximal_joint": 37,
    "R_middle_proximal_joint": 38, "R_pinky_proximal_joint": 39,
    "R_ring_proximal_joint": 40, "R_thumb_proximal_yaw_joint": 41,
    "L_thumb_proximal_pitch_joint": 46, "R_thumb_proximal_pitch_joint": 51,
}

# Exact controlled-joint order from scripts/config/gr1/action_joint_space.yaml.
ACTION_INDEX = {
    "left_shoulder_pitch_joint": 0, "right_shoulder_pitch_joint": 1,
    "left_shoulder_roll_joint": 2, "right_shoulder_roll_joint": 3,
    "left_shoulder_yaw_joint": 4, "right_shoulder_yaw_joint": 5,
    "left_elbow_pitch_joint": 6, "right_elbow_pitch_joint": 7,
    "left_wrist_yaw_joint": 8, "right_wrist_yaw_joint": 9,
    "left_wrist_roll_joint": 10, "right_wrist_roll_joint": 11,
    "left_wrist_pitch_joint": 12, "right_wrist_pitch_joint": 13,
    "L_index_proximal_joint": 14, "L_middle_proximal_joint": 15,
    "L_pinky_proximal_joint": 16, "L_ring_proximal_joint": 17,
    "L_thumb_proximal_yaw_joint": 18, "R_index_proximal_joint": 19,
    "R_middle_proximal_joint": 20, "R_pinky_proximal_joint": 21,
    "R_ring_proximal_joint": 22, "R_thumb_proximal_yaw_joint": 23,
    "L_thumb_proximal_pitch_joint": 28, "R_thumb_proximal_pitch_joint": 33,
}
ACTION_WIDTH = 36
ACTION_HORIZON = 16
KIT_ARGS = "--/rtx/verifyDriverVersion/enabled=false"
EXPECTED_WARP_VERSION = "1.8.1"


def fail(message):
    raise RuntimeError(message)


def kit_args(extension_folder):
    extension_folder = str(Path(extension_folder).resolve())
    if any(character.isspace() for character in extension_folder):
        fail(f"Warp Kit extension path cannot contain whitespace: {extension_folder}")
    return (
        f"{KIT_ARGS} --ext-folder {extension_folder} "
        "--enable omni.warp.core-1.8.1 --enable omni.warp-1.8.1"
    )


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def call_policy(connection, operation, **payload):
    connection.send({"operation": operation, **payload})
    response = connection.recv()
    if not isinstance(response, dict) or response.get("ok") is not True:
        detail = response.get("error") if isinstance(response, dict) else repr(response)
        fail(f"GR00T policy proxy failed: {detail}")
    return response


def policy_observation(robot_joint_pos, rgb, instruction):
    joints = robot_joint_pos.detach().cpu().numpy().astype(np.float32, copy=False)
    if joints.ndim != 2 or joints.shape[0] != 1 or joints.shape[1] < 54:
        fail(f"Pinned GR1 state contract expected shape (1, >=54), received {joints.shape}")

    frame = rgb.detach().cpu().numpy()
    if frame.ndim != 4 or frame.shape[0] != 1 or tuple(frame.shape[1:3]) != (160, 256):
        fail(f"Pinned GR1 ego camera expected shape (1, 160, 256, C), received {frame.shape}")
    if frame.shape[-1] < 3:
        fail(f"Pinned GR1 ego camera must provide RGB channels, received {frame.shape[-1]}")
    frame = np.asarray(frame[..., :3], dtype=np.uint8)
    frame = np.pad(frame, ((0, 0), (48, 48), (0, 0), (0, 0)), mode="constant")
    if frame.shape != (1, 256, 256, 3):
        fail(f"Pinned GR1 image transform produced unexpected shape {frame.shape}")

    observation = {
        "video.ego_view": frame.reshape(1, 1, 256, 256, 3),
        "annotation.human.action.task_description": [instruction],
    }
    for group, names in POLICY_JOINTS.items():
        values = np.stack([joints[:, STATE_INDEX[name]] for name in names], axis=1)
        observation[f"state.{group}"] = values.reshape(1, 1, len(names)).astype(np.float32)
    return observation, frame[0]


def simulator_actions(action):
    expected = {f"action.{group}" for group in POLICY_JOINTS}
    if not isinstance(action, dict) or set(action) != expected:
        actual = sorted(action) if isinstance(action, dict) else type(action).__name__
        fail(f"GR00T returned incompatible action keys: expected {sorted(expected)}, received {actual}")

    output = np.zeros((1, ACTION_HORIZON, ACTION_WIDTH), dtype=np.float32)
    for group, names in POLICY_JOINTS.items():
        values = np.asarray(action[f"action.{group}"])
        expected_shape = (1, ACTION_HORIZON, len(names))
        if values.shape != expected_shape or values.dtype != np.float32:
            fail(
                f"GR00T action.{group} expected float32 {expected_shape}, "
                f"received {values.dtype} {values.shape}"
            )
        for policy_index, name in enumerate(names):
            output[..., ACTION_INDEX[name]] = values[..., policy_index]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpc-host", required=True)
    parser.add_argument("--rpc-port", required=True, type=int)
    parser.add_argument("--rpc-auth", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--rollout-length", required=True, type=int)
    parser.add_argument("--feedback-actions", required=True, type=int)
    parser.add_argument("--episode-result", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--simulation-device", default="cuda:0")
    parser.add_argument("--kit-extension-folder", required=True)
    args = parser.parse_args()

    if not (1 <= args.feedback_actions <= ACTION_HORIZON):
        fail(f"feedback-actions must be between 1 and {ACTION_HORIZON}")
    if args.rollout_length < 1:
        fail("rollout-length must be positive")

    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    launch_phase_path = Path(f"{args.episode_result}.launch-phase.json")

    def mark_launch_phase(phase):
        atomic_json(
            launch_phase_path,
            {
                "schema_version": "skynet.isaac-launch-phase/v1",
                "phase": phase,
            },
        )

    launcher, simulation_app = _launch_isaac_app(
        AppLauncher,
        device=args.simulation_device,
        kit_args=kit_args(args.kit_extension_folder),
        phase_marker=mark_launch_phase,
        stage_api_loader=lambda: __import__("omni.usd", fromlist=["*"]),
    )
    env = None
    writer = None
    connection = None
    try:
        import gymnasium as gym
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        import isaaclab_eval_tasks.tasks  # noqa: F401 - registers exact task IDs
        import warp

        warp_version = str(warp.config.version)
        warp_path = Path(warp.__file__).resolve()
        expected_warp_root = (
            Path(args.kit_extension_folder).resolve() / "omni.warp.core"
        )
        if warp_version != EXPECTED_WARP_VERSION:
            fail(
                f"Isaac Kit loaded Warp {warp_version}, expected pinned "
                f"{EXPECTED_WARP_VERSION} from {expected_warp_root}"
            )
        if not warp_path.is_relative_to(expected_warp_root):
            fail(
                f"Isaac Kit loaded Warp from {warp_path}, outside pinned extension "
                f"{expected_warp_root}"
            )

        env_cfg = parse_env_cfg(args.task_id, device=args.simulation_device, num_envs=1)
        success_term = env_cfg.terminations.success
        env_cfg.terminations = {}
        env_cfg.recorders = {}
        env = gym.make(args.task_id, cfg=env_cfg).unwrapped
        env.seed(args.seed)
        env.sim.reset()
        env.reset(seed=args.seed)

        robot = env.scene["robot"]
        camera = env.scene["robot_pov_cam"]
        video_path = Path(args.video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video_path), fps=20, format="FFMPEG", macro_block_size=None)
        connection = Client(
            (args.rpc_host, args.rpc_port),
            authkey=bytes.fromhex(args.rpc_auth),
        )
        call_policy(connection, "ping")
        call_policy(connection, "reset")

        executed_steps = 0
        with torch.inference_mode():
            for _ in range(args.rollout_length):
                observation, _ = policy_observation(
                    robot.data.joint_pos,
                    camera.data.output["rgb"],
                    args.instruction,
                )
                response = call_policy(connection, "get_action", observation=observation)
                action_chunk = simulator_actions(response["action"])
                for action_index in range(args.feedback_actions):
                    action_tensor = torch.from_numpy(action_chunk[:, action_index, :]).to(args.simulation_device)
                    env.step(action_tensor)
                    rendered = camera.data.output["rgb"][0].detach().cpu().numpy()
                    rendered = np.asarray(rendered[..., :3], dtype=np.uint8)
                    writer.append_data(rendered)
                    executed_steps += 1

        success_value = success_term.func(env, **success_term.params)
        if not isinstance(success_value, torch.Tensor) or success_value.numel() != 1:
            fail(f"Pinned task success predicate returned unexpected value {success_value!r}")
        atomic_json(
            args.episode_result,
            {
                "success": bool(success_value.reshape(-1)[0].item()),
                "reward": None,
                "episode_length": executed_steps,
            },
        )
    finally:
        if connection is not None:
            connection.close()
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
'''.replace(
    "\ndef main(",
    f"\n{inspect.getsource(_launch_isaac_app)}\ndef main(",
    1,
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(mapping: dict[str, Any], key: str, where: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "" or value == []:
        fail(f"Missing required {where}.{key}")
    return value


def require_object(mapping: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = require(mapping, key, where)
    if not isinstance(value, dict):
        fail(f"{where}.{key} must be an object")
    return value


def load_json(path: Path, description: str = "JSON") -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Cannot read {description} {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{description} {path} must contain an object")
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


def validate_checkpoint_contract(checkpoint_path: Path) -> None:
    processor = load_json(checkpoint_path / "processor_config.json", "checkpoint processor config")
    if processor.get("processor_class") != "Gr00tN1d6Processor":
        fail(
            "Isaac GR1 evaluation requires a GR00T N1.6 processor; received "
            f"{processor.get('processor_class')!r}"
        )
    try:
        profile = processor["processor_kwargs"]["modality_configs"]["gr1"]
    except (KeyError, TypeError) as exc:
        fail(f"Checkpoint does not contain the required gr1 modality profile: {exc}")
    if not isinstance(profile, dict):
        fail("Checkpoint gr1 modality profile must be an object")

    for modality, expected in EXPECTED_MODALITIES.items():
        actual = profile.get(modality)
        if not isinstance(actual, dict):
            fail(f"Checkpoint gr1 profile is missing {modality}")
        actual_contract = {
            "keys": actual.get("modality_keys"),
            "delta_indices": actual.get("delta_indices"),
        }
        if actual_contract != expected:
            fail(
                f"Checkpoint gr1 {modality} contract is incompatible: "
                f"expected {expected}, received {actual_contract}"
            )

    state = profile["state"]
    if state.get("sin_cos_embedding_keys") != EXPECTED_MODALITIES["state"]["keys"]:
        fail("Checkpoint gr1 state must use sin/cos encoding for all four joint groups")
    action_configs = profile["action"].get("action_configs")
    if not isinstance(action_configs, list) or len(action_configs) != 4:
        fail("Checkpoint gr1 action contract must declare four action configurations")
    for index, config in enumerate(action_configs):
        if not isinstance(config, dict):
            fail(f"Checkpoint gr1 action configuration {index} must be an object")
        actual = (config.get("rep"), config.get("type"), config.get("format"), config.get("state_key"))
        if actual != ("ABSOLUTE", "NON_EEF", "DEFAULT", None):
            fail(
                "Checkpoint gr1 actions must be absolute non-EEF joint commands; "
                f"configuration {index} is {actual}"
            )

    embodiment_ids = load_json(checkpoint_path / "embodiment_id.json", "checkpoint embodiment map")
    if not isinstance(embodiment_ids.get("gr1"), int):
        fail("Checkpoint embodiment map does not contain a numeric gr1 ID")


def validate_suite(context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if context.get("schema_version") != SCHEMA:
        fail(f"Expected context schema {SCHEMA}, received {context.get('schema_version')!r}")
    if context.get("environment") != ENVIRONMENT:
        fail(f"This evaluator supports {ENVIRONMENT}, not {context.get('environment')!r}")
    if context.get("headless") is not True:
        fail("Pinned Isaac evaluation requires headless=true on Slurm")
    if int(context.get("parallelism", 0)) != 1:
        fail("Pinned dual-runtime Isaac evaluation currently requires parallelism=1")

    suite = require_object(context, "suite", "evaluation")
    if suite.get("name") != SUITE:
        fail(f"This evaluator supports only the exact suite {SUITE}")
    config = require_object(suite, "config", "evaluation.suite")
    if config.get("task_catalog_complete") is not True:
        fail("IsaacLabEvalTasks requires a complete immutable task catalog")
    if config.get("task_catalog_sha256") != TASK_CATALOG_HASH:
        fail("IsaacLabEvalTasks catalog hash does not match the pinned two-task catalog")
    provenance = config.get("task_catalog_provenance")
    if not isinstance(provenance, dict):
        fail("IsaacLabEvalTasks catalog provenance is missing")
    if provenance.get("repository") != EVALTASKS_REPOSITORY:
        fail("IsaacLabEvalTasks catalog repository does not match the pinned upstream")
    if provenance.get("revision") != EVALTASKS_REVISION:
        fail("IsaacLabEvalTasks catalog revision does not match the pinned upstream")

    tasks = config.get("tasks")
    if tasks != list(TASK_DEFINITIONS):
        fail(f"Pinned task catalog must be {list(TASK_DEFINITIONS)}, received {tasks}")
    options = config.get("task_options")
    if not isinstance(options, list):
        fail("Pinned task catalog must include task_options")
    option_map: dict[str, dict[str, Any]] = {}
    for option in options:
        if not isinstance(option, dict) or not isinstance(option.get("id"), str):
            fail("Pinned task catalog contains an invalid task option")
        option_map[option["id"]] = option
    if list(option_map) != list(TASK_DEFINITIONS):
        fail("Pinned task option order or IDs do not match the immutable catalog")
    for task, expected in TASK_DEFINITIONS.items():
        metadata = option_map[task].get("metadata")
        native = metadata.get("native") if isinstance(metadata, dict) else None
        if not isinstance(native, dict):
            fail(f"Pinned task option {task} has no native execution metadata")
        actual = {
            "environment_id": native.get("environment_id"),
            "instruction": native.get("instruction"),
            "rollout_length": native.get("rollout_length"),
            "feedback_actions": native.get("feedback_actions"),
            "simulation_device": native.get("simulation_device"),
        }
        if actual != expected:
            fail(f"Pinned task option {task} metadata changed: expected {expected}, received {actual}")
    return suite, option_map


def verify_checkout(path: Path, revision: str, label: str) -> None:
    if not path.is_dir():
        fail(f"{label} checkout does not exist: {path}")
    process = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    actual = process.stdout.strip().lower()
    if process.returncode != 0 or actual != revision:
        detail = process.stderr.strip() or actual or "not a Git checkout"
        fail(f"{label} checkout is not pinned to {revision}: {detail}")


def validate_evaluator_runtime(context: dict[str, Any]) -> dict[str, Any]:
    runtime = require_object(context, "evaluator_runtime", "evaluation")
    if runtime.get("backend") != "conda":
        fail("IsaacLabEvalTasks evaluator runtime must use the pinned conda backend")
    if not isinstance(runtime.get("profile"), str) or not runtime["profile"].strip():
        fail("IsaacLabEvalTasks evaluator runtime requires an exact profile ID")
    versions = runtime.get("versions")
    if not isinstance(versions, dict) or any(
        str(versions.get(key)) != value for key, value in EXPECTED_RUNTIME_VERSIONS.items()
    ):
        fail(
            "IsaacLabEvalTasks requires exact runtime versions "
            f"{EXPECTED_RUNTIME_VERSIONS}, received {versions}"
        )
    source = runtime.get("source")
    if not isinstance(source, dict) or source.get("repository") != EVALTASKS_REPOSITORY:
        fail("Evaluator runtime does not snapshot the pinned IsaacLabEvalTasks repository")
    if source.get("revision") != EVALTASKS_REVISION:
        fail("Evaluator runtime does not snapshot the pinned IsaacLabEvalTasks revision")

    environment_path = Path(str(require(runtime, "environment_path", "evaluation.evaluator_runtime")))
    python_executable = Path(str(require(runtime, "python_executable", "evaluation.evaluator_runtime")))
    source_dir = Path(str(require(runtime, "source_dir", "evaluation.evaluator_runtime")))
    for label, path in (
        ("environment_path", environment_path),
        ("python_executable", python_executable),
        ("source_dir", source_dir),
    ):
        if not path.is_absolute():
            fail(f"evaluation.evaluator_runtime.{label} must be absolute: {path}")
    if not environment_path.is_dir():
        fail(f"Pinned Isaac conda environment does not exist: {environment_path}")
    if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
        fail(f"Pinned Isaac Python is not executable: {python_executable}")
    expected_python = environment_path / "bin" / "python"
    try:
        same_python = expected_python.is_file() and python_executable.samefile(expected_python)
    except OSError:
        same_python = False
    if not same_python:
        fail(
            "Pinned Isaac Python must be the selected conda environment interpreter: "
            f"expected {expected_python}, received {python_executable}"
        )
    verify_checkout(source_dir, EVALTASKS_REVISION, "IsaacLabEvalTasks")
    required_paths = [
        source_dir / "source/isaaclab_eval_tasks/isaaclab_eval_tasks/tasks/__init__.py",
        source_dir / "scripts/config/gr1/state_joint_space.yaml",
        source_dir / "scripts/config/gr1/action_joint_space.yaml",
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        fail(f"Pinned IsaacLabEvalTasks checkout is incomplete: {missing}")
    return {
        **runtime,
        "environment_path": str(environment_path),
        "python_executable": str(python_executable),
        "source_dir": str(source_dir),
    }


def validate_policy(context: dict[str, Any], source_dir: Path) -> tuple[Path, str]:
    policy_runtime = require_object(context, "policy_runtime", "evaluation")
    if policy_runtime.get("backend") != "uv":
        fail("GR00T N1.6 policy server must run in the pinned UV policy runtime")
    policy = require_object(context, "policy", "evaluation")
    if policy.get("adapter") != "groot":
        fail("Isaac GR1 evaluation requires a pinned groot policy adapter")
    source = require_object(policy, "source", "evaluation.policy")
    if source.get("revision") != GROOT_REVISION:
        fail("Selected policy source is not the pinned GR00T N1.6 revision")
    native = policy.get("native_config")
    if not isinstance(native, dict) or str(native.get("embodiment_tag", "")).casefold() != "gr1":
        fail("Isaac GR1 evaluation requires native.config.embodiment_tag=GR1")
    verify_checkout(source_dir, GROOT_REVISION, "Isaac-GR00T")
    server_script = source_dir / "gr00t/eval/run_gr00t_server.py"
    if not server_script.is_file():
        fail(f"Pinned GR00T policy server is missing: {server_script}")

    checkpoint = require_object(context, "checkpoint", "evaluation")
    checkpoint_path = Path(str(require(checkpoint, "path", "evaluation.checkpoint"))).resolve()
    checkpoint_sha = str(require(checkpoint, "sha256", "evaluation.checkpoint")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha):
        fail("evaluation.checkpoint.sha256 must be a SHA-256 value")
    if not checkpoint_path.is_dir():
        fail(f"Pinned checkpoint does not exist: {checkpoint_path}")
    validate_checkpoint_contract(checkpoint_path)
    return checkpoint_path, checkpoint_sha


def allocate_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def tail(path: Path, limit: int = 8000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def wait_for_policy_server(
    process: subprocess.Popen[str],
    port: int,
    stderr_path: Path,
    timeout_seconds: int = 900,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            fail(f"GR00T policy server exited with code {return_code}:\n{tail(stderr_path)}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    fail(f"GR00T policy server did not become ready within {timeout_seconds} seconds")


def _modality_value(config: Any, key: str) -> list[Any] | None:
    value = config.get(key) if isinstance(config, dict) else getattr(config, key, None)
    return list(value) if value is not None else None


def validate_remote_contract(config: Any) -> None:
    if not isinstance(config, dict):
        fail(f"GR00T policy server returned invalid modality config {type(config).__name__}")
    for modality, expected in EXPECTED_MODALITIES.items():
        actual = config.get(modality)
        if actual is None:
            fail(f"GR00T policy server omitted {modality} modality")
        actual_contract = {
            "keys": _modality_value(actual, "modality_keys"),
            "delta_indices": _modality_value(actual, "delta_indices"),
        }
        if actual_contract != expected:
            fail(
                f"Live GR00T {modality} contract differs from the checkpoint: "
                f"expected {expected}, received {actual_contract}"
            )


def close_policy_client(client: Any) -> None:
    try:
        client.socket.close(linger=0)
    except Exception:
        pass
    try:
        client.context.term()
    except Exception:
        pass


def probe_policy_contract(port: int) -> None:
    from gr00t.policy.server_client import PolicyClient

    client = PolicyClient(host="127.0.0.1", port=port, strict=False)
    try:
        if client.ping() is not True:
            fail("GR00T policy server did not answer ping")
        validate_remote_contract(client.get_modality_config())
    finally:
        close_policy_client(client)


class PolicyProxy:
    def __init__(self, policy_port: int):
        self.authkey = secrets.token_bytes(32)
        self.listener = Listener(("127.0.0.1", 0), authkey=self.authkey)
        address = self.listener.address
        if not isinstance(address, tuple):
            fail("Policy proxy did not bind a TCP address")
        self.host = str(address[0])
        self.port = int(address[1])
        self.policy_port = policy_port
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._serve, name="groot-isaac-policy-proxy", daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        from gr00t.policy.server_client import PolicyClient

        client = PolicyClient(host="127.0.0.1", port=self.policy_port, strict=False)
        try:
            while True:
                connection = self.listener.accept()
                try:
                    while True:
                        try:
                            request = connection.recv()
                        except EOFError:
                            break
                        if not isinstance(request, dict):
                            connection.send({"ok": False, "error": "Policy proxy request must be an object"})
                            continue
                        operation = request.get("operation")
                        if operation == "shutdown":
                            connection.send({"ok": True})
                            return
                        try:
                            if operation == "ping":
                                if client.ping() is not True:
                                    fail("GR00T policy server ping failed")
                                response = {"ok": True}
                            elif operation == "reset":
                                response = {"ok": True, "info": client.reset()}
                            elif operation == "get_action":
                                observation = request.get("observation")
                                if not isinstance(observation, dict):
                                    fail("Policy proxy get_action requires an observation object")
                                action, info = client.get_action(observation)
                                response = {"ok": True, "action": action, "info": info}
                            else:
                                fail(f"Unknown policy proxy operation: {operation!r}")
                        except Exception as exc:
                            response = {"ok": False, "error": str(exc)}
                        connection.send(response)
                finally:
                    connection.close()
        except BaseException as exc:
            self.error = exc
        finally:
            close_policy_client(client)

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"Policy proxy failed: {self.error}") from self.error

    def close(self) -> None:
        try:
            connection = Client((self.host, self.port), authkey=self.authkey)
            connection.send({"operation": "shutdown"})
            connection.recv()
            connection.close()
        except Exception:
            pass
        self.thread.join(timeout=10)
        self.listener.close()
        self.check()


def terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_omni_warp_extension_folder(environment_path: Path) -> Path:
    """Verify the isolated Kit extension that fixes NVIDIA Warp issue #851."""

    root = environment_path.resolve() / OMNI_WARP_OVERLAY_RELATIVE
    extension_folder = root / "source" / "exts"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        fail(
            "Pinned NVIDIA Warp Kit overlay is missing. Provision the isolated "
            f"Warp {OMNI_WARP_VERSION} overlay at {root}; installing warp-lang into "
            "site-packages does not replace Isaac Sim's bundled omni.warp.core"
        )
    manifest = load_json(manifest_path, "NVIDIA Warp Kit overlay manifest")
    expected_manifest = {
        "schema_version": OMNI_WARP_OVERLAY_SCHEMA,
        "version": OMNI_WARP_VERSION,
        "source_revision": OMNI_WARP_SOURCE_REVISION,
        "wheel_sha256": OMNI_WARP_WHEEL_SHA256,
        "warp_binary_sha256": OMNI_WARP_BINARY_SHA256,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            fail(
                f"NVIDIA Warp Kit overlay manifest {key} is {manifest.get(key)!r}; "
                f"expected {expected!r}"
            )
    if any(character.isspace() for character in str(extension_folder)):
        fail(f"NVIDIA Warp Kit extension path cannot contain whitespace: {extension_folder}")
    for extension_name in ("omni.warp", "omni.warp.core"):
        config_path = extension_folder / extension_name / "config" / "extension.toml"
        if not config_path.is_file():
            fail(f"Pinned NVIDIA Warp Kit extension is missing: {config_path}")
        config_source = config_path.read_text(encoding="utf-8")
        if not re.search(
            rf'^version\s*=\s*"{re.escape(OMNI_WARP_VERSION)}"\s*$',
            config_source,
            flags=re.MULTILINE,
        ):
            fail(
                f"Pinned NVIDIA Warp Kit extension {extension_name} does not declare "
                f"version {OMNI_WARP_VERSION}"
            )
    binary_path = extension_folder / "omni.warp.core" / "warp" / "bin" / "warp.so"
    if not binary_path.is_file():
        fail(f"Pinned NVIDIA Warp native library is missing: {binary_path}")
    actual_binary_sha256 = _sha256_file(binary_path)
    if actual_binary_sha256 != OMNI_WARP_BINARY_SHA256:
        fail(
            f"Pinned NVIDIA Warp native library SHA-256 is {actual_binary_sha256}; "
            f"expected {OMNI_WARP_BINARY_SHA256}"
        )
    return extension_folder


def build_isaac_kit_args(extension_folder: Path) -> str:
    resolved = str(extension_folder.resolve())
    if any(character.isspace() for character in resolved):
        fail(f"NVIDIA Warp Kit extension path cannot contain whitespace: {resolved}")
    return (
        f"{ISAAC_KIT_ARGS} --ext-folder {resolved} "
        "--enable omni.warp.core-1.8.1 --enable omni.warp-1.8.1"
    )


def build_policy_server_command(source_dir: Path, checkpoint_path: Path, port: int) -> list[str]:
    return [
        sys.executable,
        str(source_dir / "gr00t/eval/run_gr00t_server.py"),
        "--model-path",
        str(checkpoint_path),
        "--embodiment-tag",
        EMBODIMENT,
        "--device",
        "cuda:0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--use-sim-policy-wrapper",
    ]


def build_isaac_environment(runtime: dict[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        environment.pop(key, None)
    declared = runtime.get("environment")
    if declared is None:
        snapshot = runtime.get("profile_snapshot")
        declared = snapshot.get("environment", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(declared, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in declared.items()
    ):
        fail("Evaluator runtime environment snapshot must map strings to strings")
    environment.update(
        {key: value for key, value in declared.items() if key != "OMNI_KIT_ACCEPT_EULA"}
    )
    operator_environment = runtime.get("operator_environment")
    if operator_environment is None:
        fail(
            "Set context.evaluator_runtime.operator_environment.OMNI_KIT_ACCEPT_EULA=YES "
            "only after the operator accepts the NVIDIA Isaac Sim EULA; committed runtime "
            "profiles cannot record legal consent"
        )
    if not isinstance(operator_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in operator_environment.items()
    ):
        fail("Evaluator runtime operator_environment must map strings to strings")
    if operator_environment.get("OMNI_KIT_ACCEPT_EULA") != "YES":
        fail(
            "Set context.evaluator_runtime.operator_environment.OMNI_KIT_ACCEPT_EULA=YES "
            "only after the operator accepts the NVIDIA Isaac Sim EULA; committed runtime "
            "profiles cannot record legal consent"
        )
    environment.update(operator_environment)
    environment_path = Path(runtime["environment_path"])
    source_dir = Path(runtime["source_dir"])
    snapshot = runtime.get("profile_snapshot")
    operations = runtime.get("environment_operations")
    if operations is None:
        operations = snapshot.get("environment_operations", []) if isinstance(snapshot, dict) else []
    if not isinstance(operations, list):
        fail("Evaluator runtime environment_operations snapshot must be a list")
    for operation in operations:
        if not isinstance(operation, dict):
            fail("Evaluator runtime environment operation must be an object")
        name = operation.get("name")
        kind = operation.get("operation", "set")
        value = operation.get("value", "")
        separator = operation.get("separator", ":")
        if not all(isinstance(item, str) for item in (name, kind, value, separator)):
            fail("Evaluator runtime environment operation fields must be strings")
        if name == "OMNI_KIT_ACCEPT_EULA":
            fail("Evaluator runtime profile cannot encode NVIDIA Isaac Sim EULA acceptance")
        if kind not in {"set", "prepend", "append", "unset"}:
            fail(f"Unsupported evaluator runtime environment operation: {kind}")
        if kind == "unset":
            environment.pop(name, None)
            continue
        value = value.replace("{{runtime.environment_path}}", str(environment_path))
        current = environment.get(name, "")
        if kind == "prepend" and current:
            value = f"{value}{separator}{current}"
        elif kind == "append" and current:
            value = f"{current}{separator}{value}"
        environment[name] = value
    environment["CONDA_PREFIX"] = str(environment_path)
    environment["PATH"] = f"{environment_path / 'bin'}:{environment.get('PATH', '')}"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(source_dir / "source/isaaclab_eval_tasks")
    return environment


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
        if not isinstance(row, dict) or row.get("identity") != identity:
            fail("Existing progress ledger belongs to a different pinned evaluation")
        if row.get("record_type") != "episode" or row.get("status") != "SUCCEEDED":
            continue
        key = (str(row.get("task")), int(row.get("seed")), int(row.get("episode_index")))
        if key in completed:
            fail(f"Progress ledger contains duplicate completed episode {key}")
        episode = row.get("episode")
        if not isinstance(episode, dict):
            fail(f"Progress ledger completed episode {key} has no canonical episode object")
        video_path = Path(str(episode.get("video_path", "")))
        if not video_path.is_file() or video_path.stat().st_size == 0:
            fail(f"Progress ledger video artifact is missing for completed episode {key}: {video_path}")
        completed[key] = episode
    return completed


def run_isaac_episode(
    *,
    runtime: dict[str, Any],
    proxy: PolicyProxy,
    task: str,
    task_definition: dict[str, Any],
    seed: int,
    episode_index: int,
    result_path: Path,
    video_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    kit_extension_folder = resolve_omni_warp_extension_folder(
        Path(runtime["environment_path"])
    )
    client_path = result_path.parent / "skynet-isaac-client.py"
    client_path.write_text(ISAAC_CLIENT_SOURCE, encoding="utf-8")
    command = [
        runtime["python_executable"],
        str(client_path),
        "--rpc-host",
        proxy.host,
        "--rpc-port",
        str(proxy.port),
        "--rpc-auth",
        proxy.authkey.hex(),
        "--task-id",
        str(task_definition["environment_id"]),
        "--instruction",
        str(task_definition["instruction"]),
        "--seed",
        str(seed),
        "--rollout-length",
        str(task_definition["rollout_length"]),
        "--feedback-actions",
        str(task_definition["feedback_actions"]),
        "--episode-result",
        str(result_path),
        "--video-path",
        str(video_path),
        "--simulation-device",
        str(task_definition["simulation_device"]),
        "--kit-extension-folder",
        str(kit_extension_folder),
    ]
    process = subprocess.run(
        command,
        cwd=runtime["source_dir"],
        env=build_isaac_environment(runtime),
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    proxy.check()
    if process.returncode != 0:
        detail = process.stderr[-4000:] or process.stdout[-4000:]
        fail(
            f"Isaac rollout failed for {task}, seed {seed}, episode {episode_index} "
            f"with code {process.returncode}; see {stderr_path}:\n{detail}"
        )
    episode_result = load_json(result_path, "Isaac episode result")
    if not isinstance(episode_result.get("success"), bool):
        fail(f"Isaac episode result has invalid success value: {episode_result}")
    if episode_result.get("reward") is not None and not isinstance(
        episode_result.get("reward"), (int, float)
    ):
        fail(f"Isaac episode result has invalid reward value: {episode_result}")
    if not isinstance(episode_result.get("episode_length"), int):
        fail(f"Isaac episode result has invalid episode_length: {episode_result}")
    if not video_path.is_file() or video_path.stat().st_size == 0:
        fail(f"Isaac rollout did not produce a non-empty MP4: {video_path}")
    return episode_result


def _evaluator_readiness_result(
    *,
    actual: dict[str, Any],
    errors: list[str],
    registration_module: str,
) -> dict[str, Any]:
    required_checks = actual["checks"].values()
    registered_ids = actual["gym_registrations"][registration_module]["ids"].values()
    ready = bool(
        not errors
        and all(actual["imports"].values())
        and actual["gym_registrations"][registration_module]["module_imported"]
        and all(registered_ids)
        and actual["gpu"]["cuda_available"]
        and actual["gpu"]["device_count"] >= 1
        and all(required_checks)
    )
    return {
        "schema_version": "skynet.evaluator-readiness/v1",
        "ready": ready,
        "errors": list(errors),
        "actual": actual,
    }


def run_evaluator_readiness_smoke(
    *,
    output_path: Path,
    source_dir: Path,
    simulation_device: str,
) -> int:
    """Exercise the evaluator on allocated compute without granting legal consent."""

    expected_task_ids = [
        str(definition["environment_id"])
        for definition in TASK_DEFINITIONS.values()
    ]
    registration_module = "isaaclab_eval_tasks.tasks"
    import_names = [
        "isaacsim",
        "pinocchio",
        "isaaclab.app",
        "isaaclab",
        "isaaclab_assets",
        "isaaclab_tasks",
        "torch",
        "numpy",
        "gymnasium",
        "isaaclab_tasks.utils.parse_cfg",
        registration_module,
        "warp",
        "imageio",
        "imageio_ffmpeg",
    ]
    actual: dict[str, Any] = {
        "imports": {name: False for name in import_names},
        "gym_registrations": {
            registration_module: {
                "module_imported": False,
                "ids": {task_id: False for task_id in expected_task_ids},
            }
        },
        "gpu": {
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
        },
        "checks": {
            "environment_constructed": False,
            "environment_reset": False,
            "media_writer": False,
            "media_nonempty": False,
        },
    }
    errors: list[str] = []
    launcher_app: Any = None
    environment: Any = None
    media_writer: Any = None
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    media_path = output_path.with_name(f".{output_path.name}.imageio-smoke.mp4")

    try:
        if os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES":
            fail(
                "The readiness smoke requires externally supplied "
                "OMNI_KIT_ACCEPT_EULA=YES after the operator accepts the NVIDIA Isaac "
                "Sim EULA; this adapter never accepts or defaults legal consent"
            )
        if simulation_device != "cuda:0":
            fail(
                "The pinned GR00T IsaacLab readiness smoke requires "
                f"simulation_device=cuda:0, received {simulation_device!r}"
            )

        kit_extension_folder = resolve_omni_warp_extension_folder(Path(sys.prefix))

        evaltasks_python = source_dir.resolve() / "source" / "isaaclab_eval_tasks"
        if not evaltasks_python.is_dir():
            fail(
                "Pinned IsaacLabEvalTasks Python source is missing: "
                f"{evaltasks_python}"
            )
        sys.path.insert(0, str(evaltasks_python))

        import importlib

        # Import order is a legal runtime contract for the pinned GR1 tasks:
        # Isaac Sim must bootstrap first, then Isaac Lab's Pinocchio build must be
        # selected before AppLauncher constructs the Kit application.
        importlib.import_module("isaacsim")
        actual["imports"]["isaacsim"] = True
        importlib.import_module("pinocchio")
        actual["imports"]["pinocchio"] = True
        app_module = importlib.import_module("isaaclab.app")
        actual["imports"]["isaaclab.app"] = True

        app_launcher = getattr(app_module, "AppLauncher")

        def mark_launch_phase(phase: str) -> None:
            actual["launch_phase"] = phase
            write_json(
                output_path,
                {
                    "schema_version": "skynet.evaluator-readiness/v1",
                    "ready": False,
                    "errors": [f"evaluator readiness stopped during {phase}"],
                    "actual": actual,
                },
            )

        launcher, launcher_app = _launch_isaac_app(
            app_launcher,
            device=simulation_device,
            kit_args=build_isaac_kit_args(kit_extension_folder),
            phase_marker=mark_launch_phase,
            stage_api_loader=lambda: importlib.import_module("omni.usd"),
        )
        importlib.import_module("isaaclab")
        actual["imports"]["isaaclab"] = True
        importlib.import_module("isaaclab_assets")
        actual["imports"]["isaaclab_assets"] = True
        importlib.import_module("isaaclab_tasks")
        actual["imports"]["isaaclab_tasks"] = True
        torch = importlib.import_module("torch")
        actual["imports"]["torch"] = True
        numpy = importlib.import_module("numpy")
        actual["imports"]["numpy"] = True
        gym = importlib.import_module("gymnasium")
        actual["imports"]["gymnasium"] = True
        parse_cfg_module = importlib.import_module("isaaclab_tasks.utils.parse_cfg")
        actual["imports"]["isaaclab_tasks.utils.parse_cfg"] = True
        importlib.import_module(registration_module)
        actual["imports"][registration_module] = True
        actual["gym_registrations"][registration_module]["module_imported"] = True
        warp = importlib.import_module("warp")
        actual["imports"]["warp"] = True
        warp_version = str(warp.config.version)
        warp_path = Path(warp.__file__).resolve()
        expected_warp_root = kit_extension_folder / "omni.warp.core"
        actual["omni_warp"] = {
            "version": warp_version,
            "module_path": str(warp_path),
            "extension_folder": str(kit_extension_folder),
        }
        if warp_version != OMNI_WARP_VERSION:
            fail(
                f"Isaac Kit loaded Warp {warp_version}, expected pinned "
                f"{OMNI_WARP_VERSION}"
            )
        if not warp_path.is_relative_to(expected_warp_root):
            fail(
                f"Isaac Kit loaded Warp from {warp_path}, outside pinned extension "
                f"{expected_warp_root}"
            )
        imageio = importlib.import_module("imageio.v2")
        actual["imports"]["imageio"] = True
        importlib.import_module("imageio_ffmpeg")
        actual["imports"]["imageio_ffmpeg"] = True

        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        devices = [str(torch.cuda.get_device_name(index)) for index in range(device_count)]
        actual["gpu"] = {
            "cuda_available": cuda_available,
            "device_count": device_count,
            "devices": devices,
        }
        if not cuda_available or device_count < 1:
            fail("The readiness smoke must run inside a Slurm allocation with one CUDA GPU")

        for task_id in expected_task_ids:
            gym.spec(task_id)
            actual["gym_registrations"][registration_module]["ids"][task_id] = True

        readiness_task_id = expected_task_ids[0]
        parse_env_cfg = getattr(parse_cfg_module, "parse_env_cfg")
        environment_cfg = parse_env_cfg(
            readiness_task_id,
            device=simulation_device,
            num_envs=1,
        )
        environment = gym.make(readiness_task_id, cfg=environment_cfg).unwrapped
        actual["checks"]["environment_constructed"] = True
        if hasattr(environment, "seed"):
            environment.seed(0)
        environment.sim.reset()
        environment.reset(seed=0)
        actual["checks"]["environment_reset"] = True

        media_writer = imageio.get_writer(
            str(media_path),
            fps=2,
            format="FFMPEG",
            macro_block_size=None,
        )
        media_writer.append_data(numpy.zeros((64, 64, 3), dtype=numpy.uint8))
        media_writer.append_data(numpy.full((64, 64, 3), 255, dtype=numpy.uint8))
        media_writer.close()
        media_writer = None
        actual["checks"]["media_writer"] = True
        actual["checks"]["media_nonempty"] = media_path.is_file() and media_path.stat().st_size > 0
        if not actual["checks"]["media_nonempty"]:
            fail("ImageIO ffmpeg writer did not produce a non-empty MP4")
    except BaseException as exc:
        traceback_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        actual["traceback"] = traceback_text
        print(traceback_text, file=sys.stderr, flush=True)
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if media_writer is not None:
            try:
                media_writer.close()
            except BaseException as exc:
                errors.append(f"media writer close failed: {type(exc).__name__}: {exc}")
        if environment is not None:
            try:
                environment.close()
            except BaseException as exc:
                errors.append(f"environment close failed: {type(exc).__name__}: {exc}")

        try:
            media_path.unlink()
        except FileNotFoundError:
            pass

        # Isaac SimulationApp.close() may complete by terminating the native Kit
        # process directly. Persist and flush the authoritative result first so a
        # successful native shutdown cannot leave the earlier phase marker behind.
        result = _evaluator_readiness_result(
            actual=actual,
            errors=errors,
            registration_module=registration_module,
        )
        write_json(output_path, result)
        print(json.dumps(result, sort_keys=True), flush=True)

        if launcher_app is not None:
            try:
                launcher_app.close()
            except BaseException as exc:
                errors.append(f"Isaac application close failed: {type(exc).__name__}: {exc}")
                result = _evaluator_readiness_result(
                    actual=actual,
                    errors=errors,
                    registration_module=registration_module,
                )
                write_json(output_path, result)
                print(json.dumps(result, sort_keys=True), flush=True)

    return 0 if result["ready"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--readiness-smoke-output")
    parser.add_argument("--simulation-device", default="cuda:0")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    if args.readiness_smoke_output:
        if args.context:
            fail("--context and --readiness-smoke-output are mutually exclusive")
        return run_evaluator_readiness_smoke(
            output_path=Path(args.readiness_smoke_output),
            source_dir=source_dir,
            simulation_device=args.simulation_device,
        )
    if not args.context:
        fail("--context is required for evaluation mode")

    context_path = Path(args.context).resolve()
    context = load_json(context_path, "evaluation context")
    suite, task_options = validate_suite(context)
    runtime = validate_evaluator_runtime(context)
    checkpoint_path, checkpoint_sha = validate_policy(context, source_dir)

    requested_tasks = require(context, "tasks", "evaluation")
    if not isinstance(requested_tasks, list) or not requested_tasks or not all(
        isinstance(task, str) for task in requested_tasks
    ):
        fail("evaluation.tasks must be a non-empty list of exact task IDs")
    unknown = [task for task in requested_tasks if task not in TASK_DEFINITIONS]
    if unknown:
        fail(f"Requested tasks are absent from the pinned catalog: {unknown}")
    episodes_per_task = int(require(context, "episodes_per_task", "evaluation"))
    seeds_value = require(context, "seeds", "evaluation")
    if episodes_per_task < 1 or not isinstance(seeds_value, list) or not seeds_value:
        fail("episodes_per_task and seeds must define at least one rollout")
    try:
        seeds = [int(seed) for seed in seeds_value]
    except (TypeError, ValueError) as exc:
        fail(f"evaluation.seeds must contain integers: {exc}")

    result_path = Path(str(require(context, "result_path", "evaluation"))).resolve()
    progress_path = Path(str(require(context, "progress_path", "evaluation"))).resolve()
    video_root = Path(str(require(context, "video_path", "evaluation"))).resolve()
    run_root = result_path.parent
    log_root = run_root / "logs" / f"adapter-attempt-{os.environ.get('SLURM_JOB_ID', 'local')}"
    scratch_root = run_root / "runtime" / f"adapter-attempt-{os.environ.get('SLURM_JOB_ID', 'local')}"
    log_root.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)

    evaluator = require_object(context, "evaluator", "evaluation")
    if evaluator.get("adapter") != ENVIRONMENT or not isinstance(evaluator.get("version"), str):
        fail("Evaluation context does not contain the pinned isaac_sim evaluator version")
    identity_document = {
        "run_id": context.get("run_id"),
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": require_object(context, "policy", "evaluation").get("manifest_sha256"),
        "suite_version": suite.get("version"),
        "suite_catalog_sha256": suite.get("catalog_sha256"),
        "task_catalog_sha256": TASK_CATALOG_HASH,
        "evaluator_runtime_profile": runtime.get("profile"),
        "evaluator_runtime_snapshot_sha256": runtime.get("profile_snapshot_sha256"),
        "evaluator_source_revision": EVALTASKS_REVISION,
        "tasks": requested_tasks,
        "episodes_per_task": episodes_per_task,
        "seeds": seeds,
    }
    identity = hashlib.sha256(
        json.dumps(identity_document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    completed = load_completed(progress_path, identity)

    policy_port = allocate_port()
    policy_stdout_path = log_root / "policy-server.stdout.log"
    policy_stderr_path = log_root / "policy-server.stderr.log"
    policy_stdout = policy_stdout_path.open("w", encoding="utf-8")
    policy_stderr = policy_stderr_path.open("w", encoding="utf-8")
    policy_process = subprocess.Popen(
        build_policy_server_command(source_dir, checkpoint_path, policy_port),
        cwd=source_dir,
        env=os.environ.copy(),
        text=True,
        stdout=policy_stdout,
        stderr=policy_stderr,
        start_new_session=True,
    )
    proxy: PolicyProxy | None = None
    try:
        wait_for_policy_server(policy_process, policy_port, policy_stderr_path)
        probe_policy_contract(policy_port)
        proxy = PolicyProxy(policy_port)
        for task in requested_tasks:
            metadata = task_options[task]["metadata"]["native"]
            for seed in seeds:
                for episode_index in range(episodes_per_task):
                    key = (task, seed, episode_index)
                    if key in completed:
                        continue
                    effective_seed = seed * 1_000_003 + episode_index
                    slug = f"{safe_name(task)}-seed-{seed}-episode-{episode_index:04d}"
                    episode_result_path = scratch_root / f"{slug}.json"
                    episode_video_path = video_root / f"{slug}.mp4"
                    stdout_path = log_root / f"{slug}.stdout.log"
                    stderr_path = log_root / f"{slug}.stderr.log"
                    try:
                        child_result = run_isaac_episode(
                            runtime=runtime,
                            proxy=proxy,
                            task=task,
                            task_definition=metadata,
                            seed=effective_seed,
                            episode_index=episode_index,
                            result_path=episode_result_path,
                            video_path=episode_video_path,
                            stdout_path=stdout_path,
                            stderr_path=stderr_path,
                        )
                    except Exception as exc:
                        append_jsonl(
                            progress_path,
                            {
                                "record_type": "episode",
                                "identity": identity,
                                "status": "FAILED",
                                "task": task,
                                "seed": seed,
                                "episode_index": episode_index,
                                "failure_reason": str(exc),
                            },
                        )
                        raise
                    episode = {
                        "task": task,
                        "seed": seed,
                        "episode_index": episode_index,
                        "success": child_result["success"],
                        "reward": child_result["reward"],
                        "episode_length": child_result["episode_length"],
                        "status": "SUCCEEDED",
                        "metrics": {"effective_seed": float(effective_seed)},
                        "video_path": str(episode_video_path),
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
                        },
                    )
                    completed[key] = episode
    finally:
        if proxy is not None:
            proxy.close()
        terminate(policy_process)
        policy_stdout.close()
        policy_stderr.close()

    expected_count = len(requested_tasks) * len(seeds) * episodes_per_task
    episodes = [
        completed[(task, seed, episode_index)]
        for task in requested_tasks
        for seed in seeds
        for episode_index in range(episodes_per_task)
    ]
    if len(episodes) != expected_count:
        fail(f"Evaluation ledger is incomplete: expected {expected_count}, found {len(episodes)}")

    aggregate: list[dict[str, Any]] = []
    all_values = [1.0 if episode["success"] else 0.0 for episode in episodes]
    aggregate.append(
        {
            "metric": "success_rate",
            "unit": "fraction",
            "mean": statistics.fmean(all_values),
            "std": statistics.pstdev(all_values) if len(all_values) > 1 else 0.0,
            "sample_count": len(all_values),
            "task": None,
        }
    )
    for task in requested_tasks:
        values = [
            1.0 if episode["success"] else 0.0
            for episode in episodes
            if episode["task"] == task
        ]
        aggregate.append(
            {
                "metric": "success_rate",
                "unit": "fraction",
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "sample_count": len(values),
                "task": task,
            }
        )
    artifacts = sorted(
        {
            str(policy_stdout_path),
            str(policy_stderr_path),
            *(str(path) for path in log_root.glob("*.log")),
            *(str(path) for path in video_root.glob("*.mp4")),
        }
    )
    result = {
        "schema_version": 1,
        "run_id": context["run_id"],
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "evaluator": {"adapter": ENVIRONMENT, "version": evaluator["version"]},
        "environment": {"suite": SUITE, "version": suite["version"]},
        "aggregate": aggregate,
        "episodes": episodes,
        "raw_metrics_path": str(progress_path),
        "artifacts": artifacts,
    }
    write_json(result_path, result)
    print(
        json.dumps(
            {
                "result_path": str(result_path),
                "episodes": expected_count,
                "success_rate": aggregate[0]["mean"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"GR00T Isaac Lab evaluation failed: {exc}", file=sys.stderr)
        raise


GROOT_ISAACLAB_BRIDGE_SOURCE = Path(__file__).read_text(encoding="utf-8")
