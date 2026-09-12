"""Isaac-side rollout; shared by recorded-data DP and ACT policies."""

import argparse
import hashlib
import inspect
import json
import math
from multiprocessing.connection import Client
import os
from pathlib import Path
import statistics
import traceback

from xpolicy_runtime import write_json
from policy_transport import receive_message, send_message


def identity(context):
    selected = {
        k: context.get(k)
        for k in [
            "run_id",
            "checkpoint",
            "suite",
            "tasks",
            "seeds",
            "episodes_per_task",
            "policy",
            "episode_assignments",
            "recorded_episode_sources",
        ]
    }
    return hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def observed_episodes(path, expected_identity):
    records = {}
    if Path(path).exists():
        lines = Path(path).read_text().splitlines(keepends=True)
        for line in lines:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            if row.get("identity") != expected_identity:
                raise ValueError(
                    "Evaluation progress belongs to a different checkpoint or configuration"
                )
            if row.get("status") in {"RUNNING", "SUCCEEDED"}:
                ep = row["episode"]
                key = (ep["task"], ep["seed"], ep["episode_index"])
                if (
                    key in records
                    and records[key].get("status", "SUCCEEDED") == "SUCCEEDED"
                ) or (
                    row["status"] == "SUCCEEDED"
                    and not Path(ep["video_path"]).is_file()
                ):
                    raise ValueError(
                        "Invalid completed episode or missing rollout video"
                    )
                records[key] = ep
    return records


def completed_episodes(path, expected_identity):
    return {
        key: ep
        for key, ep in observed_episodes(path, expected_identity).items()
        if ep.get("status", "SUCCEEDED") == "SUCCEEDED"
    }


def append_progress(path, row):
    # A preempted process may leave half a JSON record. Repair only that tail.
    with path.open("a+b") as stream:
        end = stream.tell()
        if end:
            stream.seek(end - 1)
            if stream.read(1) != b"\n":
                position = end
                while position:
                    start = max(0, position - 4096)
                    stream.seek(start)
                    chunk = stream.read(position - start)
                    newline = chunk.rfind(b"\n")
                    if newline >= 0:
                        position = start + newline + 1
                        break
                    position = start
                stream.truncate(position)
        stream.write((json.dumps(row, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(row), flush=True)


def validate_layout(env, capture, order, contract=None):
    import numpy as np

    names = []
    scales = []
    offsets = []
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        if (
            type(term).__name__ != "JointPositionAction"
            or term.cfg.asset_name != "robot"
        ):
            raise ValueError("Evaluation requires named robot joint-position actions")
        names.extend(term._joint_names)
        for key, target in [("_scale", scales), ("_offset", offsets)]:
            value = getattr(term, key)
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            target.extend(
                np.broadcast_to(value, (1, len(term._joint_names)))[0].tolist()
            )
    if contract is not None:
        from policy_contract import environment_mapping
        mapped = environment_mapping(contract, names, scales, offsets, env.step_dt)
        return [env.scene["robot"].joint_names.index(names[i]) for i in mapped], mapped
    if (
        names != capture["action_joint_names"]
        or not np.allclose(scales, capture["action_scale"])
        or not np.allclose(offsets, capture["action_offset"])
    ):
        raise ValueError(
            "Simulator actions differ from collection joint order, scale, or offset"
        )
    if not np.isclose(env.step_dt, capture["step_dt"]):
        raise ValueError("Simulator control frequency differs from collection")
    return [env.scene["robot"].joint_names.index(names[i]) for i in order]


def assigned_episodes(context):
    expected = {(seed, index) for seed in context["seeds"]
                for index in range(context["episodes_per_task"])}
    selected = context.get("episode_assignments")
    if selected is None:
        return expected
    assigned = {tuple(pair) for pair in selected}
    if len(assigned) != len(selected) or not assigned.issubset(expected):
        raise ValueError("Worker episode assignments do not match the evaluation")
    return assigned


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context", required=True)
    args = p.parse_args()
    context = json.loads(Path(args.context).read_text())
    manifest = json.loads(
        (
            Path(context["policy"]["native_config"]["dataset_path"]) / "manifest.json"
        ).read_text()
    )
    capture = manifest["capture"]
    sources = context.get("recorded_episode_sources") or []
    initial_state = None
    if context["suite"]["config"].get("initial_state") == "single_training_episode":
        if len(sources) != 1 or context["episodes_per_task"] != 1 or context["tasks"] != [capture["task"]]:
            raise ValueError("Recorded initial-state evaluation must use its single training episode and task")
        from recorded_scene import load_initial_state
        initial_state = load_initial_state(sources[0], capture)
    task = context["tasks"][0]
    order = manifest["policy_to_source_indices"]
    import pinocchio  # Load before Isaac's plugins, as in collection.
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        multi_gpu=False,
        headless=context["headless"],
        enable_cameras=True,
        device="cuda:0",
        # Isolate writable Kit caches/configuration between concurrent simulators.
        kit_args=(
            "--portable-root "
            + str(Path(context["result_path"]).parent / "kit")
            + " --/rtx/verifyDriverVersion/enabled=false"
        ),
    )
    app = launcher.app
    env = None
    try:
        import numpy as np
        import torch
        import gymnasium as gym
        import imageio.v2 as imageio
        from evaluation_video import compose_camera_views
        import dexverse.tasks
        from dexverse.tasks.utils import parse_env_cfg, prune_stale_obs_refs
        from isaaclab.managers import ManagerTermBase, TerminationTermCfg
        from images import (
            configure_cameras,
            camera_recipe,
            training_image_request,
            CAMERAS,
        )
        from wrist import configure_virtual_wrist
        import omni.usd

        if omni.usd.get_context().get_stage() is None:
            omni.usd.get_context().new_stage()
        cfg = parse_env_cfg(task, device="cuda:0", num_envs=1)
        cfg = type(cfg)(robot_type=capture["robot"], enable_debug_vis=False)
        cfg.sim.device = "cuda:0"
        cfg.scene.num_envs = 1
        cfg.env_name = task
        cfg.seed = context["seeds"][0]
        if ("TopDownGrasp" in task or "Lift" in task) and hasattr(
            getattr(cfg, "commands", None), "object_pose"
        ):
            cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        cfg.observations.policy.concatenate_terms = False
        recipe = camera_recipe(cfg)
        if (
            task == capture["task"]
            and os.environ["SKYNET_POLICY_IMAGES"] == "1"
            and training_image_request(recipe)["recipe_sha256"]
            != capture.get("image_recipe_sha256")
        ):
            raise ValueError("Evaluation cameras differ from recorded training images")
        configure_cameras(cfg)
        cfg = prune_stale_obs_refs(cfg)
        cfg.observations.contact = None
        cfg.recorders = {}
        success_term = cfg.terminations.success
        if success_term is None:
            raise ValueError("Task has no success criterion")
        # Keep failure terminations but require ten consecutive successful steps.
        cfg.terminations.success = None
        for name in dir(cfg.terminations):
            term = getattr(cfg.terminations, name)
            if isinstance(term, TerminationTermCfg) and term.time_out:
                setattr(cfg.terminations, name, None)
        env = gym.make(task, cfg=cfg).unwrapped
        env.reset()
        configure_virtual_wrist(
            env.scene["robot"],
            None,
            ["left", "right"] if capture["hand"] == "both" else [capture["hand"]],
        )
        contract = context.get("compatibility", {}).get("io_contract")
        if contract:
            ids, order = validate_layout(env, capture, order, contract)
        else:
            ids = validate_layout(env, capture, order)
        success_fn = success_term.func
        if inspect.isclass(success_fn) and issubclass(success_fn, ManagerTermBase):
            success_fn = success_fn(success_term, env)
        max_steps = math.ceil(cfg.episode_length_s / env.step_dt)
        root = Path(context["result_path"]).parent
        root.mkdir(parents=True, exist_ok=True)
        video_root = Path(context["video_path"])
        video_root.mkdir(parents=True, exist_ok=True)
        progress = Path(context["progress_path"])
        fingerprint = identity(context)
        completed = completed_episodes(progress, fingerprint)
        with Client(
            ("127.0.0.1", int(os.environ["SKYNET_POLICY_PORT"])),
            authkey=bytes.fromhex(os.environ["SKYNET_POLICY_AUTH"]),
        ) as conn:

            def request(value):
                send_message(conn, value)
                if not conn.poll(300):
                    raise TimeoutError("Policy did not answer within five minutes")
                response = receive_message(conn)
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response["result"]

            def observation():
                env.sim.render()
                images = {}
                for scene, sensor_name in CAMERAS.items():
                    sensor = env.scene[sensor_name]
                    sensor.update(0.0, force_recompute=True)
                    images[scene] = sensor.data.output["rgb"][0, :, :, :3].detach().cpu().numpy().copy()
                state = env.scene["robot"].data.joint_pos[0, ids].detach().cpu().numpy().copy()
                return {"state": state, "images": images}

            def advance(packed):
                action = np.empty(len(order), dtype=np.float32)
                action[order] = packed
                return env.step(torch.as_tensor(action, device=env.device)[None])

            if contract:
                from evaluation_preflight import verify_cycle
                first_seed = context["seeds"][0] * 1000003

                def reset_probe():
                    env.reset(seed=first_seed)
                    if initial_state is not None:
                        from recorded_scene import restore_state
                        restore_state(env, initial_state)
                    if isinstance(success_fn, ManagerTermBase):
                        success_fn.reset()
                    request({"command": "reset", "seed": first_seed})

                def report_probe(value):
                    receipt = {**value, "checkpoint_sha256": context["checkpoint"]["sha256"],
                               "implementation_sha256": context["compatibility"]["implementation_sha256"],
                               "node": os.environ.get("SLURMD_NODENAME"), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
                    write_json(root / "preflight.json", receipt)
                    print(json.dumps({"event": "evaluation_preflight", **receipt}), flush=True)

                with torch.no_grad():
                    verify_cycle(contract, reset=reset_probe, observe=observation,
                                 predict=lambda obs: request({"command": "step", "observation": obs, "predict": True}),
                                 advance=advance, report=report_probe)

            assignments = assigned_episodes(context)
            for seed in context["seeds"]:
                for index in range(context["episodes_per_task"]):
                    if (seed, index) not in assignments:
                        continue
                    key = (task, seed, index)
                    if key in completed:
                        continue
                    effective_seed = seed * 1000003 + index
                    worker_metrics = {
                        "worker_index": float(context.get("worker_index", 0)),
                        "slurm_job_id": float(os.environ.get("SLURM_JOB_ID", 0)),
                    }
                    append_progress(
                        progress,
                        dict(
                            identity=fingerprint,
                            status="RUNNING",
                            episode=dict(
                                task=task,
                                seed=seed,
                                episode_index=index,
                                status="RUNNING",
                                metrics=worker_metrics,
                            ),
                        ),
                    )
                    env.reset(seed=effective_seed)
                    if initial_state is not None:
                        from recorded_scene import restore_state
                        restore_state(env, initial_state)
                    if isinstance(success_fn, ManagerTermBase):
                        success_fn.reset()
                    request({"command": "reset", "seed": effective_seed})
                    video = video_root / f"episode-seed-{seed}-{index:04d}.mp4"
                    from episode_trace import EpisodeTrace
                    trace_cameras = CAMERAS if os.environ["SKYNET_POLICY_IMAGES"] == "1" else {"scene_front": CAMERAS["scene_front"]}
                    observation()  # Refresh camera poses after restoring the episode.
                    trace_context = context if task == capture["task"] else {**context, "recorded_episode_sources": []}
                    trace = EpisodeTrace(trace_context, capture, env, trace_cameras, video, policy_order=order)
                    pending = []
                    success = False
                    streak = 0
                    reward_sum = 0.0
                    reason = "time_limit"
                    # Simulator buffers must remain mutable for the next reset.
                    # Inference mode is confined to the separate policy process.
                    with (
                        imageio.get_writer(
                            video,
                            fps=1 / (2 * env.step_dt),
                            codec="libx264",
                            macro_block_size=1,
                        ) as writer,
                        torch.no_grad(),
                    ):
                        for step in range(max_steps):
                            obs = observation()
                            images = obs["images"]
                            if step % 2 == 0:
                                video_views = (images if os.environ["SKYNET_POLICY_IMAGES"] == "1"
                                               else {"scene_front": images["scene_front"]})
                                writer.append_data(compose_camera_views(video_views))
                            if contract:
                                from policy_contract import validate_observation, validate_actions
                                validate_observation(contract, obs)
                            predicted = request(
                                {
                                    "command": "step",
                                    "observation": {
                                        "state": obs["state"],
                                        "images": (
                                            images
                                            if os.environ["SKYNET_POLICY_IMAGES"] == "1"
                                            else {}
                                        ),
                                    },
                                    "predict": not pending,
                                }
                            )
                            if not pending:
                                if contract:
                                    validate_actions(contract, predicted)
                                pending = list(predicted)
                            packed = pending.pop(0)
                            if step % 2 == 0:
                                trace.append(step, packed)
                            _, reward, terminated, truncated, _ = advance(packed)
                            reward_sum += float(reward[0])
                            if bool(terminated[0]) or bool(truncated[0]):
                                reason = "task_termination"
                                break
                            hit = bool(success_fn(env, **success_term.params)[0])
                            streak = streak + 1 if hit else 0
                            if streak >= 10:
                                success = True
                                reason = "success"
                                break
                            if not app.is_running():
                                raise RuntimeError(
                                    "Simulator closed before the episode ended"
                                )
                    trace.finish(step + 1)
                    episode = dict(
                        task=task,
                        seed=seed,
                        episode_index=index,
                        success=success,
                        reward=reward_sum,
                        episode_length=step + 1,
                        status="SUCCEEDED",
                        metrics={
                            "effective_seed": float(effective_seed),
                            "success_hold_steps": float(streak),
                            **worker_metrics,
                        },
                        video_path=str(video),
                        failure_reason=None,
                    )
                    row = dict(
                        record_type="episode",
                        identity=fingerprint,
                        status="SUCCEEDED",
                        task=task,
                        seed=seed,
                        episode_index=index,
                        episode=episode,
                        termination_reason=reason,
                    )
                    append_progress(progress, row)
                    completed[key] = episode
        episodes = [
            completed[(task, s, i)]
            for s in context["seeds"]
            for i in range(context["episodes_per_task"])
            if (task, s, i) in completed
        ]
        values = [float(e["success"]) for e in episodes]
        aggregate = [
            dict(
                metric="success_rate",
                unit="fraction",
                mean=statistics.fmean(values),
                std=statistics.pstdev(values),
                sample_count=len(values),
                task=t,
            )
            for t in [None, task]
        ]
        write_json(
            context["result_path"],
            dict(
                schema_version=1,
                run_id=context["run_id"],
                checkpoint={k: context["checkpoint"][k] for k in ["path", "sha256"]},
                evaluator=context["evaluator"],
                environment={
                    "suite": context["suite"]["name"],
                    "version": context["suite"]["version"],
                },
                aggregate=aggregate,
                episodes=episodes,
                raw_metrics_path=str(progress),
                artifacts=[e["video_path"] for e in episodes],
            ),
        )
        Path(os.environ["SKYNET_SIMULATOR_COMPLETION"]).touch()
    except BaseException:
        # Kit's shutdown can terminate the interpreter before Python prints it.
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
