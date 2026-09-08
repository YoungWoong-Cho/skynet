"""Convert immutable live episodes using DexVerse's observation and HDF5 APIs.

Restore states, never advance physics or change recorded actions. Contact forces
are deliberately excluded: scene snapshots cannot reconstruct contact impulses.
"""

import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import traceback

import numpy as np

FORMAT = "dexverse-demo-hdf5/v1"


def file_info(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "size_bytes": Path(path).stat().st_size}


def load_recording(source, profile, unpickler):
    path = Path(source["path"])
    if not 0 < path.stat().st_size <= 100 * 1024 * 1024:
        raise ValueError("Recording is empty or exceeds 100 MB")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("Recording checksum changed: " + path.name)
    payload = unpickler(io.BytesIO(raw)).load()
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "dexverse_trajectory"
        or payload.get("schema_version") != 3
    ):
        raise ValueError(
            "Unsupported recording format; expected DexVerse trajectory v3"
        )
    if (payload.get("task"), payload.get("robot_type")) != (
        profile["task"],
        profile["robot"],
    ):
        raise ValueError("Recording hand/task differs from the collection session")
    episodes = payload.get("episodes")
    if (
        not isinstance(episodes, list)
        or not episodes
        or payload.get("num_episodes") != len(episodes)
    ):
        raise ValueError("Recording contains no complete episodes")
    for ep in episodes:
        actions = np.asarray(ep.get("actions"))
        if (
            actions.ndim != 2
            or not all(actions.shape)
            or actions.dtype.kind not in "fiu"
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Recording contains invalid actions")
        states = ep.get("states")
        if (
            ep.get("num_steps") != len(actions)
            or not isinstance(states, list)
            or len(states) != len(actions) + 1
            or ep.get("success") is not True
        ):
            raise ValueError(
                "Recording needs a successful episode with T actions and T+1 states"
            )
        for state in states:
            if not isinstance(state, dict) or not state.get("articulation"):
                raise ValueError("Recording is missing robot scene state")
            validate_state(state)
    return payload


def validate_state(value, depth=0):
    if depth > 12:
        raise ValueError("Scene state nesting is too deep")
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("Invalid scene state keys")
        for child in value.values():
            validate_state(child, depth + 1)
    else:
        a = np.asarray(value)
        if a.dtype.kind not in "fiub" or not np.isfinite(a).all():
            raise ValueError("Nonfinite or nonnumeric scene state")


def snapshot(capture):
    # GPU .numpy() buffers can alias simulator state; freeze every frame now.
    result = {k: np.asarray(v).copy() for k, v in capture.capture(0).items()}
    if not result or any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError("Simulator produced invalid observations")
    return result


def validate_hdf5(path, episodes, observation_shapes):
    """Reject truncation, changed controls, and off-by-one observation exports."""
    import h5py

    with h5py.File(path, "r") as h5:
        if len(h5["data"]) != len(episodes):
            raise ValueError("HDF5 episode count is incomplete")
        for index, ep in enumerate(episodes):
            g = h5[f"data/demo_{index}"]
            if not np.array_equal(
                g["actions"][:], np.asarray(ep["actions"], dtype=np.float32)
            ):
                raise ValueError("Converted action values changed")
            for key, shape in observation_shapes.items():
                a, b = g["obs/" + key][:], g["next_obs/" + key][:]
                expected = (ep["num_steps"], *shape)
                if (
                    a.shape != expected
                    or b.shape != expected
                    or not np.isfinite(a).all()
                    or not np.isfinite(b).all()
                    or not np.array_equal(a[1:], b[:-1])
                    or not np.array_equal(a[0], g["initial_obs/" + key][:])
                    or not np.array_equal(b[-1], g["final_obs/" + key][:])
                ):
                    raise ValueError(
                        "Converted observations are incomplete or misaligned"
                    )


def convert(request):
    from arrays import ArrayUnpickler

    root = Path(request["root"])
    profile, sources = request["profile"], request["sources"]
    status_path = root / "status.json"

    def publish(**value):
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(value))
        temp.replace(status_path)

    app = env = writer = None
    output = root / "dataset.hdf5"
    partial = root / "dataset.partial.hdf5"
    try:
        publish(state="RUNNING", detail="Checking saved recordings…")
        payloads = [load_recording(s, profile, ArrayUnpickler) for s in sources]
        episodes = [
            (s, p, ep) for s, p in zip(sources, payloads) for ep in p["episodes"]
        ]
        total_steps = sum(ep["num_steps"] for _, _, ep in episodes)
        publish(
            state="RUNNING",
            detail="Loading recorded hand and task…",
            total=len(episodes),
            completed=0,
        )
        converter = (
            Path(profile["repository"])
            / "scripts/demo_tools/create_demo_files_sequential.py"
        )
        revision = subprocess.check_output(
            ["git", "-C", profile["repository"], "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != profile["source_revision"]:
            raise ValueError("DexVerse source revision differs from the recording")
        sys.path.insert(0, str(converter.parent))
        sys.argv = [
            str(converter),
            "--obs-groups",
            "state",
            "--set-state",
            "--enable-pinocchio",
            "--headless",
            "--device",
            "cuda:0",
        ]
        ns = runpy.run_path(str(converter), run_name="skynet_dataset_converter")
        app = ns["simulation_app"]
        import gymnasium as gym
        import torch
        from dexverse.tasks.utils import (
            parse_env_cfg,
            strip_camera_cfgs,
            prune_stale_obs_refs,
        )
        from wrist import configure_virtual_wrist

        manifest = None
        if profile.get("hand_bundle"):
            sys.path.insert(0, profile["hand_bundle"]["root"])
            from runtime import install, validate_environment

            manifest = install(Path(profile["hand_bundle"]["root"]))
        template = parse_env_cfg(profile["task"], device="cuda:0", num_envs=1)
        cfg = type(template)(robot_type=profile["robot"])
        cfg.scene.num_envs = 1
        cfg.sim.device = "cuda:0"
        cfg.env_name = profile["task"]
        cfg._apply_observation_preset("state")
        cfg.observations.contact = None
        cfg.observations.debug_vis = None
        cfg = prune_stale_obs_refs(strip_camera_cfgs(cfg))
        if ns["_has_multi_asset_or_usd"](cfg.scene):
            raise ValueError(
                "Conversion of randomized multi-asset scenes is unsupported; no asset substitution was made"
            )
        cfg.recorders, cfg.terminations = {}, {}
        env = gym.make(profile["task"], cfg=cfg).unwrapped
        env.reset()
        if manifest:
            validate_environment(env, manifest)
        sides = ["left", "right"] if profile["hand"] == "both" else [profile["hand"]]
        configure_virtual_wrist(env.scene["robot"], manifest, sides)
        groups = list(env.observation_manager.active_terms)
        if not {"proprio", "state"}.issubset(groups) or set(groups) - {
            "proprio",
            "state",
            "policy",
            "goal",
        }:
            raise ValueError("Task does not expose the supported state observations")
        capture = ns["ObsGroupCapture"](env, groups, "uint8", "float32")
        writer = ns["PerPickleH5Writer"](
            partial,
            task_name=profile["task"],
            source_pickles=[Path(s["path"]) for s in sources],
            obs_groups=groups,
            rgb_dtype="uint8",
            depth_dtype="float32",
            compression="gzip",
            compression_opts=4,
            observation_preset="state",
        )
        observation_shapes, completed_steps = None, 0
        episode_manifest = []
        with torch.inference_mode():
            for index, (source, payload, ep) in enumerate(episodes):
                actions = np.asarray(ep["actions"], dtype=np.float32)
                if actions.shape[1] != env.action_space.shape[-1]:
                    raise ValueError(
                        "Saved action dimensions differ from the original hand"
                    )
                env.reset()
                if "goal" in groups:
                    goal = np.asarray(ep.get("goal_pose"), dtype=np.float32)
                    command = env.command_manager.get_command("object_pose")
                    if (
                        goal.shape != tuple(command.shape[1:])
                        or not np.isfinite(goal).all()
                    ):
                        raise ValueError(
                            "Recording is missing its task goal; a new random goal cannot be substituted"
                        )
                    command.copy_(torch.as_tensor(goal, device=env.device).unsqueeze(0))
                env.action_manager.action.zero_()

                def restore(state):
                    live = env.scene.get_state(is_relative=True)
                    if set(state) != set(live) or any(
                        set(state[k]) != set(live[k]) for k in live
                    ):
                        raise ValueError(
                            "Saved scene entities differ from the simulator; conversion stopped"
                        )
                    env.scene.reset_to(
                        ns["_tensorize_state"](state, env.device), is_relative=True
                    )
                    ns["_refresh_after_set_state"](env)

                restore(ep["states"][0])
                initial = previous = snapshot(capture)
                shapes = {k: list(v.shape) for k, v in initial.items()}
                if observation_shapes is not None and observation_shapes != shapes:
                    raise ValueError("Observation shapes changed between episodes")
                observation_shapes = shapes
                obs, next_obs = {k: [] for k in shapes}, {k: [] for k in shapes}
                for step, action in enumerate(actions):
                    if not app.is_running() or app.is_exiting():
                        raise ValueError("Simulator exited before conversion finished")
                    # last_action observations must track the saved controls,
                    # even though reset_to intentionally does not call env.step.
                    env.action_manager.action.copy_(
                        torch.as_tensor(action, device=env.device).unsqueeze(0)
                    )
                    restore(ep["states"][step + 1])
                    current = snapshot(capture)
                    for key in shapes:
                        obs[key].append(previous[key])
                        next_obs[key].append(current[key])
                    previous = current
                    if step % 60 == 0:
                        publish(
                            state="RUNNING",
                            detail=f"Converting episode {index + 1} of {len(episodes)}…",
                            completed=index,
                            total=len(episodes),
                            steps=completed_steps + step,
                            total_steps=total_steps,
                        )
                writer.write_episode(
                    episode_index=index,
                    episode_name=f"demo_{index}",
                    success=True,
                    actions=list(actions),
                    source_actions=list(actions),
                    obs=obs,
                    next_obs=next_obs,
                    initial_obs=initial,
                    final_obs=previous,
                )
                g = writer._data[f"demo_{index}"]
                g.attrs["source_sha256"] = source["sha256"]
                g.attrs["source_recording_index"] = source["index"]
                g.attrs["retargeting"] = json.dumps(
                    ep.get("skynet_retargeting", {"status": "not recorded"})
                )
                g.create_dataset(
                    "timestamps", data=np.arange(len(actions), dtype=np.float64) / 60
                )
                episode_manifest.append(
                    dict(
                        episode=index,
                        recording_index=source["index"],
                        sha256=source["sha256"],
                        steps=len(actions),
                    )
                )
                completed_steps += len(actions)
        action_terms = {}
        for name in env.action_manager.active_terms:
            term = env.action_manager.get_term(name)
            names = getattr(term, "_joint_names", None)
            if names is None:
                raise ValueError("Action term does not expose its joint order: " + name)
            action_terms[name] = list(names)
        metadata = dict(
            format=FORMAT,
            task=profile["task"],
            robot=profile["robot"],
            hand=profile["hand"],
            episodes=len(episodes),
            steps=completed_steps,
            action_dim=actions.shape[1],
            action_terms=action_terms,
            joint_names=env.scene["robot"].joint_names,
            observation_shapes=observation_shapes,
            observation_groups=groups,
            excluded_observations={
                "contact": "Contact impulses are not saved in scene snapshots",
                "images": "State-only conversion",
            },
            simulation_hz=60,
            replay="saved_states",
            source_revision=profile["source_revision"],
            converter_sha256=request["converter_sha256"],
            session_id=request["session_id"],
            sources=sources,
            episode_sources=episode_manifest,
        )
        writer._h5.attrs["skynet_metadata"] = json.dumps(metadata)
        writer._h5.attrs["robot_type"] = profile["robot"]
        writer.flush()
        writer = None
        # Reopen the artifact and verify every episode before publishing it.
        validate_hdf5(partial, [ep for _, _, ep in episodes], observation_shapes)
        partial.replace(output)
        metadata["artifact"] = file_info(output)
        (root / "manifest.json").write_text(json.dumps(metadata, indent=2))
        publish(
            state="READY",
            completed=len(episodes),
            total=len(episodes),
            metadata=metadata,
            manifest=file_info(root / "manifest.json"),
        )
    except BaseException as exc:
        traceback.print_exc()
        publish(state="FAILED", error=str(exc) or type(exc).__name__)
    finally:
        if writer:
            writer._h5.close()
        if env:
            env.close()
        if app:
            app.close()


if __name__ == "__main__":
    convert(json.loads(Path(sys.argv[1]).read_text()))
