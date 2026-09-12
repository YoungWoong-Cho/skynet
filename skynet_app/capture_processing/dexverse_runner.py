"""Standalone GPU worker: saved tracking -> DexVerse -> state BC -> rollout.

This is a Skynet state-based behavior-cloning baseline, not a GR00T or pi policy.
The HDF5 observation/action contract, checkpoint and evaluation all share the same
ordered keys. Upstream task success is measured, never inferred from job exit.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

TASK = "Dexverse-PickUpStick-v0"
ROBOT = "skynet_shadow_right"
REVISION = "30cc673e27684b9f10186fa6bea731aed246bc9f"
SCHEMA = "skynet.dexverse-state-actions/v1"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking", required=True)
    parser.add_argument("--hand-bundle-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--probe", action="store_true")
    # Pinocchio must precede the simulator's plugin/library loading.
    import pinocchio  # noqa: F401
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 1000 or not 1 <= args.eval_episodes <= 20:
        raise ValueError("Epochs must be 1–1000 and evaluation episodes 1–20")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise ValueError(
            "This output already contains a completed cycle; use a new output directory"
        )
    stages = {}

    def stage(name, status, **details):
        stages[name] = {"status": status, **details}
        save_json(out / "progress.json", {"stages": stages, "updated_at": time.time()})
        print("SKYNET_STAGE " + json.dumps({"stage": name, **stages[name]}), flush=True)

    app = None
    env = None
    current = "simulation"
    try:
        track = json.loads(Path(args.tracking).read_text())
        if (
            track.get("schema") != "skynet.dexverse-tracking/v1"
            or track.get("source_sha256") != args.source_sha256
        ):
            raise ValueError(
                "Converted tracking schema or original capture checksum mismatch"
            )
        if track.get("hand") != "right" or track.get("fps") != 60:
            raise ValueError(
                "This pipeline supports Shadow right-hand recordings at 60 Hz"
            )
        stage(
            current,
            "RUNNING",
            detail="Starting Isaac Sim; first launch compiles GPU shaders",
        )
        app = AppLauncher(args).app
        import numpy as np
        import torch
        import gymnasium as gym
        import h5py
        import imageio.v2 as imageio
        import dexverse.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab.managers import TerminationTermCfg
        from isaaclab.devices.device_base import DeviceBase
        from dexverse.devices.retargeters.simple_relative_retargeting import (
            SimpleRelativeRetargeter,
            SimpleRelativeRetargeterCfg,
        )

        import runpy

        adapter = runpy.run_path(str(Path(args.hand_bundle_root) / "runtime.py"))
        manifest = adapter["install"](args.hand_bundle_root)
        if manifest["robot"] != ROBOT:
            raise ValueError("Prepared hand differs from the capture pipeline")
        cfg = type(parse_env_cfg(TASK, device=args.device, num_envs=1))(
            robot_type=ROBOT
        )
        cfg.sim.device = args.device
        cfg.scene.num_envs = 1
        if cfg.robot_type != ROBOT:
            raise ValueError(f"Unexpected task embodiment {cfg.robot_type}")
        cfg.seed = args.seed
        cfg.recorders = {}
        terms = {
            k: v
            for k in dir(cfg.terminations)
            if not k.startswith("_")
            and isinstance(v := getattr(cfg.terminations, k), TerminationTermCfg)
            and not v.time_out
        }
        if "success" not in terms:
            raise ValueError("Task has no measurable success term")
        cfg.terminations = {}
        state_cfg = cfg.observations.state
        cfg.observation_preset = "rgb"
        cfg._apply_observation_preset("rgb")
        cfg.observations.state = state_cfg
        cfg.observations.proprio.history_length = 0
        cfg.observations.proprio.enable_corruption = False
        cfg.observations.state.enable_corruption = False
        cfg.num_rerenders_on_reset = 4
        env = gym.make(TASK, cfg=cfg, render_mode="rgb_array").unwrapped
        adapter["validate_environment"](env, manifest)
        if abs(env.step_dt - 1 / track["fps"]) > 1e-7:
            raise ValueError(
                "Simulator control rate differs from the recorded resampling rate"
            )
        obs, _ = env.reset(seed=args.seed)
        print(
            "OBSERVATIONS",
            {
                k: (list(v.keys()) if isinstance(v, dict) else list(v.shape))
                for k, v in obs.items()
            },
            flush=True,
        )
        if args.probe:
            save_json(
                out / "probe.json",
                {
                    "task": TASK,
                    "robot": cfg.robot_type,
                    "action_dim": env.action_manager.total_action_dim,
                    "observations": {
                        k: (list(v.keys()) if isinstance(v, dict) else list(v.shape))
                        for k, v in obs.items()
                    },
                },
            )
            return
        # The task exposes non-privileged state and robot proprioception. Explicit
        # keys prevent silently replacing state with images or privileged inputs.
        keys = ("proprio", "state")

        def state_vector(observation):
            values = []
            for key in keys:
                if key not in observation or not torch.is_tensor(observation[key]):
                    raise ValueError(
                        f"Expected concatenated tensor observation group {key}"
                    )
                values.append(observation[key].reshape(1, -1))
            result = torch.cat(values, dim=1)
            if not torch.isfinite(result).all():
                raise ValueError("Nonfinite simulator observations")
            return result

        def outcome():
            return {
                k: bool(term.func(env, **term.params)[0].item())
                for k, term in terms.items()
            }

        def image_frame():
            image = env.render()
            if image is None:
                raise ValueError("Simulator returned no replay image")
            if hasattr(image, "cpu"):
                image = image.cpu().numpy()
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[2] < 3:
                raise ValueError(f"Unexpected replay image shape {image.shape}")
            return image[:, :, :3].astype(np.uint8)

        def scene_arrays(group, value):
            for key, item in value.items():
                if isinstance(item, dict):
                    scene_arrays(group.create_group(key), item)
                else:
                    array = (
                        item.detach().cpu().numpy()
                        if torch.is_tensor(item)
                        else np.asarray(item)
                    )
                    if array.dtype.kind not in "bifu":
                        raise ValueError(f"Unsupported scene-state field {key}")
                    group.create_dataset(key, data=array)

        target = DeviceBase.TrackingTarget.HAND_RIGHT
        retarget = SimpleRelativeRetargeter(
            SimpleRelativeRetargeterCfg(
                robot_type=ROBOT, sim_device=args.device, bound_hand=target
            )
        )
        if (
            not retarget._dex_retgt
            or env.action_manager.total_action_dim != manifest["action_dimension"]
        ):
            raise ValueError(
                "DexPilot retargeter or the prepared hand action layout is unavailable"
            )
        if any(
            len(indices) != len(manifest["finger_joints"])
            or any(i is None or i < 0 for i in indices)
            for indices in retarget._dex_to_action_finger_indices.values()
        ):
            raise ValueError("Shadow finger joint mapping is incomplete")

        def retarget_sample(sample):
            data = {
                target: {
                    name: np.asarray(pose, dtype=np.float32)
                    for name, pose in sample["joints"].items()
                }
            }
            return retarget.retarget(data).reshape(1, -1)

        retarget_sample(track["samples"][0])
        retarget.calibrate_wrist_pose()
        for marker in (
            retarget._markers,
            retarget._canonical_markers,
            retarget._wrist_markers,
        ):
            marker.set_visibility(False)
        xs = []
        ys = []
        rewards = []
        outcomes = []
        source_indices = []
        joint_states = []
        success_streak = 0
        capture_success = False

        def snapshot(value):
            return {
                key: snapshot(item)
                if isinstance(item, dict)
                else item.detach().cpu().numpy().copy()
                for key, item in value.items()
            }

        initial_state = snapshot(env.scene.get_state(is_relative=True))
        scene_states = [initial_state]
        next_xs = []
        video = imageio.get_writer(
            str(out / "replay.mp4"), fps=30, codec="libx264", macro_block_size=1
        )
        try:
            for i, sample in enumerate(track["samples"]):
                if not app.is_running():
                    raise RuntimeError("Simulator stopped during recording replay")
                action = retarget_sample(sample)
                if (
                    action.shape != (1, manifest["action_dimension"])
                    or not torch.isfinite(action).all()
                ):
                    raise ValueError("Retargeter produced invalid robot actions")
                xs.append(state_vector(obs).detach().cpu().numpy()[0])
                ys.append(action.detach().cpu().numpy()[0])
                source_indices.append(sample["source_index"])
                joint_states.append(
                    env.scene["robot"].data.joint_pos[0].detach().cpu().numpy().copy()
                )
                obs, reward, terminated, truncated, info = env.step(action)
                result = outcome()
                outcomes.append(result)
                rewards.append(float(reward[0]))
                next_xs.append(state_vector(obs).detach().cpu().numpy()[0])
                scene_states.append(snapshot(env.scene.get_state(is_relative=True)))
                success_streak = success_streak + 1 if result["success"] else 0
                capture_success = capture_success or success_streak >= 10
                if i % 2 == 0:
                    video.append_data(image_frame())
                if i % 60 == 0:
                    stage(
                        current,
                        "RUNNING",
                        frames=i + 1,
                        total_frames=len(track["samples"]),
                    )
                failures = [
                    name
                    for name, value in result.items()
                    if name != "success" and value
                ]
                if failures:
                    # End at a physical failure; do not reset and stitch two scenes
                    # into one supposedly continuous demonstration.
                    stage(current, "RUNNING", frames=i + 1, ended_by=failures)
                    break
        finally:
            video.close()
        if len(xs) < 2:
            raise ValueError(
                "Replay ended before two observation/action pairs could be collected"
            )
        metadata = {
            "schema": SCHEMA,
            "task": TASK,
            "robot": ROBOT,
            "hand_asset": manifest["hand_asset"],
            "hand_adapter_digest": manifest["digest"],
            "action_joint_names": manifest["wrist_joints"] + manifest["finger_joints"],
            "dexverse_revision": REVISION,
            "source_sha256": args.source_sha256,
            "tracking_sha256": sha(args.tracking),
            "observation_groups": list(keys),
            "observation_dimensions": {k: int(obs[k].numel()) for k in keys},
            "action_dimension": manifest["action_dimension"],
            "action_terms": env.action_manager.active_terms,
            "fps": 60,
            "frames": len(xs),
            "episodes": 1,
            "capture_success": capture_success,
            "outcomes": outcomes[-1],
            "scene_seed": args.seed,
            "demonstration_kind": "offline human motion replay; no scene feedback during capture",
            "training_policy": "skynet-state-bc/v1",
            "includes_unsuccessful_demonstrations": not capture_success,
            "calibration": {k: v for k, v in track.items() if k != "samples"},
        }
        dataset = out / "dataset.hdf5"
        with h5py.File(dataset, "w") as f:
            f.attrs["schema"] = SCHEMA
            f.attrs["metadata"] = json.dumps(metadata, sort_keys=True)
            demo = f.create_group("data/demo_0")
            demo.attrs["num_samples"] = len(xs)
            demo.attrs["success"] = capture_success
            demo.create_dataset(
                "obs/state", data=np.asarray(xs, dtype=np.float32), compression="gzip"
            )
            demo.create_dataset(
                "actions", data=np.asarray(ys, dtype=np.float32), compression="gzip"
            )
            demo.create_dataset(
                "next_obs/state",
                data=np.asarray(next_xs, dtype=np.float32),
                compression="gzip",
            )
            demo.create_dataset(
                "success",
                data=np.asarray([item["success"] for item in outcomes], dtype=np.bool_),
            )
            for name in terms:
                demo.create_dataset(
                    "outcomes/" + name,
                    data=np.asarray([item[name] for item in outcomes], dtype=np.bool_),
                )
            demo.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float32))
            demo.create_dataset(
                "source_indices", data=np.asarray(source_indices, dtype=np.int64)
            )
            demo.create_dataset(
                "robot_joint_positions",
                data=np.asarray(joint_states, dtype=np.float32),
                compression="gzip",
            )
            scene_arrays(demo.create_group("initial_scene"), initial_state)

            def stack_states(items):
                return {
                    key: stack_states([v[key] for v in items])
                    if isinstance(items[0][key], dict)
                    else np.stack([v[key] for v in items])
                    for key in items[0]
                }

            scene_arrays(demo.create_group("scene_states"), stack_states(scene_states))
        metadata["dataset_sha256"] = sha(dataset)
        save_json(out / "dataset-manifest.json", metadata)
        stage(
            current,
            "SUCCEEDED",
            frames=len(xs),
            task_success=capture_success,
            outcome=outcomes[-1],
        )
        current = "training"
        stage(current, "RUNNING")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # Train by reading the saved dataset, not by reusing unregistered arrays.
        with h5py.File(dataset, "r") as f:
            x = torch.as_tensor(f["data/demo_0/obs/state"][:], device=args.device)
            y = torch.as_tensor(f["data/demo_0/actions"][:], device=args.device)
        x_mean = x.mean(0)
        x_std = x.std(0, unbiased=False).clamp_min(1e-4)
        y_mean = y.mean(0)
        y_std = y.std(0, unbiased=False).clamp_min(1e-4)
        net = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, manifest["action_dimension"]),
        ).to(args.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        with torch.enable_grad():
            for epoch in range(args.epochs):
                prediction = net((x - x_mean) / x_std)
                loss = ((prediction - (y - y_mean) / y_std) ** 2).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if epoch % 20 == 0:
                    stage(
                        current,
                        "RUNNING",
                        epoch=epoch + 1,
                        total_epochs=args.epochs,
                        training_mse=float(loss.detach()),
                    )
        checkpoint = out / "state-bc.pt"
        torch.save(
            {
                "schema": "skynet-state-bc/v1",
                "state_dict": net.state_dict(),
                "input_dim": x.shape[1],
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "action_min": y.min(0).values,
                "action_max": y.max(0).values,
                "metadata": metadata,
                "epochs": args.epochs,
            },
            checkpoint,
        )
        stage(
            current,
            "SUCCEEDED",
            epochs=args.epochs,
            training_mse=float(loss.detach()),
            checkpoint_sha256=sha(checkpoint),
            validation="Training fit only; generalization is measured by the separate simulator rollout",
        )
        current = "evaluation"
        stage(current, "RUNNING")
        saved = torch.load(checkpoint, map_location=args.device, weights_only=True)
        if saved["metadata"]["dataset_sha256"] != sha(dataset):
            raise ValueError("Checkpoint dataset checksum mismatch")
        net.load_state_dict(saved["state_dict"])
        net.eval()
        results = []
        for episode in range(args.eval_episodes):
            seed = args.seed + 1 + episode
            obs, _ = env.reset(seed=seed)
            success = False
            streak = 0
            clamped = 0
            writer = imageio.get_writer(
                str(out / f"evaluation-{episode}.mp4"),
                fps=30,
                codec="libx264",
                macro_block_size=1,
            )
            try:
                for i in range(len(xs)):
                    # Simulator buffers remain mutable across episode resets.
                    with torch.no_grad():
                        raw = (
                            net((state_vector(obs) - saved["x_mean"]) / saved["x_std"])
                            * saved["y_std"]
                            + saved["y_mean"]
                        )
                        if not torch.isfinite(raw).all():
                            raise ValueError("Policy produced a nonfinite action")
                        action = raw.clamp(saved["action_min"], saved["action_max"])
                        clamped += int((raw != action).any())
                        obs, _, _, _, _ = env.step(action)
                    result = outcome()
                    streak = streak + 1 if result["success"] else 0
                    success = success or streak >= 10
                    if i % 2 == 0:
                        writer.append_data(image_frame())
                    if success or any(
                        value for key, value in result.items() if key != "success"
                    ):
                        break
            finally:
                writer.close()
            results.append(
                {
                    "seed": seed,
                    "success": success,
                    "steps": i + 1,
                    "outcomes": result,
                    "actions_clamped_to_training_range": clamped,
                }
            )
            stage(
                current,
                "RUNNING",
                episodes_completed=episode + 1,
                total_episodes=args.eval_episodes,
            )
        stage(
            current,
            "SUCCEEDED",
            successes=sum(r["success"] for r in results),
            episodes=len(results),
        )
        artifacts = {
            p.name: {"sha256": sha(p), "size_bytes": p.stat().st_size}
            for p in out.iterdir()
            if p.is_file()
            and p.name not in ("progress.json", "result.json", "error.json")
        }
        save_json(
            out / "result.json",
            {
                "schema": "skynet.dexverse-cycle/v1",
                "status": "SUCCEEDED",
                "stages": stages,
                "dataset": metadata,
                "evaluation": results,
                "artifacts": artifacts,
                "limitations": [
                    "State-based behavior cloning; GR00T/pi0.5 require their own embodiment and observation adapters.",
                    "Offline motion capture has no simulator feedback. Review replay before using it as task demonstration.",
                    "A single motion-test episode cannot establish policy generalization.",
                ],
            },
        )
    except Exception as error:
        stage(current, "FAILED", error=str(error))
        save_json(
            out / "error.json",
            {
                "stage": current,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
