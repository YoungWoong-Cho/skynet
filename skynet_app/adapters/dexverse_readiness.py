"""GPU runtime check: construct a native scene, render RGB, and encode video."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    actual = {"checks": {}, "imports": {}, "gym_registrations": {}}
    errors = []
    app = env = None
    try:
        if os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES":
            raise ValueError("Operator license acceptance is required")
        if (
            Path(args.source_dir) / ".skynet-source-revision"
        ).read_text().strip() != args.source_revision:
            raise ValueError(
                "DexVerse source revision differs from the configured runtime"
            )
        import pinocchio
        from isaaclab.app import AppLauncher

        app = AppLauncher(
            headless=True,
            enable_cameras=True,
            device="cuda:0",
            kit_args="--/rtx/verifyDriverVersion/enabled=false",
        ).app
        import torch
        import gymnasium as gym
        import imageio.v2 as imageio
        import dexverse.tasks
        from dexverse.tasks.utils import parse_env_cfg, prune_stale_obs_refs
        import omni.usd

        for module in [
            "pinocchio",
            "torch",
            "gymnasium",
            "imageio",
            "isaaclab",
            "dexverse.tasks",
        ]:
            actual["imports"][module] = True
        actual["gpu"] = dict(
            cuda_available=torch.cuda.is_available(),
            device_count=torch.cuda.device_count(),
            devices=[torch.cuda.get_device_name(0)],
        )
        ids = {name: True for name in gym.registry if name.startswith("Dexverse-")}
        if not ids:
            raise ValueError("DexVerse task registration is empty")
        actual["gym_registrations"]["dexverse.tasks"] = dict(
            module_imported=True, ids=ids
        )
        if omni.usd.get_context().get_stage() is None:
            omni.usd.get_context().new_stage()
        cfg = parse_env_cfg("Dexverse-PickCube-v0", device="cuda:0", num_envs=1)
        cfg._apply_observation_preset("state")
        cfg._apply_multiview_cameras(True)
        cfg.scene.third_person_camera.width = cfg.scene.third_person_camera.height = 64
        cfg.observations.contact = None
        cfg.observations.debug_vis = None
        cfg.recorders, cfg.terminations = {}, {}
        cfg = prune_stale_obs_refs(cfg)
        env = gym.make("Dexverse-PickCube-v0", cfg=cfg).unwrapped
        actual["checks"]["environment_constructed"] = True
        env.reset()
        actual["checks"]["environment_reset"] = True
        video = output.with_suffix(".mp4")
        with (
            torch.inference_mode(),
            imageio.get_writer(
                video, fps=30, codec="libx264", macro_block_size=1
            ) as writer,
        ):
            for _ in range(4):
                env.step(
                    torch.zeros(
                        (1, env.action_manager.total_action_dim), device=env.device
                    )
                )
                camera = env.scene["third_person_camera"]
                camera.update(0.0, force_recompute=True)
                frame = camera.data.output["rgb"][0, :, :, :3].cpu().numpy()
                if frame.shape != (64, 64, 3) or frame.max() == frame.min():
                    raise ValueError("Camera did not render a scene")
                writer.append_data(frame)
        actual["checks"]["camera_rendered"] = True
        actual["checks"]["media_nonempty"] = video.stat().st_size > 100
        if not actual["checks"]["media_nonempty"]:
            raise ValueError("Video encoding produced no media")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
        raise
    finally:
        output.write_text(
            json.dumps(
                dict(
                    schema_version="skynet.evaluator-readiness/v1",
                    ready=not errors,
                    errors=errors,
                    actual=actual,
                )
            )
        )
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
