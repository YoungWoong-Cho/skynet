"""Render training images only after XR has exited; original captures are immutable."""
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import traceback

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_source(root, receipt, profile):
    from arrays import ArrayUnpickler
    from images import training_image_request
    path = (root / receipt["path"]).resolve()
    if not path.is_relative_to(root / "recordings") or path.suffix != ".pkl" or not 0 < path.stat().st_size <= 100_000_000:
        raise ValueError("Invalid original recording")
    if digest(path) != receipt["sha256"]:
        raise ValueError("Original recording checksum changed")
    payload = ArrayUnpickler(io.BytesIO(path.read_bytes())).load()
    if (payload.get("format"), payload.get("schema_version"), payload.get("task"), payload.get("robot_type")) != ("dexverse_trajectory", 3, profile["task"], profile["robot"]):
        raise ValueError("Recording task or hand differs from the session")
    requested = payload.get("skynet_training_images", {})
    if requested != training_image_request(requested.get("recipe")):
        raise ValueError("Missing or invalid frozen training image recipe")
    episodes = payload.get("episodes", [])
    if payload.get("num_episodes") != 1 or len(episodes) != 1:
        raise ValueError("Training image rendering requires one episode per recording")
    ep = episodes[0]
    actions = np.asarray(ep.get("actions"))
    if actions.ndim != 2 or not 0 < len(actions) <= 6000 or not np.isfinite(actions).all() or ep.get("success") is not True or len(ep.get("states", [])) != len(actions) + 1 or ep.get("num_steps") != len(actions):
        raise ValueError("Recording actions and saved states are incomplete")
    wall = np.asarray(ep.get("skynet_wall_times"))
    if wall.shape != (len(actions),) or not np.isfinite(wall).all() or (np.diff(wall) < 0).any():
        raise ValueError("Recording is missing its original frame times")
    return payload, ep


def restore_state(env, state):
    import torch
    current = env.scene.get_state(is_relative=True)
    def tensors(saved, actual):
        if isinstance(actual, dict):
            if not isinstance(saved, dict) or set(saved) != set(actual):
                raise ValueError("Saved scene entities differ from the renderer")
            return {k: tensors(saved[k], actual[k]) for k in actual}
        value = np.asarray(saved)
        if value.shape != tuple(actual.shape) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError("Invalid saved scene state")
        return torch.as_tensor(value, device=env.device, dtype=actual.dtype)
    env.scene.reset_to(tensors(state, current), is_relative=True)
    env.sim.forward()
    env.sim.render()
    env.scene.update(dt=env.physics_dt)


def render(config, root):
    from images import ImageRecorder, camera_recipe, training_image_request
    root = Path(root).resolve()
    profile = json.loads(Path(config).read_text())
    progress = root / "image-progress.json"
    def publish(**value):
        temp = progress.with_suffix(".tmp")
        temp.write_text(json.dumps(value))
        temp.replace(progress)
    app = env = images = None
    try:
        if (root / "episodes.json").stat().st_size > 1_000_000:
            raise ValueError("Episode receipts exceed their size limit")
        receipts = json.loads((root / "episodes.json").read_text())
        if not 1 <= len(receipts) <= 1000:
            raise ValueError("No bounded set of saved episodes to render")
        publish(state="RUNNING", completed=0, total=len(receipts), detail="Loading training image renderer")
        recorder = str(Path(profile["repository"]) / "scripts/record_demos.py")
        if (Path(profile["repository"]) / ".skynet-source-revision").read_text().strip() != profile["source_revision"]:
            raise ValueError("Renderer source revision differs from the recording")
        sys.argv = [recorder, "--task", profile["task"], "--robot_type", profile["robot"], "--teleop_device", "keyboard", "--enable_pinocchio", "--enable_cameras", "--headless", "--device", "cuda:0", "--teleop_retargeter", "absolute"]
        ns = runpy.run_path(recorder, run_name="skynet_training_image_renderer")
        app = ns["simulation_app"]
        from images import configure_cameras
        from wrist import configure_virtual_wrist
        import torch
        manifest = None
        if profile.get("hand_bundle"):
            sys.path.insert(0, profile["hand_bundle"]["root"])
            from runtime import install, validate_environment
            manifest = install(Path(profile["hand_bundle"]["root"]))
            profile["hand_manifest"] = manifest
        cfg, _ = ns["create_environment_config"]()
        recipe = camera_recipe(cfg)
        configure_cameras(cfg)
        cfg.observations.contact = None
        cfg.recorders, cfg.terminations = {}, {}
        env = ns["create_environment"](cfg)
        env.reset()
        if manifest:
            validate_environment(env, manifest)
        configure_virtual_wrist(env.scene["robot"], manifest, ["left", "right"] if profile["hand"] == "both" else [profile["hand"]])
        images = ImageRecorder(root, env, profile, [])
        images.metadata.update(render_mode="saved_states", image_recipe_sha256=training_image_request(recipe)["recipe_sha256"], alignment="rendered from each saved pre-action scene state after XR exits; physics is not advanced")
        rendered = {}
        with torch.inference_mode():
            for index, receipt in enumerate(receipts):
                payload, ep = load_source(root, receipt, profile)
                if payload["skynet_training_images"]["recipe"] != recipe or not np.isclose(payload.get("skynet_step_dt", 0), env.step_dt):
                    raise ValueError("Camera recipe or control timing differs from collection")
                images.metadata["source_sha256"] = receipt["sha256"]
                env.reset()
                # Restore a saved task goal when the task has one; never replace it with a new random goal.
                if "object_pose" in env.command_manager.active_terms:
                    goal = np.asarray(ep.get("goal_pose"))
                    target = env.command_manager.get_command("object_pose")
                    if goal.shape != tuple(target.shape[1:]) or not np.isfinite(goal).all():
                        raise ValueError("Recording is missing its task goal")
                    target.copy_(torch.as_tensor(goal, device=env.device)[None])
                restore_state(env, ep["states"][0])
                images.begin()
                for frame, action in enumerate(ep["actions"]):
                    if not app.is_running() or app.is_exiting():
                        raise ValueError("Renderer exited before completing the images")
                    restore_state(env, ep["states"][frame])
                    images.append(action, wall_time=float(ep["skynet_wall_times"][frame]))
                    if frame % 60 == 0:
                        publish(state="RUNNING", completed=index, total=len(receipts), frame=frame, frames=ep["num_steps"], detail=f"Preparing images for episode {index + 1} of {len(receipts)} ({frame}/{ep['num_steps']} frames)")
                image = images.finish()
                image["source_sha256"] = receipt["sha256"]
                rendered[receipt["path"]] = image
                # Sidecar receipts are separate; never rewrite the original episode or its receipt.
                temp = root / "image-receipts.tmp"
                temp.write_text(json.dumps(rendered, sort_keys=True))
                temp.replace(root / "image-receipts.json")
        publish(state="READY", completed=len(receipts), total=len(receipts), detail="Training images are ready")
        return rendered
    except BaseException as exc:
        traceback.print_exc()
        publish(state="FAILED", error=str(exc), detail="Training image preparation failed; originals are preserved")
        raise
    finally:
        if images: images.discard()
        if env: env.close()
        if app: app.close()


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
