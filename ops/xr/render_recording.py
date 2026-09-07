"""Render exact saved scene states; never rerun actions or train a policy."""

import hashlib
import json
from pathlib import Path
import runpy
import sys
import traceback

request = json.loads(Path(sys.argv[1]).read_text())
output = Path(request["output"])
status_path = output.with_suffix(".json")


def publish(**value):
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(status_path)


app = env = encoder = camera = None
try:
    import io
    import numpy as np
    from arrays import ArrayUnpickler

    source = Path(request["recording"])
    if not 0 < source.stat().st_size <= 100 * 1024 * 1024:
        raise ValueError("Recording exceeds the replay size limit")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != request["sha256"]:
        raise ValueError("The saved recording changed; video rendering was stopped")
    payload = ArrayUnpickler(io.BytesIO(raw)).load()
    profile = request["profile"]
    if (
        payload.get("task") != profile["task"]
        or payload.get("robot_type") != profile["robot"]
    ):
        raise ValueError("Recording hand/task differs from its session")
    episode = payload["episodes"][request["episode"]]
    states = episode["states"]
    if len(states) != episode["num_steps"] + 1:
        raise ValueError("Recording is missing scene states")
    publish(state="RENDERING", frame=0, total=len(states))
    recorder = str(Path(profile["repository"]) / "scripts/record_demos.py")
    sys.argv = [
        recorder,
        "--task",
        profile["task"],
        "--robot_type",
        profile["robot"],
        "--teleop_device",
        "keyboard",
        "--enable_pinocchio",
        "--enable_cameras",
        "--headless",
        "--device",
        "cuda:0",
    ]
    if profile.get("hand_bundle"):
        sys.path.insert(0, profile["hand_bundle"]["root"])
    ns = runpy.run_path(recorder, run_name="skynet_video_replay")
    app = ns["simulation_app"]
    import torch
    from video import SceneCamera, VideoEncoder
    from wrist import configure_virtual_wrist

    manifest = None
    if profile.get("hand_bundle"):
        from runtime import install, validate_environment

        manifest = install(Path(profile["hand_bundle"]["root"]))
    cfg, _ = ns["create_environment_config"]()
    cfg = ns["prune_stale_obs_refs"](ns["strip_camera_cfgs"](cfg))
    env = ns["create_environment"](cfg)
    env.reset()
    if manifest:
        validate_environment(env, manifest)
    sides = ["left", "right"] if profile["hand"] == "both" else [profile["hand"]]
    configure_virtual_wrist(env.scene["robot"], manifest, sides)
    camera = SceneCamera(env)
    encoder = VideoEncoder(output)

    def tensors(value):
        if isinstance(value, dict):
            return {k: tensors(v) for k, v in value.items()}
        return torch.as_tensor(
            np.asarray(value), device=env.device, dtype=torch.float32
        )

    for index in range(0, len(states), 2):
        env.scene.reset_to(tensors(states[index]), is_relative=True)
        # Refresh Fabric transforms after writing recorded poses; no physics is advanced.
        env.scene.update(env.physics_dt)
        encoder.append(camera.frame())
        if index % 60 == 0:
            publish(state="RENDERING", frame=index, total=len(states))
    metadata = encoder.finish()
    encoder = None
    publish(state="READY", kind="replay", source_sha256=request["sha256"], **metadata)
except BaseException as exc:
    traceback.print_exc()
    publish(state="FAILED", error=str(exc))
finally:
    if encoder:
        encoder.finish(discard=True)
    if camera:
        camera.close()
    if env:
        env.close()
    if app:
        app.close()
