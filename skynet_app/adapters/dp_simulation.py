"""Skynet scene binding for official DP's unchanged joint-space policy.

The same collected hand assets and measured RGB calibration are restored for
rollout. No Cartesian codec, fingertip conversion or retargeter is involved.
"""
import numpy as np

try:
    from .dp_data import validate_joint_manifest
    from .policy_contract import recorded_contract
    from .recording_dataset import verify_dataset
except ImportError:
    from dp_data import validate_joint_manifest
    from policy_contract import recorded_contract
    from recording_dataset import verify_dataset


def validate_target(target, config):
    for field, native in (("path", "dataset_path"), ("manifest_sha256", "dataset_manifest_sha256")):
        if not target.get(field) or target[field] != config.get(native):
            raise ValueError("DP rollout target must match its frozen training dataset path and SHA256")
    metadata = target.get("metadata") or {}
    validate_joint_manifest(metadata)
    validate_cameras(recorded_contract(metadata, control_hz=config.get("control_hz")))


def validate_cameras(contract):
    try:
        from observation_render import camera_recipe_from_capture
    except ImportError:
        from ops.datasets.observation_render import camera_recipe_from_capture
    for name, camera in contract["cameras"].items():
        if not all(key in camera for key in ("sensor", "intrinsic_matrix", "position_world", "quaternion_world_ros")):
            raise ValueError("DP rollout requires the measured recording camera calibration")
        camera_recipe_from_capture(camera, contract["source_revision"], camera_id=name)


def load_target(context):
    selected = context.get("target_dataset") or {}
    config = context["policy"]["native_config"]
    for field, native in (("path", "dataset_path"), ("manifest_sha256", "dataset_manifest_sha256")):
        if not selected.get(field) or selected[field] != config.get(native):
            raise ValueError("DP rollout target must match its frozen training dataset path and SHA256")
    if context.get("unseen_embodiment"):
        raise ValueError("DP joint rollout cannot evaluate an unseen embodiment")
    manifest = verify_dataset(selected["path"], selected["manifest_sha256"])
    validate_target({**selected, "metadata": manifest}, config)
    return manifest


def configure_scene(cfg, contract):
    from observation_render import configure_cameras, camera_recipe_from_capture
    validate_cameras(contract)
    jobs = []
    for name, capture in contract["cameras"].items():
        camera = camera_recipe_from_capture(capture, contract["source_revision"], camera_id=name)
        jobs.append(dict(camera_id=name, modality="rgb", recipe=dict(camera=camera, width=camera["width"], height=camera["height"])))
    configure_cameras(cfg, jobs, replay=False)
    return {name: camera["sensor"] for name, camera in contract["cameras"].items()}


def observation_from_sensors(env, contract, joint_ids):
    from observation_render import read_camera, capture_current_frame
    from observation_geometry import pose_from_ros
    capture_current_frame(env)
    images = {}
    for name, capture in contract["cameras"].items():
        view = read_camera(env.scene[capture["sensor"]], env.sim.render, ("rgb",), render_mode=env.sim.render_mode)
        if not np.allclose(view["world_from_camera"], pose_from_ros(capture["position_world"], capture["quaternion_world_ros"]), atol=1e-5):
            raise ValueError("Live DP camera pose differs from the recording")
        if not np.allclose(view["intrinsics"], capture["intrinsic_matrix"], atol=1e-5):
            raise ValueError("Live DP camera intrinsics differ from the recording")
        images[name] = view["rgb"]
    state = env.scene["robot"].data.joint_pos[0, joint_ids].detach().cpu().numpy().copy()
    return dict(state=state, images=images)
