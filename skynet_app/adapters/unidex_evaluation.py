"""Skynet bridge from native UniDex FAAS chunks to an exact simulated hand.

The policy/loss stay upstream. Camera reconstruction and asset-bound actuator
decoding reuse the same verified conversion geometry rather than a new mapping.
"""
import math
import random

import numpy as np

try:
    from .policy_contract import unidex_contract, validate_actions, validate_observation
    from .unidex_runtime import UniDexPolicy
    from .dataset_inputs import validate_selection_sources
except ImportError:
    from policy_contract import unidex_contract, validate_actions, validate_observation
    from unidex_runtime import UniDexPolicy
    from dataset_inputs import validate_selection_sources
try:
    from ops.datasets.action_codecs.unidex import encode_absolute, load_codec
    from ops.datasets.action_codecs.geometry import inverse_transform
    from ops.datasets.observation_geometry import point_cloud, pose_from_ros
except ImportError:
    from action_codecs.unidex import encode_absolute, load_codec
    from action_codecs.geometry import inverse_transform
    from observation_geometry import point_cloud, pose_from_ros


def target_manifest_path(context):
    selected = context.get("target_dataset") or {}
    if not selected.get("path") or not selected.get("manifest_sha256"):
        raise ValueError("UniDex simulation needs a separately frozen evaluation dataset")
    return selected["path"], selected["manifest_sha256"]


def training_inputs(context):
    config = context["policy"]["native_config"]
    inputs = config.get("datasets")
    if not isinstance(inputs, list) or not inputs:
        # Existing v2 single-dataset checkpoints remain valid evaluation inputs.
        if config.get("dataset_path") and config.get("dataset_manifest_sha256"):
            inputs = [dict(position=0, version_id=None, path=config["dataset_path"],
                           manifest_sha256=config["dataset_manifest_sha256"])]
        else:
            raise ValueError("UniDex evaluation requires the checkpoint's frozen training datasets")
    return inputs


def observation_sampling_identity(contract, task, seed, episode_index):
    """Common observation sampling across checkpoints and evaluation workers.

    This identity is intentionally separate from progress/resume provenance.
    The shared point-cloud builder adds the physical frame and recipe seed.
    """
    return dict(schema="skynet.unidex-rollout-observation-sampling/v1", task=task,
                robot=contract["robot"], codec_sha256=contract["codec_sha256"],
                seed=seed, episode_index=episode_index)


class RecordedPolicy:
    mode = "pointcloud"

    def __init__(self, context, source_dir, manifest, *, device="cuda"):
        self.manifest = manifest
        expected = context["compatibility"]["io_contract"]
        self.contract = unidex_contract(manifest, control_hz=1 / expected["step_dt"])
        if self.contract != expected:
            raise ValueError("Target hand or point-cloud geometry differs from the planned evaluation")
        selected = training_inputs(context)
        target = {**context["target_dataset"], "metadata": manifest}
        if context.get("unseen_embodiment"):
            if any(episode.get("hand_id") == self.contract["robot"] for item in selected
                   for episode in (item.get("metadata") or {}).get("episodes", [])):
                raise ValueError("The requested unseen hand occurs in a selected training dataset")
            validate_selection_sources([*selected, target])
        prompts = {episode.get("prompt") for episode in manifest["episodes"]}
        if len(prompts) != 1 or not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
            raise ValueError("The evaluation dataset must declare one unambiguous task instruction")
        self.prompt = prompts.pop()
        checkpoint = context["checkpoint"]
        self.policy = UniDexPolicy(source_dir, checkpoint["path"], checkpoint["sha256"],
                                  device=device, expected_datasets=selected)
        if self.policy.pointcloud_recipe != self.contract["pointcloud_recipe"]:
            raise ValueError("Checkpoint point-cloud recipe differs from the frozen evaluation target")
        self.control_hz, self.horizon = self.policy.control_hz, self.policy.horizon
        if not math.isclose(1 / self.control_hz, self.contract["step_dt"], rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("Checkpoint control frequency differs from the evaluation contract")
        self.codec = load_codec(self.contract["robot"], manifest["episodes"][0]["capture"])

    def reset(self, seed):
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def step(self, observation, predict):
        if not predict:
            return None
        validate_observation(self.contract, observation)
        joints = np.asarray(observation["state"], dtype=np.float64)
        camera_from_root = (np.diag([1., -1., -1., 1.]) @ inverse_transform(observation["world_from_camera"])
                            @ observation["world_from_root"])
        state = encode_absolute(self.codec, joints, camera_from_root=camera_from_root)
        chunk = self.policy.predict(observation["pointcloud"], state, self.prompt)
        # Preserve native raw position-command semantics; no new clipping policy.
        return validate_actions(self.contract, self.codec.decode_actions(chunk, joints)).astype(np.float32)


def observation_from_sensors(env, contract, joint_ids, *, identity, frame_id, include_pointcloud=True):
    """Capture every sensor tick, deriving policy inputs only when consumed.

    Rendering, pixel reads and pose validation stay identical between control
    ticks. The deterministic FPS operation has no sensor or global RNG effects.
    """
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    recipe = contract["cameras"]["scene_front"]
    sensor = env.scene[recipe["sensor"]]
    try:
        from observation_render import read_camera, capture_current_frame
    except ImportError:
        from ops.datasets.observation_render import read_camera, capture_current_frame
    capture_current_frame(env)
    view = dict(camera_id="scene_front", **read_camera(sensor, env.sim.render, ("rgb", "depth"),
                                                      render_mode=env.sim.render_mode))
    if "intrinsic_matrix" in recipe:
        expected_pose = pose_from_ros(recipe["position_world"], recipe["quaternion_world_ros"])
        if (not np.allclose(view["intrinsics"], recipe["intrinsic_matrix"], rtol=1e-5, atol=1e-5)
                or not np.allclose(view["world_from_camera"], expected_pose, rtol=1e-5, atol=1e-5)):
            raise ValueError("Live camera calibration differs from the frozen evaluation dataset")
    robot = env.scene["robot"]
    result = {"state": array(robot.data.joint_pos)[0, joint_ids].copy(), "images": {"scene_front": view["rgb"]},
              "world_from_camera": view["world_from_camera"],
              "world_from_root": pose_from_ros(array(robot.data.root_pos_w)[0], array(robot.data.root_quat_w)[0])}
    if include_pointcloud:
        result["pointcloud"], _ = point_cloud([view], contract["pointcloud_recipe"], identity=identity, frame_id=frame_id)
    return result


def configure_pointcloud_scene(cfg, contract):
    try:
        from observation_render import configure_cameras, camera_recipe_from_capture
    except ImportError:
        from ops.datasets.observation_render import configure_cameras, camera_recipe_from_capture
    camera = camera_recipe_from_capture(contract["cameras"]["scene_front"], contract["source_revision"])
    jobs = [dict(camera_id="scene_front", modality=modality,
                 recipe=dict(camera=camera, width=camera["width"], height=camera["height"]))
            for modality in ("rgb", "depth")]
    # Rollout keeps the task's startup/reset randomization and termination rules.
    configure_cameras(cfg, jobs, replay=False)
    return {"scene_front": camera["sensor"]}
