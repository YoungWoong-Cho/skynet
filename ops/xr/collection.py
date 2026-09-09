"""Repeated, manually started episodes for the pinned DexVerse recorder.

This module is frozen into each session capsule. Isaac is imported only by main.
"""

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import runpy
import signal
import sys
import time
import uuid

import numpy as np


def collection_success_term(task, term):
    """A stick has two equivalent ends; cups and other oriented tasks do not."""
    if task != "Dexverse-PickUpStick-v0":
        return term
    if (
        term is None
        or getattr(term.func, "__name__", None) != "lift_and_tilt"
        or term.params.get("tilt_ge") is not False
    ):
        raise ValueError(
            "Unsupported stick success condition in this DexVerse revision"
        )
    original = term.func

    def lift_stick_either_end(env, **params):
        axis = params.get("world_axis", (0.0, 0.0, 1.0))
        reverse = dict(params, world_axis=tuple(-value for value in axis))
        return original(env, **params) | original(env, **reverse)

    configured = copy.copy(term)
    configured.func = lift_stick_either_end
    return configured


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class EpisodeStore:
    """One immutable file per successful episode; earlier saves are never rewritten."""

    def __init__(self, root, metadata):
        self.root, self.metadata = Path(root), metadata
        self.directory = self.root / "recordings/live"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.receipts = []

    def save(self, episode):
        def finite(value):
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite(v) for v in value)
            if isinstance(value, (np.ndarray, float, int, np.number)):
                return bool(np.isfinite(value).all())
            return value is None or isinstance(value, str)

        actions = np.asarray(episode["actions"])
        if (
            episode.get("success") is not True
            or actions.ndim != 2
            or not all(actions.shape)
            or not np.isfinite(actions).all()
            or len(episode.get("states", [])) != len(actions) + 1
            or not finite(episode)
        ):
            raise ValueError("Episode has invalid actions or incomplete scene states")
        number = len(self.receipts) + 1
        path = self.directory / f"episode-{number:06d}.pkl"
        if path.exists():
            raise FileExistsError("Refusing to replace a saved episode")
        payload = dict(self.metadata, num_episodes=1, episodes=[episode])
        raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        if len(raw) > 100_000_000:
            raise ValueError("Episode exceeds the 100 MB review limit")
        temporary = path.with_suffix(".tmp")
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory_fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        receipt = dict(
            path=str(path.relative_to(self.root)),
            sha256=hashlib.sha256(raw).hexdigest(),
            size_bytes=len(raw),
            episodes=1,
            steps=len(actions),
        )
        if episode.get("skynet_images"):
            receipt["images"] = episode["skynet_images"]
        self.receipts.append(receipt)
        atomic_json(self.root / "episodes.json", self.receipts)
        return receipt


class ManualStart:
    """A click belongs to one attempt; stale retries cannot start another one."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.attempt_id = str(uuid.uuid4())
        self.requested = False

    def request(self, attempt_id):
        if attempt_id == self.attempt_id:
            self.requested = True

    def consume(self, tracking):
        requested, self.requested = self.requested, False
        return requested and tracking


def install_robot_point_display(retargeter, read_points, is_visible):
    """Blue markers show actual robot joint locations, in the same world frame."""
    marker = retargeter._canonical_markers
    visible = False
    marker.set_visibility(False)

    def visualize(_hand_data):
        nonlocal visible
        show = bool(is_visible())
        if show:
            marker.visualize(translations=read_points())
        if show != visible:
            marker.set_visibility(show)
            visible = show

    retargeter._visualize_canonical_hand_keypoints = visualize
    retargeter._visualize_wrist_poses = lambda: None
    retargeter._wrist_markers.set_visibility(False)


def run_loop(
    ns,
    cfg,
    root,
    env,
    teleop,
    success_term,
    recorder,
    multi_assets=None,
    multi_usds=None,
    pose_validity=None,
):
    import carb
    import torch
    from dexverse.devices.retargeters.simple_relative_retargeting import (
        DEX_RETARGETING_HAND_JOINT_NAMES,
    )
    from isaaclab.devices.device_base import DeviceBase

    if pose_validity is None:
        from omni.kit.xr.core import XRPoseValidityFlags

        pose_validity = XRPoseValidityFlags

    from wrist import configure_virtual_wrist, DexVerseWristContinuity

    task = cfg.get("task", recorder._metadata.get("task"))
    success_term = collection_success_term(task, success_term)
    if task == "Dexverse-PickUpStick-v0":
        recorder._metadata["skynet_success_orientation"] = "stick_either_end_up_v1"
    store = EpisodeStore(root, recorder._metadata)
    executor = ThreadPoolExecutor(max_workers=1)
    start = ManualStart()
    phase, instruction = "ready", "Tap Start to begin"
    tracking = False
    goal = cfg["instructions"]
    last_write, heartbeat, save_future, stop_requested = 0.0, 0.0, None, False
    saving_since = 0.0
    interrupted_since = 0.0
    tracking_lost_since = None
    reset_requested, success_count = False, 0
    connected_once = False
    raw_markers_visible = True
    tracked_sides = ["left", "right"] if cfg["hand"] == "both" else [cfg["hand"]]
    raw_data = {}
    get_raw = teleop._get_raw_data
    manifest = cfg.get("hand_manifest", {})
    robot = env.scene["robot"]
    configure_virtual_wrist(robot, manifest, tracked_sides)
    wrist_commands = DexVerseWristContinuity(teleop._retargeters, tracked_sides)
    if cfg.get("image_capture"):
        from images import training_image_request
        recorder._metadata["skynet_training_images"] = training_image_request(cfg["image_recipe"])
        recorder._metadata["skynet_step_dt"] = float(env.step_dt)
    if manifest:
        marker_names = [
            manifest["palm"],
            *manifest["joint_child_links"].values(),
            *manifest["tips"],
        ]
    else:
        marker_names = env.cfg.robot_config.hand_tips_body_names
    marker_indices = [robot.body_names.index(n) for n in dict.fromkeys(marker_names)]

    def robot_points():
        return robot.data.body_pos_w[0, marker_indices].detach().cpu().numpy()

    for retargeter in teleop._retargeters:
        install_robot_point_display(
            retargeter, robot_points, lambda: phase not in {"ended", "error"}
        )

    def capture_raw():
        nonlocal raw_data
        raw_data = get_raw()
        return raw_data

    teleop._get_raw_data = capture_raw

    def publish(force=False):
        nonlocal last_write
        now = time.monotonic()
        if not force and now - last_write < 0.2:
            return
        last_write = now
        atomic_json(
            root / "collection-status.json",
            dict(
                schema="skynet.collection/v1",
                phase=phase,
                instruction=instruction,
                goal=goal,
                saved=len(store.receipts),
                episode=len(store.receipts) + 1,
                can_start=phase == "ready" and tracking,
                attempt_id=start.attempt_id,
                control_alive=now - heartbeat < 2.0,
                updated_at=time.time(),
                session_id=os.environ.get("SKYNET_LIVE_SESSION_ID", ""),
            ),
        )

    def command(event):
        nonlocal heartbeat, reset_requested, stop_requested, connected_once
        message = event.payload.get("message", "{}")
        if isinstance(message, str):
            message = json.loads(message)
        verb = message.get("command")
        if verb == "heartbeat":
            heartbeat, connected_once = time.monotonic(), True
        elif verb == "start" and phase == "ready":
            start.request(message.get("attempt_id"))
        elif verb == "restart":
            reset_requested = True
        elif verb == "end":
            stop_requested = True

    teleop._skynet_subscription = (
        teleop._xr_core.get_message_bus().create_subscription_to_pop_by_type(
            carb.events.type_from_string("skynet_collection_command"), command
        )
    )

    def stop(*_):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    teleop.add_callback(
        "START", lambda: start.request(start.attempt_id) if phase == "ready" else None
    )
    teleop.add_callback("STOP", stop)
    teleop.add_callback(
        "RESET",
        lambda: command(
            type("Event", (), {"payload": {"message": {"command": "restart"}}})()
        ),
    )

    def reset():
        nonlocal phase, instruction, success_count
        recorder.discard_episode()
        ns["handle_reset"](env)
        teleop.reset()
        wrist_commands.reset()
        success_count = 0
        start.reset()
        phase, instruction = "ready", "Tap Start to begin"
        publish(True)

    def read_hands():
        points_by_side = {}
        for side in tracked_sides:
            device = teleop._xr_core.get_input_device("/user/hand/" + side)
            if device is None:
                return None
            poses = device.get_all_virtual_world_poses()
            for name in DEX_RETARGETING_HAND_JOINT_NAMES:
                pose = poses.get(name)
                if (
                    pose is None
                    or not (pose.validity_flags & pose_validity.POSITION_VALID)
                    or (
                        name == "wrist"
                        and not (pose.validity_flags & pose_validity.POSITION_TRACKED)
                    )
                ):
                    return None
            key = (
                DeviceBase.TrackingTarget.HAND_RIGHT
                if side == "right"
                else DeviceBase.TrackingTarget.HAND_LEFT
            )
            data = raw_data.get(key, {})
            if any(n not in data for n in DEX_RETARGETING_HAND_JOINT_NAMES):
                return None
            points_by_side[side] = np.asarray(
                [data[n][:3] for n in DEX_RETARGETING_HAND_JOINT_NAMES]
            )
        return points_by_side

    # Reuse the pinned recorder's state conversion and finalization, replacing only persistence.
    recorder.flush = lambda: (
        store.save(recorder._episodes[-1]) if recorder._episodes else None
    )
    try:
        env.sim.reset()
        reset()
        print("Teleop Device: handtracking · manual Start collection", flush=True)
        with torch.inference_mode():
            while ns["simulation_app"].is_running():
                now = time.monotonic()
                if (root / "stop.request").exists():
                    stop_requested = True
                if save_future is not None:
                    env.sim.render()
                    publish()
                    if not save_future.done() or now - saving_since < 0.8:
                        continue
                    try:
                        save_future.result()
                    except Exception as exc:
                        raise RuntimeError(
                            "Episode could not be saved: " + str(exc)
                        ) from exc
                    recorder._episodes.clear()
                    save_future = None
                    if not stop_requested:
                        reset()
                if stop_requested:
                    break
                if reset_requested:
                    reset_requested = False
                    reset()
                action = teleop.advance()
                points = read_hands()
                tracking = points is not None and now - heartbeat < 2.0
                show_raw = tracking
                if show_raw != raw_markers_visible:
                    for retargeter in teleop._retargeters:
                        retargeter._markers.set_visibility(show_raw)
                    raw_markers_visible = show_raw
                if phase == "recording" and not tracking:
                    if tracking_lost_since is None:
                        tracking_lost_since = now
                    if now - tracking_lost_since < 0.5:
                        instruction = "Tracking paused. Keep your hand visible."
                        env.sim.render()
                        publish()
                        continue
                    reset()
                    phase, interrupted_since = "interrupted", now
                    instruction = (
                        "Tracking lost. Episode discarded. Tap Start to retry."
                    )
                elif tracking:
                    tracking_lost_since = None
                    if phase == "recording":
                        instruction = goal
                if phase == "interrupted":
                    env.sim.render()
                    if tracking and now - interrupted_since >= 1.5:
                        phase, instruction = (
                            "ready",
                            "Tap Start to begin",
                        )
                if phase == "ready":
                    if not connected_once:
                        instruction = "Connect the headset to begin"
                    elif not tracking:
                        instruction = "Keep your hand visible to enable Start"
                    else:
                        instruction = "Tap Start to begin"
                    if start.consume(tracking):
                        wrist_commands.reset()
                        recorder.start_episode(
                            initial_state=env.scene.get_state(is_relative=True),
                            goal_pose=ns["_get_goal_pose_from_env"](env),
                            multi_assets=multi_assets,
                            multi_usds=multi_usds,
                            active_object_metadata=ns[
                                "_get_active_object_metadata_from_env"
                            ](env),
                        )
                        recorder._active_episode["episode_index"] = len(store.receipts)
                        recorder._active_episode["episode_name"] = (
                            f"demo_{len(store.receipts)}"
                        )
                        recorder._active_episode["skynet_start_pose"] = {
                            s: p.tolist() for s, p in points.items()
                        }
                        recorder._active_episode["skynet_retargeting"] = dict(
                            provider="dexverse",
                            mode="dexpilot",
                            wrist="absolute",
                            wrist_continuity="equivalent_euler_angles",
                        )
                        if cfg.get("image_capture"):
                            recorder._active_episode["skynet_wall_times"] = []
                        phase, instruction = "recording", goal
                        publish(True)
                        continue  # Begin simulation on the next tracked frame.
                    env.sim.render()
                elif phase == "recording":
                    action = torch.as_tensor(
                        wrist_commands.update(action.detach().cpu().numpy()),
                        device=action.device,
                        dtype=action.dtype,
                    )
                    if not torch.isfinite(action).all():
                        raise ValueError(
                            "Hand tracking produced an invalid robot action"
                        )
                    if cfg.get("image_capture"):
                        wall_times = recorder._active_episode["skynet_wall_times"]
                        if len(wall_times) >= 6000:
                            reset()
                            phase, interrupted_since = "interrupted", now
                            instruction = "Episode exceeded 100 seconds. Tap Start for a shorter demonstration."
                            continue
                        wall_times.append(time.time())
                    recorder.record_action(action.detach().clone())
                    result = env.step(action.repeat(env.num_envs, 1))
                    recorder.record_state(env.scene.get_state(is_relative=True))
                    success_count, success = ns["check_success"](
                        env, success_term, success_count
                    )
                    if success:
                        phase, instruction = "saving", "Episode ended. Saving…"
                        saving_since = time.monotonic()
                        publish(True)
                        env.sim.render()
                        save_future = executor.submit(recorder.finalize_episode, True)
                    elif bool(result[2].any()) or bool(result[3].any()):
                        reset()
                        phase, interrupted_since = "interrupted", now
                        instruction = "Object left the task area. Episode discarded. Tap Start to retry."
                publish()
                if env.sim.is_stopped():
                    break
        phase, instruction = (
            "ended",
            f"Collection ended. {len(store.receipts)} episodes saved.",
        )
        publish(True)
        # Let the connected client receive the final acknowledgement before closing the stream.
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and ns["simulation_app"].is_running():
            env.sim.render()
        return len(store.receipts)
    except Exception as exc:
        phase, instruction = "error", str(exc)
        publish(True)
        raise
    finally:
        teleop._skynet_subscription = None
        executor.shutdown(wait=True)


def main():
    cfg = json.loads(Path(os.environ["SKYNET_LIVE_CONFIG"]).read_text())
    root = Path(os.environ["SKYNET_COLLECTION_ROOT"])
    ns = runpy.run_path(
        str(Path(cfg["repository"]) / "scripts/record_demos.py"),
        run_name="skynet_upstream_recorder",
    )
    app = ns["simulation_app"]
    try:
        if cfg.get("image_capture"):
            from images import camera_recipe
            original_strip = ns["strip_camera_cfgs"]
            # Capture the recipe before upstream's XR path removes all cameras.
            def strip_cameras(env_cfg):
                cfg["image_recipe"] = camera_recipe(env_cfg)
                return original_strip(env_cfg)
            ns["main"].__globals__["strip_camera_cfgs"] = strip_cameras
            original_config = ns["create_environment_config"]
            def create_config():
                env_cfg, success = original_config()
                if "image_recipe" not in cfg:
                    cfg["image_recipe"] = camera_recipe(env_cfg)
                env_cfg = ns["prune_stale_obs_refs"](original_strip(env_cfg))
                return env_cfg, success
            ns["main"].__globals__["create_environment_config"] = create_config
        if cfg.get("hand_bundle"):
            sys.path.insert(0, cfg["hand_bundle"]["root"])
            from runtime import install, validate_environment

            cfg["hand_manifest"] = install(Path(cfg["hand_bundle"]["root"]))
            if cfg["hand_manifest"]["robot"] != cfg["robot"]:
                raise ValueError(
                    "Selected robot does not match the prepared hand bundle"
                )
            original_create = ns["create_environment"]

            def create(*args, **kwargs):
                env = original_create(*args, **kwargs)
                validate_environment(env, cfg["hand_manifest"])
                return env

            ns["main"].__globals__["create_environment"] = create
            original_init = ns["TrajectoryPickleRecorder"].__init__

            def init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                manifest = cfg["hand_manifest"]
                self._metadata["skynet_hand"] = {
                    k: manifest[k]
                    for k in (
                        "robot",
                        "hand_key",
                        "side",
                        "source_revision",
                        "digest",
                        "source_names",
                        "mimic_joints",
                    )
                }
                self._metadata["skynet_hand"]["action_joint_names"] = (
                    manifest["wrist_joints"] + manifest["finger_joints"]
                )

            ns["TrajectoryPickleRecorder"].__init__ = init
        ns["main"].__globals__["run_simulation_loop"] = lambda *a, **k: run_loop(
            ns, cfg, root, *a, **k
        )
        ns["main"]()
    except BaseException as exc:
        import traceback

        traceback.print_exc()
        print("SKYNET_COLLECTION_ERROR: " + str(exc), flush=True)
        if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
            atomic_json(root / "collection-error.json", {"error": str(exc)})
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
