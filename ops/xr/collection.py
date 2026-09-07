"""Repeated, alignment-gated episodes for the pinned DexVerse recorder.

This module is frozen into each session capsule. Isaac is imported only by main.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import runpy
import signal
import sys
import time

import numpy as np


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
        self.receipts.append(receipt)
        atomic_json(self.root / "episodes.json", self.receipts)
        return receipt


class AlignmentGate:
    def __init__(self, duration=0.8):
        self.duration = duration
        self.since = None

    def update(self, aligned, now):
        if not aligned:
            self.since = None
            return 0.0
        if self.since is None:
            self.since = now
        return min(1.0, max(0.0, (now - self.since) / self.duration))


def aligned_hand(points, side, target):
    from anatomy import palm_frame

    p = np.asarray(points)
    frame = palm_frame(p, side)
    if np.linalg.norm(p[0] - target) > 0.08:
        return False
    if frame[0, 0] < math.cos(math.radians(20)) or frame[2, 2] < math.cos(
        math.radians(25)
    ):
        return False
    # Check extension without requiring the human to match robot finger lengths.
    for base, tip in ((5, 8), (9, 12), (13, 16), (17, 20)):
        vector = p[tip] - p[base]
        if (
            np.linalg.norm(vector) < 0.025
            or np.dot(vector, frame[:, 0]) / np.linalg.norm(vector) < 0.75
        ):
            return False
    return True


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

    store = EpisodeStore(root, recorder._metadata)
    executor = ThreadPoolExecutor(max_workers=1)
    gate = AlignmentGate()
    phase, instruction, progress = "aligning", "Align the hand with the robot hand", 0.0
    goal = cfg["instructions"]
    last_write, heartbeat, save_future, stop_requested = 0.0, 0.0, None, False
    saving_since = 0.0
    interrupted_since = 0.0
    tracking_lost_since = None
    reset_requested, success_count = False, 0
    connected_once = False
    previous_positions, previous_sample = {}, 0.0
    tracked_sides = ["left", "right"] if cfg["hand"] == "both" else [cfg["hand"]]
    targets = {}
    raw_data = {}
    get_raw = teleop._get_raw_data

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
                alignment=progress,
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
    # Older clients cannot bypass the alignment gate with Play.
    teleop.add_callback("START", lambda: None)
    teleop.add_callback("STOP", stop)
    teleop.add_callback(
        "RESET",
        lambda: command(
            type("Event", (), {"payload": {"message": {"command": "restart"}}})()
        ),
    )

    def reset():
        nonlocal phase, instruction, success_count, progress, previous_positions
        recorder.discard_episode()
        ns["handle_reset"](env)
        teleop.reset()
        success_count, progress, previous_positions = 0, 0.0, {}
        gate.update(False, time.monotonic())
        robot = env.scene["robot"]
        for side in tracked_sides:
            # Imported palms retain their source name; native Shadow names differ by side.
            if cfg.get("hand_bundle"):
                name = cfg["hand_manifest"]["palm"]
            else:
                candidates = (
                    ("rh_palm", "lh_palm")
                    if side == "right"
                    else ("lh_palm", "rh_palm")
                )
                name = next((n for n in candidates if n in robot.body_names), None)
                if name is None:
                    name = env.cfg.robot_config.palm_body_name
            targets[side] = (
                robot.data.body_pos_w[0, robot.body_names.index(name)]
                .cpu()
                .numpy()
                .copy()
            )
        phase, instruction = "aligning", "Align the hand with the robot hand"
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
        print("Teleop Device: handtracking · automatic episode collection", flush=True)
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
                        "Tracking lost. Episode discarded. Align the hand again."
                    )
                elif tracking:
                    tracking_lost_since = None
                    if phase == "recording":
                        instruction = goal
                if phase == "interrupted":
                    env.sim.render()
                    if tracking and now - interrupted_since >= 1.5:
                        phase, instruction = (
                            "aligning",
                            "Align the hand with the robot hand",
                        )
                if phase == "aligning":
                    aligned = tracking
                    if points:
                        try:
                            for side, p in points.items():
                                aligned = aligned and aligned_hand(
                                    p, side, targets[side]
                                )
                                if side in previous_positions and now > previous_sample:
                                    aligned = (
                                        aligned
                                        and np.linalg.norm(
                                            p[0] - previous_positions[side]
                                        )
                                        / (now - previous_sample)
                                        < 0.08
                                    )
                                previous_positions[side] = p[0].copy()
                        except ValueError:
                            aligned = False
                    previous_sample = now
                    progress = gate.update(aligned, now)
                    if not connected_once:
                        instruction = "Connect the headset to begin"
                    elif not tracking:
                        instruction = "Keep your hand visible to the headset"
                    elif progress > 0:
                        instruction = "Hold still…"
                    else:
                        instruction = "Align the hand with the robot hand"
                    if progress >= 1:
                        for retargeter in teleop._retargeters:
                            retargeter.calibrate_wrist_pose()
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
                        recorder._active_episode["skynet_alignment"] = {
                            s: p.tolist() for s, p in points.items()
                        }
                        phase, instruction = "recording", goal
                        publish(True)
                        continue  # Recompute actions after wrist calibration before stepping physics.
                    env.sim.render()
                elif phase == "recording":
                    if not torch.isfinite(action).all():
                        raise ValueError(
                            "Hand tracking produced an invalid robot action"
                        )
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
                        instruction = "Object left the task area. Episode discarded. Align to retry."
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
