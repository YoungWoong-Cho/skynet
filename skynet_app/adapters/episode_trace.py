"""Save replay geometry next to a rollout video, without retaining GPU tensors."""

import io
import json
from pathlib import Path

import numpy as np

from scene_geometry import scene_snapshot
from episode_geometry import HandKinematics, camera_layout, recorded_urdf, replay_urdf


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


class EpisodeTrace:
    def __init__(self, context, capture, env, cameras, video, *, policy_order):
        self.path = Path(video).with_suffix(".review.json")
        self.capture, self.env = capture, env
        self.order = list(policy_order)
        self.frames, self.warnings = [], []
        self.scene = scene_snapshot(env.scene)
        self.kinematics = None
        self.xml = None
        self.demo_dt = float(capture.get("step_dt", env.step_dt))
        self.demonstration = None
        self.demo_source = None
        self.action_ids = [env.scene["robot"].joint_names.index(name) for name in capture["action_joint_names"]]
        actual_cameras = {}
        for name, sensor_name in cameras.items():
            sensor = env.scene[sensor_name]
            d = sensor.data
            actual_cameras[name] = dict(width=int(d.output["rgb"].shape[2]), height=int(d.output["rgb"].shape[1]),
                                       intrinsic_matrix=array(d.intrinsic_matrices)[0].tolist(),
                                       position_world=array(d.pos_w)[0].tolist(), quaternion_world_ros=array(d.quat_w_ros)[0].tolist())
        self.views = camera_layout(actual_cameras, labels=True)
        try:
            xml = recorded_urdf(capture, context["evaluator_runtime"]["source_dir"], context.get("hand_bundle_path"))
            wrists = [capture["action_joint_names"][i] for g in capture.get("groups", []) for i in g["wrist_indices"]]
            self.kinematics = HandKinematics(xml, capture["action_joint_names"], wrists)
            self.xml = replay_urdf(xml)
        except (ValueError, OSError, KeyError) as error:
            self.warnings.append("Hand replay unavailable: " + str(error))
        sources = context.get("recorded_episode_sources") or []
        if len(sources) == 1 and self.kinematics:
            try:
                from arrays import ArrayUnpickler
                from recorded_scene import load_initial_state
                load_initial_state(sources[0], capture)  # Reuse the checksum and scene identity checks.
                self.demonstration = ArrayUnpickler(io.BytesIO(Path(sources[0]["path"]).read_bytes())).load()["episodes"][0]["states"]
                self.demo_source = {k: sources[0][k] for k in ("session_id", "source_index", "sha256")}
            except (OSError, ValueError, KeyError) as error:
                self.warnings.append("Demonstration unavailable: " + str(error))
        else:
            self.warnings.append("No single original demonstration is linked to this rollout.")

    def append(self, step, packed):
        frame = {"time": step * float(self.env.step_dt), "index": step}
        if self.kinematics:
            robot = self.env.scene["robot"]
            root = np.concatenate([array(robot.data.root_pos_w)[0], array(robot.data.root_quat_w)[0]])
            actual = array(robot.data.joint_pos)[0, self.action_ids]
            # The policy outputs raw commands in policy order; convert exactly once.
            command = np.empty(len(self.order), dtype=np.float32)
            command[self.order] = np.asarray(packed)
            target = command * np.asarray(self.capture["action_scale"]) + np.asarray(self.capture["action_offset"])
            frame["actual"] = self.kinematics.points(actual, root)
            frame["prediction"] = self.kinematics.points(target, root)
            frame["hand_poses"] = {"actual": {"joints": actual.tolist(), "root": root.tolist()},
                                   "prediction": {"joints": target.tolist(), "root": root.tolist()}}
            demo_index = int(round(frame["time"] / self.demo_dt))
            if self.demonstration is not None and demo_index < len(self.demonstration):
                state = self.demonstration[demo_index]["articulation"]["robot"]
                names = self.capture["robot_joint_names"]
                ids = [names.index(name) for name in self.capture["action_joint_names"]]
                value = np.asarray(state["joint_position"]).reshape(-1)[ids]
                frame["demonstration"] = self.kinematics.points(value, state["root_pose"])
                frame["hand_poses"]["demonstration"] = {"joints": value.tolist(), "root": array(state["root_pose"]).reshape(-1).tolist()}
        objects = getattr(self.env.scene, "rigid_objects", {})
        frame["objects"] = {name: np.concatenate([array(obj.data.root_pos_w)[0], array(obj.data.root_quat_w)[0]]).tolist()
                            for name, obj in objects.items()}
        self.frames.append(frame)

    def finish(self, steps):
        result = dict(schema="skynet.episode-viewer/v1", kind="rollout", robot=self.capture["robot"], hand=self.capture["hand"],
                      kinematics_urdf=self.xml, joint_names=self.capture["action_joint_names"],
                      scene_objects=self.scene["objects"], scene_appearance=self.scene.get("appearance"),
                      frames=self.frames, views=self.views, edges=self.kinematics.edges if self.kinematics else [],
                      point_names=self.kinematics.links if self.kinematics else [], duration=steps * float(self.env.step_dt),
                      layers={"prediction":"Commanded joint target", "actual":"Observed hand before the command", "demonstration":"Original recorded hand"},
                      demonstration=self.demo_source, alignment="Elapsed simulation time; demonstration is hidden after its final frame.",
                      warnings=self.warnings + self.scene.get("warnings", []))
        pending = self.path.with_suffix(".tmp")
        pending.write_text(json.dumps(result, allow_nan=False))
        pending.replace(self.path)
