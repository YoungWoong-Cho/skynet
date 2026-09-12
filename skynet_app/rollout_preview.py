"""Recover a linked demonstration for existing rollout traces on cluster storage."""
import hashlib
import io
import numpy as np
import inspect
import json
from pathlib import Path, PurePosixPath
import shlex

from .adapters import episode_geometry
from .adapters.episode_geometry import recorded_urdf, replay_urdf, HandKinematics, camera_layout
from .cluster_config import CLUSTER
from .live_xr_review import ArrayUnpickler
from .recorded_evaluation import recorded_episode_sources


def enrich_demonstration(database, cluster, spec, viewer, gateway, *, duration=0, task=None):
    from .evaluation_compatibility import dataset_metadata
    capture = dataset_metadata(spec).get("capture") or {}
    if task and task != capture.get("task"):
        return viewer
    if not capture or (viewer and viewer.get("demonstration") and viewer.get("kinematics_urdf")):
        return viewer
    try:
        sources = recorded_episode_sources(database, cluster, spec)
    except ValueError:
        return viewer
    runtime = next((p for p in CLUSTER.runtime_profiles.values() if "isaac_lab" in p.versions and p.environment_path), None)
    if runtime is None:
        return viewer
    repository = str(PurePosixPath(CLUSTER.paths.repositories) / "skynet-dexverse" / str(capture.get("source_revision", "")))
    request = dict(source=sources[0], capture=capture, viewer=viewer, repository=repository, duration=duration)
    program = (Path(episode_geometry.__file__).read_text() + "\nimport pickle,io,json,hashlib\n"
               + inspect.getsource(ArrayUnpickler) + "\n" + inspect.getsource(attach_demonstration)
               + "\nprint(json.dumps(attach_demonstration(" + repr(request) + "),allow_nan=False))\n")
    _, output = cluster.run_with_fallback(shlex.quote(runtime.environment_path + "/bin/python") + " -", gateway, stdin=program, timeout=45)
    return json.loads(output)


def attach_demonstration(request):
    source, capture = request["source"], request["capture"]
    path = Path(source["path"])
    if not 0 < path.stat().st_size <= 100_000_000:
        raise ValueError("Original recording exceeds the preview limit")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("Original demonstration checksum changed")
    payload = ArrayUnpickler(io.BytesIO(raw)).load()
    if payload.get("task") != capture["task"] or payload.get("robot_type") != capture["robot"]:
        raise ValueError("Original demonstration belongs to a different scene")
    episodes = payload.get("episodes", [])
    if len(episodes) != 1:
        raise ValueError("Original demonstration is ambiguous")
    states = episodes[0]["states"]
    if payload.get("format") != "dexverse_trajectory" or payload.get("schema_version") != 3 or len(states) != source["steps"] + 1 or len(states) > 6001:
        raise ValueError("Original demonstration has an unsupported recording layout")
    xml = recorded_urdf(capture, request["repository"])
    wrists = [capture["action_joint_names"][i] for g in capture.get("groups", []) for i in g["wrist_indices"]]
    model = HandKinematics(xml, capture["action_joint_names"], wrists)
    viewer = request["viewer"] or dict(schema="skynet.episode-viewer/v1", kind="rollout", robot=capture["robot"],
                                       hand=capture["hand"], views=camera_layout(capture.get("cameras", {}), labels=True), warnings=["This older rollout has no saved Actual or Prediction poses."],
                                       duration=request["duration"], frames=[])
    if viewer.get("point_names") and viewer["point_names"] != model.links:
        raise ValueError("Original demonstration and rollout have different hand layouts")
    dt = float(capture["step_dt"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Original demonstration has invalid timing")
    if not viewer["frames"]:
        viewer["frames"] = [{"time": i * dt} for i in range(min(len(states), 6001))]
    ids = [capture["robot_joint_names"].index(n) for n in capture["action_joint_names"]]
    for frame in viewer["frames"]:
        index = int(round(frame["time"] / dt))
        if index >= len(states):
            continue
        state = states[index]["articulation"]["robot"]
        joints = np.asarray(state["joint_position"]).reshape(-1)[ids]
        root = np.asarray(state["root_pose"]).reshape(-1)
        frame["demonstration"] = model.points(joints, root)
        frame.setdefault("hand_poses", {})["demonstration"] = {"joints": joints.tolist(), "root": root.tolist()}
    viewer.update(kinematics_urdf=replay_urdf(xml), joint_names=capture["action_joint_names"], edges=model.edges, point_names=model.links,
                  demonstration={k:source[k] for k in ("session_id","source_index","sha256")})
    viewer["warnings"] = [w for w in viewer.get("warnings", []) if not w.startswith("No single original demonstration")]
    if any(f.get("actual") for f in viewer["frames"]) and not any(f.get("hand_poses", {}).get("actual") for f in viewer["frames"]):
        viewer["warnings"].append("This older rollout saved Actual keypoints but no Actual joint poses. The robot mesh shows Demonstration; Actual remains a keypoint overlay.")
    return viewer
