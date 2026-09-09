"""Bounded-memory XPolicyLab exports, using its pinned packing/decoding helpers.

DP/ACT schemas and resize conventions follow the unmodified converters in xpolicylab/.
All output is built in a private job directory and published only after validation.
"""

from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "xpolicylab"))
from XPolicyLab.utils.process_data import pack_robot_state, decode_image_bit
from arrays import ArrayUnpickler

FORMATS = json.loads((Path(__file__).parent / "formats.json").read_text())
SLOTS = {
    "cam_head": "scene_front",
    "cam_left_wrist": "scene_left",
    "cam_right_wrist": "scene_right",
}


from artifacts import digest as sha256, pack


def raw_episode(path):
    path = Path(path)
    if not 0 < path.stat().st_size <= 100_000_000:
        raise ValueError("Recording exceeds its size limit")
    payload = ArrayUnpickler(io.BytesIO(path.read_bytes())).load()
    if (
        payload.get("format") != "dexverse_trajectory"
        or payload.get("schema_version") != 3
        or payload.get("num_episodes") != 1
        or len(payload.get("episodes", [])) != 1
    ):
        raise ValueError(
            "Visual exports require one complete native episode per recording"
        )
    episode = payload["episodes"][0]
    if episode.get("success") is not True:
        raise ValueError("Only successful demonstrations can be exported")
    return payload, episode


def validate(source, h5):
    if h5.attrs.get("schema") != "skynet.rgb-trajectory/v1" or not h5.attrs.get(
        "complete"
    ):
        raise ValueError("Incomplete or unsupported image recording")
    meta = json.loads(h5.attrs["metadata"])
    payload, episode = raw_episode(source["recording"])
    if payload["robot_type"] != meta["robot"] or payload["task"] != meta["task"]:
        raise ValueError("Images belong to a different robot or task")
    image_receipt = episode.get("skynet_images", {})
    requested = payload.get("skynet_training_images")
    if requested is not None:
        recipe_hash = hashlib.sha256(
            json.dumps(
                requested.get("recipe"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        if (
            requested.get("schema") != "skynet.training-images/v2"
            or requested.get("mode") != "saved_states"
            or requested.get("recipe_sha256") != recipe_hash
            or meta.get("image_recipe_sha256") != recipe_hash
            or meta.get("render_mode") != "saved_states"
            or meta.get("source_sha256") != source["sha256"]
        ):
            raise ValueError(
                "Rendered images do not match the original recording and its frozen camera recipe"
            )
        if payload.get("skynet_step_dt") != meta["step_dt"] or not np.array_equal(
            h5["wall_times"][:], episode.get("skynet_wall_times")
        ):
            raise ValueError("Rendered image timing differs from collection")
    elif image_receipt.get("sha256") != source["image_sha256"]:
        raise ValueError("Image sidecar does not match its native recording")
    actions = np.asarray(episode["actions"])
    n, dimension = h5["action"].shape
    if (
        not 0 < n <= 6000
        or actions.shape != (n, dimension)
        or len(episode["states"]) != n + 1
        or episode["num_steps"] != n
    ):
        raise ValueError("Frames and native actions are not aligned")
    if not np.array_equal(h5["action"][:], actions):
        raise ValueError("Image actions differ from the original commands")
    names = meta["action_joint_names"]
    if len(names) != dimension or len(set(names)) != dimension:
        raise ValueError("Invalid action joint names")
    ids = [meta["robot_joint_names"].index(name) for name in names]
    states = np.stack(
        [
            np.asarray(s["articulation"]["robot"]["joint_position"])[0, ids]
            for s in episode["states"][:-1]
        ]
    )
    if h5["state"].shape != actions.shape or not np.array_equal(h5["state"][:], states):
        raise ValueError("Image states differ from the original pre-action states")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("Nonfinite robot data")
    dt = meta["step_dt"]
    if (
        not isinstance(dt, (int, float))
        or not np.isfinite(dt)
        or dt <= 0
        or h5["timestamps"].shape != (n,)
        or not np.allclose(h5["timestamps"][:], np.arange(n) * dt, rtol=0, atol=1e-9)
    ):
        raise ValueError("Image timestamps do not match control steps")
    if meta["color_space"] != "RGB":
        raise ValueError("Unsupported image color space")
    for name in SLOTS.values():
        ds = h5["images/" + name]
        if ds.dtype != np.uint8 or ds.shape != (n, 256, 256, 3):
            raise ValueError("Missing or misaligned 256 × 256 RGB camera: " + name)
    groups = meta["groups"]
    if [g["side"] for g in groups] != (
        ["left", "right"] if meta["hand"] == "both" else [meta["hand"]]
    ):
        raise ValueError("Unsupported hand grouping")
    order = [i for g in groups for i in g["wrist_indices"] + g["finger_indices"]]
    if any(
        len(g["wrist_indices"]) != 6 or not g["finger_indices"] for g in groups
    ) or sorted(order) != list(range(dimension)):
        raise ValueError("Joint mapping would lose or duplicate commands")
    return meta, n, order


@contextmanager
def source_data(source, visual):
    if sha256(source["recording"]) != source["sha256"]:
        raise ValueError("Source checksum verification failed")
    payload, episode = raw_episode(source["recording"])
    metadata = payload.get("skynet_state_metadata")
    if visual or not metadata:
        if not source.get("images"):
            raise ValueError("This older recording has no joint-layout metadata. Use its prepared dataset or finish its existing image extraction. New recordings save state metadata without images.")
        if sha256(source["images"]) != source["image_sha256"]:
            raise ValueError("Source checksum verification failed")
        with h5py.File(source["images"], "r") as src:
            meta, n, order = validate(source, src)
            yield meta, n, order, src
        return
    meta = metadata
    actions = np.asarray(episode["actions"])
    names = meta["action_joint_names"]
    if (actions.ndim != 2 or actions.shape[1] != len(names) or len(set(names)) != len(names)
            or len(episode["states"]) != len(actions) + 1 or episode["num_steps"] != len(actions)
            or not 0 < len(actions) <= 6000 or payload["robot_type"] != meta["robot"] or payload["task"] != meta["task"]):
        raise ValueError("Invalid joint state/action alignment")
    ids = [meta["robot_joint_names"].index(name) for name in names]
    states = np.stack([np.asarray(v["articulation"]["robot"]["joint_position"])[0, ids] for v in episode["states"][:-1]])
    if states.shape != actions.shape or not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite or misaligned state/action data")
    from images import joint_layout
    groups = joint_layout(names, meta["hand"], [names[i] for g in meta["groups"] for i in g["wrist_indices"]])
    if groups != meta["groups"]:
        raise ValueError("Joint mapping differs from the declared layout")
    order = [i for g in groups for i in g["wrist_indices"] + g["finger_indices"]]
    dt = meta["step_dt"]
    if not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid control timing")
    yield meta, len(actions), order, dict(state=states, action=actions, timestamps=np.arange(len(actions)) * dt)


def split(values, groups):
    result = {}
    for g in groups:
        arm = "joint_states" if len(groups) == 1 else g["side"] + "_arm_joint_states"
        hand = "ee_joint_states" if len(groups) == 1 else g["side"] + "_ee_joint_states"
        result[arm], result[hand] = (
            values[:, g["wrist_indices"]],
            values[:, g["finger_indices"]],
        )
    return result


def export(request):
    kind = request["format"]
    if kind == "egoverse":
        from egoverse_export import export as export_egoverse
        return export_egoverse(request)
    visual = kind != "dp-state"
    if visual:
        import cv2
    if kind not in FORMATS:
        raise ValueError("Unsupported export format")
    output = Path(request["output"])
    output.mkdir(exist_ok=False)
    dataset = output / "dataset"
    dataset.mkdir()
    dp = None
    total, episodes, common = 0, [], None
    training_values = {"state": [], "action": []}
    selection = request.get(
        "split", {"train": list(range(len(request["sources"]))), "validation": []}
    )
    for index, source in enumerate(request["sources"]):
        with source_data(source, visual) as (meta, n, order, src):
            if not visual:
                meta = {k: v for k, v in meta.items() if k not in {"cameras", "color_space", "image_recipe_sha256", "render_mode", "source_sha256"}}
            calibration = {k: v for k, v in meta.items() if k != "source_sha256"}
            if common is not None and common != calibration:
                raise ValueError(
                    "Camera calibration, robot layout, or timing differs between episodes"
                )
            common = calibration
            groups = meta["groups"]
            dims = dict(
                arm_dim=[len(g["wrist_indices"]) for g in groups],
                ee_dim=[len(g["finger_indices"]) for g in groups],
            )
            packed = {
                k: pack_robot_state(
                    {k: split(src[k][:], groups)},
                    "joint",
                    dims,
                    source_type="dataset",
                    state_type=k,
                ).astype("f4")
                for k in ("state", "action")
            }
            for key in packed:
                if not np.array_equal(packed[key][:, np.argsort(order)], src[key][:]):
                    raise ValueError(
                        "Joint reordering did not preserve the original data"
                    )
            if index in selection["train"]:
                for key in packed:
                    training_values[key].append(packed[key])
            if kind in {"dp", "dp-state"}:
                import zarr

                if dp is None:
                    dp = zarr.open_group(str(dataset / "demonstrations.zarr"), mode="w")
                    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
                    for key in ("state", "action"):
                        dp.create_dataset(
                            "data/" + key,
                            shape=(0, len(order)),
                            chunks=(100, len(order)),
                            dtype="f4",
                            compressor=compressor,
                        )
                    for key in (("head_camera", "left_camera", "right_camera") if visual else []):
                        dp.create_dataset(
                            "data/" + key,
                            shape=(0, 3, 240, 320),
                            chunks=(16, 3, 240, 320),
                            dtype="u1",
                            compressor=compressor,
                        )
                    dp.create_dataset(
                        "meta/episode_ends", shape=(0,), chunks=(100,), dtype="i8"
                    )
                for key in packed:
                    dp["data/" + key].append(packed[key])
                for slot, key in zip(
                    SLOTS if visual else [], ["head_camera", "left_camera", "right_camera"]
                ):
                    target = dp["data/" + key]
                    target.resize(total + n, 3, 240, 320)
                    for frame in range(n):
                        img = decode_image_bit(src["images/" + SLOTS[slot]][frame])
                        target[total + frame] = np.moveaxis(
                            cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA),
                            -1,
                            0,
                        )
                dp["meta/episode_ends"].append(np.array([total + n], dtype="i8"))
            else:
                raw = kind == "xpolicylab"
                path = dataset / (
                    f"episode_{index:07d}.hdf5" if raw else f"episode_{index}.hdf5"
                )
                with h5py.File(path, "x") as dst:
                    dst.attrs["skynet_metadata"] = json.dumps(meta)
                    if raw:
                        for key in packed:
                            for name, values in split(src[key][:], groups).items():
                                dst.create_dataset(
                                    key + "/" + name, data=values, dtype="f4"
                                )
                    else:
                        dst["action"] = packed["action"]
                        dst["observations/qpos"] = packed["state"]
                    dst["timestamps"] = src["timestamps"][:]
                    for slot, source_camera in SLOTS.items():
                        shape = (256, 256, 3) if raw else (480, 640, 3)
                        path = (
                            "vision/" + slot + "/colors"
                            if raw
                            else "observations/images/" + slot
                        )
                        target = dst.create_dataset(
                            path,
                            shape=(n, *shape),
                            dtype="u1",
                            chunks=(1, *shape),
                            compression="lzf",
                        )
                        for frame in range(n):
                            img = decode_image_bit(
                                src["images/" + source_camera][frame]
                            )
                            target[frame] = img if raw else cv2.resize(img, (640, 480))
            episodes.append(
                dict(
                    index=index,
                    steps=n,
                    sha256=source["sha256"],
                    image_sha256=source.get("image_sha256"),
                    session_id=source.get("session_id"),
                    source_index=source["index"],
                )
            )
            print(
                json.dumps(
                    {
                        "episodes_done": index + 1,
                        "episodes_total": len(request["sources"]),
                    }
                ),
                flush=True,
            )
            total += n
    if common is None:
        raise ValueError("No complete episodes to export")
    statistics = {}
    for key, chunks in training_values.items():
        values = np.concatenate(chunks)
        statistics[key] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
        }
    (output / "split.json").write_text(json.dumps(selection, indent=2))
    (output / "normalization.json").write_text(
        json.dumps(
            {
                "fit": "training_episodes_only",
                "train": selection["train"],
                "statistics": statistics,
            },
            indent=2,
        )
    )
    config = dict(robot=common["robot"], **dims)
    (output / "robot_config.json").write_text(json.dumps(config, indent=2))
    # Provide both configuration locations used by XPolicyLab data conversion and train.sh.
    for folder in [output / "env_cfg/robot", output / "XPolicyLab/utils/robot"]:
        folder.mkdir(parents=True)
        (folder / "_robot_info.json").write_text(json.dumps({"skynet": dims}))
    (output / "env_cfg/skynet.yml").write_text("config:\n  robot: skynet\n")
    if kind == "act":
        (output / "TASK_CONFIGS.json").write_text(
            json.dumps(
                {
                    "skynet": {
                        "dataset_dir": "dataset",
                        "num_episodes": len(episodes),
                        "episode_len": max(e["steps"] for e in episodes),
                        "camera_names": [
                            "cam_head",
                            "cam_right_wrist",
                            "cam_left_wrist",
                        ],
                    }
                },
                indent=2,
            )
        )
    readme = f"""Skynet policy dataset ({FORMATS[kind]})

{len(episodes)} successful episodes, {total} synchronized control steps.
The original commands are preserved. Bimanual vectors are reordered left then right;
use manifest.json policy_to_source_indices to restore simulator action order.
Wrist coordinates are six virtual joints (XYZ metres and Euler radians), not a physical arm.
Finger joints use radians. Raw commands retain the scales/offsets in capture metadata.

Camera slots: cam_head = fixed scene front; cam_left_wrist = fixed scene left;
cam_right_wrist = fixed scene right. These are three different SCENE cameras,
not wrist-mounted cameras. Evaluation must use the same calibrated views and robot mapping.

Training files are in dataset/. For DP use dataset/demonstrations.zarr. For ACT merge
TASK_CONFIGS.json into the policy configuration and set dataset_dir to the extracted
absolute dataset directory. For XPolicyLab shared demonstrations, copy dataset/ to
data/<benchmark>/<run>/skynet/data before running the selected policy converter.
Merge the provided skynet entries into env_cfg/robot/_robot_info.json and
XPolicyLab/utils/robot/_robot_info.json; copy env_cfg/skynet.yml for conversion.
Do not replace your existing robot registry files wholesale.

This export does not install or launch a policy trainer. Custom robot/action/camera
configuration is required; it does not imply compatibility with every pretrained policy.
"""
    if not visual:
        readme = f"Skynet state/action dataset ({FORMATS[kind]}). {len(episodes)} episodes, {total} steps.\nNo images are required. State is pre-action joint position; action is the original raw joint command.\nUse manifest.json for robot layout, action scale/offset, joint order, timing and split.\nThis is joint-state training, not a reproduction of task-specific benchmark observations.\n"
    (output / "README.txt").write_text(readme)
    provenance = json.loads(
        (Path(__file__).parent / "xpolicylab/provenance.json").read_text()
    )
    manifest = dict(
        format=FORMATS[kind],
        episodes=episodes,
        steps=total,
        capture=common,
        camera_slots=SLOTS if visual else {},
        observations=["state", "rgb"] if visual else ["state"],
        policy_to_source_indices=order,
        upstream=provenance,
        implementation="skynet-streaming-v1",
        converter_sha256=request.get("converter_sha256", sha256(__file__)),
        contract=request.get("contract"),
        split=selection,
        source_revision=request.get("source_revision"),
        validation={
            "status": "PASSED",
            "checks": [
                "source_checksums",
                "frame_alignment",
                "robot_and_camera_mapping",
                "finite_values",
                "lossless_joint_order",
                "training_only_normalization",
            ],
        },
        files={},
    )
    for p in sorted(output.rglob("*")):
        if p.is_file():
            manifest["files"][str(p.relative_to(output))] = dict(
                sha256=sha256(p), size_bytes=p.stat().st_size
            )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False)
    )
    pack(output)
    return manifest


if __name__ == "__main__":
    export(json.loads(Path(sys.argv[1]).read_text()))
