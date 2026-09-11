"""Stream verified recordings to the pinned EgoVerse native episode writer."""

import json
from pathlib import Path

import numpy as np
import simplejpeg

from artifacts import digest, pack
from policy_export import source_data
from egoverse_zarr_writer import ZarrWriter
from egoverse_splits import OVERFIT_MODE, validate_split


def export(request):
    output = Path(request["output"])
    output.mkdir(exist_ok=False)
    split = request["split"]
    validate_split(split, len(request["sources"]))
    overfit = split.get("mode") == OVERFIT_MODE
    episodes, common = [], None
    for index, source in enumerate(request["sources"]):
        with source_data(source, True) as (meta, n, order, src):
            capture = {k: v for k, v in meta.items() if k != "source_sha256"}
            if common is not None and capture != common:
                raise ValueError(
                    "Robot, camera calibration or timing differs between episodes"
                )
            common = capture
            partition = "train" if index in split["train"] else "validation"
            relative = f"dataset/{partition}/episode_{index:07d}.zarr"
            encoded = {}
            for name in ("scene_front", "scene_left", "scene_right"):
                # Only compressed frames are accumulated; full RGB episodes need not fit in RAM.
                frames = np.empty(n, dtype=object)
                for frame in range(n):
                    frames[frame] = simplejpeg.encode_jpeg(
                        np.ascontiguousarray(src["images/" + name][frame]),
                        quality=ZarrWriter.JPEG_QUALITY,
                        colorspace="RGB",
                    )
                encoded[name] = (frames, [256, 256, 3])
            numeric = {
                "joint_positions": src["state"][:].astype("f4"),
                "actions_joints": src["action"][:].astype("f4"),
                "timestamps": src["timestamps"][:],
            }
            for key, original in (
                ("joint_positions", "state"),
                ("actions_joints", "action"),
            ):
                if not np.array_equal(numeric[key], src[original][:]):
                    raise ValueError(
                        "Numeric conversion would change recorded joint values"
                    )
            writer = ZarrWriter(
                output / relative,
                embodiment="skynet_joints",
                fps=1 / meta["step_dt"],
                task_name=meta["task"],
                task_description=meta["task"],
                annotations=[(meta["task"], 0, n - 1)],
            )
            writer.write(
                numeric_data=numeric,
                pre_encoded_image_data=encoded,
                metadata_override={
                    "skynet_capture": capture,
                    "source_sha256": source["sha256"],
                },
            )
            # Check the physical arrays independently of the writer's return value.
            import zarr

            saved = zarr.open_group(str(output / relative), mode="r")
            for key, values in numeric.items():
                if not np.array_equal(saved[key][:n], values):
                    raise ValueError("Native Zarr numeric roundtrip differs")
            episodes.append(
                dict(
                    index=index,
                    steps=n,
                    path=relative,
                    sha256=source["sha256"],
                    image_sha256=source["image_sha256"],
                    session_id=source["session_id"],
                    source_index=source["index"],
                )
            )
            print(
                json.dumps(
                    dict(
                        episodes_done=index + 1, episodes_total=len(request["sources"])
                    )
                ),
                flush=True,
            )
    (output / "split.json").write_text(json.dumps(split, indent=2))
    (output / "README.txt").write_text(
        "EgoVerse Zarr episodes. Native joint order and three calibrated scene cameras.\n"
        + ("Single-episode overfit: validation reuses the training episode; it is not a held-out score.\n"
           if overfit else "Train and validation episodes are separate.\n")
        + "Native training computes normalization from training data only.\n"
        "Compatible with native EgoVerse ACT and the HPT recorded-joints configuration.\n"
    )
    manifest = dict(
        format="egoverse-episodes-zarr/v1",
        contract=request["contract"],
        capture=common,
        episodes=episodes,
        steps=sum(e["steps"] for e in episodes),
        split=split,
        observations=["state", "rgb"],
        policy_to_source_indices=list(range(len(common["action_joint_names"]))),
        camera_slots={k: k for k in ("scene_front", "scene_left", "scene_right")},
        converter_sha256=request["converter_sha256"],
        source_revision=request["source_revision"],
        upstream=json.loads(
            (Path(__file__).parent / "egoverse-provenance.json").read_text()
        ),
        validation=dict(
            status="PASSED",
            checks=[
                "source_checksums",
                "frame_alignment",
                "native_zarr_roundtrip",
                "explicit_single_episode_overfit" if overfit else "disjoint_episode_split",
            ],
        ),
        files={},
    )
    for file in sorted(output.rglob("*")):
        if file.is_file():
            manifest["files"][str(file.relative_to(output))] = dict(
                sha256=digest(file), size_bytes=file.stat().st_size
            )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    pack(output)
    return manifest
