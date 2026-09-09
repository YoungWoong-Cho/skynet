"""Portable policy data contract, shared by training and rollout capsules."""

import json
import subprocess
import sys
from pathlib import Path


def repository(root, revision, policy):
    root = Path(root).resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision:
        raise ValueError("Policy source differs from the pinned revision")
    directory = root / "policy" / policy
    if not directory.is_dir():
        raise ValueError("Policy implementation is missing: " + policy)
    sys.path.insert(0, str(directory))
    return directory


def validate_manifest(manifest):
    order = manifest["policy_to_source_indices"]
    capture = manifest["capture"]
    names = capture["action_joint_names"]
    if sorted(order) != list(range(len(names))) or len(set(names)) != len(names):
        raise ValueError("Dataset joint mapping is not a permutation")
    split = manifest["split"]
    if (
        sorted(split["train"] + split["validation"])
        != list(range(len(manifest["episodes"])))
        or not split["train"]
        or not split["validation"]
    ):
        raise ValueError(
            "Dataset requires disjoint, complete training and validation episodes"
        )
    if (
        capture.get("action_semantics")
        != "raw_joint_position_command; target = action * scale + offset"
    ):
        raise ValueError("Unsupported action representation")
    return manifest


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def normalization(root, manifest):
    import numpy as np

    value = json.loads((Path(root) / "normalization.json").read_text())
    if (
        value["fit"] != "training_episodes_only"
        or value["train"] != manifest["split"]["train"]
    ):
        raise ValueError("Normalization must use only the registered training split")
    result = {}
    for kind in ("state", "action"):
        stats = value["statistics"][kind]
        for field in ("mean", "std"):
            array = np.asarray(stats[field], dtype=np.float32)
            if (
                array.shape != (len(manifest["policy_to_source_indices"]),)
                or not np.isfinite(array).all()
            ):
                raise ValueError("Invalid normalization statistics")
            result[kind + "_" + field] = (
                np.maximum(array, 0.01) if field == "std" else array
            )
    return result
