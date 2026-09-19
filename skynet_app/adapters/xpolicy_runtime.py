"""Portable policy data contract, shared by training and rollout capsules."""

import json
import subprocess
import sys
from pathlib import Path

from recording_dataset import validate_manifest as validate_recording_manifest
from recording_time import resolve_sampling


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
    validate_recording_manifest(manifest)
    order = manifest["policy_to_source_indices"]
    capture = manifest["capture"]
    names = capture["action_joint_names"]
    if (not names or any(type(i) is not int for i in order)
            or sorted(order) != list(range(len(names))) or len(set(names)) != len(names)):
        raise ValueError("Dataset joint mapping is not a permutation")
    split = manifest["split"]
    if (
        sorted(split["train"] + split["validation"])
        != list(range(len(manifest["episodes"])))
        or not split["train"]
    ):
        raise ValueError(
            "Dataset requires disjoint, complete training and validation episodes"
        )
    if (
        capture.get("action_semantics")
        != "raw_joint_position_command; target = action * scale + offset"
    ):
        raise ValueError("Unsupported action representation")
    for episode in manifest["episodes"]:
        if episode.get("policy_to_source_indices") != order:
            raise ValueError("Recorded joint policies require the same joint mapping in every episode")
        for key in ("robot", "hand", "action_joint_names", "action_semantics", "action_scale", "action_offset"):
            if episode.get("capture", {}).get(key) != capture.get(key):
                raise ValueError("Recorded joint policies require the same capture contract in every episode: " + key)
        for key in ("state", "action"):
            if episode["streams"].get(key, {}).get("shape") != [episode["steps"], len(names)]:
                raise ValueError("Recorded state/action dimensions differ from the joint mapping")
    return manifest


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def checkpoint_sampling(manifest, config, saved, *, require_validation=False):
    """A checkpoint must execute the same frame grid and chunk used for training."""
    expected = resolve_sampling(manifest, config.get("control_hz"),
        action_steps=config.get("action_steps", 50), window_policy="pad",
        require_validation=require_validation)
    if saved != expected:
        raise ValueError("ACT checkpoint sampling differs from the experiment frequency, action chunk or dataset")
    return expected
