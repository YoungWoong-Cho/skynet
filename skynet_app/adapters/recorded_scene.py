"""Verify an immutable recording and restore its initial scene for policy rollout."""

import hashlib
import io
from pathlib import Path

from render_images import restore_state
from arrays import ArrayUnpickler


def load_recorded_episode(source, capture):
    path = Path(source["path"])
    if path.suffix != ".pkl" or not 0 < path.stat().st_size <= 100_000_000:
        raise ValueError("Invalid original recording")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("Original recording checksum changed")
    payload = ArrayUnpickler(io.BytesIO(raw)).load()
    from trajectory import validate_identity
    validate_identity(payload)
    if payload.get("task") != capture["task"] or payload.get("robot_type") != capture["robot"]:
        raise ValueError("Original recording differs from the training scene")
    episodes = payload.get("episodes", [])
    if len(episodes) != 1 or payload.get("num_episodes") != 1:
        raise ValueError("Expected one episode in the selected recording")
    episode = episodes[0]
    if episode.get("num_steps") != source["steps"] or len(episode.get("states", [])) != source["steps"] + 1:
        raise ValueError("The recorded episode has incomplete saved states")
    return payload, episode


def load_initial_state(source, capture):
    _, episode = load_recorded_episode(source, capture)
    return episode["states"][0]

