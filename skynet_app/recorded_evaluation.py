"""Pin the original scene belonging to an immutable, single-episode dataset."""

import json
from pathlib import PurePosixPath

from .evaluation_contracts import bound_metadata
from .live_xr_archive import archive_descriptor


def recorded_episode_sources(database, cluster, spec):
    binding = {"role": "training_data", "metadata_path": "episodes"}
    episodes = bound_metadata(spec, binding)
    split = bound_metadata(spec, {**binding, "metadata_path": "split"}) or {}
    if not isinstance(episodes, list) or len(episodes) != 1 or split.get("train") != [0]:
        raise ValueError("Recorded initial-state evaluation requires a single training episode")
    episode = episodes[0]
    identifier = episode.get("session_id")
    with database.connection() as connection:
        row = connection.execute(
            "SELECT payload_json FROM live_xr_sessions WHERE id = ?", (identifier,)
        ).fetchone()
    if row is None:
        raise ValueError("The training episode's original recording is no longer registered")
    job = json.loads(row[0])
    archive = archive_descriptor(job, cluster)
    index = episode.get("source_index")
    recordings = job.get("recordings", [])
    if type(index) is not int or not 0 <= index < len(recordings):
        raise ValueError("The training episode has no matching original recording")
    relative = recordings[index]
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or str(path) != relative or path.suffix != ".pkl":
        raise ValueError("Invalid original recording path")
    matching = [f for f in archive["manifest"]["files"] if f["path"] == "output/" + relative]
    if len(matching) != 1 or matching[0]["sha256"] != episode.get("sha256"):
        raise ValueError("Archived recording differs from the training episode")
    return [{
        "session_id": identifier,
        "source_index": index,
        "path": archive["root"] + "/" + relative,
        "sha256": episode["sha256"],
        "steps": episode["steps"],
        "archive_sha256": archive["manifest_sha256"],
    }]
