"""Skynet source-frame budgets; immutable upstream data and model stay unchanged.

A counted frame is a distinct (raw recording SHA256, original frame index) used
by an observation or action target. A 30 Hz view of 60 Hz data counts the even
indices it consumes, not the unconsumed frames between them. Whole-episode
validation splits are untouched. Segments never cross episode boundaries.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json

SCHEMA = "skynet.unidex-source-subset/v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _allocate(capacities, budget, horizon):
    """Select deterministic episode lengths, each zero or >= one full chunk.

    Suffix bitsets avoid a greedy final fragment shorter than the action chunk.
    Feasibility is exact, even when many episodes are only one chunk long.
    """
    if budget > sum(capacities):
        raise ValueError("Source-frame budget exceeds the available training frames")
    mask = (1 << (budget + 1)) - 1
    suffix = [0] * (len(capacities) + 1)
    suffix[-1] = 1
    for index in range(len(capacities) - 1, -1, -1):
        bits = suffix[index + 1]
        possible = bits
        for length in range(horizon, min(capacities[index], budget) + 1):
            possible |= bits << length
        suffix[index] = possible & mask
    if not (suffix[0] >> budget) & 1:
        raise ValueError("Source-frame budget cannot form complete within-episode action chunks")
    remaining, lengths = budget, []
    for index, capacity in enumerate(capacities):
        choices = list(range(min(capacity, remaining), horizon - 1, -1)) + [0]
        length = next(length for length in choices if (suffix[index + 1] >> (remaining - length)) & 1)
        lengths.append(length)
        remaining -= length
    assert remaining == 0
    return lengths


def apply_frame_budget(selections, sampling, unique_frames=None, *, selection_seed=20260920):
    """Return an exact equal-hand training subset, or the unmodified full view.

    The selection seed is independent of the model's initialization seed. A
    source set therefore receives the same data view across training repeats.
    No target/validation values or model results participate in selection.
    """
    if unique_frames is None:
        return sampling
    if type(unique_frames) is not int or not 1 <= unique_frames <= 10_000_000:
        raise ValueError("Unique training source-frame budget must be an integer in [1, 10000000]")
    if type(selection_seed) is not int or selection_seed < 0:
        raise ValueError("Source subset seed must be a nonnegative integer")
    if sampling.get("window_policy") != "complete":
        raise ValueError("Source frame budgets require complete action windows")
    rows = sampling.get("datasets", [])
    if len(rows) != len(selections):
        raise ValueError("Source subset selection and sampling identities differ")
    horizon = sampling["action_steps"]
    groups = defaultdict(list)
    sources, declared_hands = set(), set()
    for dataset_index, (selection, row) in enumerate(zip(selections, rows)):
        if any(selection.get(key) != row.get(key) for key in ("position", "version_id", "manifest_sha256")):
            raise ValueError("Source subset selection and sampling identities differ")
        metadata = selection["metadata"]
        for episode_index in metadata["split"]["train"]:
            episode = metadata["episodes"][episode_index]
            declared_hands.add(episode["hand_id"])
            source = (episode.get("source") or {}).get("sha256")
            if not isinstance(source, str) or len(source) != 64 or any(c not in "0123456789abcdef" for c in source):
                raise ValueError("A source subset requires every raw recording SHA256")
            if source in sources:
                raise ValueError("Duplicate raw recording in training source subset")
            sources.add(source)
            timing = row["sampling"]["episodes"][episode_index]
            if timing["windows"]:
                groups[episode["hand_id"]].append(dict(
                    dataset_index=dataset_index, episode_index=episode_index,
                    episode_id=episode["id"], source_sha256=source,
                    stride=timing["stride"], capacity=timing["sampled_steps"],
                    priority=_digest([selection_seed, source])))
    if set(groups) != declared_hands:
        raise ValueError("Every selected training hand needs at least one complete action window")
    if not groups or unique_frames % len(groups):
        raise ValueError("Source-frame budget must divide equally across the training hands")
    per_hand = unique_frames // len(groups)
    if per_hand < horizon:
        raise ValueError("Each hand needs at least one complete action chunk in its frame budget")
    result = deepcopy(sampling)
    for row in result["datasets"]:
        row["sampling"]["training_subset"] = {"schema": SCHEMA, "segments": []}
    hand_summaries, frame_identity = {}, []
    for hand in sorted(groups):
        candidates = sorted(groups[hand], key=lambda item: (item["priority"], item["source_sha256"]))
        lengths = _allocate([item["capacity"] for item in candidates], per_hand, horizon)
        total_windows = 0
        for item, length in zip(candidates, lengths):
            if not length:
                continue
            # A stable random offset avoids always taking an episode's opening
            # frames when only part fits, while staying on its original stride.
            offset = int(_digest([selection_seed, item["source_sha256"], "offset"]), 16) % (item["capacity"] - length + 1)
            start = offset * item["stride"]
            windows = length - horizon + 1
            segment = {key: item[key] for key in ("episode_index", "episode_id", "source_sha256", "stride")}
            segment.update(start=start, frames=length, windows=windows)
            result["datasets"][item["dataset_index"]]["sampling"]["training_subset"]["segments"].append(segment)
            frame_identity.extend((item["source_sha256"], frame) for frame in range(start, start + length * item["stride"], item["stride"]))
            total_windows += windows
        hand_summaries[hand] = dict(unique_source_frames=per_hand, windows=total_windows,
                                    episodes=sum(length > 0 for length in lengths))
    if len(frame_identity) != unique_frames or len(set(frame_identity)) != unique_frames:
        raise ValueError("Source subset does not have the requested unique source-frame union")
    for selection, row in zip(selections, result["datasets"]):
        plan = row["sampling"]
        segments = plan["training_subset"]["segments"]
        selected_ids = {segment["episode_id"] for segment in segments}
        plan["splits"]["train"] = dict(windows=sum(segment["windows"] for segment in segments),
            episodes=len(selected_ids), excluded_episodes=plan["splits"]["train"]["excluded_episodes"],
            subset_excluded_episodes=[plan["episodes"][index]["id"]
                                     for index in selection["metadata"]["split"]["train"]
                                     if plan["episodes"][index]["id"] not in selected_ids])
    result["splits"]["train"] = dict(windows=sum(row["windows"] for row in hand_summaries.values()),
                                      episodes=sum(row["episodes"] for row in hand_summaries.values()),
                                      excluded_episodes=sampling["splits"]["train"]["excluded_episodes"])
    result["source_subset"] = dict(schema=SCHEMA, selection_seed=selection_seed,
        unique_source_frames=unique_frames, per_hand=hand_summaries,
        source_frame_union_sha256=_digest(sorted(frame_identity)),
        unit="unique raw recording SHA256 and consumed original frame index",
        validation="unchanged whole-episode split")
    return result


def subset_windows(manifest, sampling):
    """Expand and validate a frozen training view before the shared reader runs."""
    subset = sampling.get("training_subset")
    if subset is None:
        return None
    if subset.get("schema") != SCHEMA:
        raise ValueError("Unknown UniDex source subset schema")
    allowed = set(manifest["split"]["train"])
    windows, seen = [], set()
    horizon = sampling["action_steps"]
    for segment in subset.get("segments", []):
        index, start, frames, stride = (segment.get(key) for key in ("episode_index", "start", "frames", "stride"))
        if any(type(value) is not int for value in (index, start, frames, stride)) or index not in allowed:
            raise ValueError("Source subset must reference training episodes only")
        episode = manifest["episodes"][index]
        if (index in seen or start < 0 or frames < horizon or stride != sampling["episodes"][index]["stride"]
                or start % stride or start + (frames - 1) * stride >= episode["steps"]
                or segment.get("episode_id") != episode["id"]
                or segment.get("source_sha256") != (episode.get("source") or {}).get("sha256")
                or segment.get("windows") != frames - horizon + 1):
            raise ValueError("Invalid or changed UniDex source subset segment")
        seen.add(index)
        windows.extend((index, offset) for offset in range(start, start + (frames - horizon + 1) * stride, stride))
    return windows
