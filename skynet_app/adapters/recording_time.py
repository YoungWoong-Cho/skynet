"""Skynet experiment-time sampling; stored recording streams stay unchanged.

This module has no array/GPU dependencies so submission and compute workers use
the same eligibility checks against the immutable dataset manifest.
"""
import math


def source_frequency(episode, manifest=None):
    capture = episode.get("capture") or (manifest or {}).get("capture") or {}
    dt = capture.get("step_dt")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Recording capture.step_dt must be finite and positive")
    hz = 1.0 / dt
    if not math.isfinite(hz):
        raise ValueError("Recording frequency must be finite")
    return hz


def frequency_stride(source_hz, control_hz=None):
    if control_hz is None:
        return 1
    if isinstance(control_hz, bool) or not isinstance(control_hz, (int, float)) or not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("Control frequency must be finite and positive")
    ratio = source_hz / control_hz
    if not math.isfinite(ratio):
        raise ValueError("Control frequency is too small for the recording frequency")
    stride = round(ratio)
    if stride < 1 or not math.isclose(ratio, stride, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Control frequency {control_hz:g} Hz must divide the recording frequency {source_hz:g} Hz exactly; interpolation is not supported")
    return stride


def resolve_sampling(manifest, control_hz=None, action_steps=1, *, window_policy="complete", require_validation=False, require_training=True):
    if type(action_steps) is not int or action_steps < 1:
        raise ValueError("Action chunk must be a positive integer")
    if window_policy not in {"complete", "pad"}:
        raise ValueError("Unknown recording window policy")
    episodes = manifest.get("episodes") or []
    split = manifest.get("split") or {}
    train, validation = split.get("train", []), split.get("validation", [])
    if not episodes or not train or any(type(i) is not int for i in train + validation) or sorted(train + validation) != list(range(len(episodes))):
        raise ValueError("Dataset splits must be disjoint, complete and contain training episodes")
    rows, ids, rates = [], set(), []
    for i, episode in enumerate(episodes):
        identifier, steps = episode.get("id"), episode.get("steps")
        if not isinstance(identifier, str) or not identifier or identifier in ids or episode.get("index") != i or type(steps) is not int or steps < 1:
            raise ValueError("Recording episodes require unique IDs, consecutive indices and positive lengths")
        ids.add(identifier)
        hz = source_frequency(episode, manifest)
        stride = frequency_stride(hz, control_hz)
        count = (steps + stride - 1) // stride
        windows = max(0, count - action_steps + 1) if window_policy == "complete" else count
        rows.append(dict(index=i, id=identifier, source_hz=hz, stride=stride,
                         source_steps=steps, sampled_steps=count, windows=windows))
        rates.append(hz)
    if control_hz is None and any(not math.isclose(hz, rates[0], rel_tol=1e-6) for hz in rates[1:]):
        raise ValueError("Recordings have different source frequencies; choose a common control frequency for this experiment")
    effective_hz = rates[0] if control_hz is None else float(control_hz)
    summaries = {}
    for name, indices in (("train", train), ("validation", validation)):
        selected = [rows[i] for i in indices]
        summaries[name] = dict(windows=sum(row["windows"] for row in selected),
                               episodes=sum(row["windows"] > 0 for row in selected),
                               excluded_episodes=[row["id"] for row in selected if not row["windows"]])
        if ((name == "train" and require_training) or (name == "validation" and require_validation)) and not summaries[name]["windows"]:
            raise ValueError(f"No usable {name} windows at {effective_hz:g} Hz with action chunk {action_steps}. Choose a shorter chunk, a higher supported frequency, or different episode splits.")
    return dict(schema="skynet.recording-sampling/v1", control_hz=effective_hz,
                action_steps=action_steps, window_policy=window_policy,
                episodes=rows, splits=summaries)


def resolve_collection_sampling(selections, control_hz=None, action_steps=1, *, window_policy="complete", require_validation=False):
    """One physical time scale, independent windows in each frozen dataset."""
    try:
        from .dataset_inputs import validate_selection_sources
    except ImportError:
        from dataset_inputs import validate_selection_sources
    if not selections:
        raise ValueError("Choose at least one training dataset")
    validate_selection_sources(selections)
    rates = [source_frequency(episode, item["metadata"])
             for item in selections for episode in item["metadata"].get("episodes", [])]
    if not rates:
        raise ValueError("Training datasets require recording episodes")
    if control_hz is None:
        if any(not math.isclose(rate, rates[0], rel_tol=1e-6) for rate in rates[1:]):
            raise ValueError("Recordings have different source frequencies; choose a common control frequency for this experiment")
        control_hz = rates[0]
    datasets = []
    summaries = {name: dict(windows=0, episodes=0, excluded_episodes=[]) for name in ("train", "validation")}
    for item in selections:
        plan = resolve_sampling(item["metadata"], control_hz=control_hz, action_steps=action_steps,
                                window_policy=window_policy, require_training=False, require_validation=False)
        datasets.append({key: item.get(key) for key in ("version_id", "manifest_sha256", "position")} | {"sampling": plan})
        for name, summary in plan["splits"].items():
            summaries[name]["windows"] += summary["windows"]
            summaries[name]["episodes"] += summary["episodes"]
            identity = item.get("version_id") or item["manifest_sha256"]
            summaries[name]["excluded_episodes"].extend(f"{identity}/{episode}" for episode in summary["excluded_episodes"])
    for name, summary in summaries.items():
        if (name == "train" or require_validation) and not summary["windows"]:
            raise ValueError(f"No usable {name} windows across selected datasets at {control_hz:g} Hz with action chunk {action_steps}.")
    return dict(schema="skynet.recording-sampling-collection/v1", control_hz=float(control_hz),
                action_steps=action_steps, window_policy=window_policy, datasets=datasets, splits=summaries)
