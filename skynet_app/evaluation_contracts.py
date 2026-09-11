"""Resolve dataset-bound evaluation tasks from an immutable training spec."""

import copy
from collections.abc import Mapping
from skynet_app.experiments import evaluation_task_catalog_sha256


def bound_metadata(spec, binding):
    assignments = ((spec.get("data") or {}).get("bundle") or {}).get("assignments", [])
    candidates = [a for a in assignments if a.get("role") == binding["role"]]
    if len(candidates) != 1:
        raise ValueError("Evaluation requires one registered training dataset")
    value = candidates[0].get("version", {}).get("metadata", {})
    for part in binding["metadata_path"].split("."):
        value = value.get(part) if isinstance(value, Mapping) else None
    return value


def bind_suite_to_dataset(suite, spec):
    result = copy.deepcopy(suite)
    config = result["config_json"]
    if config.get("initial_state") == "single_training_episode":
        binding = {"role": "training_data", "metadata_path": "episodes"}
        episodes = bound_metadata(spec, binding)
        split = bound_metadata(spec, {**binding, "metadata_path": "split"}) or {}
        if not isinstance(episodes, list) or len(episodes) != 1 or split.get("train") != [0]:
            raise ValueError("Recorded initial-state evaluation requires a single training episode")
    episode_binding = config.get("dataset_episode_binding")
    if episode_binding:
        if episode_binding.get("metadata_path") == "split.validation" and bound_metadata(
            spec, {**episode_binding, "metadata_path": "split.mode"}
        ) == "single_episode_overfit":
            raise ValueError("This overfit dataset reuses its training episode; there are no held-out episodes to evaluate")
        episodes = bound_metadata(spec, episode_binding)
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("The dataset has no held-out episodes" if episode_binding["metadata_path"] == "split.validation" else "The dataset has no training episodes")
        config["maximum_episodes_per_task"] = len(episodes)
    binding = config.get("dataset_task_binding")
    if not binding:
        return result
    value = bound_metadata(spec, binding)
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(c in value for c in "\x00\n\r")
    ):
        raise ValueError("The training dataset does not identify its simulation task")
    if config.get("task_selection_mode") == "subset" and config.get("tasks"):
        config["default_tasks"] = [value] if value in config["tasks"] else []
        return result
    config.update(
        tasks=[value],
        task_options=[{"id": value, "label": value}],
        task_catalog_complete=True,
        task_catalog_sha256=evaluation_task_catalog_sha256([value]),
    )
    return result
