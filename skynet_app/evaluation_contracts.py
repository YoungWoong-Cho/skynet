"""Resolve dataset-bound evaluation tasks from an immutable training spec."""

import copy
from collections.abc import Mapping
from skynet_app.experiments import evaluation_task_catalog_sha256


def bind_suite_to_dataset(suite, spec):
    result = copy.deepcopy(suite)
    config = result["config_json"]
    binding = config.get("dataset_task_binding")
    if not binding:
        return result
    assignments = (spec.get("data") or {}).get("bundle", {}).get("assignments", [])
    candidates = [a for a in assignments if a.get("role") == binding["role"]]
    if len(candidates) != 1:
        raise ValueError("Evaluation requires one registered training dataset")
    value = candidates[0].get("version", {}).get("metadata", {})
    for part in binding["metadata_path"].split("."):
        value = value.get(part) if isinstance(value, Mapping) else None
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(c in value for c in "\x00\n\r")
    ):
        raise ValueError("The training dataset does not identify its simulation task")
    config.update(
        tasks=[value],
        task_options=[{"id": value, "label": value}],
        task_catalog_complete=True,
        task_catalog_sha256=evaluation_task_catalog_sha256([value]),
    )
    return result
