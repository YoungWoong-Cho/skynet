"""Submission-time view of the same sampling plan used by recording loaders."""
from .adapters.recording_time import resolve_sampling, resolve_collection_sampling
from .adapters.dataset_inputs import resolve_data_selections
from .training_contracts import RECORDING_DATASET_FORMAT


def experiment_sampling(document, manifest=None):
    declaration = (manifest.model_dump(mode="json") if manifest is not None else
                   document.get("source", {}).get("adapter_manifest") or {})
    requirement = (declaration.get("train", {}).get("data_requirements") or {}).get("recording_sampling")
    if not requirement:
        return None
    assignments = ((document.get("data") or {}).get("bundle") or {}).get("assignments") or []
    selected = [item for item in assignments if item.get("role") == "training_data"]
    if not selected:
        return None
    many = any((field.get("data_binding") or {}).get("role") == "training_data"
               and (field.get("data_binding") or {}).get("cardinality") == "many"
               for field in declaration.get("train", {}).get("input_fields", []))
    if len(selected) != 1 and not many:
        raise ValueError("Recording sampling requires one training dataset manifest for this adapter")
    metadata = (selected[0].get("version") or {}).get("metadata") or {}
    config = (document.get("native") or {}).get("config") or {}
    formats = [(item.get("version", {}).get("metadata") or {}).get("format") for item in selected]
    if many and any(value != RECORDING_DATASET_FORMAT for value in formats):
        raise ValueError("Multiple recording datasets must all use the shared recording format")
    if metadata.get("format") != RECORDING_DATASET_FORMAT:
        if config.get("control_hz") not in (None, ""):
            raise ValueError("Frequency overrides require a shared recording dataset")
        if declaration.get("slug", "").startswith("egoverse-") and config.get("action_steps") not in (None, ""):
            raise ValueError("Action chunk overrides require a shared recording dataset for this adapter")
        return None
    if metadata.get("contract") == "skynet.egoverse-rgb-joints/v1":
        from .adapters.egoverse_runtime import reject_sampling_overrides
        reject_sampling_overrides(config.get("model_overrides") or {}, config.get("model_preset"))
    hz = config.get("control_hz")
    chunk = config.get("action_steps")
    resolver = resolve_collection_sampling if many else resolve_sampling
    value = resolve_data_selections(document) if many else metadata
    sampling = resolver(value, control_hz=None if hz == "" else hz,
                        action_steps=requirement.get("default_action_steps", 1) if chunk in (None, "") else chunk,
                        window_policy=config.get("window_policy", requirement["window_policy"]) if declaration.get("slug") == "human-policy-hat" else requirement["window_policy"],
                        require_validation=requirement.get("require_validation", False))
    if many and declaration.get("slug") in {"unidex", "human-policy-hat"}:
        from .adapters.unidex_subset import apply_frame_budget
        if declaration.get("slug") == "unidex":
            from .adapters.unidex_input import collection_pointcloud_recipe
            collection_pointcloud_recipe(value)
        sampling = apply_frame_budget(value, sampling, config.get("unique_source_frames"),
                                      selection_seed=config.get("data_selection_seed", 20260920))
    return sampling
