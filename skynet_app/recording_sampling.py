"""Submission-time view of the same sampling plan used by recording loaders."""
from .adapters.recording_time import resolve_sampling
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
    if len(selected) != 1:
        raise ValueError("Recording sampling requires one training dataset manifest")
    metadata = (selected[0].get("version") or {}).get("metadata") or {}
    config = (document.get("native") or {}).get("config") or {}
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
    return resolve_sampling(metadata, control_hz=None if hz == "" else hz,
                            action_steps=requirement.get("default_action_steps", 1) if chunk in (None, "") else chunk,
                            window_policy=requirement["window_policy"],
                            require_validation=requirement.get("require_validation", False))
