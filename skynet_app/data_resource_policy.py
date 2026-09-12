"""Keep file-set registration separate from recording-derived datasets."""

# API and UI use this catalog; migration constraints enforce the same pairs.
RESOURCE_TYPES = {
    "dataset": {
        "dataset": "Dataset",
        "demonstrations": "Demonstrations",
        "evaluation_data": "Evaluation data",
        "raw_capture": "Raw capture",
    },
    "file": {
        "simulation_assets": "Simulation assets",
        "embodiment_assets": "Hand / robot assets",
        "calibration": "Calibration",
        "checkpoint": "Checkpoint",
        "model": "Model",
        "file": "Other files",
    },
}
FILE_RESOURCE_KINDS = frozenset(RESOURCE_TYPES["file"])
RECORDING_LINK_KEYS = frozenset({"session_id", "recording_session_id"})


def validate_resource_type(category, kind):
    if kind not in RESOURCE_TYPES.get(category, {}):
        raise ValueError(f"Unsupported type {kind!r} for category {category!r}")


def validate_resource_metadata(category, metadata):
    metadata = metadata or {}
    sources = metadata.get("sources")
    linked_sources = isinstance(sources, list) and any(
        isinstance(source, dict) and RECORDING_LINK_KEYS.intersection(source)
        for source in sources
    )
    if category == "file" and (RECORDING_LINK_KEYS.intersection(metadata) or linked_sources):
        raise ValueError("Files cannot be linked to a recording; use a dataset for recorded episodes")


def resource_recording_ids(resource, versions):
    """Distinct original recording sessions across a dataset's conversion results."""
    if resource.get("category") != "dataset":
        return []
    metadata = resource.get("metadata") or {}
    owner = metadata.get("recording_session_id")
    legacy = metadata.get("session_id")
    identifiers = set()
    primary = owner or (legacy if resource.get("provider") == "collection" or metadata.get("managed_dataset") else None)
    if isinstance(primary, str) and primary:
        identifiers.add(primary)
    for version in versions:
        for source in (version.get("metadata") or {}).get("sources", []):
            if not isinstance(source, dict):
                continue
            identifier = source.get("session_id")
            # A copied one-episode recording owns its own conversion. Older
            # conversion receipts may still name its parent session.
            if owner and identifier == legacy:
                identifier = owner
            if isinstance(identifier, str) and identifier:
                identifiers.add(identifier)
    return sorted(identifiers)
