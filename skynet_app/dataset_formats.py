"""Recording conversion is declared by immutable registered adapter versions."""

from copy import deepcopy

from .training_contracts import RECORDING_DATASET_FORMAT, RecordingConversion

# Source identities used by adapter declarations, not a second recipe catalog.
XPL_COMMIT = "9c98a3aaf02d05c6f9999a5a0a7a42090555ddf3"
XPL_REPOSITORY = "https://github.com/XPolicyLab/XPolicyLab"


def _entry(adapter, version):
    manifest = version.get("manifest") or {}
    requirements = manifest.get("train", {}).get("data_requirements") or {}
    declared = requirements.get("recording_conversion")
    identity = dict(
        id=adapter["id"], adapter_id=adapter["id"],
        adapter_version_id=version["id"],
        adapter_version_number=version["version_number"],
        adapter_manifest_sha256=version.get("manifest_sha256"),
        adapter=manifest.get("slug"),
        name=manifest.get("display_name") or version.get("name") or adapter["name"],
        format=RECORDING_DATASET_FORMAT,
    )
    if not declared:
        return dict(identity, available=False, trainable=False, data_presets=[],
                    default_data_preset=None, description=requirements.get("description", ""),
                    reason="This adapter does not declare a recording conversion.")
    conversion = RecordingConversion.model_validate(declared)
    presets = [preset.model_dump(mode="json") for preset in conversion.presets]
    return dict(identity, available=True, trainable=True, data_presets=presets,
                default_data_preset=conversion.default_preset,
                description=requirements.get("description", ""), reason=None)


def catalog(database):
    """One registry query. Merely opening Convert performs no preparation."""
    return [
        _entry(adapter, adapter["latest_version"])
        for adapter in database.list_adapter_registry()
        if not adapter.get("archived_at") and adapter.get("enabled", True)
    ]


def resolve_adapter(database, adapter_id, adapter_version_id, preset=None):
    """Resolve the exact selected version; never silently upgrade it."""
    adapter = database.get_adapter(adapter_id, include_versions=True)
    if not adapter or adapter["id"] != adapter_id:
        raise ValueError("Choose a registered adapter")
    if adapter.get("archived_at") or not adapter.get("enabled", True):
        raise ValueError("The selected adapter is archived or unavailable")
    version = next((v for v in adapter.get("versions", []) if v["id"] == adapter_version_id), None)
    if version is None:
        raise ValueError("The selected adapter version is unavailable; refresh the adapter list")
    entry = _entry(adapter, version)
    if not entry["available"]:
        raise ValueError(entry["reason"])
    selected = preset if preset is not None else entry["default_data_preset"]
    choice = next((p for p in entry["data_presets"] if p["id"] == selected), None)
    if choice is None:
        raise ValueError("Choose a declared adapter data preset")
    result = {k:deepcopy(v) for k,v in entry.items() if k not in {"data_presets", "default_data_preset", "reason"}}
    result.update(deepcopy(choice))
    result.update(id=adapter_id, adapter_id=adapter_id, adapter_data_preset=selected,
                  name=entry["name"], data_preset_name=choice["name"],
                  format=RECORDING_DATASET_FORMAT,
                  capsule_files=deepcopy(version["manifest"].get("train", {}).get("capsule_files", {})))
    return result
