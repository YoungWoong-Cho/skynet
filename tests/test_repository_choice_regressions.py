from __future__ import annotations

import copy

import pytest

from skynet_app.adapters import (
    AdapterDefaults,
    adapter_manifest_sha256,
    builtin_adapter_manifests,
    canonical_adapter_manifest,
)
from skynet_app.pipeline_api import PipelineService
from skynet_app.source_control import SourceDiscovery


class AdapterRegistry:
    def __init__(self, manifest):
        self.version = {
            "id": "adapter-version-openpi-v1",
            "version_number": 1,
            "manifest": canonical_adapter_manifest(manifest),
        }
        self.record = {
            "id": "adapter-openpi",
            "selected_version": self.version,
            "versions": [self.version],
        }

    def get_adapter(self, adapter_id, *, include_versions=False):
        assert include_versions is True
        return self.record if adapter_id == self.record["id"] else None


def openpi_manifest():
    return next(
        manifest
        for manifest in builtin_adapter_manifests()
        if manifest.slug == "openpi"
    )


def service_with_registered_openpi():
    service = PipelineService.__new__(PipelineService)
    manifest = openpi_manifest()
    service.database = AdapterRegistry(manifest)
    return service, manifest


def test_registered_adapter_snapshot_replay_is_bound_to_exact_version_and_digest():
    service, manifest = service_with_registered_openpi()
    snapshot = canonical_adapter_manifest(manifest)
    source = {
        "adapter": "openpi",
        "adapter_id": "adapter-openpi",
        "adapter_version_id": "adapter-version-openpi-v1",
        "adapter_version": 1,
        "adapter_manifest": snapshot,
        "adapter_manifest_sha256": adapter_manifest_sha256(manifest),
    }

    resolved, selected_manifest, adapter_id = service._snapshot_adapter(
        source, "adapter-openpi"
    )

    assert adapter_id == "adapter-openpi"
    assert selected_manifest == manifest
    assert resolved["adapter_version_id"] == "adapter-version-openpi-v1"
    assert resolved["adapter_manifest"] == snapshot
    assert resolved["adapter_manifest_sha256"] == adapter_manifest_sha256(manifest)


def test_adapter_manifest_digest_is_stable_for_reordered_set_derived_fields():
    manifest = openpi_manifest()
    canonical = canonical_adapter_manifest(manifest)
    reordered = copy.deepcopy(canonical)
    reordered["runtime"]["allowed_backends"] = list(
        reversed(reordered["runtime"]["allowed_backends"])
    )
    reordered["capabilities"]["runtime_backends"] = list(
        reversed(reordered["capabilities"]["runtime_backends"])
    )

    assert canonical_adapter_manifest(reordered) == canonical
    assert adapter_manifest_sha256(reordered) == adapter_manifest_sha256(manifest)


def test_adapter_selection_by_slug_snapshots_both_stable_registry_ids():
    service, manifest = service_with_registered_openpi()
    record = service.database.record
    version = service.database.version
    service._adapter_selection = lambda requested, version_number: (
        record,
        version,
        manifest,
    )

    resolved, _, _ = service._snapshot_adapter(
        {"adapter": "openpi", "adapter_version": 1}, "openpi"
    )

    assert resolved["adapter_id"] == "adapter-openpi"
    assert resolved["adapter_version_id"] == "adapter-version-openpi-v1"


def test_canonical_caller_cannot_replace_registered_adapter_snapshot():
    service, manifest = service_with_registered_openpi()
    forged = copy.deepcopy(canonical_adapter_manifest(manifest))
    forged["display_name"] = "Caller-controlled OpenPI"
    source = {
        "adapter": "openpi",
        "adapter_id": "adapter-openpi",
        "adapter_version_id": "adapter-version-openpi-v1",
        "adapter_version": 1,
        "adapter_manifest": forged,
    }

    with pytest.raises(
        ValueError, match="does not match the immutable registered adapter version"
    ):
        service._snapshot_adapter(source, "adapter-openpi")


@pytest.mark.parametrize(
    "sweep_path",
    ["native.config.config_name", "native.config", "native"],
)
def test_repository_choice_fields_cannot_be_changed_after_validation_by_sweep(
    sweep_path,
):
    manifest = openpi_manifest()

    with pytest.raises(ValueError, match="repository-discovered choice fields cannot be swept"):
        PipelineService._reject_repository_choice_sweeps(
            {"sweep": {"parameters": {sweep_path: ["debug", "unregistered"]}}},
            manifest,
        )


def test_unrelated_sweep_path_remains_supported():
    PipelineService._reject_repository_choice_sweeps(
        {"sweep": {"parameters": {"train.learning_rate": [1e-4, 3e-4]}}},
        openpi_manifest(),
    )


@pytest.mark.parametrize("value", ["/outside", " //server/share "])
def test_project_subdirectories_reject_absolute_paths_before_normalization(value):
    with pytest.raises(ValueError, match="relative to the repository"):
        SourceDiscovery.project_subdirectory(value)
    with pytest.raises(ValueError, match="relative to the repository"):
        AdapterDefaults(workdir=value)
    with pytest.raises(ValueError, match="relative to the repository"):
        AdapterDefaults(project_subdirectory=value)
