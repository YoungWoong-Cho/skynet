"""Adapter definitions name cluster resources that must exist in the cluster configuration.

Built-in manifests carry GPU types and queue policies as part of their identity, so
the guard test exempts them; this test keeps those literals consistent with
config/clusters instead.
"""

import pytest

from skynet_app.adapters import builtin_adapter_manifests
from skynet_app.cluster_config import CLUSTER
from skynet_app.collection import ADAPTER_SEED_ROOT, CollectionAdapterManifest, CollectionResources

CONCRETE_GPU_ALIASES = {alias for alias, target in CLUSTER.gpu_aliases.items() if target}
PREREQUISITE_PATHS = {prerequisite.path for profile in CLUSTER.runtime_profiles.values()
                      for prerequisite in profile.source_prerequisites}


@pytest.mark.parametrize("manifest", builtin_adapter_manifests(), ids=lambda manifest: manifest.slug)
def test_builtin_adapter_resources_are_configured(manifest):
    resources = manifest.defaults.resources
    if resources.gpu_type is not None:
        assert resources.gpu_type in CONCRETE_GPU_ALIASES, resources.gpu_type
    if resources.queue_policy is not None:
        assert resources.queue_policy in CLUSTER.queues, resources.queue_policy
    if resources.gateway is not None:
        assert resources.gateway in ("auto", *CLUSTER.gateways), resources.gateway
    for recommendation in resources.gpu_recommendations:
        if recommendation.gpu_type is not None:
            assert recommendation.gpu_type in CONCRETE_GPU_ALIASES, recommendation.id


@pytest.mark.parametrize("path", sorted(ADAPTER_SEED_ROOT.glob("*.json")), ids=lambda path: path.stem)
def test_collection_seed_resources_and_checkouts_are_configured(path):
    defaults = CollectionAdapterManifest.model_validate_json(path.read_text()).defaults
    # The seed's resources must validate as a session's resources, including its GPU alias and gateway.
    resources = CollectionResources.model_validate(defaults["resources"])
    assert resources.gpu_type in CONCRETE_GPU_ALIASES, resources.gpu_type
    software = defaults["software"]
    for key, value in software.items():
        if key.endswith("_path") and value:
            assert value in PREREQUISITE_PATHS, f"{key} is not a configured runtime prerequisite: {value}"
