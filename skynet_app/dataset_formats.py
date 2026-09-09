"""Versioned conversion recipes and the training contracts they satisfy.

Format names and conversion outputs are defined here. Training requirements
and supported settings belong to the versioned adapter declarations.
An adapter can consume a recipe by declaring its contract in a data binding.
"""

XPL_COMMIT = "9c98a3aaf02d05c6f9999a5a0a7a42090555ddf3"
XPL_REPOSITORY = "https://github.com/XPolicyLab/XPolicyLab"
RECIPES = {
    "dp-state": dict(
        id="dp-state",
        name="Diffusion Policy · state",
        format="xpolicylab-dp-zarr/v1",
        container="Zarr",
        contract="skynet.dp-joints/v1",
        adapter="xpolicylab-dp",
        trainable=True,
        observations=["state"],
        description="Joint states and commands; no images required.",
    ),
    "dp": dict(
        id="dp",
        name="Diffusion Policy · RGB",
        observations=["state", "rgb"],
        format="xpolicylab-dp-zarr/v1",
        container="Zarr",
        contract="skynet.dp-rgb-joints/v1",
        adapter="xpolicylab-dp",
        trainable=True,
        description="Train a diffusion policy from the recorded joint commands and three calibrated scene views.",
    ),
    "act": dict(
        id="act",
        name="ACT",
        observations=["state", "rgb"],
        format="xpolicylab-act-hdf5/v1",
        container="HDF5",
        contract="skynet.act-rgb-joints/v1",
        adapter=None,
        trainable=False,
        description="ACT HDF5 episodes. Training in Skynet is not connected yet.",
    ),
    "xpolicylab": dict(
        id="xpolicylab",
        name="XPolicyLab intermediate HDF5",
        observations=["state", "rgb"],
        format="xpolicylab-demonstrations/v1",
        container="HDF5",
        contract="skynet.xpl-rgb-joints/v1",
        adapter=None,
        trainable=False,
        description="Shared images and joints for further policy-specific conversion.",
    ),
}


def catalog(database):
    adapters = [a for a in database.list_adapter_registry() if not a.get("archived_at")]
    manifests = {(a.get("latest_version") or {}).get("manifest", {}).get("slug", a.get("slug")): (a.get("latest_version") or {}).get("manifest", {}) for a in adapters}
    entries = [dict(value, available=True) for value in RECIPES.values()]
    for entry in entries:
        entry["compatible_adapters"] = [slug for slug, manifest in manifests.items() if any(
            entry["format"] in binding.get("formats", [])
            and entry["contract"] in (binding.get("contracts") or [binding.get("contract")])
            for field in manifest.get("train", {}).get("input_fields", [])
            if (binding := field.get("data_binding"))
        )]
        if entry["adapter"]:
            entry["trainable"] = entry["adapter"] in entry["compatible_adapters"]
            if entry["trainable"]:
                entry["training_setup"] = dict(
                    adapter=entry["adapter"],
                    repository=XPL_REPOSITORY,
                    revision=XPL_COMMIT,
                    runtime="existing",
                    runtime_profile="skynet-dp",
                    preset="dexverse-state/v1"
                    if entry["id"] == "dp-state"
                    else "rgb-joints/v2",
                )
            else:
                entry["description"] += (
                    " The training adapter is archived or missing; export remains available."
                )
    supported = {value["adapter"] for value in entries}
    for adapter in adapters:
        if adapter.get("archived_at"):
            continue
        manifest = (adapter.get("latest_version") or {}).get("manifest", {})
        slug = manifest.get("slug", adapter.get("slug", ""))
        if slug in supported:
            continue
        entries.append(
            dict(
                id=slug,
                name=manifest.get("display_name") or adapter.get("name") or slug,
                available=False,
                trainable=False,
                description=(
                    manifest.get("train", {}).get("data_requirements") or {}
                ).get(
                    "description",
                    "No converter is registered for this adapter’s declared data contract.",
                ),
            )
        )
    return entries
