"""Versioned conversion recipes and the training contracts they satisfy.

Format names, compatibility explanations and UI choices are defined here only.
An adapter can consume a recipe by declaring its contract in a data binding.
"""

XPL_COMMIT = "9c98a3aaf02d05c6f9999a5a0a7a42090555ddf3"
XPL_REPOSITORY = "https://github.com/XPolicyLab/XPolicyLab"
RECIPES = {
    "dp": dict(
        id="dp",
        name="Diffusion Policy",
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
        format="xpolicylab-act-hdf5/v1",
        container="HDF5",
        contract="skynet.act-rgb-joints/v1",
        adapter=None,
        trainable=False,
        description="Prepare ACT episode files. An ACT training adapter is not installed yet.",
    ),
    "xpolicylab": dict(
        id="xpolicylab",
        name="XPolicyLab demonstrations",
        format="xpolicylab-demonstrations/v1",
        container="HDF5",
        contract="skynet.xpl-rgb-joints/v1",
        adapter=None,
        trainable=False,
        description="Prepare shared demonstrations for another XPolicyLab converter; additional policy requirements still apply.",
    ),
}
ADAPTER_REQUIREMENTS = {
    "dexmimicgen": "Requires a Robomimic dataset and a matching robot/environment configuration. A converter for these recordings is not available yet.",
    "egoverse": "Requires the selected EgoVerse configuration's observation and action layout. A converter for these recordings is not available yet.",
    "groot": "Requires GR00T LeRobot data and an embodiment mapping. The current GR1 bridge cannot consume Shadow joint commands.",
    "openpi": "Requires LeRobot data and matching normalization. The current LIBERO bridge expects state[8], actions[7] and a wrist camera; these recordings have another robot/camera layout.",
    "get_zero": "The current simulation and distillation workflows do not consume these teleoperation recordings.",
    "generic": "This adapter must declare its trainer's data contract before Skynet can select a converter.",
}


def catalog(database):
    adapters = [a for a in database.list_adapter_registry() if not a.get("archived_at")]
    active = {
        (a.get("latest_version") or {}).get("manifest", {}).get("slug", a.get("slug"))
        for a in adapters
    }
    entries = [dict(value, available=True) for value in RECIPES.values()]
    for entry in entries:
        if entry["adapter"]:
            entry["trainable"] = entry["adapter"] in active
            if entry["trainable"]:
                entry["training_setup"] = dict(
                    adapter=entry["adapter"],
                    repository=XPL_REPOSITORY,
                    revision=XPL_COMMIT,
                    runtime="existing",
                    runtime_profile="skynet-dp",
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
                description=ADAPTER_REQUIREMENTS.get(
                    slug,
                    "No converter is registered for this adapter's declared data contract.",
                ),
            )
        )
    return entries
