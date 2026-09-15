"""Historical DexVerse release identities; not additional live collection choices."""

from .dexverse_versions import V1_TASKS, environment_profile

V0_NAMES = {
    "GraspBleach": "Grasp bleach bottle",
    "GraspPan": "Grasp pan",
    "GraspKettle": "Grasp kettle",
    "GraspCup": "Grasp cup",
    "RemoveCupFromRack": "Remove cup from rack",
    "FunctionalPourCan": "Pour can",
    "FunctionalPourMug": "Pour mug",
    "FunctionalHammerStrike": "Hammer strike",
    "OpenFaucet": "Open faucet",
    "OpenDoor": "Open door",
    "OpenLaptop": "Open laptop",
    "SqueezeScissors": "Squeeze scissors",
    "SlideUtilityKnife": "Slide utility knife",
    "OpenStapler": "Open stapler",
    "OpenFlatFolder": "Open flat folder",
    "BimanualLiftTray": "Lift tray",
    "BimanualLiftCarton": "Lift carton",
    "InsertPen": "Insert pen",
    "PushSmallSphereObstacleSlope": "Push sphere through obstacles",
    "PushT": "Push T to target",
}


def historical_task(task):
    for version, names in ((0, V0_NAMES), (1, dict(V1_TASKS))):
        for key, label in names.items():
            if task == f"Dexverse-{key}-v{version}":
                return {
                    "key": task,
                    "name": f"{label} · v{version}",
                    "task_version": version,
                    "recording_schema_version": 5 if version else 3,
                }
    return None


def release_profile(target, task, cluster_root):
    info = historical_task(task)
    if info is None:
        raise ValueError("Unknown published DexVerse baseline task")
    profile = environment_profile(target, task, cluster_root=cluster_root)
    # Imported archives have no collection workstation or session work root.
    profile.pop("work_root", None)
    return dict(
        profile,
        task=task,
        task_name="DexVerse · " + info["name"],
        provider="huggingface",
        execution="slurm",
        gateway="sky2",
        image_capture=False,
    )
