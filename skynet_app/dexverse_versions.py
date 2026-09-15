"""Pinned collection environments; task identity selects code and recording format."""

from pathlib import PurePosixPath

V1_REVISION = "917092d28764549f0f6a77872022c93ef3373c48"
V1_REPOSITORY = "repos/DexVerse-ce974ac6/" + V1_REVISION

# Ordered as upstream benchmark.BASELINE_V1_TASKS at V1_REVISION. Labels describe
# the v1 tasks, whose historical Gym names do not always describe the new goal.
V1_TASKS = (
    ("GraspBleach", "Grasp bleach bottle"),
    ("GraspPan", "Grasp pan"),
    ("GraspKettle", "Grasp kettle"),
    ("GraspCup", "Place cup under dispenser"),
    ("RemoveCupFromRack", "Remove cup from rack"),
    ("FunctionalPourCan", "Pour can"),
    ("FunctionalPourMug", "Pour mug"),
    ("FunctionalHammerStrike", "Hammer strike"),
    ("OpenFaucet", "Open faucet"),
    ("OpenDoor", "Open door"),
    ("OpenLaptop", "Open laptop"),
    ("SqueezeScissors", "Cut strip with scissors"),
    ("SlideUtilityKnife", "Cut seam with utility knife"),
    ("OpenStapler", "Reload stapler"),
    ("OpenFlatFolder", "Close folder and insert into shelf"),
    ("BimanualLiftTray", "Retrieve tray from rack"),
    ("BimanualLiftCarton", "Reorient carton"),
    ("InsertPen", "Insert pen"),
    ("PushSmallSphereObstacleSlope", "Push sphere through obstacles"),
    ("PushT", "Push T to target"),
)


def collection_tasks():
    return [
        {
            "key": f"Dexverse-{key}-v1",
            "name": f"{name} · v1",
            "task_version": 1,
            "recording_schema_version": 5,
            "required_hand": "both" if key.startswith("Bimanual") else "right",
            "instructions": f"{name}. Follow the goal and visual cues in the scene.",
        }
        for key, name in V1_TASKS
    ]


def environment_profile(profile, task, cluster_root=None):
    """Resolve a new session's environment; saved profiles remain immutable."""
    if task not in {f"Dexverse-{key}-v1" for key, _ in V1_TASKS}:
        return dict(profile, task_version=0, recording_schema_version=3)
    root = cluster_root or profile["work_root"]
    return dict(
        profile,
        repository=str(PurePosixPath(root) / V1_REPOSITORY),
        source_revision=V1_REVISION,
        task_version=1,
        recording_schema_version=5,
    )
