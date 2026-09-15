"""Live choices audited against one pinned DexVerse release; no simulator imports."""

from copy import deepcopy
from .dexverse_release import REVISION, TASK
from .hand_bundles import DEFAULT_ROBOT as ROBOT
from .hand_bundles import definitions
from .retargeting import choices, DEFAULT
from .dexverse_versions import collection_tasks

HANDS = []

for hand in definitions():
    HANDS.append(
        {
            "key": hand["robot"],
            "name": hand["name"],
            "side": hand["side"],
            "available": True,
            "imported": True,
            "hand_key": hand["key"],
            "source_revision": hand["revision"],
        }
    )
HANDS.append(
    {
        "key": "skynet_allegro_v4_left",
        "name": "Allegro Hand V4 · left hand",
        "available": False,
        "reason": "The pinned Allegro-left URDF references a missing thumb mesh. The right hand is available; no mirrored model is substituted.",
    }
)

TASKS = [
    {
        "key": TASK,
        "name": "Pick up stick",
        "instructions": "Lift the stick at least 20 cm, hold it vertical (within 30°, either end up), and keep it steady briefly.",
    },
    {
        "key": "Dexverse-PickCube-v0",
        "name": "Pick up cube",
        "instructions": "Grasp the blue cube, lift it at least 20 cm above its starting position, and hold it briefly.",
    },
    {
        "key": "Dexverse-StackCube-v0",
        "name": "Stack cubes",
        "instructions": "Place the movable cube on top of the other cube, align their centers, and hold the stack briefly.",
    },
    {
        "key": "Dexverse-RelocateSphere-v0",
        "name": "Move sphere to target",
        "instructions": "Grasp the sphere and move it to the displayed target marker. Keep it at the target briefly.",
    },
]


TASKS.extend(collection_tasks())

def catalog():
    return deepcopy(
        {
            "source_revision": REVISION,
            "hands": HANDS,
            "tasks": TASKS,
            "retargeters": choices(),
            "default_retargeter": DEFAULT,
            "default_robot": ROBOT,
            "default_task": TASK,
            "verified_pairs": [],
            "note": "All hands use the pinned Hands models with six wrist axes per hand. Simulator and teleoperation settings belong to the DexVerse adapter.",
        }
    )


def selection(task, robot, *, historical=False):
    task_info = next((item for item in TASKS if item["key"] == task), None)
    if task_info is None and historical:
        # Published v0 baselines are reviewable without adding live choices.
        from .dexverse_release import historical_task
        task_info = historical_task(task)
    hand = next((item for item in HANDS if item["key"] == robot), None)
    if (
        hand is None
        and historical
        and robot
        in {"floating_shadow_right", "floating_shadow_left", "floating_shadow_bimanual"}
    ):
        side = "both" if robot.endswith("bimanual") else robot.rsplit("_", 1)[1]
        hand = {
            "key": robot,
            "name": "Shadow · " + ("both hands" if side == "both" else side + " hand"),
            "side": side,
            "available": True,
            "historical": True,
        }
    if not task_info:
        raise ValueError(
            "Unsupported live task. Choose a task listed in Live teleoperation."
        )
    if not hand or not hand["available"]:
        raise ValueError(
            "Unsupported live hand. "
            + (
                hand["reason"]
                if hand
                else "Choose a hand listed in Live teleoperation."
            )
        )
    required = task_info.get('required_hand')
    if not historical and required and hand['side'] != required:
        label = 'both hands' if required == 'both' else required + ' hand'
        raise ValueError('This task requires ' + label)
    return deepcopy(task_info), deepcopy(hand)
