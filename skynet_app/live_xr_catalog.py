"""Live choices audited against one pinned DexVerse release; no simulator imports."""

from copy import deepcopy
from .capture_processing.dexverse_runner import REVISION, TASK, ROBOT

HANDS = [
    {
        "key": "floating_shadow_right",
        "name": "Shadow · right hand",
        "side": "right",
        "available": True,
    },
    {
        "key": "floating_shadow_left",
        "name": "Shadow · left hand",
        "side": "left",
        "available": True,
    },
    {
        "key": "floating_shadow_bimanual",
        "name": "Shadow · both hands",
        "side": "both",
        "available": True,
    },
]
for key, name in (
    ("wuji-1", "WUJI Hand 1"),
    ("wuji-2", "WUJI Hand 2 (Beta 2)"),
    ("sharpa", "Sharpa Wave"),
    ("allegro-v4", "Allegro Hand V4"),
    ("leap-v1", "LEAP Hand V1"),
    ("inspire-rh56", "Inspire RH56"),
):
    HANDS.append(
        {
            "key": key,
            "name": name,
            "available": False,
            "reason": "A viewable hand model is stored in the Hands library, but this DexVerse release does not ship its simulation and live retargeting adapter.",
        }
    )

TASKS = [
    {
        "key": TASK,
        "name": "Pick up stick",
        "instructions": "Lift the stick at least 20 cm above where it started, turn it vertical (within 30°), and hold it briefly.",
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


def catalog():
    return deepcopy(
        {
            "source_revision": REVISION,
            "hands": HANDS,
            "tasks": TASKS,
            "default_robot": ROBOT,
            "default_task": TASK,
            "verified_pairs": [{"robot": ROBOT, "task": TASK}],
            "note": "Shadow right hand + Pick up stick has passed a real headset capture. Other listed Shadow combinations are provided by this release and await a headset test. Additional DexVerse tasks are not yet configured here.",
        }
    )


def selection(task, robot):
    task_info = next((item for item in TASKS if item["key"] == task), None)
    hand = next((item for item in HANDS if item["key"] == robot), None)
    if not task_info:
        raise ValueError(
            "Unsupported live task. Choose a task listed in Live teleoperation."
        )
    if not hand or not hand["available"]:
        raise ValueError(
            "Unsupported live hand. "
            + (hand["reason"] if hand else "Choose a listed Shadow hand.")
        )
    return deepcopy(task_info), deepcopy(hand)
