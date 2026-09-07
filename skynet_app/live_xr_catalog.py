"""Live choices audited against one pinned DexVerse release; no simulator imports."""

from copy import deepcopy
from .capture_processing.dexverse_runner import REVISION, TASK, ROBOT
from .simulation_hands import definitions

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
# These adapters use the pinned mesh/physics descriptions from the Hands library.

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
            "note": "Shadow right hand + Pick up stick has passed a real headset capture. Other Shadow combinations and imported hand adapters await a headset test. Imported hands reuse the stored URDFs and meshes; first startup converts them to simulator assets and checks their joint/body mappings. Additional DexVerse tasks are not yet configured here.",
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
            + (
                hand["reason"]
                if hand
                else "Choose a hand listed in Live teleoperation."
            )
        )
    return deepcopy(task_info), deepcopy(hand)
