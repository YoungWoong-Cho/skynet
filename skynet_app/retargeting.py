"""Selectable collection solvers; physical hand assets stay in the Hands library."""

from copy import deepcopy

REVISION = "3846d3fa207165bb0d498145aac8b885a28ea923"
UTILS_REVISION = "2d8dc1a5abf5899069f9ec73c13de73674f4c897"
DEFAULT = "dexpilot"
VECTOR = "vector-wrist-joint"
METHODS = (
    {
        "key": DEFAULT,
        "name": "DexPilot",
        "description": "Direct wrist tracking with DexPilot finger retargeting.",
    },
    {
        "key": VECTOR,
        "name": "Vector Wrist Joint",
        "description": "Jointly adjusts the wrist and fingers to match fingertip position, direction and pinch.",
        "repository": "https://github.com/Mingrui-Yu/retargeting",
        "revision": REVISION,
    },
)


def choices():
    return deepcopy(METHODS)


def configuration(key, work_root):
    selected = next((m for m in METHODS if m["key"] == key), None)
    if selected is None:
        raise ValueError("Unknown retargeting method. Choose a listed method.")
    result = deepcopy(selected)
    result.update(schema="skynet.retargeting/v1", adapter_version=1)
    if key == VECTOR:
        result.update(
            adapter_version=4,
            utils_revision=UTILS_REVISION,
            runtime_root=f"{work_root}/retargeters/{VECTOR}/{REVISION}",
            # Rapid poses need more than the old 8 ms cutoff to converge. This
            # is the shared solver budget, not a guarantee of 60 Hz rendering.
            solve_budget_seconds=0.016,
            finger_velocity_weight=0.01,
            wrist_velocity_weight=0.1,
            hand_overrides={"wuji-2": {"finger_velocity_weight": 0.001}},
        )
    return result


def validate_hand(config, manifest):
    if config["key"] == DEFAULT:
        return
    if manifest.get("mimic_joints"):
        raise ValueError(
            "Vector Wrist Joint does not support this hand's coupled joints. Choose DexPilot."
        )
    for hand in manifest.get("hands", {}).values():
        if len(hand["wrist_joints"]) != 6 or not 2 <= len(hand["tips"]) <= 5:
            raise ValueError(
                "Vector Wrist Joint needs six wrist axes and two to five fingertips."
            )
    if not manifest.get("hands"):
        raise ValueError("Vector Wrist Joint requires a pinned Skynet Hands model.")
