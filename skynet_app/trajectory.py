"""Recording identity and task-state restoration shared by packaged workers.

No simulator dependency is imported until a v1 episode is restored. Keep this
module self-contained: workers receive an immutable copy named trajectory.py.
"""


def validate_recorded_identity(payload, task=None, robot=None, schema=None):
    """Reading stored states does not require a runnable simulator action layout."""
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "dexverse_trajectory"
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] not in {3, 4, 5}
        or (schema is not None and payload["schema_version"] != schema)
        or (task is not None and payload.get("task") != task)
        or (robot is not None and payload.get("robot_type") != robot)
    ):
        raise ValueError("Unsupported native recording format, task or robot")
    layout = payload.get("action_layout")
    if layout is not None:
        if (
            not isinstance(layout, dict)
            or type(layout.get("dimension")) is not int
            or layout["dimension"] < 1
        ):
            raise ValueError("Invalid recorded action layout")
        for episode in payload.get("episodes", []):
            shape = getattr(episode.get("actions"), "shape", ())
            if len(shape) != 2 or shape[1] != layout["dimension"]:
                raise ValueError(
                    "Recorded actions differ from their saved action layout"
                )
    return payload["schema_version"]


def validate_identity(payload, task=None, robot=None, schema=None):
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "dexverse_trajectory"
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] not in {3, 5}
        or (schema is not None and payload["schema_version"] != schema)
        or (task is not None and payload.get("task") != task)
        or (robot is not None and payload.get("robot_type") != robot)
    ):
        raise ValueError("Unsupported native recording format, task or robot")
    if str(payload.get("task", "")).endswith("-v1") and payload["schema_version"] != 5:
        raise ValueError(
            "This v1 recording lacks the schema 5 replay contract; its saved values can still be reviewed"
        )
    if payload["schema_version"] == 5:
        layout = payload.get("action_layout")
        if (
            not isinstance(layout, dict)
            or type(layout.get("dimension")) is not int
            or layout["dimension"] < 1
            or not isinstance(layout.get("robot_joint_names"), list)
            or not layout["robot_joint_names"]
            or not isinstance(payload.get("benchmark_revision"), str)
            or not payload["benchmark_revision"]
            or type(payload.get("task_version")) is not int
            or payload["task_version"] not in {0, 1}
        ):
            raise ValueError(
                "Recording is missing its schema 5 environment or action layout"
            )
        for episode in payload.get("episodes", []):
            shape = getattr(episode.get("actions"), "shape", ())
            if len(shape) != 2 or shape[1] != layout["dimension"]:
                raise ValueError(
                    "Recorded actions differ from their saved action layout"
                )
            if "task_state" in episode and not isinstance(episode["task_state"], dict):
                raise ValueError("Invalid recorded task state")
    return payload["schema_version"]


def restore_episode_conditions(env, payload, episode):
    """Restore hidden goal/task buffers after scene reset, before observations."""
    if payload.get("schema_version") != 5:
        return
    from dexverse.benchmark import validate_replay_identity
    from dexverse.teleop_utils.episode_task_state import (
        reset_managers_after_state_restore,
        restore_episode_task_state,
    )

    task = getattr(env.cfg, "env_name", None) or payload["task"]
    validate_replay_identity(payload, task)
    skipped = restore_episode_task_state(
        env, episode.get("task_state"), legacy_goal_pose=episode.get("goal_pose")
    )
    if skipped:
        raise ValueError(
            "Could not restore recorded task conditions: " + ", ".join(skipped)
        )
    reset_managers_after_state_restore(env)
