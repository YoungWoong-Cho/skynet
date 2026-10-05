"""Freeze evaluation data independently of a policy's immutable training inputs."""

from copy import deepcopy

from . import data_selection
from .adapters.dataset_inputs import resolve_data_selections


def evaluation_target_contract(spec):
    """One contract for browser selection and server-side target validation."""
    return {
        "human-policy-hat": "skynet.hat-rgb-fingertips/v1",
    }.get((spec.get("source") or {}).get("adapter"))


def embodiment_ids(metadata):
    """Use recorded robot identities, including every episode of a mixture."""
    episodes = metadata.get("episodes") or []
    identities = set()
    for episode in episodes:
        identity = episode.get("hand_id") or (episode.get("capture") or {}).get("robot")
        identity = identity or (metadata.get("capture") or {}).get("robot")
        if not isinstance(identity, str) or not identity:
            raise ValueError("Every dataset episode must identify its recorded hand")
        identities.add(identity)
    if not identities:
        identity = (metadata.get("capture") or {}).get("robot")
        if not identity:
            raise ValueError("The dataset does not identify its recorded hand")
        identities.add(identity)
    return identities


def validate_evaluation_target(spec, target, unseen_embodiment=False):
    if not target:
        raise ValueError("Choose an evaluation dataset to fix the target hand, scene and camera")
    metadata = target.get("metadata") or {}
    if metadata.get("contract") != evaluation_target_contract(spec):
        raise ValueError("The evaluation target must be a prepared HAT RGB dataset")
    if metadata.get("format") != "skynet.recording-dataset/v1" or (metadata.get("validation") or {}).get("status") != "PASSED":
        raise ValueError("The evaluation dataset must pass shared recording-format validation")
    hands = embodiment_ids(metadata)
    if len(hands) != 1:
        raise ValueError("Choose an evaluation dataset containing exactly one hand")
    capture = metadata.get("capture") or {}
    for episode in metadata.get("episodes") or []:
        episode_capture = episode.get("capture") or capture
        for field in ("robot", "hand", "task", "source_revision", "action_joint_names", "action_scale", "action_offset"):
            if episode_capture.get(field) != capture.get(field):
                raise ValueError(f"Evaluation target episodes have different {field}; choose one consistent target")
    training = resolve_data_selections(spec)
    if not training:
        raise ValueError("The policy has no frozen training dataset inputs")
    if unseen_embodiment:
        training_hands = set().union(*(embodiment_ids(item["metadata"]) for item in training))
        if hands & training_hands:
            raise ValueError("The evaluation hand occurs in this run's training inputs; choose a held-out hand or disable Unseen hand")
    return next(iter(hands))


def attach_evaluation_target(database, suite, spec, target_dataset_id=None, unseen_embodiment=False):
    """Snapshot registry locations once; never mutate the training spec or suite row."""
    result = deepcopy(suite)
    adapter = (spec.get("source") or {}).get("adapter")
    if adapter == "diffusion-policy":
        # A joint policy retains its training embodiment; freeze its collected
        # scene and custom-hand bundle just as for Cartesian rollout targets.
        from .adapters.dp_simulation import validate_target
        selected = resolve_data_selections(spec)
        if len(selected) != 1 or unseen_embodiment:
            raise ValueError("DP joint rollout requires its single training hand")
        target = selected[0]
        if target_dataset_id and target_dataset_id != target.get("version_id"):
            raise ValueError("DP joint rollout cannot use a separate evaluation dataset")
        validate_target(target, (spec.get("native") or {}).get("config") or {})
        data_selection.assert_available(database, target)
        result["config_json"].update(target_dataset=target, unseen_embodiment=False)
        return result
    if adapter == "human-policy-hat" and not target_dataset_id:
        selected = resolve_data_selections(spec)
        if len(selected) != 1 or unseen_embodiment:
            raise ValueError("Choose a separate HAT evaluation target hand")
        validate_evaluation_target(spec, selected[0], False)
        data_selection.assert_available(database, selected[0])
        result["config_json"].update(target_dataset=selected[0], unseen_embodiment=False)
        return result
    if not target_dataset_id:
        if unseen_embodiment:
            raise ValueError("Unseen-hand evaluation requires a registered cross-embodiment policy bridge")
        return result
    if adapter != "human-policy-hat":
        raise ValueError("A separate evaluation dataset requires a registered HAT bridge")
    version = database.get_data_resource_version(target_dataset_id)
    if not version:
        raise ValueError("The evaluation dataset is unavailable")
    location = data_selection.verified_cluster_location(version)
    if not location:
        raise ValueError("Choose a verified cluster copy of the evaluation dataset")
    bundle = data_selection.snapshot(database, [dict(role="evaluation_target", position=0, version_id=target_dataset_id, location_id=location["id"])])
    target = resolve_data_selections({"data": {"bundle": bundle}}, role="evaluation_target")[0]
    validate_evaluation_target(spec, target, unseen_embodiment)
    data_selection.assert_available(database, target)
    result["config_json"].update(target_dataset=target, unseen_embodiment=unseen_embodiment)
    return result
