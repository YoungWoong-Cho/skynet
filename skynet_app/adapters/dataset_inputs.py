"""Portable resolution of immutable experiment dataset selections.

Submission and job capsules read the same frozen input document; this module
never looks up current registry rows or rewrites datasets.
"""
import hashlib
import json
import re
from copy import deepcopy


_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


def validate_selection_sources(selections):
    """Reject repeated inputs and source episodes, including split leakage."""
    versions, manifests, sources = set(), set(), {}
    for selection in selections:
        identity = selection.get("version_id")
        digest = selection.get("manifest_sha256")
        if (identity and identity in versions) or (digest and digest in manifests):
            raise ValueError("Duplicate dataset selection; select each dataset once")
        if identity:
            versions.add(identity)
        if digest:
            manifests.add(digest)
        metadata = selection.get("metadata") or {}
        episodes = metadata.get("episodes")
        if not isinstance(episodes, list):
            continue
        split = metadata.get("split") or {}
        membership = {}
        for name in ("train", "validation"):
            for index in split.get(name, []):
                if index in membership:
                    raise ValueError("Source episode appears in both training and validation splits")
                membership[index] = name
        for index, episode in enumerate(episodes):
            source = (episode.get("source") or {}).get("sha256")
            if not source:
                continue
            current_split = membership.get(index)
            if source in sources:
                if current_split != sources[source]:
                    raise ValueError("Source episode appears in both training and validation datasets")
                raise ValueError("Duplicate source recording across selected datasets; select each episode once")
            sources[source] = current_split


def _runtime_metadata(metadata):
    """Keep worker identity checks without copying a prepared dataset manifest.

    HAT verifies and loads the actual manifest using its frozen SHA256 before
    reading data. Its native configuration only needs episode provenance for
    leakage/held-out-hand checks and timing for the shared sampling resolver.
    Other contracts keep their metadata until their loaders declare the same
    behavior; custom adapter metadata must not be silently discarded.
    """
    if (metadata.get("format") != "skynet.recording-dataset/v1"
            or metadata.get("contract") != "skynet.hat-rgb-fingertips/v1"
            or (metadata.get("validation") or {}).get("status") != "PASSED"):
        return deepcopy(metadata)

    result = {key: deepcopy(metadata[key]) for key in (
        "format", "contract", "registered_version_id", "display_name", "validation", "split",
    ) if key in metadata}

    def capture_identity(capture):
        return {key: deepcopy(capture[key]) for key in ("robot", "step_dt") if key in capture}

    if "capture" in metadata:
        result["capture"] = capture_identity(metadata["capture"] or {})
    if "episodes" in metadata:
        result["episodes"] = []
        for episode in metadata["episodes"]:
            row = {key: deepcopy(episode[key]) for key in ("id", "index", "steps", "hand_id") if key in episode}
            if "source" in episode:
                row["source"] = {key: deepcopy(episode["source"][key]) for key in ("sha256",) if key in (episode["source"] or {})}
            if "capture" in episode:
                row["capture"] = capture_identity(episode["capture"] or {})
            result["episodes"].append(row)
    return result


def runtime_data_selection(selection):
    """Project a verified selection for workers that reopen its pinned manifest."""
    result = deepcopy(selection)
    if isinstance(result.get("metadata"), dict):
        result["metadata"] = _runtime_metadata(result["metadata"])
    return result


def runtime_evaluation_context(context):
    """Keep evaluation capsules small without changing immutable planning data.

    The HAT and DP bridges verify and reopen target/training manifests on compute
    nodes. Full camera, geometry and rendering receipts remain in the saved stage
    and those checksum-pinned files. Workers retain hand/source identities for
    leakage checks and the independently planned I/O and simulation contracts.
    Other evaluators keep their full context until they declare this behavior.
    """
    result = deepcopy(context)
    if (result.get("compatibility") or {}).get("policy_loader") not in {"hat_cartesian", "diffusion_policy_joints"}:
        return result
    native = (result.get("policy") or {}).get("native_config") or {}
    if isinstance(native.get("datasets"), list):
        native["datasets"] = [runtime_data_selection(item) for item in native["datasets"]]
    if isinstance(result.get("target_dataset"), dict):
        result["target_dataset"] = runtime_data_selection(result["target_dataset"])
    suite = result.get("suite") or {}
    config = suite.get("config") or {}
    if isinstance(config.get("target_dataset"), dict):
        projected = runtime_data_selection(config["target_dataset"])
        if projected != config["target_dataset"]:
            # Keep the original frozen configuration's provenance while ensuring
            # config_sha256 still describes the actual emitted configuration.
            suite["planning_config_sha256"] = suite.get("config_sha256") or _sha256(config)
            config["target_dataset"] = projected
            suite["config_sha256"] = _sha256(config)
    return result


def _sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def evaluation_context_json(context):
    """One serialization path for preview, initial dispatch and later attempts."""
    return json.dumps(runtime_evaluation_context(context), indent=2, sort_keys=True) + "\n"


def resolve_data_selections(document, role="training_data", *, runtime=False):
    """Return ordered, verified cluster selections from data.bundle.assignments."""
    assignments = ((document.get("data") or {}).get("bundle") or {}).get("assignments") or []
    if not isinstance(assignments, list) or any(not isinstance(item, dict) for item in assignments):
        raise ValueError("Dataset assignments must be objects")
    selected = [item for item in assignments if item.get("role") == role]
    positions = [item.get("position", 0) for item in selected]
    if any(type(value) is not int or value < 0 for value in positions) or len(positions) != len(set(positions)):
        raise ValueError("Dataset inputs require unique nonnegative positions")
    result = []
    for assignment in sorted(selected, key=lambda item: item.get("position", 0)):
        version = assignment.get("version") or {}
        metadata = version.get("metadata") or {}
        location = (assignment.get("config") or {}).get("location") or {}
        digest = version.get("manifest_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("Dataset selection requires a valid manifest SHA256")
        if (location.get("kind") != "cluster" or location.get("status") != "AVAILABLE"
                or location.get("manifest_sha256") != digest):
            raise ValueError("Choose a verified training-cluster copy of every dataset")
        path = location.get("path")
        if not isinstance(path, str) or not path or any(c in path for c in ("\0", "\n", "\r")):
            raise ValueError("Dataset selection requires a valid cluster path")
        if metadata.get("storage_location") == "workstation":
            raise ValueError("Dataset is stored on the collection workstation, not the training cluster")
        if version.get("format") == "skynet.episodes/v1":
            raise ValueError("Choose a prepared dataset, not its original recording reference")
        result.append(dict(position=assignment.get("position", 0),
                           version_id=metadata.get("registered_version_id"),
                           path=path, manifest_sha256=digest, metadata=metadata))
    validate_selection_sources(result)
    if runtime:
        result = [runtime_data_selection(item) for item in result]
    return result
