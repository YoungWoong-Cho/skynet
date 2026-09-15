from __future__ import annotations

import base64
import copy
import json
import os
import re
import shlex
import threading
from pathlib import PurePosixPath
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode
from urllib.request import Request, urlopen

from .cluster_runtime import ClusterClient, WORK_ROOT


PREVIEW_SCHEMA = "skynet.data-bundle-preview/v1"
METADATA_LIMITS = {
    "meta/info.json": 524_288,
    "meta/modality.json": 524_288,
    "meta/tasks.jsonl": 1_048_576,
    "meta/episodes.jsonl": 2_097_152,
}
EVALUATION_ROLES = {"evaluation_data", "validation_data", "test_data"}
MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
_PREVIEW_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_PREVIEW_CACHE_LOCK = threading.Lock()


def _safe_registered_root(value: Any) -> str:
    root = PurePosixPath(str(value or ""))
    work_root = PurePosixPath(WORK_ROOT)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("registered data version path must be an absolute normalized path")
    if root != work_root and work_root not in root.parents:
        raise ValueError(f"registered data version path must be below {WORK_ROOT}")
    return str(root)


def _safe_relative_path(value: Any) -> str:
    raw = str(value or "")
    if not raw or any(character in raw for character in ("\x00", "\r", "\n")):
        raise ValueError("preview media path must be a nonempty single-line relative path")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("preview media path must stay inside its registered data version")
    return str(path)


def _bounded_http_text(url: str, limit: int) -> str:
    headers = {"User-Agent": "skynet-control-dataset-preview/1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=8) as response:
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"metadata file exceeds the {limit}-byte preview limit")
    return payload.decode("utf-8")


def _huggingface_content_base(
    version: Mapping[str, Any],
    resource: Mapping[str, Any],
) -> str | None:
    if str(resource.get("provider") or "").lower() != "huggingface":
        return None
    namespace = str(resource.get("namespace") or "").strip()
    source_key = str(resource.get("source_key") or "").strip()
    revision_value = str(version.get("revision") or "")
    revision, _, fragment = revision_value.partition("#")
    if not namespace or not source_key or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return None
    metadata = version.get("metadata") if isinstance(version.get("metadata"), dict) else {}
    subset = str(metadata.get("subset") or "").strip()
    if not subset and fragment:
        subset = str(parse_qs(fragment).get("subset", [""])[0]).strip()
    base = (
        "https://huggingface.co/datasets/"
        f"{quote(namespace, safe='')}/{quote(source_key, safe='')}/resolve/{revision.lower()}"
    )
    if subset:
        base += "/" + "/".join(quote(part, safe="") for part in PurePosixPath(subset).parts)
    return base


def _huggingface_metadata(
    version: Mapping[str, Any],
    resource: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    base = _huggingface_content_base(version, resource)
    if base is None:
        return {}, {}

    files: dict[str, str] = {}
    errors: dict[str, str] = {}
    for relative_path, limit in METADATA_LIMITS.items():
        try:
            files[relative_path] = _bounded_http_text(
                f"{base}/{quote(relative_path, safe='/')}", limit
            )
        except HTTPError as error:
            if error.code != 404:
                errors[relative_path] = f"Hugging Face returned HTTP {error.code}"
        except (URLError, TimeoutError, UnicodeDecodeError, ValueError) as error:
            errors[relative_path] = str(error)
    return files, errors


def _materialized_metadata(
    cluster: ClusterClient,
    root: str,
    gateway: str,
    wanted: set[str],
) -> tuple[dict[str, str], list[str]]:
    if not wanted:
        return {}, []
    root = _safe_registered_root(root)
    work_root_q = shlex.quote(WORK_ROOT)
    commands = [
        "set -u",
        f"root={shlex.quote(root)}",
        'resolved_root=$(realpath -e -- "$root") || exit 44',
        'test -d "$resolved_root" || exit 44',
        f'case "$resolved_root" in {work_root_q}|{work_root_q}/*) ;; *) exit 46 ;; esac',
    ]
    for relative_path, limit in METADATA_LIMITS.items():
        if relative_path not in wanted:
            continue
        commands.extend(
            [
                f"rel={shlex.quote(relative_path)}",
                'candidate="$resolved_root/$rel"',
                'if test -f "$candidate"; then',
                '  resolved=$(realpath -e -- "$candidate") || { printf "WARN\\t%s\\trealpath_failed\\n" "$rel"; continue; }',
                f'  case "$resolved" in {work_root_q}|{work_root_q}/*) ;; *) printf "WARN\\t%s\\toutside_work_root\\n" "$rel"; continue ;; esac',
                '  size=$(stat -c %s -- "$resolved") || { printf "WARN\\t%s\\tstat_failed\\n" "$rel"; continue; }',
                f'  if test "$size" -gt {limit}; then printf "WARN\\t%s\\ttoo_large\\n" "$rel"; continue; fi',
                '  encoded=$(timeout 4s base64 -w0 -- "$resolved") || { printf "WARN\\t%s\\tread_timeout\\n" "$rel"; continue; }',
                '  printf "FILE\\t%s\\t%s\\n" "$rel" "$encoded"',
                "fi",
            ]
        )
    host, output = cluster.run_with_fallback("\n".join(commands), gateway, timeout=25)
    files: dict[str, str] = {}
    warnings: list[str] = []
    for line in output.splitlines():
        fields = line.split("\t", 2)
        if len(fields) == 3 and fields[0] == "FILE":
            try:
                files[fields[1]] = base64.b64decode(fields[2], validate=True).decode("utf-8")
            except (ValueError, UnicodeDecodeError) as error:
                warnings.append(f"{fields[1]} from {host} was not valid UTF-8 metadata: {error}")
        elif len(fields) == 3 and fields[0] == "WARN":
            warnings.append(f"{fields[1]} could not be read from {host}: {fields[2]}")
    return files, warnings


def _load_metadata(
    cluster: ClusterClient,
    version: Mapping[str, Any],
    resource: Mapping[str, Any],
    gateway: str,
) -> tuple[dict[str, str], str, list[str]]:
    files, provider_errors = _huggingface_metadata(version, resource)
    provider_paths = set(files)
    warnings: list[str] = []
    missing = set(METADATA_LIMITS) - set(files)
    if missing and version.get("path"):
        try:
            materialized, materialized_warnings = _materialized_metadata(
                cluster, str(version["path"]), gateway, missing
            )
            files.update(materialized)
            warnings.extend(materialized_warnings)
        except Exception as error:
            warnings.append(f"Materialized metadata inspection failed: {error}")
    for relative_path, message in provider_errors.items():
        if relative_path not in files:
            warnings.append(f"{relative_path} could not be read from the pinned provider revision: {message}")
    source = (
        "pinned_provider"
        if provider_paths
        else "materialized_version"
        if files
        else "unavailable"
    )
    return files, source, warnings


def _json_object(files: Mapping[str, str], path: str, warnings: list[str]) -> dict[str, Any]:
    raw = files.get(path)
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        warnings.append(f"{path} is not valid JSON: {error.msg}")
        return {}
    if not isinstance(value, dict):
        warnings.append(f"{path} must contain one JSON object")
        return {}
    return value


def _json_lines(
    files: Mapping[str, str],
    path: str,
    *,
    limit: int,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], bool]:
    raw = files.get(path)
    if raw is None:
        return [], False
    rows: list[dict[str, Any]] = []
    lines = [line for line in raw.splitlines() if line.strip()]
    for index, line in enumerate(lines[:limit], start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            warnings.append(f"{path} line {index} is not valid JSON: {error.msg}")
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            warnings.append(f"{path} line {index} is not a JSON object")
    return rows, len(lines) > limit


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _render_template(template: Any, *, episode_index: int, chunk_size: int, video_key: str = "") -> str | None:
    if not isinstance(template, str) or not template:
        return None
    try:
        rendered = template.format(
            episode_chunk=episode_index // max(1, chunk_size),
            episode_index=episode_index,
            video_key=video_key,
        )
        return _safe_relative_path(rendered)
    except (KeyError, ValueError):
        return None


def _feature_rows(info: Mapping[str, Any]) -> list[dict[str, Any]]:
    features = info.get("features")
    if not isinstance(features, dict):
        return []
    rows: list[dict[str, Any]] = []
    for name, raw in features.items():
        spec = raw if isinstance(raw, dict) else {}
        rows.append(
            {
                "name": str(name),
                "dtype": str(spec.get("dtype") or "unknown"),
                "shape": spec.get("shape") if isinstance(spec.get("shape"), list) else [],
                "names": spec.get("names"),
                "video_info": spec.get("video_info") if isinstance(spec.get("video_info"), dict) else None,
            }
        )
    return rows


def _lerobot_dataset_preview(
    *,
    bundle_id: str,
    role: str,
    position: int,
    format_name: str,
    files: Mapping[str, str],
    gateway: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[str], str]:
    warnings: list[str] = []
    info = _json_object(files, "meta/info.json", warnings)
    tasks, tasks_truncated = _json_lines(
        files, "meta/tasks.jsonl", limit=100, warnings=warnings
    )
    episodes, episodes_truncated = _json_lines(
        files, "meta/episodes.jsonl", limit=20, warnings=warnings
    )
    is_groot = format_name.lower().startswith("groot-lerobot-v2")
    normalized_tasks = []
    for task in tasks:
        label = str(task.get("task") or "")
        normalized_tasks.append(
            {
                "task_index": _integer(task.get("task_index")),
                "task": label,
                "kind": "annotation" if is_groot and label.strip().lower() == "valid" else "task",
            }
        )
    normalized_episodes = [
        {
            "episode_index": _integer(episode.get("episode_index")),
            "tasks": episode.get("tasks") if isinstance(episode.get("tasks"), list) else [],
            "length": _integer(episode.get("length")),
            "trajectory_id": episode.get("trajectory_id"),
        }
        for episode in episodes
    ]
    features = _feature_rows(info)
    modality = _json_object(files, "meta/modality.json", warnings) if is_groot else {}
    summary = {
        "adapter": "groot-lerobot-v2" if is_groot else "lerobot-v2",
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": _number(info.get("fps")),
        "total_episodes": _integer(info.get("total_episodes")),
        "total_frames": _integer(info.get("total_frames")),
        "total_tasks": _integer(info.get("total_tasks")),
        "total_videos": _integer(info.get("total_videos")),
        "total_chunks": _integer(info.get("total_chunks")),
        "splits": info.get("splits") if isinstance(info.get("splits"), dict) else {},
    }
    dataset = {
        "summary": summary,
        "tasks": normalized_tasks,
        "tasks_truncated": tasks_truncated,
        "episodes": normalized_episodes,
        "episodes_truncated": episodes_truncated,
        "features": features,
        "modalities": modality,
    }
    chunk_size = _integer(info.get("chunks_size")) or 1000
    sample_files: list[dict[str, Any]] = []
    sample_media: list[dict[str, Any]] = []
    sampled_episodes = [
        episode for episode in normalized_episodes
        if episode["episode_index"] is not None
    ][:3]
    for episode in sampled_episodes:
        episode_index = int(episode["episode_index"])
        data_path = _render_template(
            info.get("data_path"), episode_index=episode_index, chunk_size=chunk_size
        )
        if data_path:
            sample_files.append(
                {
                    "kind": "episode_data",
                    "episode_index": episode_index,
                    "relative_path": data_path,
                }
            )
    for feature in features:
        dtype = feature["dtype"].lower()
        if dtype == "video":
            for episode in sampled_episodes:
                episode_index = int(episode["episode_index"])
                media_path = _render_template(
                    info.get("video_path"),
                    episode_index=episode_index,
                    chunk_size=chunk_size,
                    video_key=feature["name"],
                )
                if not media_path:
                    continue
                query = urlencode(
                    {
                        "role": role,
                        "position": position,
                        "path": media_path,
                        "gateway": gateway,
                    }
                )
                sample_media.append(
                    {
                        "kind": "video",
                        "mime_type": MEDIA_TYPES.get(PurePosixPath(media_path).suffix.lower()),
                        "feature": feature["name"],
                        "episode_index": episode_index,
                        "relative_path": media_path,
                        "available": True,
                        "url": f"/api/data/bundles/{quote(bundle_id, safe='')}/preview/media?{query}",
                    }
                )
        elif dtype == "image":
            sample_media.append(
                {
                    "kind": "embedded_image",
                    "mime_type": None,
                    "feature": feature["name"],
                    "episode_index": sampled_episodes[0]["episode_index"] if sampled_episodes else None,
                    "relative_path": None,
                    "available": False,
                    "note": "Images are embedded in Parquet. Configure a trusted row decoder to render pixels.",
                }
            )
    if not info:
        warnings.append("meta/info.json was unavailable; only registered version metadata can be shown")
    return dataset, sample_files[:12], sample_media[:8], warnings, summary["adapter"]


def _usage_summary(assignments: list[Mapping[str, Any]]) -> dict[str, Any]:
    def references(roles: set[str]) -> list[dict[str, Any]]:
        return [
            {
                "role": str(item.get("role") or ""),
                "position": _integer(item.get("position")) or 0,
                "version_id": str((item.get("version") or {}).get("id") or item.get("version_id") or ""),
            }
            for item in assignments
            if str(item.get("role") or "") in roles
        ]

    training = references({"training_data"})
    evaluation = references(EVALUATION_ROLES)
    simulation = references({"simulation_assets"})
    training_ids = {item["version_id"] for item in training if item["version_id"]}
    evaluation_ids = {item["version_id"] for item in evaluation if item["version_id"]}
    if not evaluation:
        relationship = (
            "This bundle declares training data but no evaluation dataset. "
            "Simulator evaluation uses the selected evaluation suite and its environment inputs."
            if training
            else "This bundle does not declare training_data or evaluation_data roles."
        )
    elif training_ids & evaluation_ids:
        relationship = "Training and evaluation roles share at least one exact immutable data version."
    else:
        relationship = "Training and evaluation roles use different immutable data versions."
    return {
        "training": {"declared": bool(training), "assignments": training},
        "evaluation": {"declared": bool(evaluation), "assignments": evaluation},
        "simulation": {"declared": bool(simulation), "assignments": simulation},
        "relationship": relationship,
    }


def _with_current_resource_labels(preview, bundle):
    """Mutable registry labels are read live, outside the immutable preview cache."""
    result = copy.deepcopy(preview)
    labels = {
        (str(item.get("role") or ""), _integer(item.get("position")) or 0):
            ((item.get("version") or {}).get("resource") or {}).get("display_name")
        for item in bundle.get("assignments", []) if isinstance(item, dict)
    }
    for item in result["assignments"]:
        item["resource"]["display_name"] = labels.get((item["role"], item["position"]))
    return result


def build_data_bundle_preview(
    bundle: Mapping[str, Any],
    cluster: ClusterClient,
    gateway: str = "auto",
) -> dict[str, Any]:
    bundle_id = str(bundle.get("id") or bundle.get("bundle_id") or "")
    if not bundle_id:
        raise ValueError("data bundle has no ID")
    cache_identity = str(bundle.get("manifest_sha256") or bundle_id)
    cache_key = (cache_identity, gateway)
    with _PREVIEW_CACHE_LOCK:
        cached = _PREVIEW_CACHE.get(cache_key)
    if cached is not None:
        return _with_current_resource_labels(cached, bundle)

    assignments = bundle.get("assignments")
    if not isinstance(assignments, list):
        assignments = []
    preview_assignments: list[dict[str, Any]] = []
    bundle_warnings: list[str] = []
    for raw_assignment in assignments:
        assignment = raw_assignment if isinstance(raw_assignment, dict) else {}
        version = assignment.get("version") if isinstance(assignment.get("version"), dict) else {}
        resource = version.get("resource") if isinstance(version.get("resource"), dict) else {}
        role = str(assignment.get("role") or "")
        position = _integer(assignment.get("position")) or 0
        format_name = str(version.get("format") or "")
        inspection_warnings: list[str] = []
        dataset: dict[str, Any] = {
            "summary": {
                "adapter": None,
                "robot_type": (version.get("metadata") or {}).get("robot_type")
                if isinstance(version.get("metadata"), dict)
                else None,
                "fps": (version.get("metadata") or {}).get("fps")
                if isinstance(version.get("metadata"), dict)
                else None,
                "total_episodes": (version.get("metadata") or {}).get("total_episodes")
                if isinstance(version.get("metadata"), dict)
                else None,
                "total_frames": (version.get("metadata") or {}).get("total_frames")
                if isinstance(version.get("metadata"), dict)
                else None,
                "total_tasks": (version.get("metadata") or {}).get("total_tasks")
                if isinstance(version.get("metadata"), dict)
                else None,
            },
            "tasks": [],
            "tasks_truncated": False,
            "episodes": [],
            "episodes_truncated": False,
            "features": [],
            "modalities": {},
        }
        sample_files: list[dict[str, Any]] = []
        sample_media: list[dict[str, Any]] = []
        normalized_format = format_name.lower()
        if "lerobot-v2" in normalized_format:
            files, metadata_source, read_warnings = _load_metadata(
                cluster, version, resource, gateway
            )
            dataset, sample_files, sample_media, parser_warnings, adapter_name = (
                _lerobot_dataset_preview(
                    bundle_id=bundle_id,
                    role=role,
                    position=position,
                    format_name=format_name,
                    files=files,
                    gateway=gateway,
                )
            )
            inspection_warnings.extend(read_warnings)
            inspection_warnings.extend(parser_warnings)
            inspection_status = "ready" if dataset["summary"].get("total_episodes") is not None else "unavailable"
        else:
            metadata_source = "registry"
            adapter_name = None
            inspection_status = "unsupported"
            inspection_warnings.append(
                f'No static dataset preview adapter is registered for format "{format_name or "undeclared"}".'
            )
        preview_assignments.append(
            {
                "role": role,
                "position": position,
                "required": assignment.get("required") is not False,
                "mount_path": assignment.get("mount_path"),
                "config": assignment.get("config") if isinstance(assignment.get("config"), dict) else {},
                "resource": {
                    "id": resource.get("id"),
                    "provider": resource.get("provider"),
                    "namespace": resource.get("namespace"),
                    "source_key": resource.get("source_key"),
                    "kind": resource.get("kind"),
                },
                "version": {
                    "id": version.get("id") or assignment.get("version_id"),
                    "revision": version.get("revision"),
                    "format": format_name,
                    "path": version.get("path"),
                    "source_uri": version.get("source_uri"),
                    "manifest_sha256": version.get("manifest_sha256"),
                    "status": version.get("status"),
                    "size_bytes": version.get("size_bytes"),
                },
                "inspection": {
                    "status": inspection_status,
                    "adapter": adapter_name,
                    "metadata_source": metadata_source,
                    "warnings": inspection_warnings,
                },
                "dataset": dataset,
                "sample_files": sample_files,
                "sample_media": sample_media,
            }
        )
        bundle_warnings.extend(f"{role}[{position}]: {warning}" for warning in inspection_warnings)

    result = {
        "schema_version": PREVIEW_SCHEMA,
        "bundle": {
            "id": bundle_id,
            "name": bundle.get("name"),
            "version": bundle.get("version"),
            "manifest_sha256": bundle.get("manifest_sha256"),
            "created_at": bundle.get("created_at"),
            "archived_at": bundle.get("archived_at"),
            "roles": sorted({str(item.get("role") or "") for item in assignments if item.get("role")}),
            "usage": _usage_summary(assignments),
        },
        "assignments": preview_assignments,
        "warnings": bundle_warnings,
    }
    with _PREVIEW_CACHE_LOCK:
        if len(_PREVIEW_CACHE) >= 128:
            _PREVIEW_CACHE.pop(next(iter(_PREVIEW_CACHE)))
        _PREVIEW_CACHE[cache_key] = result
    return _with_current_resource_labels(result, bundle)


def resolve_data_bundle_preview_media(
    bundle: Mapping[str, Any],
    cluster: ClusterClient,
    *,
    role: str,
    position: int,
    relative_path: str,
    gateway: str = "auto",
) -> dict[str, str]:
    relative_path = _safe_relative_path(relative_path)
    suffix = PurePosixPath(relative_path).suffix.lower()
    media_type = MEDIA_TYPES.get(suffix)
    if media_type is None:
        raise ValueError("preview media type is not allowed")
    preview = build_data_bundle_preview(bundle, cluster, gateway)
    allowed = False
    registered_root = ""
    source_version: Mapping[str, Any] = {}
    source_resource: Mapping[str, Any] = {}
    for item in preview["assignments"]:
        if item["role"] != role or int(item["position"]) != position:
            continue
        if any(
            media.get("available") is True
            and media.get("relative_path") == relative_path
            for media in item.get("sample_media", [])
        ):
            allowed = True
            registered_root = _safe_registered_root(item["version"].get("path"))
            for original in bundle.get("assignments", []):
                if (
                    str(original.get("role") or "") == role
                    and int(original.get("position") or 0) == position
                ):
                    source_version = original.get("version") or {}
                    source_resource = source_version.get("resource") or {}
                    break
            break
    if not allowed:
        raise ValueError("requested file is not a registered sample media item for this bundle")

    provider_base = _huggingface_content_base(source_version, source_resource)
    if provider_base is not None:
        return {
            "kind": "provider",
            "url": f"{provider_base}/{quote(relative_path, safe='/')}",
            "media_type": media_type,
        }

    lexical_candidate = str(PurePosixPath(registered_root) / PurePosixPath(relative_path))
    work_root_q = shlex.quote(WORK_ROOT)
    command = (
        f"candidate=$(realpath -e -- {shlex.quote(lexical_candidate)}) || exit 44; "
        f"case \"$candidate\" in {work_root_q}|{work_root_q}/*) ;; *) exit 46 ;; esac; "
        "test -f \"$candidate\" || exit 44; printf '%s' \"$candidate\""
    )
    host, resolved_path = cluster.run_with_fallback(command, gateway, timeout=20)
    resolved_path = resolved_path.strip()
    if not resolved_path:
        raise ValueError("preview media did not resolve to a file")
    return {
        "kind": "cluster",
        "host": host,
        "path": resolved_path,
        "media_type": media_type,
    }


__all__ = [
    "PREVIEW_SCHEMA",
    "build_data_bundle_preview",
    "resolve_data_bundle_preview_media",
]
