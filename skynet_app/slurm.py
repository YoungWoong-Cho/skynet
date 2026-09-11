from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
from pathlib import Path, PurePosixPath
from textwrap import dedent
from typing import Any, Literal, Mapping

from pydantic import Field

from skynet_app.adapters import AdapterPlan, resolve_gpu_count, resolve_gpu_type
from skynet_app.cluster_config import CLUSTER
from skynet_app.cluster_runtime import HOME_ROOT, SLURM_BIN
from skynet_app.experiments import CanonicalModel, ExperimentSpec, canonical_sha256
from skynet_app.evaluation_placement import resolve_evaluation_resources
from skynet_app.workspace_storage import paths_for_root


RUNNER_SOURCE = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import signal
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TOKENS = {
    "{{SKYNET_RUN_DIR}}": lambda: os.environ["SKYNET_RUN_DIR"],
    "{{SKYNET_SOURCE_DIR}}": lambda: os.environ["SKYNET_SOURCE_DIR"],
}
child = None
termination_signal = None


def replace_tokens(value, resume_checkpoint):
    for token, provider in TOKENS.items():
        value = value.replace(token, provider())
    if "{{SKYNET_RESUME_CHECKPOINT}}" in value:
        if not resume_checkpoint:
            raise RuntimeError("resume argv requires a validated checkpoint")
        value = value.replace("{{SKYNET_RESUME_CHECKPOINT}}", resume_checkpoint)
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def safe_run_path(run_dir, relative, label):
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise RuntimeError(f"{label} must stay inside the run directory: {relative}")
    root = run_dir.resolve()
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes the run directory: {relative}") from error
    return candidate


def ensure_native_tracking_resume_ids(execution, resume_checkpoint):
    if not resume_checkpoint and not execution.get("native_tracking_resume"):
        return
    run_dir = Path(os.environ["SKYNET_RUN_DIR"]).resolve()
    for binding in execution.get("native_tracking", []):
        template = binding.get("run_id_file")
        remote_id = binding.get("remote_id")
        provider = binding.get("provider", "tracking")
        if not template or not remote_id:
            continue
        candidate = Path(replace_tokens(str(template), resume_checkpoint)).resolve()
        try:
            candidate.relative_to(run_dir)
        except ValueError as error:
            raise RuntimeError(
                f"native {provider} run ID file escapes the run directory: {candidate}"
            ) from error
        if candidate.exists():
            existing = candidate.read_text(encoding="utf-8").strip()
            if existing != str(remote_id):
                raise RuntimeError(
                    f"native {provider} run ID file conflicts with the central run"
                )
            continue
        candidate.parent.mkdir(parents=True, exist_ok=True)
        temporary = candidate.with_name(f".{candidate.name}.{os.getpid()}.tmp")
        temporary.write_text(str(remote_id) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(candidate)


def preparation_outputs(working_directory, patterns):
    outputs = {}
    for pattern in patterns:
        pattern_path = Path(pattern)
        if pattern_path.is_absolute() or ".." in pattern_path.parts:
            raise RuntimeError(f"preparation output glob must be relative: {pattern}")
        matches = sorted(path for path in working_directory.glob(pattern) if path.is_file())
        if not matches:
            return None
        for path in matches:
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(working_directory.resolve())
            except ValueError as error:
                raise RuntimeError(f"preparation output escapes its working directory: {path}") from error
            outputs[str(relative)] = sha256_file(resolved)
    return dict(sorted(outputs.items()))


def expected_preparation_outputs_match(outputs, expected):
    if outputs is None:
        return False
    if not isinstance(expected, dict):
        raise RuntimeError("preparation expected_output_sha256 must be an object")
    for path, digest in expected.items():
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
        ):
            raise RuntimeError("invalid preparation expected output SHA-256 declaration")
        if outputs.get(path) != digest.lower():
            return False
    return True


def run_preparation_steps(execution, run_dir, capsule_dir):
    global child
    records = []
    state_path = capsule_dir / "state" / "preparation.json"
    marker_root = run_dir / "state" / "preparation"
    marker_root.mkdir(parents=True, exist_ok=True)
    for step in execution.get("preparation_steps", []):
        step_id = str(step.get("id", ""))
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,95}", step_id):
            raise RuntimeError(f"invalid preparation step ID: {step_id}")
        fingerprint = canonical_sha256(step)
        working_directory = safe_run_path(
            run_dir, step.get("working_directory", "."), "preparation working directory"
        )
        working_directory.mkdir(parents=True, exist_ok=True)
        argv = [replace_tokens(item, None) for item in step.get("argv", [])]
        if not argv:
            raise RuntimeError(f"preparation step {step_id} has empty argv")
        marker_path = marker_root / f"{step_id}.json"
        outputs = preparation_outputs(working_directory, step.get("output_globs", []))
        expected_outputs = step.get("expected_output_sha256", {})
        expected_outputs_match = expected_preparation_outputs_match(
            outputs, expected_outputs
        )
        marker = None
        if marker_path.is_file():
            try:
                marker = json.loads(marker_path.read_text())
            except (OSError, ValueError):
                marker = None
        record = {
            "id": step_id,
            "step_sha256": fingerprint,
            "argv": argv,
            "working_directory": str(working_directory),
        }
        if (
            marker is not None
            and marker.get("step_sha256") == fingerprint
            and outputs is not None
            and expected_outputs_match
            and marker.get("outputs") == outputs
        ):
            record.update({"status": "skipped", "outputs": outputs})
            records.append(record)
            atomic_json(state_path, {"schema_version": 1, "steps": records})
            print(f"Preparation step {step_id}: verified existing outputs; skipping")
            continue
        record["status"] = "running"
        records.append(record)
        atomic_json(state_path, {"schema_version": 1, "steps": records})
        print(f"Preparation step {step_id}: running")
        child = subprocess.Popen(argv, cwd=working_directory, start_new_session=True)
        return_code = wait_for_child()
        child = None
        if termination_signal is not None:
            raise SystemExit(interrupted_exit_code())
        if return_code != 0:
            record.update({"status": "failed", "return_code": return_code})
            atomic_json(state_path, {"schema_version": 1, "steps": records})
            raise RuntimeError(f"preparation step {step_id} failed with exit code {return_code}")
        outputs = preparation_outputs(working_directory, step.get("output_globs", []))
        if outputs is None:
            record.update({"status": "failed", "error": "declared outputs were not created"})
            atomic_json(state_path, {"schema_version": 1, "steps": records})
            raise RuntimeError(f"preparation step {step_id} did not create every declared output")
        if not expected_preparation_outputs_match(outputs, expected_outputs):
            record.update(
                {
                    "status": "failed",
                    "error": "an output did not match its declared SHA-256",
                    "outputs": outputs,
                }
            )
            atomic_json(state_path, {"schema_version": 1, "steps": records})
            raise RuntimeError(
                f"preparation step {step_id} output SHA-256 mismatch"
            )
        marker = {
            "schema_version": 1,
            "id": step_id,
            "step_sha256": fingerprint,
            "outputs": outputs,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(marker_path, marker)
        record.update({"status": "completed", "outputs": outputs})
        atomic_json(state_path, {"schema_version": 1, "steps": records})


def write_runtime_manifest(run_dir):
    packages = {}
    for distribution in importlib.metadata.distributions():
        try:
            name = distribution.metadata.get("Name") or distribution.name
            packages[str(name)] = distribution.version
        except Exception:
            continue
    environment_keys = (
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONPATH",
        "UV_PROJECT_ENVIRONMENT",
        "VIRTUAL_ENV",
    )
    manifest = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": sys.version,
        },
        "packages": [
            {"name": name, "version": packages[name]}
            for name in sorted(packages, key=str.casefold)
        ],
        "sys_path": sys.path,
        "environment": {
            key: os.environ[key] for key in environment_keys if key in os.environ
        },
    }
    target = run_dir / "runtime-manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def latest_checkpoint(run_dir, initial_checkpoint=None):
    descriptor = run_dir / "checkpoints" / "latest.json"
    if not descriptor.is_file():
        if not initial_checkpoint:
            return None
        path = Path(initial_checkpoint)
        if not path.exists():
            raise RuntimeError(f"initial checkpoint is missing: {path}")
        return str(path)
    payload = json.loads(descriptor.read_text())
    path = Path(payload["path"])
    if not path.is_absolute():
        path = run_dir / path
    if not path.exists():
        raise RuntimeError(f"resume checkpoint is missing: {path}")
    expected = payload.get("sha256")
    if expected and path.is_file() and sha256_file(path) != expected:
        raise RuntimeError(f"resume checkpoint hash mismatch: {path}")
    return str(path)


def _validate_checkpoint_pattern(pattern, label):
    pattern_path = Path(pattern)
    if (
        not isinstance(pattern, str)
        or not pattern
        or pattern_path.is_absolute()
        or ".." in pattern_path.parts
    ):
        raise RuntimeError(f"{label} must be a non-empty relative glob: {pattern}")


def _paths_overlap(first, second):
    if first == second:
        return True
    try:
        second.relative_to(first)
        return True
    except ValueError:
        pass
    try:
        first.relative_to(second)
        return True
    except ValueError:
        return False


def _reject_overlapping_paths(paths, label):
    ordered = sorted(set(paths), key=lambda path: (len(path.parts), str(path)))
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if _paths_overlap(first, second):
                raise RuntimeError(
                    f"{label} must be atomic and non-overlapping: {first} overlaps {second}"
                )


def checkpoint_candidates(run_dir, project_dir, execution):
    candidates = {}
    patterns = execution.get("checkpoint_globs", [])
    candidate_kind = execution.get("checkpoint_candidate_kind", "any")
    required_patterns = execution.get("checkpoint_inference_required_globs", [])
    if candidate_kind not in {"any", "file", "directory"}:
        raise RuntimeError(f"invalid checkpoint candidate kind: {candidate_kind}")
    if not isinstance(required_patterns, list) or any(
        not isinstance(pattern, str) for pattern in required_patterns
    ):
        raise RuntimeError("checkpoint inference required globs must be a list of strings")
    if candidate_kind == "directory" and patterns and not required_patterns:
        raise RuntimeError(
            "directory checkpoint candidates require adapter-declared inference paths or markers"
        )
    basename_pattern = execution.get("checkpoint_basename_regex")
    try:
        basename_regex = re.compile(basename_pattern) if basename_pattern else None
    except re.error as error:
        raise RuntimeError(f"invalid checkpoint basename regex: {error}") from error
    for root in (run_dir, project_dir):
        root = root.resolve()
        for pattern in patterns:
            _validate_checkpoint_pattern(pattern, "checkpoint candidate glob")
            for path in root.glob(pattern):
                if (
                    (not path.exists() and not path.is_symlink())
                    or path.name in {"latest.json", "selected-for-inference.json"}
                ):
                    continue
                try:
                    if path.is_symlink():
                        raise RuntimeError(
                            f"checkpoint candidate root cannot be a symlink: {path}"
                        )
                    resolved = path.resolve(strict=True)
                    resolved.relative_to(root)
                    if candidate_kind == "file" and not path.is_file():
                        continue
                    if candidate_kind == "directory" and not path.is_dir():
                        continue
                    if basename_regex and basename_regex.fullmatch(path.name) is None:
                        continue
                    if path.is_dir():
                        if not required_patterns:
                            continue
                        if not _checkpoint_satisfies_required_content(
                            resolved, required_patterns
                        ):
                            continue
                    candidates[str(resolved)] = (resolved.stat().st_mtime_ns, resolved)
                except (OSError, ValueError) as error:
                    raise RuntimeError(f"checkpoint candidate escapes its search root: {path}") from error
    ordered = [item[1] for item in sorted(candidates.values(), key=lambda item: (item[0], str(item[1])))]
    _reject_overlapping_paths(ordered, "checkpoint candidates")
    return ordered


def _checkpoint_sidecar(candidate):
    if candidate.is_dir():
        return candidate / "skynet-checkpoint.json"
    return candidate.with_name(candidate.name + ".skynet-checkpoint.json")


def _checkpoint_score(candidate):
    sidecar = _checkpoint_sidecar(candidate)
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid checkpoint score sidecar {sidecar}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint score sidecar must contain an object: {sidecar}")
    selection = payload.get("selection", payload)
    if not isinstance(selection, dict):
        raise RuntimeError(f"checkpoint score selection must contain an object: {sidecar}")
    score = selection.get("score")
    mode = selection.get("mode", "max")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or mode not in {"min", "max"}:
        raise RuntimeError(f"checkpoint score sidecar has no valid numeric score and min/max mode: {sidecar}")
    return {"score": float(score), "mode": mode, "sidecar": str(sidecar)}


def _select_checkpoint(candidates, requested, *, final):
    latest = candidates[-1]
    if not final:
        return latest, {
            "requested": "latest",
            "actual": "latest",
            "reason": "resume retention uses latest modification time",
        }
    if requested not in {"best", "latest"}:
        raise RuntimeError(f"invalid final checkpoint selector: {requested}")
    if requested == "latest":
        return latest, {
            "requested": "latest",
            "actual": "latest",
            "reason": "selected latest candidate by modification time",
        }
    scored = [(candidate, _checkpoint_score(candidate)) for candidate in candidates]
    missing_scores = [str(candidate) for candidate, metadata in scored if metadata is None]
    if missing_scores:
        raise RuntimeError(
            "best checkpoint selection requires a valid skynet-checkpoint.json score for every candidate: "
            + ", ".join(missing_scores)
        )
    modes = {metadata["mode"] for _, metadata in scored}
    if len(modes) != 1:
        raise RuntimeError("checkpoint sidecars disagree on best-score mode")
    mode = next(iter(modes))
    selected, metadata = (
        max(scored, key=lambda item: (item[1]["score"], str(item[0])))
        if mode == "max"
        else min(scored, key=lambda item: (item[1]["score"], str(item[0])))
    )
    return selected, {
        "requested": "best",
        "actual": "best",
        "reason": f"selected {mode} score from skynet-checkpoint.json",
        **metadata,
    }


def _checkpoint_glob_matches(root, patterns, label, *, require_each=False):
    root = root.resolve()
    matches = {}
    for pattern in patterns:
        _validate_checkpoint_pattern(pattern, label)
        found = []
        for path in root.glob(pattern):
            if not path.exists() and not path.is_symlink():
                continue
            logical = Path(os.path.abspath(path))
            try:
                logical.relative_to(root)
                path.resolve().relative_to(root)
            except (OSError, ValueError) as error:
                raise RuntimeError(f"{label} escapes selected checkpoint: {path}") from error
            found.append(logical)
            matches[str(logical)] = logical
        if require_each and not found:
            raise RuntimeError(f"{label} matched no checkpoint content: {pattern}")
    return [matches[key] for key in sorted(matches)]


def _checkpoint_satisfies_required_content(candidate, patterns):
    for pattern in patterns:
        if not _checkpoint_glob_matches(
            candidate,
            [pattern],
            "required inference glob",
        ):
            return False
    return True


def _checkpoint_weight_index_files(selected):
    # A wildcard matching one shard does not prove that a sharded model is complete.
    required = []
    if not selected.is_dir():
        return required
    for index in selected.glob("*.safetensors.index.json"):
        try:
            weights = json.loads(index.read_text())["weight_map"]
            if not isinstance(weights, dict) or not weights:
                raise ValueError("empty weight map")
            for name in set(weights.values()):
                path = selected / name
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError(f"unsafe shard path: {name}")
                path.resolve().relative_to(selected.resolve())
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"missing or empty shard: {name}")
                required.append(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeError(f"invalid inference weight index {index}: {error}") from error
        required.append(index)
    return required


def _checkpoint_cleanup_plan(selected, execution, *, final):
    checkpoint = execution.get("checkpoint", {})
    required_patterns = execution.get("checkpoint_inference_required_globs", [])
    required = (
        _checkpoint_glob_matches(
            selected,
            required_patterns,
            "required inference glob",
            require_each=True,
        )
        if final and required_patterns
        else []
    )
    if final:
        required.extend(_checkpoint_weight_index_files(selected))
    cleanup_requested = bool(
        final and checkpoint.get("remove_training_state_after_success", False)
    )
    prune_patterns = execution.get("checkpoint_prune_globs", [])
    prune = (
        _checkpoint_glob_matches(selected, prune_patterns, "checkpoint prune glob")
        if cleanup_requested and prune_patterns
        else []
    )
    if any(path == selected.resolve() for path in prune):
        raise RuntimeError("checkpoint prune glob cannot remove the selected checkpoint root")
    _reject_overlapping_paths(prune, "checkpoint prune targets")
    for prune_path in prune:
        for required_path in required:
            if _paths_overlap(prune_path, required_path):
                raise RuntimeError(
                    "checkpoint prune targets overlap required inference content: "
                    f"{prune_path} and {required_path}"
                )
    return required_patterns, required, prune, {
        "requested": cleanup_requested,
        "performed": False,
        "prune_globs": list(prune_patterns),
        "removed": [],
        "reason": (
            "cleanup disabled by checkpoint policy"
            if not cleanup_requested
            else "adapter declared no checkpoint prune globs"
            if not prune_patterns
            else "pending"
        ),
    }


def _training_output_cleanup_plan(run_dir, selected, execution, cleanup):
    patterns = execution.get("training_output_prune_globs", [])
    cleanup["output_prune_globs"] = list(patterns)
    cleanup["removed_outputs"] = []
    if not cleanup["requested"] or not patterns:
        return []
    if not execution.get("checkpoint_inference_required_globs"):
        raise RuntimeError("training output cleanup requires an inference content contract")
    targets = _checkpoint_glob_matches(run_dir, patterns, "training output prune glob")
    _reject_overlapping_paths(targets, "training output prune targets")
    for target in targets:
        if _paths_overlap(target, selected):
            raise RuntimeError(f"training output cleanup overlaps selected checkpoint: {target}")
    return targets


def _remove_checkpoint_path(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def checkpoint_identity(path):
    path = Path(path)
    if path.is_symlink():
        target = os.readlink(path)
        record = {"path": ".", "type": "symlink", "target": target}
        return {
            "sha256": canonical_sha256([record]),
            "size_bytes": len(target.encode("utf-8")),
            "file_count": 1,
            "is_directory": False,
        }
    if path.is_file():
        return {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "file_count": 1,
            "is_directory": False,
        }
    if not path.is_dir():
        raise RuntimeError(f"unsupported checkpoint candidate type: {path}")
    records = []
    size_bytes = 0
    file_count = 0
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        relative = child.relative_to(path).as_posix()
        if child.is_symlink():
            target = os.readlink(child)
            encoded_size = len(target.encode("utf-8"))
            records.append(
                {"path": relative, "type": "symlink", "target": target, "size_bytes": encoded_size}
            )
            size_bytes += encoded_size
            file_count += 1
        elif child.is_file():
            child_size = child.stat().st_size
            records.append(
                {
                    "path": relative,
                    "type": "file",
                    "sha256": sha256_file(child),
                    "size_bytes": child_size,
                }
            )
            size_bytes += child_size
            file_count += 1
        elif child.is_dir():
            records.append({"path": relative, "type": "directory"})
        else:
            records.append({"path": relative, "type": "other"})
    return {
        "sha256": canonical_sha256(records),
        "size_bytes": size_bytes,
        "file_count": file_count,
        "is_directory": True,
    }


def snapshot_checkpoints(run_dir, project_dir, execution, *, final=False):
    run_dir = run_dir.resolve()
    project_dir = project_dir.resolve()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    candidates = checkpoint_candidates(run_dir, project_dir, execution)
    if not candidates:
        if final and execution.get("checkpoint_globs"):
            raise RuntimeError(
                "training succeeded but declared checkpoint globs produced no valid candidate"
            )
        return None
    keep_last = int(execution.get("checkpoint", {}).get("keep_last", 3))
    requested_selector = execution.get("checkpoint", {}).get("final_selector", "latest")
    selected, selection = _select_checkpoint(candidates, requested_selector, final=final)
    required_patterns, _required, prune, cleanup = _checkpoint_cleanup_plan(
        selected, execution, final=final
    )
    output_prune = _training_output_cleanup_plan(run_dir, selected, execution, cleanup)
    retained = {selected} if final else set(candidates[-max(1, keep_last) :])
    stale_candidates = [candidate for candidate in candidates if candidate not in retained]
    _reject_overlapping_paths([*stale_candidates, *prune, *output_prune], "cleanup targets")
    # Validate every destructive target before removing anything. Otherwise a later
    # unsafe candidate could leave retention only partially applied.
    for stale in stale_candidates:
        try:
            stale.relative_to(run_dir)
        except ValueError as error:
            raise RuntimeError(
                f"checkpoint retention cannot delete a candidate outside the run directory: {stale}"
            ) from error
        for retained_root in retained:
            if _paths_overlap(stale, retained_root):
                raise RuntimeError(
                    "checkpoint retention target overlaps a retained checkpoint root: "
                    f"{stale} and {retained_root}"
                )
    for target in prune:
        try:
            target.relative_to(run_dir)
        except ValueError as error:
            raise RuntimeError(
                f"checkpoint cleanup cannot modify content outside the run directory: {target}"
            ) from error
    for stale in stale_candidates:
        _remove_checkpoint_path(stale)
    if selected not in retained or (not selected.exists() and not selected.is_symlink()):
        raise RuntimeError(f"selected checkpoint disappeared during retention: {selected}")
    removed = []
    for target in prune:
        relative = target.relative_to(selected.resolve()).as_posix()
        _remove_checkpoint_path(target)
        removed.append(relative)
    for target in output_prune:
        _remove_checkpoint_path(target)
        cleanup["removed_outputs"].append(target.relative_to(run_dir).as_posix())
    if required_patterns:
        _checkpoint_glob_matches(
            selected,
            required_patterns,
            "required inference glob",
            require_each=True,
        )
    cleanup["performed"] = bool(removed or output_prune)
    cleanup["removed"] = removed
    if cleanup["requested"] and (cleanup["prune_globs"] or cleanup["output_prune_globs"]):
        cleanup["reason"] = (
            "removed adapter-declared training state and redundant outputs"
            if cleanup["performed"]
            else "adapter prune globs matched no content"
        )
    identity = checkpoint_identity(selected)
    payload = {
        "schema_version": 2,
        "run_id": execution.get("run_id"),
        "path": str(selected),
        **identity,
        "final": final,
        "resumable": not final and not cleanup["performed"],
        "selection": selection,
        "cleanup": cleanup,
        "contract": {
            "candidate_globs": list(execution.get("checkpoint_globs", [])),
            "candidate_kind": execution.get("checkpoint_candidate_kind", "any"),
            "basename_regex": execution.get("checkpoint_basename_regex"),
            "inference_required_globs": list(required_patterns),
        },
    }
    atomic_json(checkpoint_dir / "latest.json", payload)
    if final:
        atomic_json(checkpoint_dir / "selected-for-inference.json", payload)
    return selected


def forward(signum, _frame):
    global termination_signal
    if termination_signal is not None:
        return
    termination_signal = signum
    marker = Path(os.environ["SKYNET_RUN_DIR"]) / "state" / "checkpoint-requested"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(signum))
    if signum == signal.SIGUSR1:
        capsule = Path(os.environ.get("SKYNET_CAPSULE_DIR", os.environ["SKYNET_RUN_DIR"]))
        atomic_json(capsule / "state" / "interruption.json", {
            "schema_version": 1,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "run_id": os.environ.get("SKYNET_RUN_ID"),
            "reason": "time_limit_warning",
            "exit_code": 124,
        })
        print("[skynet] Time-limit warning: stopping training and preserving its latest checkpoint.", flush=True)
    if child is not None and child.poll() is None:
        try:
            # USR1 is a scheduler warning, not a portable trainer checkpoint API.
            os.killpg(child.pid, signal.SIGTERM if signum == signal.SIGUSR1 else signum)
        except ProcessLookupError:
            pass


def interrupted_exit_code():
    return 124 if termination_signal == signal.SIGUSR1 else 128 + termination_signal


def check_timeout_warning():
    capsule = Path(os.environ.get("SKYNET_CAPSULE_DIR", os.environ["SKYNET_RUN_DIR"]))
    if termination_signal is None and (capsule / "state" / "time-limit-warning").is_file():
        forward(signal.SIGUSR1, None)


def wait_for_child():
    while True:
        check_timeout_warning()
        try:
            return child.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            continue


def main():
    global child
    parser = argparse.ArgumentParser()
    parser.add_argument("execution")
    args = parser.parse_args()
    execution = json.loads(Path(args.execution).read_text())
    stage = execution.get("stage", "train")
    resume_mode = execution.get("resume_mode", "checkpoint_argv_suffix")
    if stage not in {"train", "eval"}:
        raise RuntimeError(f"invalid execution stage: {stage}")
    if resume_mode not in {"checkpoint_argv_suffix", "evaluation_ledger"}:
        raise RuntimeError(f"invalid resume mode: {resume_mode}")
    if stage == "train" and resume_mode != "checkpoint_argv_suffix":
        raise RuntimeError("training execution requires checkpoint argv suffix resume")
    if stage == "eval" and resume_mode != "evaluation_ledger":
        raise RuntimeError("evaluation execution requires ledger resume")
    run_dir = Path(os.environ["SKYNET_RUN_DIR"])
    capsule_dir = Path(os.environ.get("SKYNET_CAPSULE_DIR", str(run_dir)))
    project_dir = Path(os.environ["SKYNET_PROJECT_DIR"])
    write_runtime_manifest(capsule_dir)
    if stage == "train":
        snapshot_checkpoints(run_dir, project_dir, execution)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, forward)
    check_timeout_warning()
    if termination_signal is not None:
        return interrupted_exit_code()
    run_preparation_steps(execution, run_dir, capsule_dir)
    resume_checkpoint = (
        latest_checkpoint(run_dir, execution.get("initial_checkpoint"))
        if stage == "train" and execution["auto_resume"]
        else None
    )
    ensure_native_tracking_resume_ids(execution, resume_checkpoint)
    argv = [replace_tokens(item, resume_checkpoint) for item in execution["argv"]]
    if stage == "train" and resume_checkpoint:
        argv.extend(replace_tokens(item, resume_checkpoint) for item in execution["resume_argv"])
    if not argv:
        raise RuntimeError("refusing to execute an empty argv")
    os.environ["SKYNET_RESUME_CHECKPOINT"] = resume_checkpoint or ""
    evaluation_resume = execution.get("evaluation_resume", {})
    os.environ["SKYNET_EVALUATION_RESUME"] = (
        "1" if stage == "eval" and bool(evaluation_resume.get("enabled")) else "0"
    )
    evaluation_progress = evaluation_resume.get("progress_path")
    if stage == "eval" and evaluation_progress:
        os.environ["SKYNET_EVAL_PROGRESS_PATH"] = str(evaluation_progress)
    (capsule_dir / "state").mkdir(parents=True, exist_ok=True)
    (capsule_dir / "state" / "effective-argv.json").write_text(json.dumps(argv, indent=2) + "\n")
    child = subprocess.Popen(argv, cwd=project_dir, start_new_session=True)
    sampler = None
    if stage == "train" and os.environ.get("SLURM_JOB_ID"):
        try:
            from gpu_metrics import GPUSampler
            sampler = GPUSampler(run_dir / "state" / "gpu-stats" / os.environ["SLURM_JOB_ID"]).start()
        except Exception as error:
            print(f"GPU statistics unavailable: {type(error).__name__}", file=sys.stderr)
    try:
        return_code = wait_for_child()
    finally:
        if sampler is not None:
            sampler.stop()
    if stage == "train":
        snapshot_checkpoints(
            run_dir, project_dir, execution,
            final=return_code == 0 and termination_signal is None,
        )
    if termination_signal is not None:
        return interrupted_exit_code()
    return return_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"skynet runtime preflight/execution failure: {error}", file=sys.stderr)
        raise
'''


class SlurmCompileError(ValueError):
    pass


BATCH_WARNING_HANDLER = '''mkdir -p "$SKYNET_CAPSULE_DIR/state"
rm -f "$SKYNET_CAPSULE_DIR/state/time-limit-warning" "$SKYNET_CAPSULE_DIR/state/interruption.json"
trap 'skynet_wait_interrupted=1; printf "%s\\n" USR1 > "$SKYNET_CAPSULE_DIR/state/time-limit-warning"' USR1'''


def _supervise_runtime(runtime_lines: list[str]) -> list[str]:
    """Wait for finalization even when the batch shell receives a warning."""
    return [
        "skynet_run_runtime() {",
        *runtime_lines,
        "}",
        "skynet_run_runtime &",
        "skynet_runtime_pid=$!",
        "while true; do",
        "  skynet_wait_interrupted=0",
        '  if wait "$skynet_runtime_pid"; then skynet_runtime_rc=0; else skynet_runtime_rc=$?; fi',
        '  if [[ "$skynet_wait_interrupted" == 0 ]]; then break; fi',
        "done",
        'exit "$skynet_runtime_rc"',
    ]


class CompiledSlurmJob(CanonicalModel):
    run_id: str
    stage: Literal["train", "eval"]
    job_name: str
    script: str
    script_sha256: str
    spec_sha256: str
    argv_sha256: str
    run_directory: str
    source_directory: str
    gpu_count: int
    gpu_type: str
    stdout_path_template: str
    stderr_path_template: str
    files: dict[str, str] = Field(default_factory=dict)


def _shell(value: str) -> str:
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise SlurmCompileError("shell literals must be single-line values")
    return shlex.quote(value)


def _safe_identifier(value: str, *, maximum: int = 96) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not normalized:
        raise SlurmCompileError("identifier contains no usable characters")
    return normalized[:maximum]


def resolve_slurm_log_path(
    template: str | None,
    job_id: str,
    *,
    job_name: str | None = None,
    logs_root: str | None = None,
) -> str | None:
    """Resolve the bounded Slurm substitutions used by Skynet log paths."""

    if not template or any(character in template for character in ("\x00", "\n", "\r")):
        return None
    if not re.fullmatch(r"\d+(?:_[0-9]+)?", job_id):
        return None
    if job_name is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", job_name
    ):
        return None

    resolved: list[str] = []
    index = 0
    while index < len(template):
        character = template[index]
        if character != "%":
            resolved.append(character)
            index += 1
            continue
        if index + 1 >= len(template):
            return None
        token = template[index + 1]
        if token == "%":
            resolved.append("%")
        elif token == "j":
            resolved.append(job_id)
        elif token == "x" and job_name is not None:
            resolved.append(job_name)
        else:
            return None
        index += 2

    value = "".join(resolved)
    if not re.fullmatch(r"/[A-Za-z0-9._/%+:-]+", value):
        return None
    path = PurePosixPath(value)
    log_root = PurePosixPath(logs_root or CLUSTER.paths.logs)
    if not path.is_absolute() or ".." in path.parts:
        return None
    try:
        path.relative_to(log_root)
    except ValueError:
        return None
    return str(path)


def resolve_slurm_log_paths_from_sbatch(
    script: str,
    job_id: str,
    *,
    logs_root: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve canonical log directives from a pinned, archived sbatch script."""

    values: dict[str, str] = {}
    ambiguous: set[str] = set()
    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith("#"):
            break
        if not line.startswith("#SBATCH"):
            continue
        exact = re.fullmatch(
            r"#SBATCH[ \t]+--(job-name|output|error)=([^ \t]+)", line
        )
        if exact is None:
            candidate = re.match(
                r"#SBATCH[ \t]+--(job-name|output|error)(?:=|[ \t])", line
            )
            if candidate:
                ambiguous.add(candidate.group(1))
            continue
        name, value = exact.groups()
        if name in values:
            ambiguous.add(name)
            continue
        values[name] = value
    for name in ambiguous:
        values.pop(name, None)

    job_name = values.get("job-name")
    return (
        resolve_slurm_log_path(values.get("output"), job_id, job_name=job_name, logs_root=logs_root),
        resolve_slurm_log_path(values.get("error"), job_id, job_name=job_name, logs_root=logs_root),
    )


def _repo_slug(repository: str) -> str:
    tail = repository.rstrip("/").rsplit("/", 1)[-1]
    tail = tail.removesuffix(".git")
    digest = hashlib.sha256(repository.encode("utf-8")).hexdigest()[:8]
    return f"{_safe_identifier(tail, maximum=48)}-{digest}"


def _encode_file(path: str, content: str) -> str:
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    delimiter = f"SKYNET_{hashlib.sha256(path.encode()).hexdigest()[:16]}"
    return dedent(
        f"""\
        mkdir -p "$(dirname {_shell(path)})"
        base64 --decode > {_shell(path)} <<'{delimiter}'
        {encoded}
        {delimiter}
        """
    ).rstrip()


def _execution_files(
    spec: ExperimentSpec,
    plan: AdapterPlan,
    run_id: str,
    capsule_files: Mapping[str, str] | None = None,
    *,
    stage: Literal["train", "eval"],
    stage_auto_resume: bool,
    evaluation_progress_path: str,
    native_tracking_run_ids: Mapping[str, str] | None = None,
    native_tracking_resume: bool = False,
) -> dict[str, str]:
    is_training = stage == "train"
    execution = {
        "schema_version": 5,
        "run_id": run_id,
        "stage": stage,
        "resume_mode": "checkpoint_argv_suffix" if is_training else "evaluation_ledger",
        "argv": plan.argv,
        "preparation_steps": [
            step.model_dump(mode="json") for step in plan.preparation_steps
        ],
        "resume_argv": plan.resume_argv if is_training else [],
        "auto_resume": stage_auto_resume,
        "checkpoint": spec.train.checkpoint.model_dump(mode="json") if is_training else {},
        "checkpoint_globs": plan.checkpoint_globs if is_training else [],
        "checkpoint_candidate_kind": getattr(plan, "checkpoint_candidate_kind", "any"),
        "checkpoint_basename_regex": getattr(plan, "checkpoint_basename_regex", None),
        "checkpoint_prune_globs": (
            getattr(plan, "checkpoint_prune_globs", []) if is_training else []
        ),
        "training_output_prune_globs": (
            getattr(plan, "training_output_prune_globs", []) if is_training else []
        ),
        "checkpoint_inference_required_globs": getattr(
            plan, "checkpoint_inference_required_globs", []
        ),
        "initial_checkpoint": (
            plan.native_config.get("initial_checkpoint") if is_training else None
        ),
        "native_tracking_resume": bool(native_tracking_resume),
        "native_tracking": [
            {
                "provider": integration.provider,
                "run_id_file": integration.run_id_file,
                "remote_id": (native_tracking_run_ids or {}).get(integration.provider),
            }
            for integration in plan.native_tracking
            if is_training
            and integration.run_id_file
            and (native_tracking_run_ids or {}).get(integration.provider)
        ],
        "evaluation_resume": {
            "enabled": stage_auto_resume if not is_training else False,
            "progress_path": evaluation_progress_path,
            "episode_identity": ["checkpoint", "suite", "version", "task", "seed", "episode_index"],
        },
    }
    files = {
        "requested-spec.json": json.dumps(spec.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True) + "\n",
        "resolved-spec.json": json.dumps(spec.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True) + "\n",
        "native-config.json": json.dumps(plan.native_config, indent=2, sort_keys=True) + "\n",
        "argv.json": json.dumps(plan.argv, indent=2) + "\n",
        "preparation.json": json.dumps(
            [step.model_dump(mode="json") for step in plan.preparation_steps],
            indent=2,
            sort_keys=True,
        ) + "\n",
        "execution.json": json.dumps(execution, indent=2, sort_keys=True) + "\n",
        "runtime-wrapper.py": RUNNER_SOURCE,
        "gpu_metrics.py": Path(__file__).with_name("gpu_metrics.py").read_text(),
        "adapter-plan.json": json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    }
    if spec.source.adapter_manifest is not None:
        files["adapter-manifest.json"] = json.dumps(
            spec.source.adapter_manifest, indent=2, sort_keys=True
        ) + "\n"
    if spec.runtime.resolution:
        files["runtime-resolution.json"] = json.dumps(
            spec.runtime.resolution, indent=2, sort_keys=True
        ) + "\n"
    for name, content in (capsule_files or {}).items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
            raise SlurmCompileError(f"invalid capsule file path: {name}")
        normalized = str(path)
        if normalized in files or normalized == "checksums.sha256":
            raise SlurmCompileError(f"capsule file already exists: {normalized}")
        files[normalized] = content
    checksums = [
        f"{hashlib.sha256(content.encode('utf-8')).hexdigest()}  {name}"
        for name, content in sorted(files.items())
    ]
    files["checksums.sha256"] = "\n".join(checksums) + "\n"
    return files


def _runtime_launch(spec: ExperimentSpec, wrapper_path: str, execution_path: str) -> list[str]:
    runtime = spec.runtime
    wrapper = _shell(wrapper_path)
    execution = _shell(execution_path)
    if runtime.backend == "uv":
        explicit = _shell(runtime.uv_executable) if runtime.uv_executable else "''"
        lines = [
            f"UV_EXPLICIT={explicit}",
            "UV_BIN=",
            'if [[ -n "$UV_EXPLICIT" && -x "$UV_EXPLICIT" ]]; then UV_BIN="$UV_EXPLICIT"; fi',
            'if [[ -z "$UV_BIN" && -x "$WORK_ROOT/.local/bin/uv" ]]; then UV_BIN="$WORK_ROOT/.local/bin/uv"; fi',
            'if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then UV_BIN="$HOME/.local/bin/uv"; fi',
            'if [[ -z "$UV_BIN" ]] && command -v uv >/dev/null 2>&1; then UV_BIN="$(command -v uv)"; fi',
            'if [[ -z "$UV_BIN" ]] && python3 -c "import uv" >/dev/null 2>&1; then UV_BIN="python3 -m uv"; fi',
        ]
        if runtime.bootstrap_uv:
            lines.extend(
                [
                    'if [[ -z "$UV_BIN" ]]; then',
                    '  UV_BOOTSTRAP_ROOT="$WORK_ROOT/.cache/uv/bootstrap-' + runtime.uv_version + '"',
                    '  python3 -m venv "$UV_BOOTSTRAP_ROOT" || { echo "TODO/preflight: python venv unavailable for uv bootstrap" >&2; exit 69; }',
                    f'  "$UV_BOOTSTRAP_ROOT/bin/python" -m pip install --disable-pip-version-check --no-input "uv=={runtime.uv_version}" || {{ echo "TODO/preflight: pinned uv bootstrap failed" >&2; exit 69; }}',
                    '  UV_BIN="$UV_BOOTSTRAP_ROOT/bin/uv"',
                    "fi",
                ]
            )
        lines.extend(
            [
                'if [[ -z "$UV_BIN" ]]; then echo "TODO/preflight: uv not found; configure runtime.uv_executable, an existing env, or a container" >&2; exit 69; fi',
                'if [[ "$UV_BIN" != "python3 -m uv" ]]; then',
                '  export SKYNET_UV_EXECUTABLE="$UV_BIN"',
                '  export PATH="$(dirname "$UV_BIN"):$PATH"',
                "else",
                '  unset SKYNET_UV_EXECUTABLE',
                "fi",
                'if [[ "$UV_BIN" == "python3 -m uv" ]]; then',
                f"  python3 -m uv run --frozen --project \"$SKYNET_PROJECT_DIR\" python3 {wrapper} {execution}",
                "else",
                f"  \"$UV_BIN\" run --frozen --project \"$SKYNET_PROJECT_DIR\" python3 {wrapper} {execution}",
                "fi",
            ]
        )
        return lines
    if runtime.backend == "conda":
        assert runtime.environment_path is not None
        if runtime.lock_file:
            lock_lines = [
                'command -v conda >/dev/null 2>&1 || { echo "preflight: conda not found" >&2; exit 69; }',
                'command -v conda-lock >/dev/null 2>&1 || { echo "preflight: conda-lock not found" >&2; exit 69; }',
                f"SKYNET_CONDA_ENV={_shell(runtime.environment_path)}",
                f"SKYNET_CONDA_LOCK=\"$SKYNET_PROJECT_DIR\"/{_shell(runtime.lock_file)}",
                '[[ -f "$SKYNET_CONDA_LOCK" ]] || { echo "preflight: conda lock file missing" >&2; exit 69; }',
                'mkdir -p "$(dirname "$SKYNET_CONDA_ENV")"',
                '(',
                '  flock -x 9',
                '  if [[ ! -x "$SKYNET_CONDA_ENV/bin/python" ]]; then',
                '    conda-lock install --prefix "$SKYNET_CONDA_ENV" "$SKYNET_CONDA_LOCK"',
                '  fi',
                ') 9>"$SKYNET_CONDA_ENV.lock"',
                f"conda run --no-capture-output -p \"$SKYNET_CONDA_ENV\" python3 {wrapper} {execution}",
            ]
            if runtime.lock_sha256:
                lock_lines.insert(
                    4,
                    f"printf '%s  %s\\n' {_shell(runtime.lock_sha256)} \"$SKYNET_CONDA_LOCK\" | sha256sum -c -",
                )
            return lock_lines
        return [
            'command -v conda >/dev/null 2>&1 || { echo "preflight: conda not found" >&2; exit 69; }',
            f"conda run --no-capture-output -p {_shell(runtime.environment_path)} python3 {wrapper} {execution}",
        ]
    if runtime.backend == "apptainer":
        assert runtime.container_image is not None
        return [
            'command -v apptainer >/dev/null 2>&1 || { echo "preflight: apptainer not found" >&2; exit 69; }',
            f"apptainer exec --nv {_shell(runtime.container_image)} python3 {wrapper} {execution}",
        ]
    if runtime.backend == "existing":
        if runtime.environment_path:
            activate = str(PurePosixPath(runtime.environment_path) / "bin" / "activate")
            return [
                f"[[ -r {_shell(activate)} ]] || {{ echo \"preflight: existing environment activation script missing\" >&2; exit 69; }}",
                f"source {_shell(activate)}",
                f"python3 {wrapper} {execution}",
            ]
        return [f"python3 {wrapper} {execution}"]
    raise SlurmCompileError("runtime backend must be resolved before sbatch compilation")


def compile_slurm_placement_directives(resources: Any) -> list[str]:
    """Render placement directives from a validated ResourceSpec."""

    if resources.node.mode != "manual":
        return []
    assert resources.node.name is not None
    return [f"#SBATCH --nodelist={resources.node.name}"]


def compile_sbatch(
    spec: ExperimentSpec,
    plan: AdapterPlan,
    *,
    run_id: str,
    stage: Literal["train", "eval"] = "train",
    capsule_files: Mapping[str, str] | None = None,
    stage_auto_resume: bool | None = None,
    runtime_environment: Mapping[str, str] | None = None,
    secret_environment_files: Mapping[str, str] | None = None,
    native_tracking_run_ids: Mapping[str, str] | None = None,
    native_tracking_resume: bool = False,
    work_root: str | None = None,
) -> CompiledSlurmJob:
    paths = paths_for_root(work_root) if work_root is not None else CLUSTER.paths
    work_root = paths.work_root
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise SlurmCompileError("run_id must be a safe stable identifier")
    if not plan.runnable:
        raise SlurmCompileError("adapter plan is blocked: " + "; ".join(plan.blockers))
    if plan.adapter != spec.source.adapter or plan.adapter_version != spec.source.adapter_version:
        raise SlurmCompileError("adapter plan does not match the experiment source")
    if (
        plan.checkpoint_globs
        and plan.checkpoint_candidate_kind == "directory"
        and not plan.checkpoint_inference_required_globs
    ):
        raise SlurmCompileError(
            "directory checkpoint candidates require adapter-declared inference paths or markers"
        )

    effective_auto_resume = (
        spec.train.checkpoint.auto_resume
        if stage == "train" and stage_auto_resume is None
        else bool(stage_auto_resume)
    )
    canonical_evaluation = plan.native_config.get("canonical_evaluation", {})
    configured_progress_path = (
        canonical_evaluation.get("progress_path")
        if isinstance(canonical_evaluation, Mapping)
        else None
    )
    evaluation_progress_path = (
        configured_progress_path
        if stage == "eval" and isinstance(configured_progress_path, str) and configured_progress_path
        else f"{work_root}/eval/runs/{run_id}/progress.jsonl"
    )

    gpu_count = resolve_gpu_count(spec, plan)
    gpu_type = resolve_gpu_type(spec, plan)
    if stage == "eval":
        try:
            resources = resolve_evaluation_resources(
                spec.resources,
                canonical_evaluation if isinstance(canonical_evaluation, Mapping) else {},
                runtime_profile_id=plan.native_config.get("evaluation_runtime_profile_id"),
                runtime=spec.runtime.model_dump(mode="json"),
                gpu_count=gpu_count, gpu_type=gpu_type,
            )
        except ValueError as error:
            raise SlurmCompileError(str(error)) from error
        if resources is not spec.resources:
            spec = spec.model_copy(update={"resources": resources})
            gpu_type = resources.gpu.gpu_type
    if gpu_count > 1 and not plan.capabilities.supports_multi_gpu_single_node:
        raise SlurmCompileError("adapter cannot use multiple GPUs on one node")
    spec_sha = canonical_sha256(spec)
    argv_sha = canonical_sha256(
        {
            "argv": plan.argv,
            "preparation_steps": [
                step.model_dump(mode="json") for step in plan.preparation_steps
            ],
        }
    )
    run_directory = f"{work_root}/jobs/runs/{run_id}"
    source_directory = f"{work_root}/repos/{_repo_slug(spec.source.repository)}/{spec.source.revision}"
    project_directory = str(PurePosixPath(source_directory) / spec.source.project_subdirectory)
    injected_environment = dict(runtime_environment or {})
    secret_files = dict(secret_environment_files or {})
    for key, value in injected_environment.items():
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            raise SlurmCompileError(f"invalid injected environment variable: {key}")
        if re.search(r"(?:API_?KEY|TOKEN|PASSWORD|SECRET)$", key):
            raise SlurmCompileError(
                f"sensitive environment variable must use a secret file: {key}"
            )
        if any(character in value for character in ("\x00", "\n", "\r")):
            raise SlurmCompileError(f"injected environment value must be single-line: {key}")
    for key, path_value in secret_files.items():
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            raise SlurmCompileError(f"invalid secret environment variable: {key}")
        path = PurePosixPath(path_value)
        run_root = PurePosixPath(run_directory)
        if not path.is_absolute() or path == run_root or run_root not in path.parents:
            raise SlurmCompileError(
                f"secret environment file must be below the run directory: {key}"
            )
    files = _execution_files(
        spec,
        plan,
        run_id,
        capsule_files,
        stage=stage,
        stage_auto_resume=effective_auto_resume,
        evaluation_progress_path=evaluation_progress_path,
        native_tracking_run_ids=native_tracking_run_ids,
        native_tracking_resume=native_tracking_resume,
    )
    materializers = [
        _encode_file(f"{run_directory}/{name}", content)
        for name, content in sorted(files.items())
    ]
    attempt_archivers = [
        f"install -D -m 600 {_shell(f'{run_directory}/{name}')} \"$SKYNET_CAPSULE_DIR/{name}\""
        for name in sorted(files)
    ]

    job_name = _safe_identifier(f"{spec.identity.experiment}-{stage}", maximum=64)
    stdout_path_template = f"{paths.logs}/{job_name}-%j.out"
    stderr_path_template = f"{paths.logs}/{job_name}-%j.err"
    if gpu_type not in CLUSTER.gpu_aliases:
        raise SlurmCompileError(f"GPU type is not configured: {gpu_type}")
    gpu_alias = CLUSTER.gpu_aliases[gpu_type]
    gres = f"gpu:{gpu_count}" if gpu_alias is None else f"gpu:{gpu_alias}:{gpu_count}"
    directives = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={spec.resources.account}",
        f"#SBATCH --partition={spec.resources.partition}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --gres={gres}",
        f"#SBATCH --cpus-per-task={spec.resources.cpus_per_task}",
        f"#SBATCH --mem={spec.resources.memory_gb}G",
        f"#SBATCH --time={spec.resources.time_limit}",
        f"#SBATCH --chdir={paths.workspace}",
        f"#SBATCH --output={paths.logs}/%x-%j.out",
        f"#SBATCH --error={paths.logs}/%x-%j.err",
        "#SBATCH --export=NIL",
    ]
    directives.extend(compile_slurm_placement_directives(spec.resources))
    if stage == "train" and effective_auto_resume:
        directives.extend(
            [
                f"#SBATCH --signal=B:USR1@{spec.train.checkpoint.save_before_timeout_seconds}",
                "#SBATCH --requeue",
            ]
        )
    elif stage == "eval" and effective_auto_resume:
        directives.append("#SBATCH --requeue")

    exports = {
        "HOME": HOME_ROOT,
        "WORK_ROOT": work_root,
        "XDG_CACHE_HOME": f"{work_root}/.cache",
        "UV_CACHE_DIR": paths.uv_cache,
        "HF_HOME": paths.huggingface_cache,
        "TORCH_HOME": paths.torch_cache,
        "SKYNET_RUN_ID": run_id,
        "SKYNET_RUN_DIR": run_directory,
        "SKYNET_SOURCE_DIR": source_directory,
        "SKYNET_PROJECT_DIR": project_directory,
        "SKYNET_CHECKPOINT_DIR": f"{run_directory}/checkpoints",
        "SKYNET_CHECKPOINT_SAVE_STEPS": str(spec.train.checkpoint.save_every_steps),
        "SKYNET_CHECKPOINT_KEEP_LAST": str(spec.train.checkpoint.keep_last),
        "SKYNET_EVAL_ROOT": f"{work_root}/eval",
        "SKYNET_EVAL_PROGRESS_PATH": evaluation_progress_path,
        "SKYNET_EVALUATION_RESUME": (
            "1" if stage == "eval" and effective_auto_resume else "0"
        ),
        **spec.runtime.environment,
        **plan.environment,
    }
    for key, value in injected_environment.items():
        if key in exports and exports[key] != value:
            raise SlurmCompileError(
                f"injected environment conflicts with the pinned execution: {key}"
            )
        exports[key] = value
    for key in secret_files:
        if key in exports:
            raise SlurmCompileError(
                f"secret environment conflicts with the pinned execution: {key}"
            )
    export_lines = [f"export {key}={_shell(value)}" for key, value in sorted(exports.items())]
    secret_lines: list[str] = []
    for key, path in sorted(secret_files.items()):
        secret_lines.extend(
            [
                f"[[ -r {_shell(path)} ]] || {{ echo {_shell(f'preflight: secret file for {key} is unavailable')} >&2; exit 78; }}",
                f"export {key}=\"$(cat -- {_shell(path)})\"",
                f"rm -f -- {_shell(path)}",
                f"[[ -n \"${{{key}}}\" ]] || {{ echo {_shell(f'preflight: secret value for {key} is empty')} >&2; exit 78; }}",
            ]
        )

    submodule_setup = (
        'git -C "$SKYNET_SOURCE_DIR" submodule sync --recursive\n'
        '          git -C "$SKYNET_SOURCE_DIR" submodule update --init --recursive'
        if spec.source.include_submodules
        else ":"
    )
    lfs_setup = (
        'if command -v git-lfs >/dev/null 2>&1; then git -C "$SKYNET_SOURCE_DIR" lfs pull; fi'
        if spec.source.include_git_lfs
        else ":"
    )
    source_setup = dedent(
        f"""\
        mkdir -p "$WORK_ROOT"/{{workspace,repos,datasets,artifacts,logs,jobs}}
        mkdir -p "$XDG_CACHE_HOME" "$WORK_ROOT"/.cache/{{uv,huggingface,torch}}
        mkdir -p "$WORK_ROOT"/eval/{{catalogs,datasets,assets,runs}}
        mkdir -p "$SKYNET_RUN_DIR"/{{artifacts,checkpoints,eval,state,attempts}}
        if [[ -n "${{SKYNET_EVAL_RESULT_PATH:-}}" ]]; then
          mkdir -p "$(dirname "$SKYNET_EVAL_RESULT_PATH")"
        fi
        mkdir -p "$(dirname "$SKYNET_SOURCE_DIR")"

        (
          flock -x 9
          if [[ ! -d "$SKYNET_SOURCE_DIR/.git" ]]; then
            rm -rf "$SKYNET_SOURCE_DIR"
            git clone --no-checkout -- {_shell(spec.source.repository)} "$SKYNET_SOURCE_DIR"
          fi
          current_origin="$(git -C "$SKYNET_SOURCE_DIR" remote get-url origin)"
          if [[ "$current_origin" != {_shell(spec.source.repository)} ]]; then
            echo "preflight: cached repository origin mismatch" >&2
            exit 65
          fi
          git -C "$SKYNET_SOURCE_DIR" fetch --force origin {_shell(spec.source.revision)}
          git -C "$SKYNET_SOURCE_DIR" checkout --detach --force {_shell(spec.source.revision)}
          actual_commit="$(git -C "$SKYNET_SOURCE_DIR" rev-parse HEAD)"
          [[ "$actual_commit" == {_shell(spec.source.revision.lower())} ]] || {{ echo "preflight: source commit mismatch" >&2; exit 65; }}
          {submodule_setup}
          {lfs_setup}
        ) 9>"$SKYNET_SOURCE_DIR.lock"

        [[ -d "$SKYNET_PROJECT_DIR" ]] || {{ echo "preflight: project subdirectory missing" >&2; exit 66; }}
        """
    ).rstrip()

    provenance = dedent(
        r"""
        export SKYNET_SOURCE_MANIFEST="$SKYNET_CAPSULE_DIR/source-manifest.json"
        export SKYNET_DIRTY_PATCH="$SKYNET_CAPSULE_DIR/dirty.patch"
        python3 - <<'PY'
        import json, os, platform, shutil, subprocess
        from datetime import datetime, timezone
        from pathlib import Path

        source = os.environ["SKYNET_SOURCE_DIR"]
        def git(*args):
            return subprocess.run(["git", "-C", source, *args], check=True, text=True, capture_output=True).stdout.strip()
        status = git("status", "--porcelain=v1", "--untracked-files=all")
        patch = git("diff", "--binary", "HEAD")
        lfs_process = subprocess.run(
            ["git", "-C", source, "lfs", "ls-files", "--long"],
            check=False, text=True, capture_output=True,
        )
        Path(os.environ["SKYNET_DIRTY_PATCH"]).write_text(patch)
        manifest = {
            "schema_version": 1,
            "repository": git("remote", "get-url", "origin"),
            "commit": git("rev-parse", "HEAD"),
            "submodules": git("submodule", "status", "--recursive").splitlines(),
            "dirty_status": status.splitlines(),
            "dirty": bool(status or patch),
            "git_lfs": lfs_process.stdout.strip().splitlines() if lfs_process.returncode == 0 else [],
            "host": platform.node(),
        }
        Path(os.environ["SKYNET_SOURCE_MANIFEST"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if manifest["dirty"] and os.environ.get("SKYNET_DIRTY_POLICY") == "reject":
            raise SystemExit("preflight: source checkout is dirty")

        def capture(argv):
            if shutil.which(argv[0]) is None:
                return {"argv": argv, "available": False}
            try:
                process = subprocess.run(
                    argv, check=False, text=True, capture_output=True, timeout=30,
                )
                return {
                    "argv": argv,
                    "available": True,
                    "returncode": process.returncode,
                    "stdout": process.stdout[-200000:],
                    "stderr": process.stderr[-200000:],
                }
            except subprocess.TimeoutExpired:
                return {"argv": argv, "available": True, "timed_out": True}

        os_release = Path("/etc/os-release")
        system_manifest = {
            "schema_version": 1,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "os_release": os_release.read_text() if os_release.is_file() else None,
            "slurm_environment": {
                key: value for key, value in sorted(os.environ.items())
                if key.startswith("SLURM_")
            },
            "commands": {
                "lscpu": capture(["lscpu", "--json"]),
                "nvidia_smi": capture([
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ]),
                "nvidia_topology": capture(["nvidia-smi", "topo", "-m"]),
                "nvcc": capture(["nvcc", "--version"]),
            },
        }
        (Path(os.environ["SKYNET_CAPSULE_DIR"]) / "system-manifest.json").write_text(
            json.dumps(system_manifest, indent=2, sort_keys=True) + "\n"
        )
        PY
        """
    ).strip()

    status_trap = dedent(
        r"""
        finalize_attempt() {
          rc=$?
          export SKYNET_FINAL_RC="$rc"
          python3 - <<'PY' || true
        import hashlib, json, os, shutil
        from datetime import datetime, timezone
        from pathlib import Path
        run_dir = Path(os.environ["SKYNET_RUN_DIR"])
        capsule_dir = Path(os.environ["SKYNET_CAPSULE_DIR"])
        payload = {
            "schema_version": 1,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "exit_code": int(os.environ["SKYNET_FINAL_RC"]),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        target = capsule_dir / "final.json"
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        for name in ("latest.json", "selected-for-inference.json"):
            source = run_dir / "checkpoints" / name
            if source.is_file():
                destination = capsule_dir / "checkpoints" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        metadata_paths = [
            "adapter-plan.json",
            "adapter-manifest.json",
            "argv.json",
            "checksums.sha256",
            "dirty.patch",
            "execution.json",
            "job.sbatch",
            "native-config.json",
            "requested-spec.json",
            "resolved-spec.json",
            "runtime-manifest.json",
            "runtime-resolution.json",
            "runtime-wrapper.py",
            "source-manifest.json",
            "state/effective-argv.json",
            "system-manifest.json",
            "checkpoints/latest.json",
            "checkpoints/selected-for-inference.json",
            "final.json",
        ]
        files = []
        for relative in metadata_paths:
            path = capsule_dir / relative
            if not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append({"path": relative, "sha256": digest, "size_bytes": path.stat().st_size})
        capsule = {
            "schema_version": 1,
            "run_id": os.environ.get("SKYNET_RUN_ID"),
            "job_id": payload["job_id"],
            "files": files,
        }
        (capsule_dir / "capsule-manifest.json").write_text(
            json.dumps(capsule, indent=2, sort_keys=True) + "\n"
        )
        latest = run_dir / "state" / "latest-attempt.json"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(json.dumps({
            "job_id": payload["job_id"],
            "capsule_directory": str(capsule_dir),
            "capsule_manifest": str(capsule_dir / "capsule-manifest.json"),
        }, indent=2, sort_keys=True) + "\n")
        PY
          exit "$rc"
        }
        trap finalize_attempt EXIT
        """
    ).strip()

    runtime_lines = _runtime_launch(
        spec,
        f"{run_directory}/runtime-wrapper.py",
        f"{run_directory}/execution.json",
    )
    script_parts = [
        "#!/usr/bin/env bash",
        f"# skynet-spec-sha256: {spec_sha}",
        f"# skynet-argv-sha256: {argv_sha}",
        f"# skynet-adapter: {plan.adapter}@{plan.adapter_version}",
        f"# skynet-source-revision: {spec.source.revision}",
        *directives,
        "",
        "set -Eeuo pipefail",
        "umask 077",
        *export_lines,
        f"export SKYNET_DIRTY_POLICY={_shell(spec.source.dirty_policy)}",
        f"export PATH={SLURM_BIN}:/usr/local/bin:/usr/bin:/bin",
        'export SKYNET_CAPSULE_DIR="$SKYNET_RUN_DIR/attempts/$SLURM_JOB_ID"',
        'mkdir -p "$SKYNET_CAPSULE_DIR"',
        "",
        status_trap,
        BATCH_WARNING_HANDLER,
        "",
        *materializers,
        *attempt_archivers,
        "",
        'if command -v scontrol >/dev/null 2>&1; then scontrol write batch_script "$SLURM_JOB_ID" "$SKYNET_CAPSULE_DIR/job.sbatch" >/dev/null 2>&1 || true; fi',
        'cd "$SKYNET_RUN_DIR" && sha256sum -c checksums.sha256',
        "",
        source_setup,
        "",
        provenance,
        "",
        'ln -sfn "' + work_root + '/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$SKYNET_CAPSULE_DIR/stdout.log"',
        'ln -sfn "' + work_root + '/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" "$SKYNET_CAPSULE_DIR/stderr.log"',
        *secret_lines,
        'cd "$SKYNET_PROJECT_DIR"',
        *_supervise_runtime(runtime_lines),
        "",
    ]
    script = "\n".join(script_parts)
    return CompiledSlurmJob(
        run_id=run_id,
        stage=stage,
        job_name=job_name,
        script=script,
        script_sha256=hashlib.sha256(script.encode("utf-8")).hexdigest(),
        spec_sha256=spec_sha,
        argv_sha256=argv_sha,
        run_directory=run_directory,
        source_directory=source_directory,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        stdout_path_template=stdout_path_template,
        stderr_path_template=stderr_path_template,
        files=files,
    )
