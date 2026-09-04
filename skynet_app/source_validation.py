from __future__ import annotations

import re
import time
from typing import Any, Mapping

from .adapters import AdapterManifest, AdapterPlan, adapter_manifest_sha256
from .experiments import ExperimentSpec, canonical_sha256


RESULT_SCHEMA = "skynet.static-repository-argument-validation/v1"
CACHE_KIND = "static-repository-argument-validation/v1"
FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _lookup(document: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = document
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def repository_argument_validation_cache_parameters(
    spec: ExperimentSpec,
    manifest: AdapterManifest,
    plan: AdapterPlan,
    repository_options: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    manifest_sha256 = str(spec.source.adapter_manifest_sha256 or "").lower()
    if not SHA256_RE.fullmatch(manifest_sha256):
        manifest_sha256 = adapter_manifest_sha256(manifest)
    return {
        "commit": spec.source.revision.lower(),
        "manifest_sha256": manifest_sha256,
        "argv_sha256": canonical_sha256(plan.argv),
        "runtime_sha256": canonical_sha256(
            spec.runtime.model_dump(mode="json", by_alias=True)
        ),
        "repository_metadata_sha256": canonical_sha256(repository_options or {}),
    }


def validate_repository_arguments(
    spec: ExperimentSpec,
    manifest: AdapterManifest,
    plan: AdapterPlan,
    repository_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a generated plan using only pinned, previously inspected metadata."""

    started = time.monotonic()
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []
    revision = spec.source.revision.lower()
    manifest_sha256 = str(spec.source.adapter_manifest_sha256 or "").lower()
    declared_manifest_sha256 = str(spec.source.adapter_manifest_sha256 or "").lower()
    if not FULL_COMMIT_RE.fullmatch(revision):
        errors.append("source revision is not a pinned 40-character commit")
    else:
        checks.append("pinned_source_commit")
    if declared_manifest_sha256 and not SHA256_RE.fullmatch(declared_manifest_sha256):
        errors.append("pinned adapter manifest digest is not a SHA-256 value")
    else:
        if not manifest_sha256:
            manifest_sha256 = adapter_manifest_sha256(manifest)
        checks.append("pinned_manifest_digest")
    if plan.adapter != manifest.slug:
        errors.append("generated plan adapter does not match the pinned manifest")
    else:
        checks.append("plan_adapter_identity")
    if not plan.argv:
        errors.append("generated adapter argv is empty")
    elif any(
        not isinstance(argument, str)
        or any(character in argument for character in ("\x00", "\n", "\r"))
        for argument in plan.argv
    ):
        errors.append("generated adapter argv contains a non-string or multiline argument")
    else:
        checks.append("generated_argv_shape")

    options = repository_options or {}
    spec_document = spec.model_dump(mode="json", by_alias=True)
    for field in manifest.train.input_fields:
        if field.choice_source is None:
            continue
        option = options.get(field.path)
        if not isinstance(option, Mapping) or not option.get("complete"):
            errors.append(f"{field.path}: pinned repository metadata is incomplete")
            continue
        source = option.get("source")
        rule = field.choice_source.model_dump(mode="json")
        if not isinstance(source, Mapping):
            errors.append(f"{field.path}: pinned repository metadata has no source evidence")
            continue
        if str(source.get("commit") or "").lower() != revision:
            errors.append(f"{field.path}: repository metadata commit does not match")
        if source.get("rule_sha256") != canonical_sha256(rule):
            errors.append(f"{field.path}: repository metadata rule digest does not match")
        files = source.get("files")
        if not isinstance(files, list) or not files or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("path"), str)
            or not SHA256_RE.fullmatch(str(item.get("sha256") or ""))
            for item in files
        ):
            errors.append(f"{field.path}: repository metadata file evidence is invalid")
        present, selected = _lookup(spec_document, field.path)
        choices = list(option.get("choices") or [])
        if (
            present
            and selected not in (None, "")
            and selected not in choices
            and not field.choice_source.allow_custom
        ):
            errors.append(
                f"{field.path}: selected value is absent from pinned repository metadata"
            )
        checks.append(f"repository_input:{field.path}")

    if manifest.train.argument_validation is not None:
        warnings.append(
            "historical executable argument-validation contract is inert; "
            "only pinned static repository metadata was evaluated"
        )
    valid = not errors
    return {
        "schema_version": RESULT_SCHEMA,
        "status": "passed" if valid else "failed",
        "phase": "static_repository_metadata",
        "validation_mode": "local_static_only",
        "valid": valid,
        "required_before_submit": True,
        "heavy_execution": False,
        "duration_seconds": round(time.monotonic() - started, 6),
        "repository": spec.source.repository,
        "commit": revision,
        "manifest_sha256": manifest_sha256,
        "argv_sha256": canonical_sha256(plan.argv),
        "runtime_sha256": canonical_sha256(
            spec.runtime.model_dump(mode="json", by_alias=True)
        ),
        "repository_metadata_sha256": canonical_sha256(options),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


def validation_failure_reason(report: Mapping[str, Any]) -> str | None:
    if bool(report.get("valid")):
        return None
    errors = [str(item) for item in report.get("errors") or [] if str(item).strip()]
    detail = "; ".join(errors) or "static validation did not produce success evidence"
    return f"static repository argument validation failed: {detail}"


__all__ = [
    "CACHE_KIND",
    "repository_argument_validation_cache_parameters",
    "validate_repository_arguments",
    "validation_failure_reason",
]
