from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTRACT_SCHEMA = "skynet.runtime-readiness-contract/v1"
ATTESTATION_SCHEMA = "skynet.runtime-readiness/v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def suite_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    if reference := config.get("runtime_readiness_suite"):
        from .experiments import get_evaluation_catalog

        shared = next((item for item in get_evaluation_catalog()
                       if item.suite == reference["suite"]
                       and item.version == reference["version"]), None)
        if shared is None:
            raise ValueError("The shared simulator readiness suite is not registered")
        base = shared.model_dump(mode="json")
        # A different reset strategy can reuse evidence for the same simulator,
        # source revision and task registration. Cross-simulator aliases cannot.
        if (base.get("runtime_readiness_suite")
                or any(base.get(key) != config.get(key) for key in (
                    "evaluator", "task_catalog_provenance", "dataset_task_binding"))
                or not set(config.get("tasks") or []).issubset(base["tasks"])):
            raise ValueError("Shared readiness requires the same simulator source and tasks")
        return suite_contract(base)
    if binding := config.get("dataset_task_binding"):
        # Runtime readiness verifies the pinned simulator and registration module.
        # Each dataset's concrete task/robot contract is checked by its rollout.
        # Binding a new dataset must not invalidate unchanged GPU runtime evidence.
        base = dict(
            config, tasks=[], task_options=[], task_catalog_sha256=_content_sha256([])
        )
        base.pop("dataset_task_binding")
        return {**suite_contract(base), "dataset_task_binding": dict(binding)}
    options = config.get("task_options")
    if not isinstance(options, list):
        options = []
    task_environments: list[dict[str, Any]] = []
    for option in options:
        if not isinstance(option, Mapping):
            continue
        metadata = option.get("metadata")
        native = metadata.get("native") if isinstance(metadata, Mapping) else None
        task_environments.append(
            {
                "id": option.get("id"),
                "environment_id": (
                    native.get("environment_id") if isinstance(native, Mapping) else None
                ),
                "simulation_device": (
                    native.get("simulation_device") if isinstance(native, Mapping) else None
                ),
            }
        )
    provenance = config.get("task_catalog_provenance")
    return {
        "evaluator": config.get("evaluator"),
        "suite": config.get("suite") or config.get("name"),
        "version": config.get("version") or config.get("suite_version"),
        "catalog_sha256": config.get("catalog_sha256"),
        "task_catalog_sha256": config.get("task_catalog_sha256"),
        "task_catalog_provenance": (
            dict(provenance) if isinstance(provenance, Mapping) else None
        ),
        "tasks": list(config.get("tasks") or []),
        "task_environments": task_environments,
    }


def suite_contract_sha256(config: Mapping[str, Any]) -> str:
    return _content_sha256(suite_contract(config))


def requires_isaac_consent(contract):
    return (contract.get("suite", {}).get("evaluator") in {"isaac_sim", "isaac_lab"}
            or "isaacsim" in contract.get("verification", {}).get("distributions", {}))


def _resolve_template(value: str, tokens: Mapping[str, str]) -> str:
    resolved = value
    for name, replacement in tokens.items():
        resolved = resolved.replace("{{" + name + "}}", replacement)
    if "{{" in resolved or "}}" in resolved:
        raise ValueError(f"unresolved runtime-readiness template: {value}")
    return resolved


def _find_capsule_source(adapter: str, asset: str) -> str:
    from .adapters import builtin_adapter_manifests

    manifest = next(
        (item for item in builtin_adapter_manifests() if item.slug == adapter),
        None,
    )
    if manifest is None:
        raise ValueError(f"readiness adapter is not registered: {adapter}")
    for declaration in manifest.evaluations:
        if declaration.command is None:
            continue
        source = declaration.command.capsule_files.get(asset)
        if source is not None:
            return source
    raise ValueError(f"adapter {adapter} does not publish readiness capsule asset {asset}")


def build_readiness_contract(profile_id: str, suite_id: str) -> tuple[dict[str, Any], str]:
    from .cluster_config import CLUSTER
    from .experiments import get_evaluation_catalog

    profile = CLUSTER.runtime_profile(profile_id)
    profile_snapshot = CLUSTER.runtime_profile_snapshot(profile_id)
    smoke = profile.verification.compute_smoke
    if smoke is None:
        raise ValueError(f"runtime profile {profile_id} does not declare compute_smoke")
    if smoke.suite != suite_id:
        raise ValueError(
            f"runtime profile {profile_id} readiness smoke is pinned to {smoke.suite}, "
            f"not {suite_id}"
        )
    suite = next(
        (
            item
            for item in get_evaluation_catalog()
            if item.suite == suite_id and item.current
        ),
        None,
    )
    if suite is None:
        raise ValueError(f"current evaluation suite is not registered: {suite_id}")
    suite_document = suite.model_dump(mode="json")
    pinned_suite = suite_contract(suite_document)
    capsule_source = _find_capsule_source(smoke.adapter, smoke.capsule_asset)
    capsule_sha256 = hashlib.sha256(capsule_source.encode("utf-8")).hexdigest()
    if capsule_sha256 != smoke.capsule_sha256:
        raise ValueError(
            f"readiness capsule {smoke.capsule_asset} has sha256 {capsule_sha256}, "
            f"expected {smoke.capsule_sha256}; publish an updated runtime profile"
        )

    expected_task_ids = [
        task_id
        for registration in profile.verification.gym_registrations
        for task_id in registration.ids
    ]
    suite_task_ids = [
        str(item.get("environment_id") or "")
        for item in pinned_suite["task_environments"] if item.get("environment_id")
    ]
    if suite_task_ids != expected_task_ids:
        raise ValueError(
            "runtime readiness Gym IDs do not exactly match the selected suite task catalog"
        )

    source_dir = ""
    provenance = pinned_suite.get("task_catalog_provenance") or {}
    revision = str(provenance.get("revision") or "")
    for prerequisite in profile.source_prerequisites:
        if prerequisite.kind in {"git_checkout", "directory"} and prerequisite.revision == revision:
            source_dir = prerequisite.path
            break
    if not source_dir:
        raise ValueError(
            "runtime profile has no exact source prerequisite for the suite catalog revision"
        )

    verification = profile.verification.model_dump(mode="json")
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "profile_id": profile_id,
        "profile_snapshot": profile_snapshot,
        "profile_snapshot_sha256": _content_sha256(profile_snapshot),
        "suite": pinned_suite,
        "suite_contract_sha256": _content_sha256(pinned_suite),
        "verification": verification,
        "runtime": {
            "environment_path": profile.environment_path,
            "python_executable": (
                f"{str(profile.environment_path).rstrip('/')}/bin/python"
                if profile.environment_path
                else None
            ),
            "source_dir": source_dir,
        },
        "hook": {
            "adapter": smoke.adapter,
            "capsule_asset": smoke.capsule_asset,
            "capsule_sha256": capsule_sha256,
            "argv": smoke.argv,
            "environment": smoke.environment,
            "result_schema": smoke.result_schema,
            "timeout_seconds": smoke.timeout_seconds,
        },
    }
    return contract, capsule_source


def _render_environment_exports(profile_snapshot: Mapping[str, Any]) -> list[str]:
    environment_path = str(profile_snapshot.get("environment_path") or "")
    lines = ["export PATH=" + shlex.quote(f"{environment_path}/bin:/usr/local/bin:/usr/bin:/bin")]
    for name, value in (profile_snapshot.get("environment") or {}).items():
        if name == "OMNI_KIT_ACCEPT_EULA":
            continue
        lines.append(f"export {name}={shlex.quote(str(value))}")
    for operation in profile_snapshot.get("environment_operations") or []:
        name = str(operation["name"])
        if name == "OMNI_KIT_ACCEPT_EULA":
            continue
        kind = str(operation.get("operation") or "set")
        if kind == "unset":
            lines.append(f"unset {name}")
            continue
        value = _resolve_template(
            str(operation.get("value") or ""),
            {"runtime.environment_path": environment_path},
        )
        quoted_value = shlex.quote(value)
        quoted_separator = shlex.quote(str(operation.get("separator") or ":"))
        if kind == "prepend":
            lines.append(
                f'if test -n "${{{name}:-}}"; then export {name}='
                f'{quoted_value}{quoted_separator}"${{{name}}}"; '
                f"else export {name}={quoted_value}; fi"
            )
        elif kind == "append":
            lines.append(
                f'if test -n "${{{name}:-}}"; then export {name}='
                f'"${{{name}}}"{quoted_separator}{quoted_value}; '
                f"else export {name}={quoted_value}; fi"
            )
        else:
            lines.append(f"export {name}={quoted_value}")
    return lines


def render_readiness_sbatch(
    profile_id: str,
    suite_id: str,
    *,
    queue_policy: str | None = None,
    gpu_type: str | None = None,
    cpus_per_task: int | None = None,
    memory_gb: int | None = None,
    time_limit: str | None = None,
    node: str | None = None,
) -> str:
    from .cluster_config import CLUSTER
    from .experiments import ResourceSpec, parse_slurm_duration
    from .slurm import compile_slurm_placement_directives
    from .evaluation_placement import resolve_evaluation_resources

    contract, capsule_source = build_readiness_contract(profile_id, suite_id)
    profile = CLUSTER.runtime_profile(profile_id)
    smoke = profile.verification.compute_smoke
    assert smoke is not None
    minimum = smoke.resources
    queue_policy = queue_policy or minimum.queue_policy
    gpu_type = gpu_type or minimum.gpu_type
    cpus_per_task = (
        minimum.cpus_per_task if cpus_per_task is None else cpus_per_task
    )
    memory_gb = minimum.memory_gb if memory_gb is None else memory_gb
    time_limit = time_limit or minimum.time_limit
    if cpus_per_task < minimum.cpus_per_task:
        raise ValueError(
            f"runtime profile {profile_id} readiness requires at least "
            f"{minimum.cpus_per_task} CPUs per task"
        )
    if memory_gb < minimum.memory_gb:
        raise ValueError(
            f"runtime profile {profile_id} readiness requires at least "
            f"{minimum.memory_gb} GB memory"
        )
    if parse_slurm_duration(time_limit) < parse_slurm_duration(minimum.time_limit):
        raise ValueError(
            f"runtime profile {profile_id} readiness requires a time limit of at least "
            f"{minimum.time_limit}"
        )
    queue = CLUSTER.queue(queue_policy)
    resources = ResourceSpec.model_validate(
        {
            "queue_policy": queue_policy,
            "account": queue.account,
            "partition": queue.partition,
            "node": (
                {"mode": "manual", "name": node}
                if node is not None
                else {"mode": "auto"}
            ),
            "gpu": {
                "mode": "explicit",
                "count": minimum.gpu_count,
                "type": gpu_type,
            },
            "cpus_per_task": cpus_per_task,
            "memory_gb": memory_gb,
            "time_limit": time_limit,
        }
    )
    resources = resolve_evaluation_resources(resources, {}, runtime_profile_id=profile_id)
    gpu_alias = CLUSTER.gpu_aliases.get(resources.gpu.gpu_type)
    if not gpu_alias:
        raise ValueError(f"runtime readiness requires a concrete configured GPU type: {gpu_type}")

    driver_source = Path(__file__).read_text(encoding="utf-8")
    encoded_driver = base64.b64encode(driver_source.encode("utf-8")).decode("ascii")
    encoded_hook = base64.b64encode(capsule_source.encode("utf-8")).decode("ascii")
    encoded_contract = base64.b64encode(
        json.dumps(contract, sort_keys=True, indent=2).encode("utf-8")
    ).decode("ascii")
    runtime = contract["runtime"]
    attestation_path = contract["verification"]["compute_attestation_path"]
    job_name = f"skynet-ready-{re.sub(r'[^a-z0-9-]+', '-', profile_id.lower())[:80]}"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={resources.partition}",
        f"#SBATCH --account={resources.account}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={resources.cpus_per_task}",
        f"#SBATCH --mem={resources.memory_gb}G",
        f"#SBATCH --gres=gpu:{gpu_alias}:{resources.gpu.count}",
        f"#SBATCH --time={resources.time_limit}",
        f"#SBATCH --output={CLUSTER.paths.logs}/%x-%j.out",
        f"#SBATCH --error={CLUSTER.paths.logs}/%x-%j.err",
        # Preserve Slurm/SPANK-provided CUDA visibility. The Python producer
        # explicitly filters the environment passed to the evaluator hook.
        "#SBATCH --export=ALL",
        *compile_slurm_placement_directives(resources),
        "set -euo pipefail",
        *(['if test "${OMNI_KIT_ACCEPT_EULA:-}" != "YES"; then',
        "  printf '%s\n' 'Read and accept the NVIDIA Isaac Sim EULA, then explicitly export OMNI_KIT_ACCEPT_EULA=YES for this sbatch submission.' >&2",
        "  exit 2",
        "fi"] if requires_isaac_consent(contract) else []),
        'probe_root="${SLURM_TMPDIR:-/tmp}/skynet-runtime-readiness-${SLURM_JOB_ID:-manual}"',
        'mkdir -p "$probe_root"',
        'mkdir -p "$probe_root/tmp"',
        'export TMPDIR="$probe_root/tmp"',
        f"printf '%s' {shlex.quote(encoded_driver)} | base64 --decode > \"$probe_root/runtime_readiness.py\"",
        f"printf '%s' {shlex.quote(encoded_hook)} | base64 --decode > \"$probe_root/evaluator-hook.py\"",
        f"printf '%s' {shlex.quote(encoded_contract)} | base64 --decode > \"$probe_root/contract.json\"",
        *_render_environment_exports(contract["profile_snapshot"]),
        f"export SKYNET_RUNTIME_READINESS_GPU_TYPE={shlex.quote(gpu_type)}",
        (
            f"{shlex.quote(str(runtime['python_executable']))} "
            '"$probe_root/runtime_readiness.py" execute '
            '--contract "$probe_root/contract.json" '
            '--hook "$probe_root/evaluator-hook.py" '
            f"--attestation {shlex.quote(str(attestation_path))}"
        ),
    ]
    return "\n".join(lines) + "\n"


def submit_readiness_sbatch(
    profile_id: str,
    suite_id: str,
    *,
    run_id: str,
    gateway: str = "auto",
    queue_policy: str | None = None,
    gpu_type: str | None = None,
    cpus_per_task: int | None = None,
    memory_gb: int | None = None,
    time_limit: str | None = None,
    node: str | None = None,
    cluster: Any | None = None,
) -> Any:
    from .cluster_runtime import ClusterClient, approved_operator_environment

    script = render_readiness_sbatch(
        profile_id,
        suite_id,
        queue_policy=queue_policy,
        gpu_type=gpu_type,
        cpus_per_task=cpus_per_task,
        memory_gb=memory_gb,
        time_limit=time_limit,
        node=node,
    )
    client = cluster or ClusterClient()
    validated_gateway, _ = client.test_script(script, gateway)
    return client.submit_script(
        script,
        run_id,
        validated_gateway,
        submission_key=run_id,
        forwarded_environment=approved_operator_environment(),
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _merge_actual(target: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _merge_actual(target[key], value)
        else:
            target[key] = value


def _hook_subprocess_environment(
    contract: Mapping[str, Any], hook: Mapping[str, Any], tokens: Mapping[str, str]
) -> tuple[dict[str, str], list[str]]:
    """Pass scheduler/runtime state to the hook without leaking application state."""

    exact_names = {
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "LOGNAME",
        "OMNI_KIT_ACCEPT_EULA",
        "PATH",
        "SHELL",
        "TERM",
        "TMPDIR",
        "USER",
    }
    prefixes = ("CUDA_", "NVIDIA_", "SLURM_")
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in exact_names or name.startswith(prefixes)
    }
    profile_snapshot = contract.get("profile_snapshot") or {}
    declared_names = set((profile_snapshot.get("environment") or {}).keys())
    declared_names.update(
        str(item.get("name") or "")
        for item in profile_snapshot.get("environment_operations") or []
        if isinstance(item, Mapping)
    )
    for name in declared_names:
        if name and name != "OMNI_KIT_ACCEPT_EULA" and name in os.environ:
            environment[name] = os.environ[name]

    errors: list[str] = []
    for name, value in hook.get("environment", {}).items():
        if name == "OMNI_KIT_ACCEPT_EULA":
            errors.append("readiness hook configuration must not encode EULA acceptance")
            continue
        environment[name] = _resolve_template(str(value), tokens)
    return environment, errors


def _run_streaming_subprocess(
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
    timeout_seconds: int,
    stdout_stream: Any | None = None,
    stderr_stream: Any | None = None,
) -> dict[str, Any]:
    """Tee hook output live while retaining bounded tails for attestation."""

    process = subprocess.Popen(
        list(argv),
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    sinks = {
        "stdout": stdout_stream or sys.stdout,
        "stderr": stderr_stream or sys.stderr,
    }
    tails = {"stdout": bytearray(), "stderr": bytearray()}

    def pump(name: str, pipe: Any) -> None:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            tail = tails[name]
            tail.extend(chunk)
            if len(tail) > 20000:
                del tail[:-20000]
            sinks[name].write(chunk.decode("utf-8", errors="replace"))
            sinks[name].flush()

    threads = [
        threading.Thread(target=pump, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=pump, args=("stderr", process.stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        process.wait()
        returncode = None
    finally:
        for thread in threads:
            thread.join(timeout=5)
    return {
        "returncode": returncode,
        "stdout": tails["stdout"].decode("utf-8", errors="replace"),
        "stderr": tails["stderr"].decode("utf-8", errors="replace"),
        "timed_out": timed_out,
    }


def execute_compute_probe(contract_path: Path, hook_path: Path, attestation_path: Path) -> int:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("unsupported runtime readiness contract")
    verification = contract["verification"]
    runtime = contract["runtime"]
    errors: list[str] = []
    actual: dict[str, Any] = {
        "python": platform.python_version(),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "distributions": {},
        "imports": {},
        "executables": {},
        "executable_providers": {},
        "shared_libraries": {},
        "gym_registrations": {},
        "source_prerequisites": [],
        "platform": {
            "system": platform.system(),
            "libc": platform.libc_ver()[0],
            "libc_version": platform.libc_ver()[1],
        },
        "gpu": {"cuda_available": False, "device_count": 0, "devices": []},
        "checks": {},
        "pip_check": {"ok": False, "output": ""},
    }

    expected_prefix = str(Path(runtime["environment_path"]).resolve())
    if actual["python_prefix"] != expected_prefix:
        errors.append(
            f"compute probe Python prefix {actual['python_prefix']} does not match {expected_prefix}"
        )
    for distribution, wanted in verification.get("distributions", {}).items():
        try:
            found = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            found = None
        actual["distributions"][distribution] = found
        if found is None:
            errors.append(f"distribution {distribution} is not installed")
        elif not (found == wanted or found.startswith(str(wanted) + ".")):
            errors.append(f"distribution {distribution} {found} does not match {wanted}")
    for distribution in verification.get("required_distributions", []):
        try:
            found = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            found = None
        actual["distributions"][distribution] = found
        if found is None:
            errors.append(f"required distribution {distribution} is not installed")
    if verification.get("pip_check"):
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        actual["pip_check"] = {
            "ok": completed.returncode == 0,
            "output": completed.stdout[-20000:],
        }
        if completed.returncode != 0:
            errors.append(
                "pip dependency consistency check failed: "
                + (completed.stdout.strip() or f"exit code {completed.returncode}")
            )

    executable_path = os.pathsep.join(
        [str(Path(sys.executable).resolve().parent), os.environ.get("PATH", "")]
    )
    for executable in verification.get("executables", []):
        found = shutil.which(executable, path=executable_path)
        actual["executables"][executable] = found
        if found is None:
            errors.append(f"required executable {executable} was not found")
    for provider in verification.get("executable_providers", []):
        name = provider["name"]
        record = {"path": None}
        actual["executable_providers"][name] = record
        try:
            module = importlib.import_module(provider["module"])
            resolver = getattr(module, provider["resolver"])
            resolved = Path(str(resolver())).expanduser().resolve()
            record["path"] = str(resolved)
            if not resolved.is_file() or not os.access(resolved, os.X_OK):
                errors.append(
                    f"executable provider {name} returned a missing/non-executable path: {resolved}"
                )
        except Exception as error:
            errors.append(
                f"executable provider {name} failed: {type(error).__name__}: {error}"
            )
    for library in verification.get("shared_libraries", []):
        try:
            ctypes.CDLL(library)
            actual["shared_libraries"][library] = True
        except OSError as error:
            actual["shared_libraries"][library] = False
            errors.append(f"required shared library {library} failed to load: {error}")

    for prerequisite in contract["profile_snapshot"].get("source_prerequisites", []):
        path = Path(prerequisite["path"])
        record = {"name": prerequisite["name"], "path": str(path), "present": path.is_dir()}
        if prerequisite.get("kind") == "git_checkout" and path.is_dir():
            try:
                head = subprocess.check_output(
                    ["git", "-C", str(path), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.STDOUT,
                ).strip()
                wanted = subprocess.check_output(
                    ["git", "-C", str(path), "rev-parse", f"{prerequisite['revision']}^{{commit}}"],
                    text=True,
                    stderr=subprocess.STDOUT,
                ).strip()
                record.update({"head": head, "expected_commit": wanted})
                if head != wanted:
                    errors.append(
                        f"{prerequisite['name']} checkout does not match {prerequisite['revision']}"
                    )
            except (OSError, subprocess.CalledProcessError) as error:
                record["probe_error"] = str(error)
                errors.append(f"could not verify {prerequisite['name']} Git revision")
        if prerequisite.get("required", True) and not path.is_dir():
            errors.append(f"required source prerequisite is missing: {path}")
        actual["source_prerequisites"].append(record)

    if not str(os.environ.get("SLURM_JOB_ID") or "").isdigit():
        errors.append("compute readiness must run inside a Slurm allocation")
    if requires_isaac_consent(contract) and os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES":
        errors.append(
            "NVIDIA Isaac Sim EULA acceptance is missing; explicitly provide "
            "OMNI_KIT_ACCEPT_EULA=YES after reviewing and accepting the EULA"
        )

    hook = contract["hook"]
    if hashlib.sha256(hook_path.read_bytes()).hexdigest() != hook["capsule_sha256"]:
        errors.append("staged evaluator readiness hook does not match its pinned SHA-256")
    hook_result_path = hook_path.parent / "evaluator-hook-result.json"
    tokens = {
        "runtime.python_executable": str(runtime["python_executable"]),
        "runtime.environment_path": str(runtime["environment_path"]),
        "runtime.source_dir": str(runtime["source_dir"]),
        "smoke.asset_path": str(hook_path),
        "smoke.result_path": str(hook_result_path),
    }
    hook_environment, hook_environment_errors = _hook_subprocess_environment(
        contract, hook, tokens
    )
    hook_environment.setdefault("PYTHONUNBUFFERED", "1")
    errors.extend(hook_environment_errors)
    hook_integrity_valid = (
        hashlib.sha256(hook_path.read_bytes()).hexdigest() == hook["capsule_sha256"]
    )
    operator_authorized = not requires_isaac_consent(contract) or os.environ.get("OMNI_KIT_ACCEPT_EULA") == "YES"
    actual["hook"] = {
        "attempted": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
    }
    # Independent preflight failures must not suppress adapter-owned GPU, import,
    # task-registration, lifecycle, and media evidence. Legal consent and capsule
    # integrity are the only hard prerequisites for executing the pinned hook.
    if hook_integrity_valid and operator_authorized and not hook_environment_errors:
        try:
            argv = [_resolve_template(str(item), tokens) for item in hook["argv"]]
            actual["hook"]["attempted"] = True
            streamed = _run_streaming_subprocess(
                argv,
                environment=hook_environment,
                timeout_seconds=int(hook["timeout_seconds"]),
            )
            actual["hook"].update(streamed)
            if streamed["timed_out"]:
                errors.append(
                    f"evaluator readiness hook timed out after {hook['timeout_seconds']}s"
                )
            elif streamed["returncode"] != 0:
                errors.append(
                    f"evaluator readiness hook exited with code {streamed['returncode']}"
                )
        except (OSError, ValueError) as error:
            errors.append(f"evaluator readiness hook failed: {type(error).__name__}: {error}")
    if hook_result_path.is_file():
        try:
            hook_result = json.loads(hook_result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"evaluator readiness hook result is invalid: {error}")
        else:
            if hook_result.get("schema_version") != hook["result_schema"]:
                errors.append("evaluator readiness hook returned the wrong schema")
            if hook_result.get("ready") is not True:
                errors.append("evaluator readiness hook did not declare ready=true")
            errors.extend(str(item) for item in hook_result.get("errors") or [])
            hook_actual = hook_result.get("actual")
            if isinstance(hook_actual, Mapping):
                _merge_actual(actual, hook_actual)
    else:
        errors.append("evaluator readiness hook did not write its declared result")

    found_python = str(actual.get("python") or "")
    wanted_python = str(verification.get("python_version") or "")
    if wanted_python and not (
        found_python == wanted_python or found_python.startswith(wanted_python + ".")
    ):
        errors.append(
            f"Python {found_python or 'missing'} does not match required {wanted_python}"
        )
    wanted_system = str(verification.get("platform_system") or "")
    if wanted_system and actual["platform"].get("system") != wanted_system:
        errors.append(
            f"compute operating system {actual['platform'].get('system')} does not match "
            f"{wanted_system}"
        )
    wanted_glibc = str(verification.get("glibc_minimum") or "")
    found_glibc = str(actual["platform"].get("libc_version") or "")
    numeric = lambda value: tuple(int(part) for part in re.findall(r"\d+", value))
    if wanted_glibc and (
        actual["platform"].get("libc") != "glibc"
        or not found_glibc
        or numeric(found_glibc) < numeric(wanted_glibc)
    ):
        errors.append(
            f"compute glibc {found_glibc or 'missing'} does not satisfy minimum "
            f"{wanted_glibc}"
        )
    for module in verification.get("python_imports", []):
        if actual["imports"].get(module) is not True:
            errors.append(f"required Python import was not verified: {module}")
    for registration in verification.get("gym_registrations", []):
        record = actual["gym_registrations"].get(registration["module"], {})
        if record.get("module_imported") is not True:
            errors.append(
                f"Gym registration module was not verified: {registration['module']}"
            )
        for task_id in registration.get("ids", []):
            if record.get("ids", {}).get(task_id) is not True:
                errors.append(f"required Gym task was not registered: {task_id}")
    if verification.get("requires_gpu") and (
        actual["gpu"].get("cuda_available") is not True
        or int(actual["gpu"].get("device_count") or 0) < 1
    ):
        errors.append("an allocated CUDA GPU was not verified")
    for check in (verification.get("compute_smoke") or {}).get(
        "required_checks", []
    ):
        if actual["checks"].get(check) is not True:
            errors.append(f"required evaluator smoke check did not pass: {check}")

    errors = list(dict.fromkeys(errors))
    attestation = {
        "schema_version": ATTESTATION_SCHEMA,
        "profile_id": contract["profile_id"],
        "profile_snapshot_sha256": contract["profile_snapshot_sha256"],
        "suite_contract_sha256": contract["suite_contract_sha256"],
        "ready": not errors,
        "errors": errors,
        "execution": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME") or platform.node(),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "gpu_type": os.environ.get("SKYNET_RUNTIME_READINESS_GPU_TYPE"),
        },
        "actual": actual,
    }
    _atomic_write_json(attestation_path, attestation)
    if errors:
        for error in errors:
            print(f"runtime readiness failed: {error}", file=sys.stderr)
        print(f"runtime readiness evidence: {attestation_path}", file=sys.stderr)
    else:
        print(f"runtime readiness verified: {attestation_path}")
    return 0 if not errors else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render or execute runtime readiness probes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render-sbatch")
    render.add_argument("--profile", required=True)
    render.add_argument("--suite", required=True)
    render.add_argument("--queue-policy")
    render.add_argument("--gpu-type")
    render.add_argument("--cpus-per-task", type=int)
    render.add_argument("--memory-gb", type=int)
    render.add_argument("--time-limit")
    render.add_argument(
        "--node",
        help="request one concrete Slurm node (default: automatic placement)",
    )
    render.add_argument("--output", type=Path)
    submit = subparsers.add_parser("submit-sbatch")
    submit.add_argument("--profile", required=True)
    submit.add_argument("--suite", required=True)
    submit.add_argument("--queue-policy")
    submit.add_argument("--gpu-type")
    submit.add_argument("--cpus-per-task", type=int)
    submit.add_argument("--memory-gb", type=int)
    submit.add_argument("--time-limit")
    submit.add_argument(
        "--node",
        help="request one concrete Slurm node (default: automatic placement)",
    )
    submit.add_argument("--gateway", default="auto")
    submit.add_argument(
        "--run-id",
        required=True,
        help="stable readiness submission ID; reuse it to recover without duplicating",
    )
    execute = subparsers.add_parser("execute")
    execute.add_argument("--contract", type=Path, required=True)
    execute.add_argument("--hook", type=Path, required=True)
    execute.add_argument("--attestation", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "submit-sbatch":
        submission = submit_readiness_sbatch(
            args.profile,
            args.suite,
            run_id=args.run_id,
            gateway=args.gateway,
            queue_policy=args.queue_policy,
            gpu_type=args.gpu_type,
            cpus_per_task=args.cpus_per_task,
            memory_gb=args.memory_gb,
            time_limit=args.time_limit,
            node=args.node,
        )
        print(submission.job_id)
        return 0
    if args.command == "render-sbatch":
        script = render_readiness_sbatch(
            args.profile,
            args.suite,
            queue_policy=args.queue_policy,
            gpu_type=args.gpu_type,
            cpus_per_task=args.cpus_per_task,
            memory_gb=args.memory_gb,
            time_limit=args.time_limit,
            node=args.node,
        )
        if args.output:
            args.output.write_text(script, encoding="utf-8")
        else:
            sys.stdout.write(script)
        return 0
    return execute_compute_probe(args.contract, args.hook, args.attestation)


if __name__ == "__main__":
    raise SystemExit(main())
