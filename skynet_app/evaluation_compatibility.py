"""Compose a policy loader and a suite executor without tying suites to training.

Planning reads immutable metadata only. Compute workers recheck the same I/O
contract against real model outputs and simulator observations before scoring.
Native integrations remain available through their existing versioned commands.
"""

from copy import deepcopy
from pathlib import Path

from .adapters.policy_contract import contract_issues, recorded_contract
from .evaluation_contracts import bind_suite_to_dataset

RECORDED_SUITES = frozenset({("isaac_lab", "dexverse_recorded"), ("isaac_lab", "dexverse_training_episode")})
RECORDED_POLICY_MODELS = {
    "egoverse-hpt": ("egoverse_joints", {"hpt_joints"}),
    "egoverse-hpt-joints": ("egoverse_joints", {"hpt_joints"}),
    "egoverse-act": ("egoverse_joints", {"act"}),
    "xpolicylab-act": ("xpolicy_joints", None),
    "xpolicylab-act-native": ("xpolicy_joints", None),
    "xpolicylab-dp": ("xpolicy_joints", None),
}
STATUS_LABELS = {"compatible": "Compatible", "mapping_required": "Mapping required", "unknown": "Missing information", "incompatible": "Incompatible"}


def policy_loader(spec):
    source = spec.get("source") or {}
    slug = source.get("adapter")
    config = (spec.get("native") or {}).get("config") or {}
    selected = RECORDED_POLICY_MODELS.get(slug)
    if selected and (selected[1] is None or config.get("model_preset") in selected[1]):
        return selected[0]
    return None


def supports_recorded_suite(manifest, suite):
    return manifest.get("slug") in RECORDED_POLICY_MODELS and (suite["evaluator_adapter"], suite["name"]) in RECORDED_SUITES


def dataset_metadata(spec):
    assignments = ((spec.get("data") or {}).get("bundle") or {}).get("assignments") or []
    candidates = [a for a in assignments if a.get("role") == "training_data"]
    return candidates[0].get("version", {}).get("metadata", {}) if len(candidates) == 1 else {}


def declaration_matches(entry, spec, suite):
    if entry.environment != suite["evaluator_adapter"] or (entry.suites and suite["name"] not in entry.suites):
        return False
    for path, choices in entry.enabled_when.items():
        value = spec
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value not in choices:
            return False
    return entry.command is not None


def inspect_compatibility(spec, manifest, suite, checkpoint=None):
    """The same report gates the picker, target validation and submission."""
    checks = []
    loader = policy_loader(spec)
    recorded = (suite["evaluator_adapter"], suite["name"]) in RECORDED_SUITES
    native = [e for e in manifest.evaluations if declaration_matches(e, spec, suite)]
    binding = None
    bound = deepcopy(suite)

    def check(field, message, status="compatible"):
        checks.append(dict(field=field, message=message, status=status))

    if recorded:
        if not loader:
            check("loader", "No recorded-joint policy loader is registered for this model preset.", "mapping_required")
        else:
            check("loader", f"Policy loader: {loader}.")
            metadata = dataset_metadata(spec)
            config = (spec.get("native") or {}).get("config") or {}
            binding = recorded_contract(metadata, images=not (loader == "xpolicy_joints" and config.get("observation_mode") == "state"))
            checks.extend(contract_issues(binding))
            if spec.get("source", {}).get("adapter") == "xpolicylab-act-native" and metadata.get("contract") != "skynet.act-rgb-joints/v1":
                check("dataset", "Native ACT simulator evaluation requires a recorded RGB/joint dataset.", "incompatible")
            if loader == "egoverse_joints" and metadata.get("contract") != "skynet.egoverse-rgb-joints/v1":
                check("dataset", "The EgoVerse loader requires a recorded RGB/joint dataset.", "incompatible")
            if not config.get("dataset_path") or not config.get("dataset_manifest_sha256"):
                check("dataset", "The training dataset path or checksum is missing.", "unknown")
            if not checks[1:]:
                check("io", "Named joints, action scaling, control period and camera inputs are recorded.")
    elif len(native) == 1:
        check("loader", "This suite has an existing native model/evaluator integration.")
    elif len(native) > 1:
        check("executor", "Multiple native evaluator commands match this suite.", "incompatible")
    else:
        check("executor", "No model loader and observation/action bridge is registered for this suite and model combination.", "mapping_required")

    try:
        bound = bind_suite_to_dataset(suite, spec)
    except ValueError as error:
        check("dataset", str(error), "incompatible")
    if checkpoint is not None:
        if checkpoint.get("status", "AVAILABLE") != "AVAILABLE" or checkpoint.get("pruned_at"):
            check("checkpoint", "The selected checkpoint is not available.", "incompatible")
        elif not checkpoint.get("sha256"):
            check("checkpoint", "The selected checkpoint has no verified checksum.", "unknown")
        else:
            check("checkpoint", "A registered checkpoint checksum is available; model loading is checked on the compute node.")
    status = next((state for state in ("incompatible", "mapping_required", "unknown") if any(c["status"] == state for c in checks)), "compatible")
    return {
        "schema_version": "skynet.evaluation-compatibility/v1",
        "status": status, "label": STATUS_LABELS[status], "checks": checks,
        "ready": status == "compatible",
        "messages": [c["message"] for c in checks if c["status"] != "compatible"],
        "policy_loader": loader if recorded else None,
        "executor": "recorded_simulator" if recorded else "native",
        "runtime_verification": "pending", "io_contract": binding,
    }, bound


def compose_evaluator(spec, manifest, suite):
    """Versioned bridge code is frozen in the evaluation capsule, never the run."""
    loader = policy_loader(spec)
    if not loader or (suite["evaluator_adapter"], suite["name"]) not in RECORDED_SUITES:
        return manifest
    from .adapters import CommandTemplate, EvaluationAdapterMetadata
    from .adapters.xpolicy_manifest import simulation_support_files, support_files
    if loader == "egoverse_joints":
        from .adapters.egoverse_manifest import support_files as egoverse_support
        files = egoverse_support()
    else:
        files = support_files("act" if spec["source"]["adapter"] in {"xpolicylab-act", "xpolicylab-act-native"} else "dp")
    if spec["source"]["adapter"] == "xpolicylab-act-native":
        from .adapters.xpolicy_native_manifest import act_support_files
        files.update(act_support_files())
    files.update(simulation_support_files())
    directory = Path(__file__).with_name("adapters")
    for name in ("recorded_policy_evaluation.py", "policy_loading.py", "policy_contract.py", "egoverse_simulation.py"):
        files["adapter-support/" + name] = (directory / name).read_text()
    entry = EvaluationAdapterMetadata(
        environment=suite["evaluator_adapter"], suites=[suite["name"]], maximum_parallelism=8,
        runtime_profile_id="isaacsim-5.1.0_isaaclab-2.3.2_py311",
        command=CommandTemplate(
            argv=["python", "{{tokens.run_dir}}/adapter-support/evaluation_workers.py", "--context", "{{tokens.run_dir}}/adapter-support/evaluation-context.json", "--source-dir", "{{tokens.source_dir}}"],
            capsule_files=files, environment={"SKYNET_EVAL_RESUME_GRANULARITY": "episode"},
            required_values=["evaluation.checkpoint.sha256", "evaluation.policy.native_config.dataset_path", "evaluation.policy.native_config.dataset_manifest_sha256", "evaluation.evaluator_runtime.python_executable", "evaluation.evaluator_runtime.source_dir"],
        ),
    )
    return manifest.model_copy(update={"evaluations": [entry]})
