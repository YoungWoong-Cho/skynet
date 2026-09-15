"""Pinned upstream XPolicyLab launchers, separate from recorded-policy bridges."""

import json

from .xpolicy_manifest import progress_contract
from pathlib import Path

from skynet_app.dataset_formats import RECIPES, XPL_COMMIT, XPL_REPOSITORY
from skynet_app.training_contracts import DatasetRequirement


def catalog():
    return json.loads(Path(__file__).with_name("xpolicy_native_catalog.json").read_text())


def act_support_files():
    root = Path(__file__).parent
    return {"adapter-support/" + name: (root / name).read_text() for name in (
        "act_native_data.py", "act_native_checkpoint.py", "act_native_evaluation.py",
    )}


def act_progress():
    progress = progress_contract()
    metrics = {**progress.source.metrics, **{key: key for key in (
        "train/l1", "train/kl", "validation/l1", "validation/kl",
    )}}
    return progress.model_copy(update={"source": progress.source.model_copy(update={"metrics": metrics})})


def act_evaluation():
    from .xpolicy_manifest import evaluation
    entry = evaluation("act")
    files = {**entry.command.capsule_files, **act_support_files()}
    return entry.model_copy(update={"command": entry.command.model_copy(update={"capsule_files": files})})


def manifests():
    from . import (
        AdapterCapabilities, AdapterCheckpointDefaults, AdapterDefaults, AdapterHyperparameterDefaults,
        AdapterInputField, AdapterManifest, AdapterPrerequisite,
        AdapterResourceDefaults, AdapterRuntimePolicy, CommandTemplate,
        DataBundleInputBinding,
    )

    root = Path(__file__).parent
    records = catalog()
    if records["revision"] != XPL_COMMIT:
        raise ValueError("XPolicyLab launcher catalog must match the pinned source revision")
    capsules = {
        "adapter-support/xpolicy_native.py": (root / "xpolicy_native.py").read_text(),
        "adapter-support/xpolicy_native_catalog.json": (root / "xpolicy_native_catalog.json").read_text(),
        "adapter-support/artifacts.py": (root.parents[1] / "ops/datasets/artifacts.py").read_text(),
    }
    result = []
    for record in records["policies"]:
        name, slug = record["policy"], record["slug"]
        data_format = f"xpolicylab-native-{name.lower()}/v1"
        formats = [data_format]
        contracts = ["skynet.xpolicylab-native/v1"]
        if name == "ACT":
            formats.append(RECIPES["act"]["format"])
            contracts.append(RECIPES["act"]["contract"])
        fields = [AdapterInputField(
            path="native.config." + key, label=label, kind="string", required=True,
            data_binding=DataBundleInputBinding(
                role="training_data", formats=formats,
                contracts=contracts, value_path=value,
            ),
            help="Policy-specific native data and configuration, prepared using the pinned XPolicyLab recipe.",
        ) for key, label, value in [
            ("dataset_path", "Native training data", "location.path"),
            ("dataset_manifest_sha256", "Dataset fingerprint", "version.manifest_sha256"),
        ]]
        if name == "ACT":
            fields.append(AdapterInputField(path="native.config.epochs", label="Epochs",
                kind="integer", default=6000, minimum=1, maximum=100000,
                canonical_path="train.max_epochs"))
        # EventVLA/Hy-VLA have no seed argument in their upstream shell API.
        seed_supported = record["entry_kind"] == "standard"
        argv = [
            "python", "{{tokens.run_dir}}/adapter-support/xpolicy_native.py",
            "--repository", "{{tokens.source_dir}}", "--revision", XPL_COMMIT,
            "--policy", name, "--dataset", "{{native.config.dataset_path}}",
            "--manifest-sha", "{{native.config.dataset_manifest_sha256}}",
            "--output", "{{tokens.run_dir}}/artifacts",
            "--gpu-count", "{{computed.gpu_count}}",
        ]
        if seed_supported:
            argv += ["--seed", "{{train.seed}}"]
        if name == "ACT":
            argv += ["--epochs", "{{native.config.epochs}}"]
        result.append(AdapterManifest(
            slug=slug, display_name=f"XPolicyLab · {name} · Native",
            description=f"Original policy/{name}/train.sh at {XPL_COMMIT[:12]}. {record['notes']}",
            default_repository=XPL_REPOSITORY,
            # Do not claim the whole repository: choosing a policy must be explicit.
            repository_patterns=[],
            runtime=AdapterRuntimePolicy(allowed_backends={"existing", "conda"}, recommended_backend="existing"),
            capabilities=AdapterCapabilities(
                name=slug, runtime_backends={"existing", "conda"},
                minimum_gpus=record["minimum_gpus"],
                recommended_gpus=record["minimum_gpus"],
                maximum_gpus=record["maximum_gpus"],
                supports_multi_gpu_single_node=record["maximum_gpus"] > 1,
                supports_resume=name == "ACT",
            ),
            defaults=AdapterDefaults(
                resources=AdapterResourceDefaults(gpu_count=record["minimum_gpus"], gpu_mode="explicit"),
                hyperparameters=AdapterHyperparameterDefaults(seed=42 if seed_supported else None),
                checkpoint=AdapterCheckpointDefaults(auto_resume=name == "ACT", max_attempts=3 if name == "ACT" else 1, final_selector="latest"),
            ),
            prerequisites=[
                AdapterPrerequisite(id="native-inputs", kind="asset", name=f"{name} native training inputs",
                    description=record["notes"], source_repository=f"{XPL_REPOSITORY}/tree/{XPL_COMMIT}/policy/{name}"),
                AdapterPrerequisite(id="native-runtime", kind="framework", name=f"{name} installed runtime",
                    description="Install this policy's own dependencies and required pretrained weights before submission. Source inspection and launcher tests do not establish GPU training compatibility."),
            ],
            train=CommandTemplate(
                argv=argv, input_fields=fields, capsule_files={**capsules, **(act_support_files() if name == "ACT" else {})},
                resume_argv=["--resume", "{{tokens.resume_checkpoint}}"] if name == "ACT" else [],
                progress=act_progress() if name == "ACT" else None,
                strict_native_config=True, strict_canonical_inputs=True,
                supported_canonical_fields=["train.seed"] if seed_supported else [],
                data_requirements=DatasetRequirement(
                    description=("ACT RGB/joint HDF5, including converted Skynet recordings. Original loader uses at least two episodes, its own 80/20 split and normalization over all episodes."
                        if name == "ACT" else f"{name} native prepared inputs ({data_format}). Skynet ACT/DP/EgoVerse exports are not this format."),
                    observations=["policy_specific"], action_representation="policy_specific",
                ),
                checkpoint_globs=["artifacts/checkpoints/last.ckpt"] if name == "ACT" else [],
            ),
            evaluations=[act_evaluation()] if name == "ACT" else [],
            warnings=[
                ("Accepts ACT HDF5 from Recordings. The upstream loader uses its own 80/20 split, statistics over all episodes, and previous-step action alignment; dataset split/normalization files are not used."
                 if name == "ACT" else "Requires this policy's native prepared dataset, weights and Runtime; current Shadow conversion outputs are not automatically compatible."),
                ("Original ACT defaults are preserved. Epochs can be configured; checkpoint hooks save optimizer, RNG, split and normalization for automatic recovery. Recorded RGB/joint inputs support Isaac Lab evaluation."
                 if name == "ACT" else "Hyperparameters come from the upstream recipe and registered native configuration. Automatic resume and simulator evaluation are not implemented for this native adapter."),
                ("ACT recording inputs passed a one-epoch L40S training and checkpoint-save check. This does not establish full training quality." if name == "ACT" else "GPU training has not been validated for this adapter. Native logs and outputs are retained under the run's artifacts/native-workspace."),
            ],
        ))
    return result
