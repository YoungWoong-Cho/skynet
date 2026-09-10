"""One declaration factory for EgoVerse model adapters."""

from skynet_app.model_io import adapter_io_contract

from pathlib import Path

REPOSITORY = "https://github.com/GaTech-RL2/EgoVerse"
REVISION = "e17cf98fe4bc234c564b37abc9e155f25e76d566"
RUNTIME = "egoverse-native"
FORMAT = "egoverse-episodes-zarr/v1"
CONTRACT = "skynet.egoverse-rgb-joints/v1"
MODEL_LEARNING_RATES = {
    "act": 5e-5,
    "hpt_bc_keypoints_base": 5e-5,
    "hpt_cotrain_enc_dec_base": 5e-5,
    "hpt_bc_flow_aria": 3e-4,
    "hpt_bc_flow_human": 3e-4,
    "hpt_bc_flow_mecka": 3e-4,
    "hpt_bc_flow_scale": 3e-4,
    "hpt_cotrain_flow_seperate_head": 3e-4,
    "hpt_bc_pickplace_qwen_pertoken": 2e-4,
    "hpt_bc_pickplace_qwen_pooled": 2e-4,
}

# The incomplete PI base and the EgoBridge alias are not duplicate models.
MODELS = {
    "act": ("ACT", True),
    "hpt_joints": ("HPT flow · recorded joints", True),
    "dp_joints": ("Diffusion Policy · recorded joints", True),
    **{
        name: (label, False)
        for name, label in [
            ("hpt_bc_flow_aria", "HPT flow · Aria"),
            ("hpt_bc_flow_eva", "HPT flow · EVA"),
            ("hpt_bc_flow_human", "HPT flow · human"),
            ("hpt_bc_flow_mecka", "HPT flow · Mecka"),
            ("hpt_bc_flow_scale", "HPT flow · Scale"),
            ("hpt_bc_keypoints_base", "HPT flow · human keypoints"),
            ("hpt_cotrain_enc_dec_base", "HPT co-training · encoder-decoder"),
            ("hpt_bc_pickplace_qwen_pertoken", "HPT · Qwen per token"),
            ("hpt_bc_pickplace_qwen_pooled", "HPT · Qwen pooled"),
            ("hpt_cotrain_flow_seperate_head", "HPT co-training · separate heads"),
            ("hpt_cotrain_flow_shared_head", "HPT co-training · shared head"),
            ("hpt_cotrain_mecka_flow_shared_head", "HPT co-training · Mecka"),
            ("hpt_cotrain_scale_flow_shared_head", "HPT co-training · Scale"),
            ("pi0.5_bc_aria", "π0.5 · Aria"),
            ("pi0.5_bc_eva", "π0.5 · EVA"),
            ("pi0.5_bc_mecka", "π0.5 · Mecka"),
            ("pi0.5_bc_scale", "π0.5 · Scale"),
            ("pi0.5_cotrain_eva_aria", "π0.5 co-training · EVA / Aria"),
            ("pi0.5_cotrain_mecka_scale", "π0.5 co-training · Mecka / Scale"),
        ]
    },
}


def support_files():
    root = Path(__file__).parent
    return {
        "adapter-support/" + name: (root / name).read_text()
        for name in (
            "egoverse_runtime.py",
            "egoverse_evaluation.py",
            "egoverse_data.py",
            "egoverse_readiness.py",
        )
    } | {
        "adapter-support/artifacts.py": (
            root.parents[1] / "ops/datasets/artifacts.py"
        ).read_text(),
    }


def manifests():
    from . import (
        AdapterManifest,
        AdapterCapabilities,
        AdapterRuntimePolicy,
        AdapterDefaults,
        AdapterHyperparameterDefaults,
        AdapterResourceDefaults,
        AdapterInputField,
        DataBundleInputBinding,
        CommandTemplate,
        ArgumentBinding,
        TrainingProgressContract,
        TrainingProgressJsonlSource,
        AdapterBatchCompatibility,
        EvaluationAdapterMetadata,
    )
    from skynet_app.training_contracts import DatasetRequirement

    for model, (label, recorded) in MODELS.items():
        slug = "egoverse-" + model.replace("_", "-").replace(".", "")
        lr = MODEL_LEARNING_RATES.get(model, 3e-5 if model.startswith("pi") else 1e-4)
        fields = [
            AdapterInputField(
                path="native.config." + key,
                label=label,
                kind="string",
                required=True,
                data_binding=DataBundleInputBinding(
                    role="training_data",
                    formats=[FORMAT],
                    contracts=(
                        [CONTRACT] if recorded else [f"egoverse.native-{model}/v1"]
                    ),
                    value_path=value,
                ),
            )
            for key, label, value in [
                ("dataset_path", "Prepared dataset", "location.path"),
                (
                    "dataset_manifest_sha256",
                    "Dataset fingerprint",
                    "version.manifest_sha256",
                ),
            ]
        ]
        settings = [
            ("epochs", "Epochs", 2000, 1),
            ("validation_every", "Validate every epochs", 200, 1),
            ("train_batches", "Training batches per epoch", 100, 1),
            ("validation_batches", "Validation batches", 80, 1),
        ]
        fields.extend(
            AdapterInputField(
                path="native.config." + key,
                label=label,
                kind="integer",
                default=value,
                minimum=minimum,
                help={
                    "validation_every": "Epochs between validation runs.",
                    "train_batches": "Maximum training batches per epoch.",
                    "validation_batches": "Maximum batches per validation run.",
                }.get(key, ""),
                canonical_path="train.max_epochs" if key == "epochs" else None,
            )
            for key, label, value, minimum in settings
        )
        if model.startswith("pi"):
            fields.append(
                AdapterInputField(
                    path="native.config.weights",
                    label="π0.5 pretrained weights",
                    kind="string",
                    required=True,
                    help="Use the EgoVerse π0.5 runtime and a cluster directory containing pretrained weights.",
                )
            )
        fields.append(
            AdapterInputField(
                path="native.config.model_overrides",
                label="Model overrides",
                kind="json",
                default={},
                help="Optional native model settings.",
            )
        )
        if recorded:
            fields.append(
                AdapterInputField(
                    path="native.config.reject_outliers",
                    label="Filter training outliers",
                    kind="boolean",
                    default=True,
                    help="Native quantile filter. May reject small datasets. Validation keeps every sample.",
                )
            )
        flags = {
            "train.batch.value": "batch-size",
            "train.learning_rate": "learning-rate",
            "train.seed": "seed",
            "train.num_workers_per_rank": "num-workers",
            "train.precision": "precision",
            "train.batch.gradient_accumulation_steps": "gradient-accumulation",
            **{"native.config." + key: key.replace("_", "-") for key, *_ in settings},
            "native.config.model_overrides": "model-overrides",
        }
        if model.startswith("pi"):
            flags["native.config.weights"] = "weights"
        if recorded:
            flags["native.config.reject_outliers"] = "reject-outliers"
        argv = [
            "python",
            "{{tokens.run_dir}}/adapter-support/egoverse_runtime.py",
            "--repository",
            "{{tokens.source_dir}}",
            "--revision",
            REVISION,
            "--dataset",
            "{{native.config.dataset_path}}",
            "--manifest-sha",
            "{{native.config.dataset_manifest_sha256}}",
            "--output",
            "{{tokens.run_dir}}/artifacts",
            "--model",
            model,
            "--gpu-count",
            "{{computed.gpu_count}}",
        ]
        yield AdapterManifest(
            slug=slug,
            display_name="EgoVerse · " + MODELS[model][0],
            description=(
                "EgoVerse diffusion head with a Skynet HPT configuration. "
                if model == "dp_joints"
                else "Native EgoVerse training and held-out evaluation. "
            )
            + (
                "RGB and recorded joints."
                if recorded
                else "Requires the model’s native data."
            ),
            default_repository=REPOSITORY,
            repository_patterns=[REPOSITORY],
            capabilities=AdapterCapabilities(
                name=slug,
                runtime_backends={"existing"},
                supports_resume=True,
                supports_multi_gpu_single_node=True,
                maximum_gpus=16,
            ),
            runtime=AdapterRuntimePolicy(
                allowed_backends={"existing"}, recommended_backend="existing"
            ),
            defaults=AdapterDefaults(
                resources=AdapterResourceDefaults(gpu_type="l40s"),
                hyperparameters=AdapterHyperparameterDefaults(
                    batch_size=32,
                    batch_semantics="per_device",
                    learning_rate=lr,
                    gradient_accumulation_steps=1,
                    num_workers_per_rank=6,
                    seed=42,
                    precision="bf16",
                ),
            ),
            train=CommandTemplate(
                model_io=adapter_io_contract(slug),
                argv=argv,
                parameter_flags={
                    k: ArgumentBinding(flag="--" + v) for k, v in flags.items()
                },
                input_fields=fields,
                strict_canonical_inputs=True,
                strict_native_config=True,
                supported_canonical_fields=[k for k in flags if k.startswith("train.")]
                + ["train.batch.declared_semantics"],
                resume_argv=["--checkpoint", "{{tokens.resume_checkpoint}}"],
                checkpoint_globs=["artifacts/checkpoints/last.ckpt"],
                data_requirements=DatasetRequirement(
                    description=(
                        "RGB and joints."
                        if recorded
                        else "Native EgoVerse data.yaml, evaluator.yaml and Zarr episodes for "
                        + model
                        + "."
                    )
                ),
                batch_compatibility=AdapterBatchCompatibility(
                    allowed_semantics=["per_device"],
                    multi_gpu_allowed_semantics=["per_device"],
                    supports_gradient_accumulation=True,
                ),
                progress=TrainingProgressContract(
                    unit="epoch",
                    total_path="native.config.epochs",
                    starts_at_zero=True,
                    source=TrainingProgressJsonlSource(
                        path="artifacts/logs.json.txt",
                        completed_key="epoch",
                        completed_offset=1,
                        required_key="epoch",
                        metrics={
                            "train_loss": "train/loss",
                            "val_loss": "validation/joint_mse",
                        },
                    ),
                ),
                capsule_files=support_files(),
            ),
            evaluations=[
                EvaluationAdapterMetadata(
                    environment="egoverse",
                    suites=["egoverse_held_out"],
                    maximum_parallelism=1,
                    runtime_profile_id=(
                        "egoverse-pi" if model.startswith("pi") else RUNTIME
                    ),
                    command=CommandTemplate(
                        argv=[
                            "python",
                            "{{tokens.run_dir}}/adapter-support/egoverse_evaluation.py",
                            "--context",
                            "{{tokens.run_dir}}/adapter-support/evaluation-context.json",
                            "--source-dir",
                            "{{tokens.source_dir}}",
                        ],
                        capsule_files=support_files(),
                        required_values=[
                            "evaluation.checkpoint.sha256",
                            "evaluation.policy.native_config.dataset_path",
                            "evaluation.policy.native_config.dataset_manifest_sha256",
                        ],
                    ),
                )
            ],
        )
