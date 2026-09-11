"""One declaration factory for EgoVerse model adapters."""

from pathlib import Path

from skynet_app.model_io import adapter_io_contract
from .egoverse_models import ALGORITHMS, MODEL_LEARNING_RATES, model_contracts

REPOSITORY = "https://github.com/GaTech-RL2/EgoVerse"
REVISION = "e17cf98fe4bc234c564b37abc9e155f25e76d566"
RUNTIME = "egoverse-native"
FORMAT = "egoverse-episodes-zarr/v1"
CONTRACT = "skynet.egoverse-rgb-joints/v1"


def support_files():
    root = Path(__file__).parent
    return {
        "adapter-support/" + name: (root / name).read_text()
        for name in (
            "egoverse_runtime.py",
            "egoverse_models.py",
            "egoverse_evaluation.py",
            "egoverse_data.py",
            "egoverse_splits.py",
            "egoverse_readiness.py",
            "evaluation_video.py",
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

    for algorithm, (label, models) in ALGORITHMS.items():
        model = models[0]
        recorded = algorithm in {"act", "hpt"}
        slug = "egoverse-" + algorithm
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
                    contracts=model_contracts(model),
                    contract_selector="native.config.model_preset",
                    contract_choices={name: model_contracts(name) for name in models},
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
        fields.insert(0, AdapterInputField(
            path="native.config.model_preset", label="Native model preset",
            kind="string", required=True, default=model, choices=models,
            help=("HPT recorded joints binds the native EVA flow model to recorded joint/camera names. " if algorithm == "hpt" else "")
            + "Other presets require their matching native dataset contract.",
        ))
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
            "native.config.model_preset": "model",
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
            "--algorithm",
            algorithm,
            "--gpu-count",
            "{{computed.gpu_count}}",
        ]
        yield AdapterManifest(
            slug=slug,
            display_name="EgoVerse · " + label,
            description=f"Native EgoVerse {label} training and held-out evaluation. Select a compatible model preset and dataset.",
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
                        "The selected model preset defines the verified dataset contract. Recorded joints are supported by ACT and the HPT recorded-joints configuration; other presets require native data.yaml, evaluator.yaml and Zarr episodes."
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
