"""Declarative XPolicyLab DP adapter; uses the ordinary experiment pipeline."""

from pathlib import Path
from skynet_app.dataset_formats import RECIPES, XPL_REPOSITORY, XPL_COMMIT


def manifest():
    from . import (
        AdapterManifest,
        AdapterRuntimePolicy,
        AdapterCapabilities,
        AdapterDefaults,
        AdapterResourceDefaults,
        AdapterHyperparameterDefaults,
        CommandTemplate,
        AdapterInputField,
        DataBundleInputBinding,
        TrainingProgressContract,
        TrainingProgressJsonlSource,
    )

    support = Path(__file__).parent
    recipe = RECIPES["dp"]
    fields = [
        AdapterInputField(
            path="native.config.dataset_path",
            label="Prepared dataset",
            kind="string",
            required=True,
            data_binding=DataBundleInputBinding(
                role="training_data",
                formats=[recipe["format"]],
                value_path="location.path",
                contract=recipe["contract"],
            ),
            help="Choose a prepared dataset with a verified copy on the training cluster.",
        ),
        AdapterInputField(
            path="native.config.dataset_manifest_sha256",
            label="Dataset fingerprint",
            kind="string",
            required=True,
            data_binding=DataBundleInputBinding(
                role="training_data",
                formats=[recipe["format"]],
                value_path="version.manifest_sha256",
                contract=recipe["contract"],
            ),
            help="Pinned automatically from the selected dataset. Every file is checked before training.",
        ),
        AdapterInputField(
            path="native.config.epochs",
            label="Training epochs",
            kind="integer",
            default=100,
            help="Number of passes over the selected training episodes.",
        ),
    ]
    return AdapterManifest(
        slug="xpolicylab-dp",
        display_name="XPolicyLab · Diffusion Policy",
        description="Image-conditioned diffusion policy using recorded joint commands and calibrated scene views.",
        default_repository=XPL_REPOSITORY,
        repository_patterns=[XPL_REPOSITORY],
        runtime=AdapterRuntimePolicy(
            allowed_backends={"conda", "existing"}, recommended_backend="existing"
        ),
        capabilities=AdapterCapabilities(
            name="xpolicylab-dp",
            runtime_backends={"conda", "existing"},
            supports_multi_gpu_single_node=False,
            supports_resume=False,
            minimum_gpus=1,
            maximum_gpus=1,
        ),
        defaults=AdapterDefaults(
            resources=AdapterResourceDefaults(gpu_type="l40s"),
            hyperparameters=AdapterHyperparameterDefaults(
                batch_size=8,
                batch_semantics="per_device",
                learning_rate=0.0001,
                seed=42,
            ),
        ),
        train=CommandTemplate(
            argv=[
                "python",
                "{{tokens.run_dir}}/adapter-support/skynet_dp_training.py",
                "--repository",
                "{{tokens.source_dir}}",
                "--revision",
                XPL_COMMIT,
                "--dataset",
                "{{native.config.dataset_path}}",
                "--manifest-sha",
                "{{native.config.dataset_manifest_sha256}}",
                "--output",
                "{{tokens.run_dir}}/artifacts",
                "--batch-size",
                "{{train.batch.value}}",
                "--learning-rate",
                "{{train.learning_rate}}",
                "--seed",
                "{{train.seed}}",
                "--epochs",
                "{{native.config.epochs}}",
            ],
            input_fields=fields,
            supported_canonical_fields=[
                "train.batch.value",
                "train.batch.declared_semantics",
                "train.learning_rate",
                "train.seed",
            ],
            checkpoint_globs=["artifacts/checkpoints/*.ckpt"],
            progress=TrainingProgressContract(
                unit="epoch",
                total_path="native.config.epochs",
                starts_at_zero=True,
                source=TrainingProgressJsonlSource(
                    path="artifacts/logs.json.txt",
                    completed_key="epoch",
                    completed_offset=1,
                    required_key="val_loss",
                    metrics={
                        "train_loss": "train/loss",
                        "val_loss": "validation/loss",
                        "lr": "train/learning_rate",
                        "global_step": "training/global_step",
                        "train_action_mse_error": "train/action_mse",
                    },
                ),
            ),
            capsule_files={
                "adapter-support/skynet_dp_training.py": (
                    support / "dp_training.py"
                ).read_text(),
                "adapter-support/artifacts.py": (
                    support.parents[1] / "ops/datasets/artifacts.py"
                ).read_text(),
            },
        ),
        warnings=[
            "Training measures validation loss. Simulation evaluation is not available yet."
        ],
    )
