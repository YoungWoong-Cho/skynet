"""ACT training and evaluation through the ordinary Skynet adapter interface."""

from skynet_app.model_io import adapter_io_contract

from skynet_app.dataset_formats import RECIPES, XPL_COMMIT, XPL_REPOSITORY
from skynet_app.training_contracts import TrainingPreset, DatasetRequirement
from .xpolicy_manifest import support_files, progress_contract, evaluation


def manifest():
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
    )

    fields = [
        AdapterInputField(
            path="native.config." + key,
            label=label,
            kind="string",
            required=True,
            help="From the selected cluster dataset.",
            data_binding=DataBundleInputBinding(
                role="training_data",
                formats=[RECIPES["act"]["format"]],
                contracts=[RECIPES["act"]["contract"]],
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
        ("epochs", "Epochs", "integer", 6000, 1, 100000),
        ("action_steps", "Action chunk", "integer", 50, 1, 200),
        ("hidden_dim", "Hidden dimension", "integer", 512, 64, 2048),
        ("feedforward_dim", "Feedforward dimension", "integer", 3200, 128, 8192),
        ("kl_weight", "KL weight", "number", 10.0, 0, 1000),
        ("backbone_learning_rate", "Backbone learning rate", "number", 1e-5, 1e-10, 1),
        ("weight_decay", "Weight decay", "number", 1e-4, 0, 1),
        ("gradient_clip", "Gradient norm limit", "number", 1.0, 1e-6, 1000),
    ]
    for key, label, kind, default, minimum, maximum in settings:
        fields.append(
            AdapterInputField(
                path="native.config." + key,
                label=label,
                kind=kind,
                default=default,
                minimum=minimum,
                maximum=maximum,
                canonical_path="train.max_epochs" if key == "epochs" else None,
            )
        )
    common = {
        "train.learning_rate": 1e-5,
        "train.batch.value": 16,
        "train.batch.gradient_accumulation_steps": 1,
        "train.num_workers_per_rank": 0,
    }
    preset = TrainingPreset(
        id="xpolicylab-act/v1",
        name="XPolicyLab ACT",
        description="RGB and joints with action chunks.",
        source=f"{XPL_REPOSITORY}/blob/{XPL_COMMIT}/policy/ACT/train.sh",
        values={**common, **{"native.config." + s[0]: s[3] for s in settings}},
    )
    fields.insert(
        0,
        AdapterInputField(
            path="native.config.training_preset",
            label="Training preset",
            kind="string",
            default=preset.id,
            choices=[preset.id, "custom"],
        ),
    )
    argv = [
        "python",
        "{{tokens.run_dir}}/adapter-support/skynet_act_training.py",
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
    ]
    for flag, path in {
        "batch-size": "train.batch.value",
        "batch-semantics": "train.batch.declared_semantics",
        "gpu-count": "computed.gpu_count",
        "learning-rate": "train.learning_rate",
        "seed": "train.seed",
        "gradient-accumulation": "train.batch.gradient_accumulation_steps",
        "num-workers": "train.num_workers_per_rank",
        "training-preset": "native.config.training_preset",
        **{s[0].replace("_", "-"): "native.config." + s[0] for s in settings},
    }.items():
        argv.extend(["--" + flag, "{{" + path + "}}"])
    return AdapterManifest(
        slug="xpolicylab-act",
        display_name="XPolicyLab · ACT",
        description="Action Chunking Transformer with RGB and joint observations.",
        default_repository=XPL_REPOSITORY,
        repository_patterns=[XPL_REPOSITORY],
        capabilities=AdapterCapabilities(
            name="xpolicylab-act",
            runtime_backends={"existing", "conda"},
            minimum_gpus=1,
            maximum_gpus=8,
            supports_resume=False,
            supports_multi_gpu_single_node=True,
        ),
        runtime=AdapterRuntimePolicy(
            allowed_backends={"existing", "conda"}, recommended_backend="existing"
        ),
        defaults=AdapterDefaults(
            resources=AdapterResourceDefaults(gpu_type="l40s"),
            hyperparameters=AdapterHyperparameterDefaults(
                batch_size=16,
                batch_semantics="per_device",
                learning_rate=1e-5,
                gradient_accumulation_steps=1,
                num_workers_per_rank=0,
                seed=42,
            ),
        ),
        train=CommandTemplate(
            model_io=adapter_io_contract("xpolicylab-act"),
            argv=argv,
            input_fields=fields,
            presets=[preset],
            default_preset=preset.id,
            strict_canonical_inputs=True,
            strict_native_config=True,
            data_requirements=DatasetRequirement(
                description="ACT HDF5 with three RGB views and joints.",
                observations=["state", "rgb"],
                action_representation="raw_joint_position_command",
            ),
            supported_canonical_fields=[
                *common,
                "train.batch.declared_semantics",
                "train.seed",
            ],
            checkpoint_globs=["artifacts/checkpoints/best.ckpt"],
            progress=progress_contract(),
            capsule_files=support_files("act"),
        ),
        evaluations=[evaluation("act")],
    )
