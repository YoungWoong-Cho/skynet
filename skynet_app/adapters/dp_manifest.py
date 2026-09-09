"""DP declarations consumed by the ordinary adapter, data, and experiment UI."""

from pathlib import Path

from skynet_app.dataset_formats import RECIPES, XPL_COMMIT, XPL_REPOSITORY
from skynet_app.training_contracts import DatasetRequirement, TrainingPreset


def manifest():
    from . import (
        AdapterCapabilities,
        AdapterDefaults,
        AdapterHyperparameterDefaults,
        AdapterInputField,
        AdapterManifest,
        AdapterResourceDefaults,
        AdapterRuntimePolicy,
        CommandTemplate,
        DataBundleInputBinding,
        TrainingProgressContract,
        TrainingProgressJsonlSource,
    )

    support = Path(__file__).parent
    contracts = [RECIPES[k]["contract"] for k in ("dp-state", "dp")]
    fields = [
        AdapterInputField(
            path="native.config." + key,
            label=label,
            kind="string",
            required=True,
            data_binding=DataBundleInputBinding(
                role="training_data",
                formats=[RECIPES["dp"]["format"]],
                value_path=value,
                contracts=contracts,
                contract_selector="native.config.observation_mode",
                contract_choices={
                    "state": contracts,
                    "rgb": [RECIPES["dp"]["contract"]],
                },
            ),
            help="Selected automatically from a verified training-cluster dataset.",
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
        (
            "observation_mode",
            "Observations",
            "string",
            "state",
            None,
            None,
            ["state", "rgb"],
            "State uses joint positions; RGB adds three calibrated scene views.",
        ),
        (
            "epochs",
            "Maximum epochs",
            "integer",
            300,
            1,
            100000,
            [],
            "Training stops earlier when validation loss stops improving.",
        ),
        (
            "observation_steps",
            "Observation history",
            "integer",
            2,
            1,
            16,
            [],
            "Consecutive observations available at each decision.",
        ),
        (
            "action_steps",
            "Action chunk",
            "integer",
            16,
            4,
            64,
            [4, 8, 16, 32, 64],
            "Future actions predicted and executed before replanning.",
        ),
        (
            "diffusion_steps",
            "Diffusion training steps",
            "integer",
            100,
            1,
            1000,
            [],
            "DDPM noise timesteps; squared-cosine schedule.",
        ),
        (
            "inference_steps",
            "Diffusion inference steps",
            "integer",
            20,
            1,
            1000,
            [],
            "Denoising steps per prediction.",
        ),
        (
            "weight_decay",
            "Weight decay",
            "number",
            1e-4,
            0,
            1,
            [],
            "AdamW weight decay.",
        ),
        (
            "gradient_clip",
            "Gradient norm limit",
            "number",
            1.0,
            0.000001,
            1000,
            [],
            "Applied at each optimizer update.",
        ),
        (
            "ema_decay",
            "EMA decay",
            "number",
            0.995,
            0,
            0.999999,
            [],
            "Fixed decay per optimizer update; validation uses EMA weights.",
        ),
        (
            "early_stopping_patience",
            "Early stopping patience",
            "integer",
            20,
            0,
            100000,
            [],
            "Epochs without validation improvement. 0 disables it. The paper does not specify patience.",
        ),
        (
            "lr_schedule",
            "Learning-rate schedule",
            "string",
            "constant",
            None,
            None,
            ["constant", "cosine"],
            "Implementation choice; the paper does not specify a schedule.",
        ),
        (
            "warmup_steps",
            "Warmup updates",
            "integer",
            0,
            0,
            1000000,
            [],
            "Optimizer updates before reaching the learning rate.",
        ),
    ]
    for key, label, kind, default, minimum, maximum, choices, help_text in settings:
        fields.append(
            AdapterInputField(
                path="native.config." + key,
                label=label,
                kind=kind,
                default=default,
                minimum=minimum,
                maximum=maximum,
                choices=choices,
                help=help_text,
                maximum_path="native.config.diffusion_steps"
                if key == "inference_steps"
                else None,
                canonical_path="train.max_epochs" if key == "epochs" else None,
            )
        )
    common = {
        "train.learning_rate": 1e-4,
        "train.batch.value": 256,
        "train.batch.gradient_accumulation_steps": 1,
        "train.num_workers_per_rank": 0,
    }
    state = TrainingPreset(
        id="dexverse-state/v1",
        name="DexVerse state DP",
        source="https://arxiv.org/html/2607.08751v1",
        description="Paper hyperparameters with joint-state observations. Patience 20, constant LR, zero warmup and seed 42 are implementation choices; task observations and rollout evaluation are not a benchmark reproduction.",
        values={**common, **{"native.config." + x[0]: x[3] for x in settings}},
    )
    rgb = TrainingPreset(
        id="rgb-joints/v2",
        name="RGB and joints DP",
        description="Three scene cameras plus joints; a smaller batch for image memory use.",
        values={
            **state.values,
            "native.config.observation_mode": "rgb",
            "train.batch.value": 8,
        },
    )
    fields.insert(
        0,
        AdapterInputField(
            path="native.config.training_preset",
            label="Training preset",
            kind="string",
            default=state.id,
            choices=[state.id, rgb.id],
            help="Loads versioned defaults. Individual settings remain editable.",
        ),
    )
    argv = [
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
    ]
    for flag, path in {
        "batch-size": "train.batch.value",
        "learning-rate": "train.learning_rate",
        "seed": "train.seed",
        "gradient-accumulation": "train.batch.gradient_accumulation_steps",
        "num-workers": "train.num_workers_per_rank",
        **{key.replace("_", "-"): "native.config." + key for key, *_ in settings},
        "training-preset": "native.config.training_preset",
    }.items():
        argv += ["--" + flag, "{{" + path + "}}"]
    return AdapterManifest(
        slug="xpolicylab-dp",
        display_name="XPolicyLab · Diffusion Policy",
        description="State or RGB-conditioned diffusion policy with explicit data and training settings.",
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
                batch_size=256,
                batch_semantics="per_device",
                learning_rate=1e-4,
                gradient_accumulation_steps=1,
                num_workers_per_rank=0,
                seed=42,
            ),
        ),
        train=CommandTemplate(
            argv=argv,
            input_fields=fields,
            presets=[state, rgb],
            default_preset=state.id,
            strict_canonical_inputs=True,
            strict_native_config=True,
            data_requirements=DatasetRequirement(
                description="Verified joint state/action Zarr; RGB mode additionally requires three calibrated scene cameras.",
                observations=["state", "optional_rgb"],
                action_representation="raw_joint_position_command",
            ),
            supported_canonical_fields=[
                *common,
                "train.batch.declared_semantics",
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
    )
