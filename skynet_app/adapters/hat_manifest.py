"""Official HAT model connected through existing Skynet recording/Slurm contracts."""
from pathlib import Path
from skynet_app.observation_contracts import rgb_requirements
from skynet_app.training_contracts import (
    RECORDING_DATASET_FORMAT, RecordingConversion, RecordingDataPreset,
    RecordingLoaderValidation, RecordingSampling, DatasetRequirement, TrainingPreset,
)
from skynet_app.model_io import ModelIOContract, IOStream, axis

REPOSITORY = "https://github.com/RogerQi/human-policy"
REVISION = "2d9d73cc5a3859094ef705f35b8f2faecfc2bc4f"
CONTRACT = "skynet.hat-rgb-fingertips/v1"
RUNTIME = "human-policy-hat"


def support_files():
    from .unidex_manifest import support_files as existing_files
    shared = existing_files()
    files = {name: content for name, content in shared.items()
             if name.startswith("adapter-support/action_codecs/") or
             name in {"adapter-support/recording_dataset.py", "adapter-support/recording_time.py", "adapter-support/dataset_inputs.py",
                      "adapter-support/unidex_subset.py", "adapter-support/unidex_data.py", "adapter-support/unidex_input.py"}}
    root = Path(__file__).parent
    files.update({"adapter-support/" + name: (root / name).read_text()
                  for name in ("hat_data.py", "hat_runtime.py", "training_parallel.py")})
    files["adapter-support/observation_geometry.py"] = (root.parents[1] / "ops/datasets/observation_geometry.py").read_text()
    return files


def recording_conversion():
    from .unidex_manifest import verified_robots
    return RecordingConversion(default_preset="rgb-fingertips", presets=[RecordingDataPreset(
        id="rgb-fingertips", name="RGB and HAT wrist/fingertips",
        description="Skynet bridge: existing RGB/absolute-wrist preparation and verified hand FK feed official HAT slots.",
        contract=CONTRACT, observations=["state", "rgb"], supported_robots=verified_robots(),
        action_representation=dict(id="skynet.unidex-faas/v1", frame="camera_opengl", action_semantics="controller_targets"),
        observation_requirements=rgb_requirements(),
        preprocessing=dict(hat_state_action_dim=128, hat_frame="camera_opengl", hat_inactive_slots="zero"),
        training_setup=dict(adapter="human-policy-hat", repository=REPOSITORY, revision=REVISION,
                            runtime="existing", runtime_profile=RUNTIME, preset="hat-fit/v1"),
        loader_validation=RecordingLoaderValidation(runtime_profile=RUNTIME, script="hat_runtime.py",
            argv=["--repository", "{repository}", "--dataset", "{dataset}", "--manifest-sha", "{manifest_sha256}",
                  "--output", "{output}", "--verify-only"], schemas=["skynet.hat-loader-validation/v1"], mode="rgb"),
    )])


def manifest():
    from . import (AdapterManifest, AdapterCapabilities, AdapterRuntimePolicy, AdapterDefaults,
        AdapterHyperparameterDefaults, AdapterResourceDefaults, AdapterCheckpointDefaults,
        AdapterInputField, DataBundleInputBinding, CommandTemplate, ArgumentBinding,
        AdapterBatchCompatibility, TrainingProgressContract, TrainingProgressJsonlSource)
    fields = [AdapterInputField(path="native.config.datasets", label="Training datasets", kind="json", required=True,
        data_binding=DataBundleInputBinding(role="training_data", cardinality="many", formats=[RECORDING_DATASET_FORMAT],
            contracts=[CONTRACT], value_path="selection"))]
    fields += [
        AdapterInputField(path="native.config.training_preset", label="Training preset", kind="string",
            default="hat-fit/v1", choices=["hat-fit/v1", "hat-fit-resnet/v1", "hat-seven-hands-main/v1", "custom"]),
        AdapterInputField(path="native.config.model_config", label="Official model config", kind="string",
            default="hat_linear", choices=["hat_linear", "act_resnet"],
            help="Upstream configs at the pinned revision: hat_linear = DINOv2 ViT-S/14 (frozen); "
                 "act_resnet = ImageNet ResNet18 (lr_backbone 1e-5). Transformer, CVAE and loss are identical."),
        AdapterInputField(path="native.config.action_steps", label="Action chunk", kind="integer", default=50, minimum=1, maximum=200,
            help="Skynet fit preset: 50. The official README example uses 100."),
        AdapterInputField(path="native.config.control_hz", label="Control frequency (Hz)", kind="number", minimum=.000001,
            help="Skynet selection; blank preserves the recorded rate."),
        AdapterInputField(path="native.config.validation_every", label="Validate every optimizer steps", kind="integer", default=200, minimum=1),
    ]
    fields += [
        AdapterInputField(path="native.config.window_policy",label="Action windows",kind="string",default="pad",choices=["pad","complete"]),
        AdapterInputField(path="native.config.mixing_policy",label="Dataset mixing",kind="string",default="window_proportional",choices=["window_proportional","hand_balanced"]),
        AdapterInputField(path="native.config.unique_source_frames",label="Unique training source frames",kind="integer",minimum=1,maximum=10000000),
        AdapterInputField(path="native.config.data_selection_seed",label="Data selection seed",kind="integer",default=20260920,minimum=0),
        AdapterInputField(path="native.config.validation_enabled",label="Run validation",kind="boolean",default=True),
    ]
    common = {"train.learning_rate": 1e-4, "train.batch.value": 16,
              "train.num_workers_per_rank": 2, "train.max_steps": 3000, "train.seed": 42}
    preset = TrainingPreset(id="hat-fit/v1", name="HAT · Fit check",
        source=f"{REPOSITORY}/blob/{REVISION}/hdt/configs/models/hat_linear.yaml",
        description="Skynet 3000-update fit preset. Official hat_linear architecture/loss; verified right-hand inputs, three scene views, 50-step chunks and fixed episode splits.",
        values={**common, "native.config.model_config": "hat_linear", "native.config.action_steps": 50,
                "native.config.validation_every": 200})
    resnet_preset = TrainingPreset(id="hat-fit-resnet/v1", name="HAT · Fit check · ResNet18 (no DINO)",
        source=f"{REPOSITORY}/blob/{REVISION}/hdt/configs/models/act_resnet.yaml",
        description="DINO ablation of the fit preset: the repository's official act_resnet vision backbone. Data, HAT128 wrist/fingertip representation, chunk and update budget are unchanged.",
        values={**preset.values, "native.config.model_config": "act_resnet"})
    main_preset = TrainingPreset(id="hat-seven-hands-main/v1", name="HAT · Seven-hand zero-shot study",
        source=preset.source, description="Skynet fixed-data study: 8000 updates, 1800 source frames, 30Hz/chunk30, hand-balanced; final checkpoint only. Original HAT architecture and optimizer.",
        values={**common,"native.config.model_config":"hat_linear","train.max_steps":8000,"train.seed":1701,"train.checkpoint.final_selector":"latest",
            "native.config.action_steps":30,"native.config.control_hz":30,"native.config.window_policy":"complete",
            "native.config.mixing_policy":"hand_balanced","native.config.unique_source_frames":1800,
            "native.config.data_selection_seed":20260920,"native.config.validation_enabled":False,
            "native.config.validation_every":1000})
    argv = ["python", "{{tokens.run_dir}}/adapter-support/hat_runtime.py", "--repository", "{{tokens.source_dir}}",
            "--data-spec", "{{tokens.run_dir}}/resolved-spec.json",
            "--output", "{{tokens.run_dir}}/artifacts"]
    for flag, path in {"batch-size":"train.batch.value", "max-steps":"train.max_steps", "learning-rate":"train.learning_rate",
        "num-workers":"train.num_workers_per_rank", "seed":"train.seed", "action-steps":"native.config.action_steps",
        "validation-every":"native.config.validation_every"}.items():
        argv += ["--" + flag, "{{" + path + "}}"]
    return AdapterManifest(slug="human-policy-hat", display_name="Human Policy · HAT",
        description="Original HAT DINOv2/CVAE Transformer and L1+KL+EEF loss; the repository's official ResNet18 config is selectable for DINO ablations. Skynet bridges verified right-hand geometries and three fixed scene views, training-only normalization and an optimizer-step loop. No human co-training is claimed. Skynet rollout uses the existing recorded simulator, exact target-hand assets and dex_retargeting PositionOptimizer.",
        default_repository=REPOSITORY, repository_patterns=[REPOSITORY],
        capabilities=AdapterCapabilities(name="human-policy-hat", runtime_backends={"existing", "conda"},
            supports_multi_gpu_single_node=False, supports_resume=False),
        runtime=AdapterRuntimePolicy(allowed_backends={"existing", "conda"}, recommended_backend="existing"),
        defaults=AdapterDefaults(resources=AdapterResourceDefaults(gpu_type="l40s", cpus_per_task=8, memory_gb=48),
            hyperparameters=AdapterHyperparameterDefaults(batch_size=16, batch_semantics="per_device", learning_rate=1e-4,
                num_workers_per_rank=2, seed=42),
            checkpoint=AdapterCheckpointDefaults(auto_resume=False, final_selector="best")),
        train=CommandTemplate(argv=argv, input_fields=fields, presets=[preset, resnet_preset, main_preset], default_preset=preset.id,
            parameter_flags={
                **{"native.config."+key:ArgumentBinding(flag="--"+key.replace("_","-"),omit_if_none=True)
                   for key in ("model_config","control_hz","window_policy","mixing_policy","unique_source_frames","data_selection_seed")},
                "native.config.validation_enabled":ArgumentBinding(flag="--validation-enabled",style="boolean",false_flag="--no-validation-enabled"),
                "train.checkpoint.final_selector":ArgumentBinding(flag="--checkpoint-selection",omit_if_none=True)},
            strict_canonical_inputs=True, strict_native_config=True,
            supported_canonical_fields=[*common, "train.batch.declared_semantics", "train.checkpoint.final_selector"],
            batch_compatibility=AdapterBatchCompatibility(allowed_semantics=["per_device"], supports_gradient_accumulation=False),
            data_requirements=DatasetRequirement(description="Verified right hands: camera-frame wrist and palm-relative fingertips in official HAT128 slots.",
                observations=["state", "rgb"], action_representation="hat128_right_wrist_and_fingertips",
                recording_conversion=recording_conversion(), recording_sampling=RecordingSampling(window_policy="pad", default_action_steps=50)),
            model_io=ModelIOContract(inputs=[IOStream(name="RGB", modality="RGB", cameras=["scene_front","scene_left","scene_right"],
                axes=[axis("views",3),axis("height",224),axis("width",308),axis("channels",3)]),
                IOStream(name="HAT state", modality="state", axes=[axis("values",128)])],
                outputs=[IOStream(name="HAT action chunk", modality="actions", axes=[axis("steps",50,"spec.native.config.action_steps"),axis("values",128)])],
                note="Original 128-D HAT; Skynet right wrist/tips in camera OpenGL, other slots zero. Three scene cameras replace stereo egocentric views."),
            checkpoint_globs=["artifacts/checkpoints/best.ckpt", "artifacts/checkpoints/latest.ckpt", "artifacts/checkpoints/step-*.ckpt"],
            progress=TrainingProgressContract(unit="step", total_path="train.max_steps", source=TrainingProgressJsonlSource(
                path="artifacts/logs.json.txt", completed_key="global_step", required_key="train_loss",
                metrics={"train/loss":"train_loss","validation/loss":"val_loss","train/lr":"lr"})),
            capsule_files=support_files()))
