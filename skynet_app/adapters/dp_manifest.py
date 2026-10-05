"""Official DP components exposed through the canonical Skynet experiment UI."""
from pathlib import Path
from skynet_app.training_contracts import DatasetRequirement, TrainingPreset, RECORDING_DATASET_FORMAT, RecordingSampling
from skynet_app.model_io import recorded_joint_io
from .dp_data import CONTRACTS
from .dp_runtime import REVISION

REPOSITORY="https://github.com/real-stanford/diffusion_policy"


def support_files():
    root=Path(__file__).parent
    return {"adapter-support/"+name:(root/name).read_text() for name in (
        "dp_runtime.py","dp_data.py","dp_evaluation.py","recording_dataset.py","recording_time.py","policy_contract.py","training_parallel.py")}


def manifest():
    from . import (AdapterManifest, AdapterCapabilities, AdapterRuntimePolicy, AdapterDefaults,
        AdapterHyperparameterDefaults, AdapterResourceDefaults, AdapterCheckpointDefaults,
        AdapterInputField, DataBundleInputBinding, CommandTemplate, ArgumentBinding,
        AdapterBatchCompatibility, TrainingProgressContract, TrainingProgressJsonlSource)
    fields=[AdapterInputField(path="native.config."+key,label=label,kind="string",required=True,
        data_binding=DataBundleInputBinding(role="training_data",formats=[RECORDING_DATASET_FORMAT],
            contracts=list(CONTRACTS),value_path=value)) for key,label,value in (
        ("dataset_path","Recorded RGB/joint dataset","location.path"),
        ("dataset_manifest_sha256","Dataset fingerprint","version.manifest_sha256"))]
    fields += [
        AdapterInputField(path="native.config.training_preset",label="Training preset",kind="string",
            default="dp-hat-fit-comparison/v1",choices=["dp-hat-fit-comparison/v1","custom"]),
        AdapterInputField(path="native.config.action_steps",label="Action chunk",kind="integer",default=50,minimum=1,maximum=200,
            help="Skynet comparison with HAT fit: 50 executed commands. Upstream image example: 8."),
        AdapterInputField(path="native.config.observation_steps",label="Observation history",kind="integer",default=2,minimum=1,maximum=8,
            help="Official DP example: 2 consecutive observations ending at the current frame."),
        AdapterInputField(path="native.config.control_hz",label="Control frequency (Hz)",kind="number",minimum=.000001,
            help="Skynet setting. Blank preserves the recorded rate."),
        AdapterInputField(path="native.config.validation_every",label="Validate every optimizer steps",kind="integer",default=200,minimum=1),
    ]
    common={"train.learning_rate":1e-4,"train.batch.value":16,"train.num_workers_per_rank":2,"train.max_steps":3000,"train.seed":42}
    preset=TrainingPreset(id="dp-hat-fit-comparison/v1",name="DP · Same-hand HAT fit comparison",
        source=f"{REPOSITORY}/blob/{REVISION}/diffusion_policy/config/train_diffusion_unet_image_workspace.yaml",
        description="Skynet matched-data fit budget: 3000 updates, batch16, chunk50. Official DP model/loss, ResNet18 without pretrained weights, native EMA and optimizer recipe.",
        values={**common,"native.config.action_steps":50,"native.config.observation_steps":2,"native.config.validation_every":200})
    argv=["python","{{tokens.run_dir}}/adapter-support/dp_runtime.py","--repository","{{tokens.source_dir}}",
        "--dataset","{{native.config.dataset_path}}","--manifest-sha","{{native.config.dataset_manifest_sha256}}",
        "--output","{{tokens.run_dir}}/artifacts"]
    for flag,key in {"batch-size":"train.batch.value","max-steps":"train.max_steps","num-workers":"train.num_workers_per_rank",
                     "seed":"train.seed","learning-rate":"train.learning_rate","action-steps":"native.config.action_steps",
                     "observation-steps":"native.config.observation_steps","validation-every":"native.config.validation_every"}.items():
        argv.extend(["--"+flag,"{{"+key+"}}"])
    return AdapterManifest(slug="diffusion-policy",display_name="Diffusion Policy · Official U-Net",
        description="Original real-stanford DiffusionUnetImagePolicy, ResNet18, DDPM100, loss and EMA. Skynet supplies registered RGB/joint data, fixed splits, timing, step budget, checkpoints and simulator rollout. The HAT data contract supplies raw joint/RGB streams; HAT fingertip conversion is not used by DP.",
        default_repository=REPOSITORY,repository_patterns=[REPOSITORY],
        capabilities=AdapterCapabilities(name="diffusion-policy",runtime_backends={"existing","conda"},supports_multi_gpu_single_node=False,supports_resume=False),
        runtime=AdapterRuntimePolicy(allowed_backends={"existing","conda"},recommended_backend="existing"),
        defaults=AdapterDefaults(resources=AdapterResourceDefaults(gpu_type="a40",gpu_count=1,cpus_per_task=8,memory_gb=64),
            hyperparameters=AdapterHyperparameterDefaults(batch_size=16,batch_semantics="per_device",learning_rate=1e-4,num_workers_per_rank=2,seed=42),
            checkpoint=AdapterCheckpointDefaults(auto_resume=False,final_selector="best")),
        train=CommandTemplate(argv=argv,input_fields=fields,presets=[preset],default_preset=preset.id,
            parameter_flags={"native.config.control_hz":ArgumentBinding(flag="--control-hz",omit_if_none=True),
                             "train.checkpoint.final_selector":ArgumentBinding(flag="--checkpoint-selection",omit_if_none=True)},
            strict_canonical_inputs=True,strict_native_config=True,
            supported_canonical_fields=[*common,"train.batch.declared_semantics","train.checkpoint.final_selector"],
            batch_compatibility=AdapterBatchCompatibility(allowed_semantics=["per_device"],supports_gradient_accumulation=False),
            data_requirements=DatasetRequirement(description="One hand's registered RGB/joint streams with unchanged immutable episode split. Accepts existing ACT or HAT prepared recordings and reads their raw joint commands.",
                observations=["state","rgb"],action_representation="native_joint_position_commands",
                recording_sampling=RecordingSampling(window_policy="pad",default_action_steps=50,require_validation=True)),
            model_io=recorded_joint_io(history=2,action_steps=50,image_size=(240,320),
                history_paths=("spec.native.config.observation_steps",),action_paths=("spec.native.config.action_steps",)),
            capsule_files=support_files(),checkpoint_globs=["artifacts/checkpoints/best.ckpt","artifacts/checkpoints/latest.ckpt"],
            progress=TrainingProgressContract(unit="step",total_path="train.max_steps",source=TrainingProgressJsonlSource(
                path="artifacts/logs.json.txt",completed_key="global_step",required_key="train_loss",
                metrics={"train/loss":"train_loss","validation/loss":"val_loss","train/lr":"lr"}))))
