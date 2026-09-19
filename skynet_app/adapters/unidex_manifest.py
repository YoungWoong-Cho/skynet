"""Pinned native UniDex training and its recording data contract."""
import json
from pathlib import Path

from skynet_app.observation_contracts import CONTRACT_SCHEMA
from skynet_app.training_contracts import (
    RECORDING_DATASET_FORMAT, DatasetRequirement, RecordingConversion,
    RecordingDataPreset, RecordingLoaderValidation, RecordingSampling,
)

REPOSITORY = "https://github.com/unidex-ai/UniDex"
REVISION = "97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d"
CONTRACT = "skynet.unidex-pointcloud-faas/v1"
RUNTIME = "unidex-native"
SPECIFICATIONS = Path(__file__).resolve().parents[2] / "config/action_representations/unidex-faas-v1"


def verified_robots():
    specifications = [json.loads(path.read_text()) for path in sorted(SPECIFICATIONS.glob("*.json"))]
    return [spec["robot"] for spec in specifications
            if spec.get("mapping_status") == "VERIFIED" and spec.get("native_wrist")]


def recording_conversion():
    return RecordingConversion(default_preset="pointcloud-faas", presets=[RecordingDataPreset(
        id="pointcloud-faas", name="Scene-front point cloud and FAAS actions",
        description="1,024 colored points from the fixed front scene camera and native UniDex FAAS82 actions.",
        contract=CONTRACT, observations=["state", "point_cloud"],
        action_representation=dict(id="skynet.unidex-faas/v1", frame="camera_opengl", action_semantics="controller_targets"),
        observation_requirements={
            "schema": CONTRACT_SCHEMA, "timing": {"alignment": "pre_action_state", "stride": 1},
            "streams": [{"name": "scene_front_pointcloud", "modality": "point_cloud", "camera_ids": ["scene_front"],
                "width": 256, "height": 256, "dtype": "float32", "channels": "XYZRGB",
                "coordinate_frame": "camera:scene_front", "camera_convention": "ros_optical", "color_range": "0_1",
                "num_points": 1024, "sampling": "farthest_point", "seed": 0, "crop": None,
                "depth_range": [0.01, 5.0], "processing_order": ["unproject", "world_transform", "merge", "crop", "sample"],
                "insufficient_points": "repeat_with_mask"}],
        },
        preprocessing=dict(pointcloud_frame="camera_ros_optical", pointcloud_native_frame="camera_opengl"),
        supported_robots=verified_robots(),
        training_setup=dict(adapter="unidex", repository=REPOSITORY, revision=REVISION, runtime="existing", runtime_profile=RUNTIME),
        loader_validation=RecordingLoaderValidation(
            runtime_profile=RUNTIME, script="unidex_runtime.py",
            argv=["--repository", "{repository}", "--dataset", "{dataset}", "--manifest-sha", "{manifest_sha256}",
                  "--output", "{output}", "--verify-only"],
            schemas=["skynet.unidex-loader-validation/v1"], mode="pointcloud",
        ),
    )])


def support_files():
    root = Path(__file__).parent
    files = {"adapter-support/" + name: (root / name).read_text() for name in (
        "unidex_data.py", "unidex_runtime.py", "unidex_weights.py", "recording_dataset.py", "recording_time.py",
    )}
    codecs = root.parents[1] / "ops/datasets/action_codecs"
    files.update({"adapter-support/action_codecs/" + name: (codecs / name).read_text() for name in (
        "__init__.py", "geometry.py", "hand_contract.py", "unidex.py",
    )})
    specifications = SPECIFICATIONS
    for path in sorted(specifications.rglob("*")):
        if path.is_file():
            files["adapter-support/action_codecs/specs/" + path.relative_to(specifications).as_posix()] = path.read_text()
    return files


def manifest():
    from . import (
        AdapterManifest, AdapterCapabilities, AdapterRuntimePolicy, AdapterDefaults,
        AdapterHyperparameterDefaults, AdapterResourceDefaults, AdapterInputField,
        DataBundleInputBinding, CommandTemplate, ArgumentBinding,
        AdapterBatchCompatibility, TrainingProgressContract, TrainingProgressJsonlSource,
    )
    conversion = recording_conversion()
    fields = [AdapterInputField(
        path="native.config." + key, label=label, kind="string", required=True,
        data_binding=DataBundleInputBinding(role="training_data", formats=[RECORDING_DATASET_FORMAT],
            contracts=[conversion.presets[0].contract], value_path=value),
    ) for key, label, value in [
        ("dataset_path", "Prepared dataset", "location.path"),
        ("dataset_manifest_sha256", "Dataset fingerprint", "version.manifest_sha256"),
    ]]
    for key, label, required in [
        ("base_weights", "PaliGemma weights directory", True),
        ("pointcloud_weights", "Uni3D weights file", True),
        ("weights_provenance", "Weights provenance manifest", True),
        ("initialize_checkpoint", "Initialize model weights from checkpoint", False),
        ("initialize_checkpoint_sha", "Initialization checkpoint SHA-256", False),
    ]:
        fields.append(AdapterInputField(path="native.config." + key, label=label, kind="string", required=required))
    fields.append(AdapterInputField(path="native.config.epochs", label="Epochs", kind="integer", default=32,
                                    minimum=1, maximum=100000, canonical_path="train.max_epochs"))
    fields.extend([
        AdapterInputField(path="native.config.control_hz", label="Training frequency (Hz)", kind="number",
                          help="Leave blank to use the recording frequency. Lower frequencies must divide each recording's frequency exactly."),
        AdapterInputField(path="native.config.action_steps", label="Action chunk length", kind="integer", default=30,
                          minimum=1, help="Number of controller targets predicted per observation. Only complete chunks are used."),
    ])
    flags = {
        **{"native.config."+key:key.replace("_", "-") for key in (
            "base_weights", "pointcloud_weights", "weights_provenance", "initialize_checkpoint", "initialize_checkpoint_sha", "epochs",
            "control_hz", "action_steps")},
        "train.batch.value":"batch-size", "train.learning_rate":"learning-rate", "train.seed":"seed",
        "train.precision":"precision", "train.num_workers_per_rank":"num-workers",
        "train.batch.gradient_accumulation_steps":"gradient-accumulation",
    }
    return AdapterManifest(
        slug="unidex", display_name="UniDex · Point cloud and FAAS", default_repository=REPOSITORY,
        repository_patterns=[REPOSITORY],
        description="Pinned native UniDex model with shared recording observations and FAAS82 controller targets.",
        runtime=AdapterRuntimePolicy(allowed_backends={"existing"}, recommended_backend="existing"),
        capabilities=AdapterCapabilities(name="unidex", runtime_backends={"existing"},
            supports_multi_gpu_single_node=True, supports_resume=True, maximum_gpus=8),
        defaults=AdapterDefaults(resources=AdapterResourceDefaults(gpu_type="l40s"),
            hyperparameters=AdapterHyperparameterDefaults(batch_size=4, batch_semantics="per_device", learning_rate=1e-4,
                gradient_accumulation_steps=4, num_workers_per_rank=1, seed=42, precision="fp32")),
        train=CommandTemplate(
            argv=["python", "{{tokens.run_dir}}/adapter-support/unidex_runtime.py", "--repository", "{{tokens.source_dir}}",
                  "--dataset", "{{native.config.dataset_path}}", "--manifest-sha", "{{native.config.dataset_manifest_sha256}}",
                  "--output", "{{tokens.run_dir}}/artifacts", "--gpu-count", "{{computed.gpu_count}}"],
            parameter_flags={key:ArgumentBinding(flag="--"+flag) for key,flag in flags.items()},
            input_fields=fields, strict_canonical_inputs=True, strict_native_config=True,
            supported_canonical_fields=[key for key in flags if key.startswith("train.")] + ["train.batch.declared_semantics"],
            resume_argv=["--resume-checkpoint", "{{tokens.resume_checkpoint}}"],
            checkpoint_globs=["artifacts/checkpoints/last.ckpt"], capsule_files=support_files(),
            data_requirements=DatasetRequirement(
                description="Fixed scene-front XYZRGB and FAAS82, restricted to verified asset-bound hand mappings.",
                recording_sampling=RecordingSampling(window_policy="complete", require_validation=True, default_action_steps=30),
                observations=["state", "point_cloud"], action_representation="skynet.unidex-faas/v1", recording_conversion=conversion),
            batch_compatibility=AdapterBatchCompatibility(allowed_semantics=["per_device"],
                multi_gpu_allowed_semantics=["per_device"], supports_gradient_accumulation=True),
            progress=TrainingProgressContract(unit="epoch", total_path="native.config.epochs", starts_at_zero=True,
                source=TrainingProgressJsonlSource(path="artifacts/logs.json.txt", completed_key="epoch", completed_offset=1,
                    required_key="train_loss", metrics={"train_loss":"train/loss", "val_loss":"validation/loss", "global_step":"training/global_step"})),
        ),
        evaluations=[],
    )
