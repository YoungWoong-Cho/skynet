"""Data/configuration bridge to the pinned, unmodified EgoVerse trainer.

This module is frozen into each run capsule. Models, optimization, normalization,
and checkpoints use EgoVerse and Lightning; no alternate training loop lives here.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    from .egoverse_models import model_algorithm, NATIVE_TARGETS, RECORDING_ACTION_STEPS
    from .egoverse_splits import OVERFIT_MODE, validate_split
except ImportError:  # Frozen run capsule modules live beside the entrypoint.
    from egoverse_models import model_algorithm, NATIVE_TARGETS, RECORDING_ACTION_STEPS
    from egoverse_splits import OVERFIT_MODE, validate_split

REVISION = "e17cf98fe4bc234c564b37abc9e155f25e76d566"
JOINT_CONTRACT = "skynet.egoverse-rgb-joints/v1"
JOINT_MODELS = {"act", "hpt_joints"}
CAMERAS = ["scene_front", "scene_left", "scene_right"]
HPT_JOINT_STATE_KEY = "state_joint_positions"
DATASET_FORMAT = "skynet.recording-dataset/v1"
NATIVE_DATASET_FORMAT = "egoverse-episodes-zarr/v1"


def register_joint_domain():
    from enum import Enum
    from egomimic.rldb.embodiment import embodiment as module

    if "SKYNET_JOINTS" not in module.EMBODIMENT.__members__:
        module.EMBODIMENT = Enum(
            "EMBODIMENT",
            {
                **{k: v.value for k, v in module.EMBODIMENT.__members__.items()},
                "SKYNET_JOINTS": 100,
            },
        )
        module.EMBODIMENT_ID_TO_KEY[100] = "SKYNET_JOINTS"


def joint_keymap(horizon=100, norm_mode=False):
    result = {
        "joint_positions": dict(key_type="proprio_keys", zarr_key="joint_positions"),
        "actions_joints": dict(
            key_type="action_keys", zarr_key="actions_joints", horizon=horizon
        ),
    }
    if not norm_mode:
        result.update({k: dict(key_type="camera_keys", zarr_key=k) for k in CAMERAS})
    return result


class ACTDataInterface:
    """Translate the pre-MultiDataset ACT API without normalizing data twice."""

    def __init__(self, norm_stats):
        self.stats = norm_stats
        self.embodiments = norm_stats.embodiments
        self.norm_stats = norm_stats.norm_stats

    def keys_of_type(self, kind):
        return self.stats.keys_of_type(kind, next(iter(self.embodiments)))

    def key_shape(self, key, embodiment):
        return self.stats.key_shape(key, embodiment)

    def lerobot_key_to_keyname(self, key, embodiment):
        return self.stats.zarr_key_to_keyname(key, embodiment)

    def normalize_data(self, data, embodiment):
        # Current native MultiDataset.__getitem__ already applies these stats.
        return data

    def unnormalize_data(self, data, embodiment):
        return self.stats.unnormalize(data, embodiment)


def make_act(norm_stats, **kwargs):
    from egomimic.algo.act import ACT

    return ACT(data_schematic=ACTDataInterface(norm_stats), **kwargs)


def joint_data(root, batch, workers, horizon=100, reject_outliers=True, *, overfit=False, validation=True, manifest_sha=None, control_hz=None):
    datasets = {}
    for split in ("train", "validation"):
        datasets[split] = {
            "skynet_joints": {
                "_target_": "egoverse_data.JointDataset._from_resolver",
                "resolver": {
                    "_target_": "egoverse_data.RecordingResolver",
                    "root": str(root),
                    "manifest_sha": manifest_sha,
                    "control_hz": control_hz,
                    "split": "train" if overfit else split,
                    "key_map": {
                        "_target_": "egoverse_runtime.joint_keymap",
                        "horizon": horizon,
                    },
                },
                "mode": "total",
                "reject_outliers": reject_outliers if split == "train" else False,
            }
        }
    return {
        "_target_": "egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper",
        "train_datasets": datasets["train"],
        "valid_datasets": datasets["validation"] if validation else {},
        **{
            k: ({"skynet_joints": {"batch_size": batch, "num_workers": workers}}
                if validation or k == "train_dataloader_params" else {})
            for k in ("train_dataloader_params", "valid_dataloader_params")
        },
    }


def map_joint_model(model, dimension):
    """Keep native HPT architecture; bind the actual joint/camera domain."""
    m = model["robomimic_model"]
    domain = "skynet_joints"
    renames = {
        "front_img_1": "scene_front",
        "left_wrist_img": "scene_left",
        "right_wrist_img": "scene_right",
        # HPT._robomimic_to_hpt_data prefixes proprioceptive keys with state_.
        "state_ee_pose": HPT_JOINT_STATE_KEY,
    }
    m["6dof"] = False
    m["domains"] = [domain]
    m["ac_keys"] = {domain: "actions_joints"}
    m["shared_obs_keys"] = [renames[k] for k in m["shared_obs_keys"]]
    for key in ("shared_stem_specs", "encoder_specs"):
        m[key] = {renames[k]: v for k, v in m[key].items()}
    stems = {renames[k]: v for k, v in m["stem_specs"]["eva_bimanual"].items()}
    stems[HPT_JOINT_STATE_KEY]["input_dim"] = dimension
    m["stem_specs"] = {domain: stems}
    head = m["head_specs"]["eva_bimanual"]
    head["infer_ac_dims"] = {domain: dimension}
    head["model"]["act_dim"] = dimension
    m["head_specs"] = {domain: head}
    return model


def validate_hpt_joint_inputs(model, *, checkpoint=False):
    """Reject the old mapping, which native HPT silently skips as missing."""
    stems = model.get("robomimic_model", {}).get("stem_specs", {}).get("skynet_joints", {})
    if HPT_JOINT_STATE_KEY not in stems or "joint_positions" in stems:
        subject = "Checkpoint" if checkpoint else "HPT recorded-joint configuration"
        raise ValueError(
            f"{subject} has an incompatible joint-input mapping. "
            "Use the latest EgoVerse HPT adapter and start a fresh training run; "
            "the previous mapping omitted joint-state inputs."
        )


def validate_episode_split(root, manifest):
    episodes, split = manifest["episodes"], manifest["split"]
    validate_split(split, len(episodes))
    identities = [ep.get("id") for ep in episodes]
    if any(not isinstance(value, str) or not value for value in identities) or len(set(identities)) != len(identities):
        raise ValueError("Recording episodes must have distinct stable identities")


def validate_native_episode_split(root, manifest):
    root = Path(root).resolve()
    episodes, split = manifest["episodes"], manifest["split"]
    validate_split(split, len(episodes))
    paths = [(root / episode["path"]).resolve() for episode in episodes]
    if len(set(paths)) != len(paths) or any(not p.is_relative_to(root) for p in paths):
        raise ValueError("Native episodes must have distinct paths inside the registered dataset")


def validate_manifest(root, sha, model):
    model_algorithm(model)
    if model not in JOINT_MODELS:
        # External native robot/human datasets are a separate input contract,
        # not compatibility for retired collection conversion output.
        try:
            from ops.datasets.artifacts import verify
        except ImportError:
            from artifacts import verify
        manifest = verify(Path(root), sha)
        if manifest.get("format") != NATIVE_DATASET_FORMAT or manifest.get("contract") != f"egoverse.native-{model}/v1":
            raise ValueError("External native dataset contract does not match the selected EgoVerse model")
        validate_native_episode_split(root, manifest)
        return manifest
    try:
        from .recording_dataset import verify_dataset
    except ImportError:
        from recording_dataset import verify_dataset
    manifest = verify_dataset(root, sha)
    if manifest.get("format") != DATASET_FORMAT:
        raise ValueError("Convert the recordings again with the current adapter; old dataset formats are no longer supported")
    validate_episode_split(root, manifest)
    if manifest.get("contract") != JOINT_CONTRACT or model not in JOINT_MODELS:
        raise ValueError("This recording reader supports native EgoVerse ACT and HPT recorded-joint presets only")
    for episode in manifest["episodes"]:
        if not {"state", "action", *CAMERAS}.issubset(episode["streams"]):
            raise ValueError("EgoVerse requires joint state/actions and three aligned scene RGB views")
        streams, steps = episode["streams"], episode["steps"]
        state_shape, action_shape = streams["state"]["shape"], streams["action"]["shape"]
        if len(state_shape) != 2 or state_shape != action_shape or state_shape[0] != steps:
            raise ValueError("EgoVerse joint state and action streams must have identical [frame,joint] shape")
        order = episode.get("policy_to_source_indices", manifest.get("policy_to_source_indices"))
        if not isinstance(order, list) or sorted(order) != list(range(state_shape[1])):
            raise ValueError("EgoVerse joint mapping must cover each source joint exactly once")
        for camera in CAMERAS:
            shape = streams[camera]["shape"]
            if len(shape) != 4 or shape[0] != steps or shape[-1] != 3 or streams[camera]["dtype"] != "uint8":
                raise ValueError("EgoVerse requires aligned uint8 RGB camera streams")
    return manifest


def validate_checkpoint_receipt(receipt, model, manifest_sha, *, sampling=None):
    if model in JOINT_MODELS and receipt.get("dataset_format") != DATASET_FORMAT:
        raise ValueError("This checkpoint predates the shared recording dataset format. Its history and weights are preserved; start a new run with the current adapter. Old-checkpoint resume/evaluation is not supported.")
    if receipt.get("model") != model or receipt.get("manifest_sha256") != manifest_sha:
        raise ValueError("Checkpoint model or dataset differs from the selected native configuration")
    if sampling is not None and receipt.get("sampling") != sampling:
        raise ValueError("Checkpoint training frequency or action chunk differs from this experiment. Start a new run; changed sampling cannot resume or evaluate this checkpoint.")


def experiment_sampling(manifest, model, *, control_hz=None, action_steps=None):
    """Skynet experiment sampling; source streams and native padding stay unchanged."""
    if model not in JOINT_MODELS:
        if control_hz is not None or action_steps is not None:
            raise ValueError("Training frequency and action chunk controls apply to shared recording datasets only. External native EgoVerse Zarr datasets keep their native reader configuration.")
        return None
    try:
        from .recording_time import resolve_sampling
    except ImportError:
        from recording_time import resolve_sampling
    return resolve_sampling(
        manifest, control_hz=control_hz,
        action_steps=RECORDING_ACTION_STEPS[model] if action_steps is None else action_steps,
        window_policy="pad",
    )


def reject_sampling_overrides(overrides, model):
    """There is exactly one experiment control for the output action horizon."""
    if model not in JOINT_MODELS:
        return
    def paths(value, prefix=""):
        for key, child in value.items():
            path = (prefix + "." if prefix else "") + key
            yield path
            if isinstance(child, dict):
                yield from paths(child, path)
    for path in paths(overrides):
        act_horizon = path in {"robomimic_model.chunk_size", "robomimic_model.style_encoder.act_len"}
        hpt_horizon = path.startswith("robomimic_model.head_specs.") and path.endswith((".action_horizon", ".model.act_seq"))
        if act_horizon or hpt_horizon:
            raise ValueError("Set Action chunk with native.config.action_steps; model_overrides must not define a second action horizon")


def apply_recording_horizon(model, name, horizon):
    """Wire one chunk value to the native model's dependent output dimensions."""
    native = model["robomimic_model"]
    if name == "act":
        native["chunk_size"] = horizon
        # Keep resolved configurations and the original ACT YAML interpolation aligned.
        native["style_encoder"]["act_len"] = horizon
    else:
        head = native["head_specs"]["skynet_joints"]
        head["action_horizon"] = horizon
        head["model"]["act_seq"] = horizon


def model_settings(value):
    overrides = json.loads(value)
    if not isinstance(overrides, dict):
        raise ValueError("Model overrides must be a JSON object")

    def changes_target(item):
        if isinstance(item, dict):
            return any("_target_" in key or changes_target(child) for key, child in item.items())
        if isinstance(item, list):
            return any(changes_target(child) for child in item)
        return False

    for key, value in overrides.items():
        if (
            not key.startswith(("robomimic_model.", "optimizer.", "scheduler."))
            or "_target_" in key
            or changes_target(value)
        ):
            raise ValueError("Use native model, optimizer or scheduler settings without replacing architecture targets")
    return overrides


def build_config(args, manifest):
    algorithm = model_algorithm(args.model)
    if getattr(args, "algorithm", None) not in {None, algorithm}:
        raise ValueError("Model preset does not belong to the selected native EgoVerse algorithm")
    sampling = experiment_sampling(manifest, args.model, control_hz=getattr(args, "control_hz", None), action_steps=getattr(args, "action_steps", None))
    overrides = model_settings(args.model_overrides)
    reject_sampling_overrides(overrides, args.model)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    native_model = (
        "hpt_bc_flow_eva" if args.model == "hpt_joints" else args.model
    )
    with initialize_config_dir(
        config_dir=str(Path(args.repository) / "egomimic/hydra_configs"),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "hydra/launcher=basic",
                "model=" + native_model,
                *(
                    ["trainer=ddp_pi", "evaluator=eval_pi"]
                    if args.model.startswith("pi")
                    else []
                ),
            ],
        )
    if cfg.model.robomimic_model._target_ != NATIVE_TARGETS[algorithm]:
        raise ValueError("The selected preset does not use its declared native EgoVerse algorithm")
    with open_dict(cfg):
        cfg.paths.output_dir = args.output
        cfg.paths.work_dir = args.repository
        cfg.paths.root_dir = args.repository
        cfg.norm_stats.save_cache_dir = args.output
        cfg.seed = args.seed
        cfg.launch_params.gpus_per_node = args.gpu_count
        cfg.launch_params.nodes = 1
        cfg.trainer.devices = args.gpu_count
        cfg.trainer.num_nodes = 1
        # Native ModelWrapper uses distributed barriers even with one GPU.
        # HPT has optional heads/encoders with unused parameters on a given batch.
        cfg.trainer.strategy = (
            "ddp" if args.model == "act" else "ddp_find_unused_parameters_true"
        )
        cfg.trainer.max_epochs = args.epochs
        cfg.trainer.min_epochs = args.epochs
        cfg.trainer.accumulate_grad_batches = args.gradient_accumulation
        if args.precision is not None:
            cfg.trainer.precision = {
                "bf16": "bf16-mixed",
                "fp16": "16-mixed",
                "fp32": "32-true",
            }[args.precision]
        cfg.trainer.check_val_every_n_epoch = args.validation_every
        cfg.trainer.default_root_dir = args.output
        if args.learning_rate is not None:
            cfg.model.optimizer.lr = args.learning_rate
        cfg.ckpt_path = args.checkpoint
        cfg.logger = None  # Skynet publishes the native metrics with its pinned tracking identity.
        if args.model in JOINT_MODELS:
            if manifest.get("format") != DATASET_FORMAT or manifest.get("contract") != JOINT_CONTRACT or args.model not in JOINT_MODELS:
                raise ValueError("Select a newly converted recording dataset and a recorded-joint EgoVerse preset")
            cfg.data = joint_data(
                args.dataset, args.batch_size, args.num_workers,
                reject_outliers=args.reject_outliers,
                overfit=manifest["split"].get("mode") == OVERFIT_MODE,
                validation=bool(manifest["split"]["validation"]),
                manifest_sha=args.manifest_sha,
                control_hz=sampling["control_hz"],
                horizon=sampling["action_steps"],
            )
            if args.model == "act":
                cfg.model.robomimic_model._target_ = "egoverse_runtime.make_act"
            else:
                cfg.model = map_joint_model(
                    OmegaConf.to_container(cfg.model, resolve=True),
                    len(manifest["policy_to_source_indices"]),
                )
            cfg.evaluator = {"_target_": "egoverse_runtime.JointEvaluator"}
        else:
            if manifest.get("format") != NATIVE_DATASET_FORMAT or manifest.get("contract") != f"egoverse.native-{args.model}/v1":
                raise ValueError("Select the model's externally imported native dataset")
            # Imported native data supplies its verified native reader configuration.
            cfg.data = OmegaConf.load(Path(args.dataset) / "data.yaml")
            evaluator_path = Path(args.dataset) / "evaluator.yaml"
            if (
                "evaluator.yaml" not in manifest["files"]
                or "data.yaml" not in manifest["files"]
            ):
                raise ValueError(
                    "Native data must include verified data.yaml and evaluator.yaml configurations"
                )
            cfg.evaluator = OmegaConf.load(evaluator_path)
            if not str(cfg.evaluator.get("_target_", "")).startswith("egomimic.eval."):
                raise ValueError(
                    "Use the model's native EgoVerse evaluator configuration"
                )
            root = Path(args.dataset).resolve()
            validate_native_episode_split(root, manifest)
            for partition in ("train_datasets", "valid_datasets"):
                actual_episodes = set()
                for data_config in cfg.data[partition].values():
                    if (
                        data_config.resolver._target_
                        != "egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver"
                    ):
                        raise ValueError(
                            "Registered EgoVerse datasets must use local, verified episode files"
                        )
                    folder = Path(data_config.resolver.folder_path)
                    folder = (
                        (root / folder).resolve()
                        if not folder.is_absolute()
                        else folder.resolve()
                    )
                    if not folder.is_relative_to(root) or not folder.is_dir():
                        raise ValueError(
                            "Native dataset folders must stay inside the registered dataset"
                        )
                    actual_episodes.update(
                        str(p.resolve()) for p in folder.iterdir() if p.is_dir()
                    )
                    data_config.resolver.folder_path = str(folder)
                    data_config.mode = "total"
                    if partition == "valid_datasets":
                        data_config._target_ = (
                            "egoverse_data.JointDataset._from_resolver"
                        )
                        data_config.reject_outliers = False
                split = "train" if partition == "train_datasets" else "validation"
                expected_episodes = {
                    str((root / manifest["episodes"][i]["path"]).resolve())
                    for i in manifest["split"][split]
                }
                if actual_episodes != expected_episodes:
                    raise ValueError(
                        f"Native {split} folders differ from the registered episode split"
                    )
            for key in ("train_dataloader_params", "valid_dataloader_params"):
                for params in cfg.data[key].values():
                    params.batch_size = args.batch_size
                    params.num_workers = args.num_workers
        cfg.callbacks.skynet_progress = {
            "_target_": "egoverse_runtime.progress_callback"
        }
        if args.train_batches is not None:
            cfg.trainer.limit_train_batches = args.train_batches
        if args.validation_batches is not None:
            cfg.trainer.limit_val_batches = args.validation_batches
        if not manifest["split"]["validation"]:
            cfg.trainer.limit_val_batches = 0
            cfg.trainer.num_sanity_val_steps = 0
            cfg.data.valid_datasets = {}
            cfg.data.valid_dataloader_params = {}
        if args.weights:
            if not Path(args.weights).is_dir():
                raise ValueError(
                    "π0.5 pretrained weights are missing from the selected cluster directory"
                )
            cfg.model.robomimic_model.config.pytorch_weight_path = args.weights
        for key, value in overrides.items():
            OmegaConf.update(cfg.model, key, value, merge=False)
        expected_target = (
            "egoverse_runtime.make_act"
            if algorithm == "act" and manifest["contract"] == JOINT_CONTRACT
            else NATIVE_TARGETS[algorithm]
        )
        if cfg.model.robomimic_model._target_ != expected_target:
            raise ValueError("Model settings changed the selected native EgoVerse algorithm")
        if manifest["contract"] == JOINT_CONTRACT:
            if args.model == "hpt_joints":
                validate_hpt_joint_inputs(cfg.model)
            horizon = sampling["action_steps"]
            apply_recording_horizon(cfg.model, args.model, horizon)
            for split in ("train_datasets", "valid_datasets"):
                if "skynet_joints" in cfg.data[split]:
                    cfg.data[split].skynet_joints.resolver.key_map.horizon = horizon
    return cfg


class JointEvaluator:
    """Held-out raw joint prediction error; no simulated success is implied."""

    override_dict = {}

    def on_validation_start(self):
        pass

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        import torch

        pred = self.model.forward_eval(batch)
        if hasattr(self.model, "data_schematic"):
            expected = self.model.data_schematic.unnormalize_data(batch, 100)[
                "actions_joints"
            ]
            actual = pred["actions_joints"]
        else:
            actual = pred["skynet_joints_actions_joints"]
            expected = self.model.norm_stats.unnormalize(batch[100], 100)[
                "actions_joints"
            ]
        loss = torch.mean((actual - expected) ** 2)
        self.trainer.lightning_module.log(
            "Validation/joint_mse", loss, on_step=False, on_epoch=True, sync_dist=True
        )

    def on_validation_end(self):
        pass


def progress_callback():
    from lightning.pytorch.callbacks import Callback

    class Progress(Callback):
        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(Path(trainer.default_root_dir) / "native-config.yaml")
            receipt = json.loads(
                (Path(trainer.default_root_dir) / "dataset-receipt.json").read_text()
            )
            checkpoint["skynet"] = {
                **receipt,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }

        def on_train_epoch_end(self, trainer, pl_module):
            if not trainer.is_global_zero:
                return
            metrics = {
                k: float(v.detach().cpu())
                for k, v in trainer.callback_metrics.items()
                if hasattr(v, "numel") and v.numel() == 1
            }
            record = {
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "train_loss": metrics.get("Train/Loss"),
                "metrics": metrics,
            }
            for key, value in metrics.items():
                if key.startswith("Train/") and "loss" in key.lower():
                    record["train_loss"] = value
                    break
            record["val_loss"] = metrics.get("Validation/joint_mse")
            with (Path(trainer.default_root_dir) / "logs.json.txt").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")

        def on_train_end(self, trainer, pl_module):
            trainer.save_checkpoint(
                str(Path(trainer.default_root_dir) / "checkpoints/last.ckpt")
            )

    return Progress()


def parser():
    p = argparse.ArgumentParser()
    for key in ("repository", "dataset", "manifest-sha", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--model", default="act")
    p.add_argument("--algorithm", choices=["act", "hpt", "pi"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--validation-every", type=int, default=200)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--checkpoint")
    p.add_argument("--weights")
    p.add_argument("--model-overrides", default="{}")
    p.add_argument("--control-hz", type=float)
    p.add_argument("--action-steps", type=int)
    p.add_argument(
        "--reject-outliers",
        type=lambda v: {"true": True, "false": False}[v.lower()],
        default=True,
    )
    p.add_argument("--train-batches", type=int)
    p.add_argument("--validation-batches", type=int)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--config-only", action="store_true")
    return p


def main():
    args = parser().parse_args()
    model_algorithm(args.model)
    actual = subprocess.check_output(
        ["git", "-C", args.repository, "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != args.revision or actual != REVISION:
        raise ValueError("EgoVerse repository revision differs from the pinned adapter")
    sys.path.insert(0, args.repository)
    os.chdir(args.repository)
    manifest = validate_manifest(args.dataset, args.manifest_sha, args.model)
    if args.verify_only:
        # Conversion verifies physical source streams, not experiment horizons.
        # No model window, resampling or minimum chunk length belongs here.
        print(json.dumps({
            "schema": "skynet.egoverse-loader-validation/v1",
            "manifest_sha256": args.manifest_sha, "observation_mode": "rgb",
        }))
        return
    sampling = experiment_sampling(manifest, args.model, control_hz=args.control_hz, action_steps=args.action_steps)
    register_joint_domain()
    # Import the native entrypoint before resolving its registered Hydra resolvers.
    from egomimic.trainHydra import train

    cfg = build_config(args, manifest)
    if args.checkpoint:
        import torch
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        receipt = saved.get("skynet") or {}
        validate_checkpoint_receipt(receipt, args.model, args.manifest_sha, sampling=sampling)
        if args.model == "hpt_joints":
            validate_hpt_joint_inputs(receipt.get("config", {}).get("model", {}), checkpoint=True)
    from omegaconf import OmegaConf

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        OmegaConf.save(cfg, output / "native-config.yaml", resolve=True)
        (output / "dataset-receipt.json").write_text(
            json.dumps(
                dict(
                    manifest_sha256=args.manifest_sha,
                    dataset_format=manifest["format"],
                    model=args.model,
                    revision=REVISION,
                    sampling=sampling,
                )
            )
        )
    if args.config_only:
        return
    # Skynet owns Slurm retries; Lightning owns the processes inside this allocation.
    # A one-task sbatch allocation must not be interpreted as one DDP worker.
    for key in list(os.environ):
        if key.startswith("SLURM_"):
            os.environ.pop(key)
    train(cfg)


if __name__ == "__main__":
    # Hydra must be able to resolve the same module in Lightning worker processes.
    sys.modules.setdefault("egoverse_runtime", sys.modules[__name__])
    main()
