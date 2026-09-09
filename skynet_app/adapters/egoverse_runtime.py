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

REVISION = "e17cf98fe4bc234c564b37abc9e155f25e76d566"
JOINT_CONTRACT = "skynet.egoverse-rgb-joints/v1"
JOINT_MODELS = {"act", "hpt_joints", "dp_joints"}
CAMERAS = ["scene_front", "scene_left", "scene_right"]


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


def joint_data(root, batch, workers, horizon=100, reject_outliers=True):
    datasets = {}
    for split in ("train", "validation"):
        datasets[split] = {
            "skynet_joints": {
                "_target_": "egoverse_data.JointDataset._from_resolver",
                "resolver": {
                    "_target_": "egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver",
                    "folder_path": str(Path(root) / "dataset" / split),
                    "key_map": {
                        "_target_": "egoverse_runtime.joint_keymap",
                        "horizon": horizon,
                    },
                    "transform_list": [],
                },
                "mode": "total",
                "reject_outliers": reject_outliers if split == "train" else False,
            }
        }
    return {
        "_target_": "egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper",
        "train_datasets": datasets["train"],
        "valid_datasets": datasets["validation"],
        **{
            k: {"skynet_joints": {"batch_size": batch, "num_workers": workers}}
            for k in ("train_dataloader_params", "valid_dataloader_params")
        },
    }


def map_joint_model(model, dimension, diffusion=False):
    """Keep native HPT architecture; bind the actual joint/camera domain."""
    m = model["robomimic_model"]
    domain = "skynet_joints"
    renames = {
        "front_img_1": "scene_front",
        "left_wrist_img": "scene_left",
        "right_wrist_img": "scene_right",
        "state_ee_pose": "joint_positions",
    }
    m["6dof"] = False
    m["domains"] = [domain]
    m["ac_keys"] = {domain: "actions_joints"}
    m["shared_obs_keys"] = [renames[k] for k in m["shared_obs_keys"]]
    for key in ("shared_stem_specs", "encoder_specs"):
        m[key] = {renames[k]: v for k, v in m[key].items()}
    stems = {renames[k]: v for k, v in m["stem_specs"]["eva_bimanual"].items()}
    stems["joint_positions"]["input_dim"] = dimension
    m["stem_specs"] = {domain: stems}
    head = m["head_specs"]["eva_bimanual"]
    head["infer_ac_dims"] = {domain: dimension}
    head["model"]["act_dim"] = dimension
    if diffusion:
        # EgoVerse supplies the head but no training YAML for it. These wiring
        # values are an explicit Skynet preset, never an upstream/paper preset.
        head = {
            "_target_": "egomimic.models.diffusion_policy.DiffusionPolicy",
            "action_horizon": 100,
            "num_inference_steps": 50,
            "pooling": "flatten",
            "infer_ac_dims": {domain: dimension},
            "noise_scheduler": {
                "_target_": "diffusers.DDPMScheduler",
                "num_train_timesteps": 100,
            },
            "model": {
                "_target_": "egomimic.models.denoising_nets.ConditionalUnet1D",
                "input_dim": dimension,
                "cond_dim": 256,
                "ac_latent_seq": 64,
            },
        }
    m["head_specs"] = {domain: head}
    return model


def validate_episode_split(root, manifest):
    root = Path(root).resolve()
    episodes, split = manifest["episodes"], manifest["split"]
    train, validation = split["train"], split["validation"]
    if (
        not train
        or not validation
        or sorted(train + validation) != list(range(len(episodes)))
    ):
        raise ValueError(
            "Every episode must belong to exactly one training or validation split"
        )
    paths = [(root / episode["path"]).resolve() for episode in episodes]
    if len(set(paths)) != len(paths) or any(not p.is_relative_to(root) for p in paths):
        raise ValueError(
            "Episodes must have distinct paths inside the registered dataset"
        )


def validate_manifest(root, sha, model):
    from artifacts import verify

    manifest = verify(Path(root), sha)
    validate_episode_split(root, manifest)
    if manifest.get("contract") == JOINT_CONTRACT:
        if model not in JOINT_MODELS:
            raise ValueError(
                "This model requires its native robot/human dataset, not recorded joint data"
            )
        if not manifest.get("split", {}).get("validation"):
            raise ValueError("A separate validation episode is required")
    elif manifest.get("contract") != f"egoverse.native-{model}/v1":
        raise ValueError("Dataset contract does not match the selected EgoVerse model")
    return manifest


def build_config(args, manifest):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    native_model = (
        "hpt_bc_flow_eva" if args.model in {"hpt_joints", "dp_joints"} else args.model
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
        if manifest["contract"] == JOINT_CONTRACT:
            cfg.data = joint_data(
                args.dataset,
                args.batch_size,
                args.num_workers,
                reject_outliers=args.reject_outliers,
            )
            if args.model == "act":
                cfg.model.robomimic_model._target_ = "egoverse_runtime.make_act"
            else:
                cfg.model = map_joint_model(
                    OmegaConf.to_container(cfg.model, resolve=True),
                    len(manifest["policy_to_source_indices"]),
                    args.model == "dp_joints",
                )
            cfg.evaluator = {"_target_": "egoverse_runtime.JointEvaluator"}
        else:
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
            validate_episode_split(root, manifest)
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
        if args.weights:
            if not Path(args.weights).is_dir():
                raise ValueError(
                    "π0.5 pretrained weights are missing from the selected cluster directory"
                )
            cfg.model.robomimic_model.config.pytorch_weight_path = args.weights
        overrides = json.loads(args.model_overrides)
        if not isinstance(overrides, dict):
            raise ValueError("Model overrides must be a JSON object")
        for key, value in overrides.items():
            if (
                not key.startswith(("robomimic_model.", "optimizer.", "scheduler."))
                or "_target_" in key
            ):
                raise ValueError(
                    "Use native model, optimizer or scheduler setting paths"
                )
            OmegaConf.update(cfg.model, key, value, merge=False)
        if manifest["contract"] == JOINT_CONTRACT:
            horizon = (
                cfg.model.robomimic_model.chunk_size
                if args.model == "act"
                else cfg.model.robomimic_model.head_specs.skynet_joints.action_horizon
            )
            for split in ("train_datasets", "valid_datasets"):
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
    actual = subprocess.check_output(
        ["git", "-C", args.repository, "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != args.revision or actual != REVISION:
        raise ValueError("EgoVerse repository revision differs from the pinned adapter")
    sys.path.insert(0, args.repository)
    os.chdir(args.repository)
    register_joint_domain()
    # Import the native entrypoint before resolving its registered Hydra resolvers.
    from egomimic.trainHydra import train

    manifest = validate_manifest(args.dataset, args.manifest_sha, args.model)
    cfg = build_config(args, manifest)
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
                    model=args.model,
                    revision=REVISION,
                )
            )
        )
    if args.config_only:
        return
    if args.verify_only:
        import hydra

        for split in ("train_datasets", "valid_datasets"):
            for config in cfg.data[split].values():
                dataset = hydra.utils.instantiate(config)
                if len(dataset) < 1:
                    raise ValueError("Empty " + split)
                dataset[0]
        print(
            json.dumps(
                {
                    "schema": "skynet.egoverse-loader-validation/v1",
                    "manifest_sha256": args.manifest_sha,
                    "observation_mode": "rgb",
                }
            )
        )
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
