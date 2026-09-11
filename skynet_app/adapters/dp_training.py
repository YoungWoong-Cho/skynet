"""Frozen DP runtime: shared state/RGB loading and one explicit training loop.

Uses the pinned XPolicyLab encoder/U-Net and DDPM implementation. Actions start
at the current observation, so a chunk of 16 means 16 future commands, even
when the observation history is longer than one frame.
"""

import argparse
import copy
from contextlib import nullcontext
import json
import math
import subprocess
import sys
from pathlib import Path

CAMERAS = {
    "head_cam": "head_camera",
    "left_cam": "left_camera",
    "right_cam": "right_camera",
}


def configure_repository(repository):
    root = Path(repository).resolve() / "policy/DP"
    if not (root / "diffusion_policy/workspace/robotworkspace.py").is_file():
        raise ValueError("Select the pinned XPolicyLab repository with policy/DP")
    sys.path.insert(0, str(root))
    return root


class RecordedDataset:
    """Spawn-safe dataset; load only observation frames that the policy uses."""
    def __init__(self, zarr_path, manifest_path, observation_mode="state", n_obs_steps=2, horizon=16, **_):
        import numpy as np
        import zarr
        from diffusion_policy.common.replay_buffer import ReplayBuffer
        group = zarr.open_group(str(zarr_path), mode="r")
        self.replay_buffer = ReplayBuffer(group)
        self.keys = ["state", "action", *(CAMERAS.values() if observation_mode == "rgb" else [])]
        for key in self.keys:
            if key not in group["data"]:
                raise ValueError("Missing required observation: " + key)
        self.n_obs_steps, self.horizon = n_obs_steps, horizon
        self.sequence_length = horizon + n_obs_steps - 1
        manifest = json.loads(Path(manifest_path).read_text())
        split = manifest["split"]
        count = self.replay_buffer.n_episodes
        if (sorted(split["train"] + split["validation"]) != list(range(count))
                or not split["train"]):
            raise ValueError("Training requires disjoint, complete train/validation episode splits")
        self.train_mask = np.zeros(count, dtype=bool)
        self.train_mask[split["train"]] = True
        self.sampler = self.make_sampler(self.train_mask)

    def make_sampler(self, mask):
        from diffusion_policy.common.sampler import SequenceSampler
        return SequenceSampler(self.replay_buffer, self.sequence_length, self.n_obs_steps - 1,
            self.horizon - 1, episode_mask=mask, keys=["state", "action"])

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.sampler = self.make_sampler(~self.train_mask)
        return result

    def __len__(self):
        return len(self.sampler)

    def sample(self, index):
        import numpy as np
        result = self.sampler.sample_sequence(index)
        result["state"] = result["state"][:self.n_obs_steps]
        start, end, pad_before, _ = self.sampler.indices[index]
        for key in CAMERAS.values():
            if key in self.keys:
                needed = max(1, min(end - start, self.n_obs_steps - pad_before))
                frames = self.replay_buffer[key][start:start + needed]
                positions = np.clip(np.arange(self.n_obs_steps) - pad_before, 0, len(frames) - 1)
                result[key] = frames[positions]
        return result

    def __getitem__(self, index):
        import numpy as np
        import torch
        if isinstance(index, (int, np.integer)):
            return {k: torch.from_numpy(v.copy()) for k, v in self.sample(int(index)).items()}
        samples = [self.sample(int(i)) for i in index]
        return {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in samples[0]}

    def postprocess(self, batch, device):
        obs = {"agent_pos": batch["state"][:, :self.n_obs_steps].to(device, non_blocking=True).float()}
        for name, key in CAMERAS.items():
            if key in self.keys:
                obs[name] = batch[key][:, :self.n_obs_steps].to(device, non_blocking=True).float() / 255.0
        start = self.n_obs_steps - 1
        return {"obs": obs, "action": batch["action"][:, start:start + self.horizon].to(device, non_blocking=True).float()}

    def get_normalizer(self, mode="limits", **kwargs):
        import numpy as np
        from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
        ends = self.replay_buffer.episode_ends[:]
        starts = np.r_[0, ends[:-1]]
        data = {out: np.concatenate([self.replay_buffer[key][int(starts[i]):int(ends[i])]
            for i in np.flatnonzero(self.train_mask)]) for key, out in [("state", "agent_pos"), ("action", "action")]}
        normalizer = LinearNormalizer()
        normalizer.fit(data, last_n_dims=1, mode=mode, **kwargs)
        for name, key in CAMERAS.items():
            if key in self.keys:
                normalizer[name] = SingleFieldLinearNormalizer.create_identity()
        return normalizer


def StateEncoder(dimension):
    import torch.nn as nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(dimension, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()
            )

        def output_shape(self):
            return (128,)

        def forward(self, obs):
            return self.layers(obs["agent_pos"])

    return Encoder()


def FutureActionPolicy(**kwargs):
    from diffusion_policy.policy.diffusion_unet_image_policy import (
        DiffusionUnetImagePolicy,
    )

    class Policy(DiffusionUnetImagePolicy):
        def predict_action(self, obs_dict):
            result = super().predict_action(obs_dict)
            result["action"] = result["action_pred"][:, : self.n_action_steps]
            return result

    return Policy(**kwargs)


def workspace_class():
    import torch
    from diffusion_policy.common.json_logger import JsonLogger
    from diffusion_policy.common.pytorch_util import optimizer_to
    from diffusion_policy.model.common.lr_scheduler import get_scheduler
    from diffusion_policy.workspace.robotworkspace import RobotWorkspace
    from hydra.utils import instantiate
    from training_parallel import PolicyLoss

    class TrainingWorkspace(RobotWorkspace):
        def save_checkpoint(self, path=None, **kwargs):
            kwargs["use_thread"] = False
            return super().save_checkpoint(
                Path(self.output_dir)
                / "checkpoints"
                / (Path(path).name if path else "latest.ckpt"),
                **kwargs,
            )

        def run(self, context):
            cfg = self.cfg
            dataset = instantiate(cfg.task.dataset)
            normalizer = dataset.get_normalizer()
            device = torch.device(cfg.training.device)
            for model in (self.model, self.ema_model):
                model.set_normalizer(normalizer)
                model.to(device)
            optimizer_to(self.optimizer, device)
            train_loss = context.wrap(PolicyLoss(self.model, "compute_loss"))
            validation_loss = PolicyLoss(self.ema_model, "compute_loss")
            torch.manual_seed(cfg.training.seed + context.rank)
            train = context.loader(dataset, cfg.dataloader.batch_size,
                cfg.dataloader.num_workers, shuffle=True, seed=cfg.training.seed)
            validation_dataset = dataset.get_validation_dataset()
            validation = (context.loader(validation_dataset,
                cfg.val_dataloader.batch_size, cfg.val_dataloader.num_workers)
                if len(validation_dataset) else None)
            accumulation = cfg.training.gradient_accumulate_every
            batches = min(len(train), cfg.training.max_train_steps or len(train))
            updates = math.ceil(batches / accumulation) * cfg.training.num_epochs
            scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                num_training_steps=updates,
            )
            best = float("inf")
            self.stop_reason = "max_epochs"
            with (JsonLogger(str(Path(self.output_dir) / "logs.json.txt")) if context.primary else nullcontext()) as logger:
                for epoch in range(cfg.training.num_epochs):
                    self.epoch = epoch
                    train.batch_sampler.set_epoch(epoch)
                    self.model.train()
                    self.optimizer.zero_grad(set_to_none=True)
                    loss_sum, examples = 0.0, 0
                    window_examples = window_expected = 0
                    for index, raw in enumerate(train):
                        batch = dataset.postprocess(raw, device)
                        size = train.batch_sampler.valid_count(index)
                        loss = train_loss(batch)[0]["loss"]
                        context.check_loss(loss)
                        context.backward(loss, size)
                        window_examples += size
                        window_expected += train.batch_sampler.global_count(index)
                        loss_sum += loss.item() * size
                        examples += size
                        if (index + 1) % accumulation == 0 or index + 1 == batches:
                            context.normalize_gradients(self.model.parameters(), window_examples, window_expected)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                cfg.training.gradient_clip,
                                error_if_nonfinite=True,
                            )
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            scheduler.step()
                            self.global_step += 1
                            window_examples = window_expected = 0
                            with torch.no_grad():
                                for target, source in zip(
                                    self.ema_model.parameters(), self.model.parameters()
                                ):
                                    target.lerp_(source, 1 - cfg.training.ema_decay)
                                for target, source in zip(
                                    self.ema_model.buffers(), self.model.buffers()
                                ):
                                    target.copy_(source)
                        if index + 1 >= batches:
                            break
                    val_loss = None
                    if validation is not None:
                        self.ema_model.eval()
                        val_sum, val_examples = 0.0, 0
                        # Keep diffusion noise fixed across epochs for comparable held-out loss.
                        with torch.random.fork_rng(devices=[context.local_rank]), torch.no_grad():
                            torch.manual_seed(cfg.training.seed + context.rank + 1)
                            for index, raw in enumerate(validation):
                                batch = dataset.postprocess(raw, device)
                                loss = validation_loss(batch)[0]["loss"]
                                context.check_loss(loss)
                                count = validation.batch_sampler.valid_count(index)
                                val_sum += loss.item() * count
                                val_examples += count
                                if (
                                    cfg.training.max_val_steps
                                    and index + 1 >= cfg.training.max_val_steps
                                ):
                                    break
                        val_loss = context.mean(val_sum, val_examples,
                            min(len(validation.dataset), (cfg.training.max_val_steps or len(validation)) * cfg.val_dataloader.batch_size))
                    epoch_loss = context.mean(loss_sum, examples, min(len(dataset), batches * cfg.dataloader.batch_size))
                    improved = val_loss is not None and val_loss < best
                    if improved:
                        best = val_loss
                    record = dict(
                        epoch=epoch,
                        global_step=self.global_step,
                        train_loss=epoch_loss,
                        lr=self.optimizer.param_groups[0]["lr"],
                        **({"val_loss": val_loss, "best_val_loss": best} if val_loss is not None else {}),
                    )
                    if context.primary:
                        logger.log(record)
                        print(json.dumps(record), flush=True)
                    self.epoch = epoch + 1
                    if context.primary:
                        if improved:
                            self.save_checkpoint("best.ckpt")
                        self.save_checkpoint()

    return TrainingWorkspace


def arguments():
    parser = argparse.ArgumentParser()
    for key in ("revision", "dataset", "manifest-sha", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--observation-mode", choices=["state", "rgb"], default="state")
    parser.add_argument("--training-preset", default="dexverse-state/v1")
    parser.add_argument(
        "--lr-schedule", choices=["constant", "cosine"], default="constant"
    )
    for key, default in {
        "batch-size": 256,
        "gpu-count": 1,
        "epochs": 300,
        "seed": 42,
        "observation-steps": 2,
        "action-steps": 16,
        "diffusion-steps": 100,
        "inference-steps": 20,
        "gradient-accumulation": 1,
        "num-workers": 0,
        "warmup-steps": 0,
    }.items():
        parser.add_argument("--" + key, type=int, default=default)
    for key, default in {
        "learning-rate": 1e-4,
        "weight-decay": 1e-4,
        "gradient-clip": 1.0,
        "ema-decay": 0.995,
    }.items():
        parser.add_argument("--" + key, type=float, default=default)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--max-train-steps",
        type=int,
        help="Bound each epoch for isolated runtime smoke checks",
    )
    parser.add_argument(
        "--batch-semantics",
        default="per_device",
        choices=[
            "per_device",
            "global_before_accumulation",
            "global_effective",
            "repository_native",
        ],
    )
    args = parser.parse_args()
    if any(
        getattr(args, k) < 1
        for k in [
            "batch_size",
            "gpu_count",
            "epochs",
            "observation_steps",
            "action_steps",
            "diffusion_steps",
            "inference_steps",
            "gradient_accumulation",
        ]
    ):
        parser.error("Counts must be positive")
    if (
        args.action_steps not in [4, 8, 16, 32, 64]
        or args.inference_steps > args.diffusion_steps
    ):
        parser.error(
            "Action chunk must be 4/8/16/32/64; inference steps cannot exceed training diffusion steps"
        )
    if (
        args.num_workers < 0
        or args.warmup_steps < 0
        or not 0 <= args.ema_decay < 1
        or args.gradient_clip <= 0
    ):
        parser.error("Invalid worker, warmup, EMA, or clipping setting")
    return args


def main():
    args = arguments()
    from training_parallel import launch_distributed, TrainingContext
    if not args.verify_only and launch_distributed(args.gpu_count):
        return
    context = TrainingContext(args.gpu_count) if not args.verify_only else None
    if context is not None:
        args.device = str(context.device)
    primary = context is None or context.primary
    revision = subprocess.check_output(
        ["git", "-C", args.repository, "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != args.revision:
        raise ValueError("XPolicyLab checkout differs from the pinned adapter revision")
    policy_dir = configure_repository(args.repository)
    import numpy as np
    import torch
    from artifacts import digest, verify
    from training_parallel import training_batch_size
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    root, output = Path(args.dataset), Path(args.output)
    if primary:
        manifest = verify(root, args.manifest_sha)
    if context is not None and context.world_size > 1:
        context.dist.barrier()
    if not primary:
        manifest = json.loads((root / "manifest.json").read_text())
    contracts = [
        "skynet.dp-rgb-joints/v1",
        *(["skynet.dp-joints/v1"] if args.observation_mode == "state" else []),
    ]
    if (
        manifest.get("contract") not in contracts
        or manifest["format"] != "xpolicylab-dp-zarr/v1"
    ):
        raise ValueError("Dataset does not satisfy the selected observation contract")
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(
        config_dir=str(policy_dir / "diffusion_policy/config"), version_base="1.3"
    ):
        cfg = compose(config_name="robot_dp")
    OmegaConf.set_struct(cfg, False)
    dim = len(manifest["policy_to_source_indices"])
    cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps = (
        args.action_steps,
        args.observation_steps,
        args.action_steps,
    )
    cfg.task.shape_meta = {
        "action": {"shape": [dim]},
        "obs": {"agent_pos": {"shape": [dim], "type": "low_dim"}},
    }
    if args.observation_mode == "rgb":
        for key in CAMERAS:
            cfg.task.shape_meta.obs[key] = {"shape": [3, 240, 320], "type": "rgb"}
    else:
        cfg.policy.obs_encoder = {
            "_target_": "skynet_dp_training.StateEncoder",
            "dimension": dim,
        }
    cfg.policy._target_ = "skynet_dp_training.FutureActionPolicy"
    cfg.policy.num_inference_steps = args.inference_steps
    cfg.policy.noise_scheduler.num_train_timesteps = args.diffusion_steps
    cfg.task.name = "registered_demonstrations"
    cfg.task.dataset = {
        "_target_": "skynet_dp_training.RecordedDataset",
        "zarr_path": str(root / "dataset/demonstrations.zarr"),
        "manifest_path": str(root / "manifest.json"),
        "observation_mode": args.observation_mode,
        "n_obs_steps": args.observation_steps,
        "horizon": args.action_steps,
    }
    batch_size = training_batch_size(
        args.batch_size,
        args.gpu_count,
        args.gradient_accumulation,
        args.batch_semantics,
    )
    cfg.dataloader = {
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "shuffle": True,
        "drop_last": False,
    }
    cfg.val_dataloader = {
        "batch_size": min(batch_size, 32 * args.gpu_count),
        "num_workers": args.num_workers,
        "shuffle": False,
        "drop_last": False,
    }
    for key, value in {
        "seed": args.seed,
        "gpu_count": args.gpu_count,
        "device": args.device,
        "num_epochs": args.epochs,
        "resume": False,
        "checkpoint_every": 1,
        "val_every": 1,
        "max_train_steps": args.max_train_steps,
        "max_val_steps": 1 if args.max_train_steps else None,
        "gradient_accumulate_every": args.gradient_accumulation,
        "gradient_clip": args.gradient_clip,
        "ema_decay": args.ema_decay,
        "lr_scheduler": args.lr_schedule,
        "lr_warmup_steps": args.warmup_steps,
        "use_ema": True,
    }.items():
        cfg.training[key] = value
    cfg.ema = {"decay": args.ema_decay, "schedule": "fixed_per_optimizer_update"}
    cfg.optimizer.lr, cfg.optimizer.weight_decay = args.learning_rate, args.weight_decay
    cfg.logging.mode = "disabled"
    cfg.skynet = {
        "training_preset": args.training_preset,
        "effective_batch_size": batch_size * args.gradient_accumulation,
        "action_alignment": "current observation then future commands",
        "image_normalization": (
            "uint8 / 255 then ImageNet once" if args.observation_mode == "rgb" else None
        ),
    }
    OmegaConf.resolve(cfg)
    dataset = RecordedDataset(
        **{k: v for k, v in cfg.task.dataset.items() if k != "_target_"}
    )
    if not len(dataset):
        raise ValueError(
            "Training must contain at least one sequence"
        )
    sample = dataset.postprocess(dataset[np.array([0])], torch.device("cpu"))
    for key, value in sample["obs"].items():
        if (
            tuple(value.shape[2:]) != tuple(cfg.task.shape_meta.obs[key].shape)
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Invalid observations: " + key)
    if (
        tuple(sample["action"].shape) != (1, args.action_steps, dim)
        or not torch.isfinite(sample["action"]).all()
    ):
        raise ValueError("Invalid action sequences")
    dataset.get_normalizer()
    receipt = dict(
        schema="skynet.dp-loader-validation/v2",
        manifest_sha256=args.manifest_sha,
        source_revision=manifest["source_revision"],
        split=manifest["split"],
        train_sequences=len(dataset),
        validation_sequences=len(dataset.get_validation_dataset()),
        observation_shapes={k: list(v.shape) for k, v in sample["obs"].items()},
        action_shape=list(sample["action"].shape),
        normalization="training_episodes_only",
        observation_mode=args.observation_mode,
    )
    if primary:
        (output / "dataset-validation.json").write_text(json.dumps(receipt, indent=2))
        OmegaConf.save(cfg, output / "training-config.yaml")
        (output / "applied-settings.json").write_text(
            json.dumps(
                dict(
                    schema="skynet.applied-settings/v1",
                    settings=vars(args),
                    effective_batch_size=batch_size * args.gradient_accumulation,
                    manifest_sha256=args.manifest_sha,
                    repository_revision=revision,
                ),
                indent=2,
            )
        )
    if args.verify_only:
        print(json.dumps(receipt))
        return
    workspace = workspace_class()(cfg, output_dir=str(output))
    workspace.run(context)
    if not primary:
        context.close()
        return
    checkpoint = workspace.save_checkpoint()
    (output / "training-result.json").write_text(
        json.dumps(
            dict(
                manifest_sha256=args.manifest_sha,
                checkpoint=checkpoint,
                checkpoint_sha256=digest(checkpoint),
                global_step=workspace.global_step,
                epochs=workspace.epoch,
                stop_reason=workspace.stop_reason,
            ),
            indent=2,
        )
    )
    print(json.dumps({"checkpoint": checkpoint, "global_step": workspace.global_step}))
    context.close()


if __name__ == "__main__":
    main()
