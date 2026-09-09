"""Skynet integration for the pinned XPolicyLab DP policy and training loop.

This file is frozen into each run capsule. Dataset selection, normalization and
checkpoint placement are adapted; the upstream model and optimizer loop remain
unchanged. Image batches stream from Zarr instead of copying all RGB into RAM.
"""

import argparse
import json
from pathlib import Path
import sys
import subprocess


def configure_repository(repository):
    root = Path(repository).resolve()
    policy = root / "policy/DP"
    if not (policy / "diffusion_policy/workspace/robotworkspace.py").is_file():
        raise ValueError("Select the pinned XPolicyLab repository with policy/DP")
    sys.path.insert(0, str(policy))
    return policy


# Hydra imports this module by name from the frozen capsule after main has
# established the repository import path.
def dataset_class():
    import numpy as np
    import torch
    import zarr
    from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset
    from diffusion_policy.common.replay_buffer import ReplayBuffer
    from diffusion_policy.common.sampler import SequenceSampler
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusion_policy.common.normalize_util import get_image_range_normalizer

    class RecordedDataset(RobotImageDataset):
        def __init__(
            self,
            zarr_path,
            manifest_path,
            batch_size=1,
            horizon=8,
            pad_before=2,
            pad_after=5,
            **_,
        ):
            self.replay_buffer = ReplayBuffer(zarr.open_group(str(zarr_path), mode="r"))
            manifest = json.loads(Path(manifest_path).read_text())
            split = manifest["split"]
            count = self.replay_buffer.n_episodes
            if (
                sorted(split["train"] + split["validation"]) != list(range(count))
                or not split["train"]
                or not split["validation"]
            ):
                raise ValueError(
                    "DP requires a disjoint, complete training/validation episode split"
                )
            self.train_mask = np.zeros(count, dtype=bool)
            self.train_mask[split["train"]] = True
            self.horizon, self.pad_before, self.pad_after = (
                horizon,
                pad_before,
                pad_after,
            )
            self.batch_size = batch_size
            self.sampler = SequenceSampler(
                self.replay_buffer,
                horizon,
                pad_before,
                pad_after,
                episode_mask=self.train_mask,
            )

        def __getitem__(self, index):
            if isinstance(index, (int, np.integer)):
                return {
                    k: torch.from_numpy(v.copy())
                    for k, v in self.sampler.sample_sequence(int(index)).items()
                }
            samples = [self.sampler.sample_sequence(int(i)) for i in index]
            return {
                k: torch.from_numpy(np.stack([s[k] for s in samples]))
                for k in samples[0]
            }

        def get_normalizer(self, mode="limits", **kwargs):
            ends = self.replay_buffer.episode_ends[:]
            starts = np.r_[0, ends[:-1]]
            data = {}
            for key, output in (("action", "action"), ("state", "agent_pos")):
                data[output] = np.concatenate(
                    [
                        self.replay_buffer[key][int(starts[i]) : int(ends[i])]
                        for i in np.flatnonzero(self.train_mask)
                    ]
                )
            normalizer = LinearNormalizer()
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
            for key in ("head_cam", "left_cam", "right_cam"):
                normalizer[key] = get_image_range_normalizer()
            return normalizer

    return RecordedDataset


def workspace_class():
    from diffusion_policy.workspace.robotworkspace import RobotWorkspace

    class TrainingWorkspace(RobotWorkspace):
        def save_checkpoint(self, path=None, **kwargs):
            # Upstream requests policy/DP/checkpoints; redirect into the run capsule.
            name = Path(path).name if path else "latest.ckpt"
            kwargs["use_thread"] = False
            return super().save_checkpoint(
                Path(self.output_dir) / "checkpoints" / name, **kwargs
            )

    return TrainingWorkspace


def RecordedDataset(**kwargs):
    return dataset_class()(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=".")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--inference-steps", type=int, default=100)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", args.repository, "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != args.revision:
        raise ValueError("XPolicyLab checkout differs from the pinned adapter revision")
    policy_dir = configure_repository(args.repository)
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from artifacts import verify, digest

    root = Path(args.dataset)
    manifest = verify(root, args.manifest_sha)
    if (
        manifest.get("contract") != "skynet.dp-rgb-joints/v1"
        or manifest["format"] != "xpolicylab-dp-zarr/v1"
    ):
        raise ValueError("Dataset does not satisfy the recorded RGB/joint DP contract")
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("Batch size and epochs must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(
        config_dir=str(policy_dir / "diffusion_policy/config"), version_base="1.3"
    ):
        cfg = compose(config_name="robot_dp")
    OmegaConf.set_struct(cfg, False)
    dim = len(manifest["policy_to_source_indices"])
    cfg.task.shape_meta = {
        "action": {"shape": [dim]},
        "obs": {
            key: {"shape": [3, 240, 320], "type": "rgb"}
            for key in ("head_cam", "left_cam", "right_cam")
        },
    }
    cfg.task.shape_meta.obs.agent_pos = {"shape": [dim], "type": "low_dim"}
    cfg.task.name = "recorded_demonstrations"
    cfg.task.dataset = {
        "_target_": "skynet_dp_training.RecordedDataset",
        "zarr_path": str(root / "dataset/demonstrations.zarr"),
        "manifest_path": str(root / "manifest.json"),
        "batch_size": args.batch_size,
        "horizon": 8,
        "pad_before": 2,
        "pad_after": 5,
    }
    cfg.dataloader.batch_size = args.batch_size
    cfg.val_dataloader.batch_size = 1
    cfg.training.seed = args.seed
    cfg.training.device = args.device
    cfg.training.num_epochs = args.epochs
    cfg.training.resume = False
    cfg.training.checkpoint_every = 1
    cfg.training.val_every = 1
    cfg.training.max_train_steps = args.max_train_steps
    cfg.training.max_val_steps = 1 if args.max_train_steps else None
    cfg.policy.num_inference_steps = args.inference_steps
    cfg.optimizer.lr = args.learning_rate
    # Skynet forwards JsonLogger metrics to the selected tracking provider.
    cfg.logging.mode = "disabled"
    OmegaConf.resolve(cfg)
    dataset = RecordedDataset(
        **{k: v for k, v in cfg.task.dataset.items() if k != "_target_"}
    )
    validation = dataset.get_validation_dataset()
    if len(dataset) < args.batch_size or not len(validation):
        raise ValueError(
            "Too few training sequences for this batch size; reduce the batch size or select more episodes"
        )
    sample = dataset.postprocess(dataset[np.array([0])], torch.device("cpu"))
    for key, value in sample["obs"].items():
        expected = tuple(cfg.task.shape_meta.obs[key].shape)
        if tuple(value.shape[2:]) != expected or not torch.isfinite(value).all():
            raise ValueError("DP data loader returned invalid observations: " + key)
    if (
        tuple(sample["action"].shape) != (1, 8, dim)
        or not torch.isfinite(sample["action"]).all()
    ):
        raise ValueError("DP data loader returned invalid actions")
    dataset.get_normalizer()
    receipt = dict(
        schema="skynet.dp-loader-validation/v1",
        manifest_sha256=args.manifest_sha,
        source_revision=manifest["source_revision"],
        split=manifest["split"],
        train_sequences=len(dataset),
        validation_sequences=len(validation),
        observation_shapes={
            key: list(value.shape) for key, value in sample["obs"].items()
        },
        action_shape=list(sample["action"].shape),
        normalization="training_episodes_only",
    )
    (output / "dataset-validation.json").write_text(json.dumps(receipt, indent=2))
    OmegaConf.save(cfg, output / "training-config.yaml")
    if args.verify_only:
        print(json.dumps(receipt))
        return
    workspace = workspace_class()(cfg, output_dir=str(output))
    workspace.run()
    checkpoint = workspace.save_checkpoint()
    (output / "training-result.json").write_text(
        json.dumps(
            {
                "manifest_sha256": args.manifest_sha,
                "checkpoint": checkpoint,
                "checkpoint_sha256": digest(checkpoint),
                "global_step": workspace.global_step,
                "epochs": workspace.epoch,
            },
            indent=2,
        )
    )
    print(json.dumps({"checkpoint": checkpoint, "global_step": workspace.global_step}))


if __name__ == "__main__":
    main()
