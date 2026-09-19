"""Pinned native UniDex (PaliGemma + Uni3D) training and tensor inference.

No alternate model, loss, random split or upstream dataset cache is introduced.
GPU setup is separate from CPU dataset verification. Simulator execution is not
advertised: predict() returns native FAAS chunks, not controller commands.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    from .unidex_data import FORMAT, NativeNormalizer, UniDexDataset, validate_manifest
    from .unidex_weights import digest, verify_weight_provenance, initialize_pretrained, load_policy_state
except ImportError:
    from unidex_data import FORMAT, NativeNormalizer, UniDexDataset, validate_manifest
    from unidex_weights import digest, verify_weight_provenance, initialize_pretrained, load_policy_state

REVISION = "97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d"
TRAIN_TARGET = "src.unidex.unidex.PointCloudUniDexTrain"
INFERENCE_TARGET = "src.unidex.unidex.PointCloudUniDexInference"


def verify_repository(repository):
    actual = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    clean = subprocess.run(["git", "-C", str(repository), "diff", "--quiet", "HEAD", "--"], check=False).returncode == 0
    if actual != REVISION or not clean:
        raise ValueError("UniDex requires the exact unmodified pinned source revision " + REVISION)


def load_native_config(repository, manifest, base_weights=None, pointcloud_weights=None):
    """Load published model/training defaults; correct only the broken targets."""
    import yaml
    from omegaconf import OmegaConf

    root = Path(repository) / "config"
    model = yaml.safe_load((root / "model/unidex.yaml").read_text())
    model.pop("defaults")
    model["pointcloud_encoder"] = yaml.safe_load((root / "model/pointcloud_encoder/uni3d_l.yaml").read_text())
    if model.get("_target_") != TRAIN_TARGET:
        raise ValueError("Pinned UniDex model target changed")
    corrections = {"projector": ("src.openmodel.modules.PaliGemmaMultiModalProjector", "src.unidex.modules.PaliGemmaMultiModalProjector"),
                   "joint": ("src.openmodel.joint_model.JointModel", "src.unidex.joint_model.JointModel")}
    for key, (old, new) in corrections.items():
        if model[key]["_target_"] != old:
            raise ValueError("Pinned UniDex config differs from the audited target correction")
        model[key]["_target_"] = new
    model["horizon_steps"] = manifest["temporal"]["action_horizon"]
    if base_weights:
        model["pretrained_model_path"] = str(Path(base_weights).resolve())
        model["tokenizer_path"] = str(Path(base_weights).resolve())
    if pointcloud_weights:
        model["pointcloud_encoder"]["pretrained_model_path"] = str(Path(pointcloud_weights).resolve())
    cfg = OmegaConf.create({"model": model})
    OmegaConf.resolve(cfg)
    normalizer = yaml.safe_load((root / "dataset/normalizer/base.yaml").read_text())
    training = yaml.safe_load((root / "train.yaml").read_text())["train"]
    return OmegaConf.to_container(cfg.model, resolve=True), normalizer, training


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def training_identity(args, model_config, normalizer, manifest, provenance=None):
    return {"schema": "skynet.unidex-run/v1", "dataset_format": FORMAT,
            "manifest_sha256": args.manifest_sha, "revision": REVISION,
            "model_config": model_config, "normalizer": normalizer,
            "temporal": manifest["temporal"], "pretrained_assets": provenance,
            "training": {key: getattr(args, key) for key in (
                "batch_size", "learning_rate", "num_workers", "seed", "precision",
                "gpu_count", "gradient_accumulation")}}


def validate_resume(saved, identity):
    receipt = saved.get("skynet", {})
    if receipt.get("dataset_format") != FORMAT or receipt.get("schema") != "skynet.unidex-run/v1":
        raise ValueError("Old or foreign checkpoints cannot resume the new UniDex recording dataset; use an explicit weights-only initialization")
    if receipt.get("identity_sha256") != stable_digest(identity):
        raise ValueError("Resume requires identical dataset, source, model, normalization and training configuration")
    if not saved.get("optimizer_states") or not saved.get("lr_schedulers"):
        raise ValueError("True resume requires optimizer and scheduler state; this is a weights-only checkpoint")


def make_training_wrapper(policy, training_config, identity, initialization):
    import hydra
    import pytorch_lightning as lightning
    import torch
    from torch.optim.lr_scheduler import LambdaLR

    class NativeTraining(lightning.LightningModule):
        def __init__(self):
            super().__init__()
            self.policy = policy
            self.schedule = hydra.utils.instantiate(training_config["scheduler"])

        def training_step(self, batch, batch_idx):
            # Exactly the upstream flow-matching objective, all 82 dimensions.
            loss, metrics = self.policy(batch)
            self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
            for name, value in metrics.items():
                self.log("train_" + name, value, on_step=False, on_epoch=True, sync_dist=True)
            return loss

        def validation_step(self, batch, batch_idx):
            target = batch["action"]
            prediction = self.policy.infer_action(dict(batch))
            loss = torch.nn.functional.mse_loss(prediction, target)
            self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
            return loss

        def configure_optimizers(self):
            optimizer = hydra.utils.instantiate(training_config["optimizer"], params=self.policy.parameters())
            return {"optimizer": optimizer, "lr_scheduler": {
                "scheduler": LambdaLR(optimizer, lr_lambda=lambda step: self.schedule(step)),
                "interval": "step", "frequency": 1}}

        def on_save_checkpoint(self, checkpoint):
            checkpoint["skynet"] = {**identity, "identity_sha256": stable_digest(identity),
                                    "initialization": initialization}

        def on_train_epoch_end(self):
            if not self.trainer.is_global_zero:
                return
            metrics = {key: float(value.detach().cpu()) for key, value in self.trainer.callback_metrics.items()
                       if hasattr(value, "numel") and value.numel() == 1}
            row = dict(epoch=self.current_epoch, global_step=self.global_step,
                       train_loss=metrics.get("train_loss"), val_loss=metrics.get("val_loss"), metrics=metrics)
            with (Path(self.trainer.default_root_dir) / "logs.json.txt").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

    return NativeTraining()


def parser():
    p = argparse.ArgumentParser()
    for name in ("repository", "dataset", "manifest-sha", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--base-weights")
    p.add_argument("--pointcloud-weights")
    p.add_argument("--weights-provenance")
    p.add_argument("--initialize-checkpoint")
    p.add_argument("--resume-checkpoint")
    p.add_argument("--initialize-checkpoint-sha")
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="fp32")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--config-only", action="store_true")
    return p


def main():
    args = parser().parse_args()
    verify_repository(args.repository)
    manifest = validate_manifest(args.dataset, args.manifest_sha)
    model_config, norm_config, train_config = load_native_config(
        args.repository, manifest, args.base_weights, args.pointcloud_weights)
    normalizer = NativeNormalizer(norm_config)
    datasets = {split: UniDexDataset(args.dataset, manifest, split, normalizer) for split in ("train", "validation")}
    for dataset in datasets.values():
        dataset[0]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        report = {"schema": "skynet.unidex-loader-validation/v1", "status": "PASSED",
                  "manifest_sha256": args.manifest_sha, "observation_mode": "pointcloud",
                  "windows": {split: len(dataset) for split, dataset in datasets.items()}}
        (output / "loader-validation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        return
    if args.config_only:
        (output / "native-config.json").write_text(json.dumps(model_config, indent=2))
        return
    if not all((args.base_weights, args.pointcloud_weights, args.weights_provenance)):
        raise ValueError("Training needs explicit local PaliGemma/Uni3D paths and their weight provenance; weights are never downloaded implicitly")
    if min(args.epochs, args.batch_size, args.gpu_count, args.gradient_accumulation) < 1 or args.num_workers < 0:
        raise ValueError("Invalid native training resource/count settings")
    provenance = verify_weight_provenance(args.base_weights, args.pointcloud_weights, args.weights_provenance)
    if args.initialize_checkpoint and not args.resume_checkpoint and (not args.initialize_checkpoint_sha or digest(args.initialize_checkpoint) != args.initialize_checkpoint_sha):
        raise ValueError("Weights-only checkpoint initialization requires its exact SHA256")
    # Native tokenizer/model imports may only read the declared local assets.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(Path(args.repository).resolve()))
    import hydra
    import torch
    import pytorch_lightning as lightning
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf

    if not torch.cuda.is_available() or torch.cuda.device_count() < args.gpu_count:
        raise ValueError("Native UniDex training requires the requested CUDA runtime and GPUs")
    lightning.seed_everything(args.seed, workers=True)
    identity = training_identity(args, model_config, norm_config, manifest, provenance)
    saved = None
    if args.resume_checkpoint:
        saved = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        validate_resume(saved, identity)
        if args.initialize_checkpoint and saved["skynet"].get("initialization", {}).get("checkpoint_sha256") != args.initialize_checkpoint_sha:
            raise ValueError("Resume initialization differs from the checkpoint receipt")
    policy = hydra.utils.instantiate(OmegaConf.create(model_config))
    initialization = {"weights": provenance}
    if saved is not None:
        policy.load_state_dict({key.removeprefix("policy."): value for key, value in saved["state_dict"].items()}, strict=True)
        initialization = saved["skynet"]["initialization"]
    elif args.initialize_checkpoint:
        initialization.update(load_policy_state(policy, args.initialize_checkpoint))
        initialization.update(mode="weights_only", checkpoint_sha256=args.initialize_checkpoint_sha)
    else:
        initialization.update(initialize_pretrained(policy, args.base_weights, args.pointcloud_weights))
        initialization["mode"] = "base_pretrained"
    train_config["optimizer"]["lr"] = args.learning_rate
    wrapper = make_training_wrapper(policy, train_config, identity, initialization)
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
                                num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
               for split, dataset in datasets.items()}
    checkpoint = ModelCheckpoint(dirpath=str(output / "checkpoints"), save_last=True, save_top_k=1,
                                 monitor="val_loss", mode="min", filename="epoch-{epoch:03d}")
    # Slurm owns the allocation; Lightning launches per-GPU workers inside it.
    for key in list(os.environ):
        if key.startswith("SLURM_"):
            os.environ.pop(key)
    trainer = lightning.Trainer(accelerator="gpu", devices=args.gpu_count,
        strategy="ddp_find_unused_parameters_true" if args.gpu_count > 1 else "auto",
        max_epochs=args.epochs, precision={"bf16": "bf16-mixed", "fp16": "16-mixed", "fp32": "32-true"}[args.precision],
        accumulate_grad_batches=args.gradient_accumulation, gradient_clip_val=1.0,
        logger=False, callbacks=[checkpoint], default_root_dir=str(output))
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        (output / "dataset-receipt.json").write_text(json.dumps({**identity, "initialization": initialization}, indent=2))
    trainer.fit(wrapper, loaders["train"], loaders["validation"], ckpt_path=args.resume_checkpoint)


class UniDexPolicy:
    """Native tensor inference; returns physical FAAS chunks for a separate decoder."""
    def __init__(self, repository, checkpoint, checkpoint_sha, *, device="cuda"):
        verify_repository(repository)
        if digest(checkpoint) != checkpoint_sha:
            raise ValueError("UniDex checkpoint checksum mismatch")
        sys.path.insert(0, str(Path(repository).resolve()))
        import torch
        import hydra
        from omegaconf import OmegaConf
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        receipt = saved.get("skynet", {})
        if receipt.get("dataset_format") != FORMAT or receipt.get("revision") != REVISION:
            raise ValueError("Select a checkpoint from the current native UniDex adapter")
        cfg = copy.deepcopy(receipt["model_config"])
        cfg["_target_"] = INFERENCE_TARGET
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.model = hydra.utils.instantiate(OmegaConf.create(cfg))
        self.model.load_state_dict({key.removeprefix("policy."): value for key, value in saved["state_dict"].items()}, strict=True)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.normalizer = NativeNormalizer(receipt["normalizer"])
        self.horizon = receipt["temporal"]["action_horizon"]

    def predict(self, pointcloud_ros, state_absolute, prompt):
        import numpy as np
        import torch
        cloud, state = np.asarray(pointcloud_ros, dtype=np.float32), np.asarray(state_absolute, dtype=np.float32)
        if cloud.shape != (1024, 6) or state.shape != (82,) or not np.isfinite(cloud).all() or not np.isfinite(state).all():
            raise ValueError("UniDex inference requires finite front-camera XYZRGB1024 and absolute FAAS82 state")
        if not isinstance(prompt, str) or not prompt.strip() or np.any(cloud[:, 3:] < 0) or np.any(cloud[:, 3:] > 1):
            raise ValueError("Invalid task prompt or XYZRGB color range")
        cloud = cloud.copy()
        cloud[:, 1:3] *= -1
        with torch.inference_mode():
            result = self.model.infer_action(
                torch.from_numpy(cloud[None, None]).to(self.device),
                torch.from_numpy(self.normalizer.normalize("state", state)[None, None]).to(self.device), [prompt])
        values = result[0].float().cpu().numpy()
        if values.shape != (self.horizon, 82) or not np.isfinite(values).all():
            raise ValueError("Native UniDex produced invalid FAAS action chunks")
        return self.normalizer.unnormalize("action", values)


if __name__ == "__main__":
    main()
