"""Pinned native UniDex (PaliGemma + Uni3D) training and tensor inference.

No alternate model, loss, random split or upstream dataset cache is introduced.
GPU setup is separate from CPU dataset verification. predict() returns native
FAAS chunks; the separate Skynet evaluation bridge decodes controller commands.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

try:
    from .unidex_data import (FORMAT, NativeNormalizer, UniDexCollectionDataset, UniDexMixtureSampler, make_training_dataloader,
                             validate_manifest, validate_recorded_values)
    from .recording_time import resolve_sampling, resolve_collection_sampling
    from .unidex_subset import apply_frame_budget
    from .unidex_input import collection_pointcloud_recipe, checkpoint_pointcloud_recipe, validate_checkpoint_inputs
    from .unidex_precision import precision_recipe, apply_policy_precision, training_precision_kwargs
    from .dataset_inputs import resolve_data_selections
    from .unidex_weights import digest, verify_weight_provenance, initialize_pretrained, load_policy_state
except ImportError:
    from unidex_data import (FORMAT, NativeNormalizer, UniDexCollectionDataset, UniDexMixtureSampler, make_training_dataloader,
                            validate_manifest, validate_recorded_values)
    from recording_time import resolve_sampling, resolve_collection_sampling
    from unidex_subset import apply_frame_budget
    from unidex_input import collection_pointcloud_recipe, checkpoint_pointcloud_recipe, validate_checkpoint_inputs
    from unidex_precision import precision_recipe, apply_policy_precision, training_precision_kwargs
    from dataset_inputs import resolve_data_selections
    from unidex_weights import digest, verify_weight_provenance, initialize_pretrained, load_policy_state

REVISION = "97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d"
TRAIN_TARGET = "src.unidex.unidex.PointCloudUniDexTrain"
INFERENCE_TARGET = "src.unidex.unidex.PointCloudUniDexInference"
RUN_SCHEMA = "skynet.unidex-run/v3"


def verify_repository(repository):
    actual = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    clean = subprocess.run(["git", "-C", str(repository), "diff", "--quiet", "HEAD", "--"], check=False).returncode == 0
    if actual != REVISION or not clean:
        raise ValueError("UniDex requires the exact unmodified pinned source revision " + REVISION)


def load_native_config(repository, action_steps, base_weights=None, pointcloud_weights=None):
    """Load published model/training defaults; correct only the broken targets."""
    from omegaconf import OmegaConf

    root = Path(repository) / "config"
    # Use the upstream Hydra/OmegaConf scalar semantics. PyYAML treats bare
    # scientific notation (e.g. optimizer eps: 1e-8) as strings; wrapping that
    # result in OmegaConf later does not recover the original numeric types.
    model = OmegaConf.load(root / "model/unidex.yaml")
    model.pop("defaults")
    model["pointcloud_encoder"] = OmegaConf.load(root / "model/pointcloud_encoder/uni3d_l.yaml")
    if model.get("_target_") != TRAIN_TARGET:
        raise ValueError("Pinned UniDex model target changed")
    corrections = {"projector": ("src.openmodel.modules.PaliGemmaMultiModalProjector", "src.unidex.modules.PaliGemmaMultiModalProjector"),
                   "joint": ("src.openmodel.joint_model.JointModel", "src.unidex.joint_model.JointModel")}
    for key, (old, new) in corrections.items():
        if model[key]["_target_"] != old:
            raise ValueError("Pinned UniDex config differs from the audited target correction")
        model[key]["_target_"] = new
    model["horizon_steps"] = action_steps
    if base_weights:
        model["pretrained_model_path"] = str(Path(base_weights).resolve())
        model["tokenizer_path"] = str(Path(base_weights).resolve())
    if pointcloud_weights:
        model["pointcloud_encoder"]["pretrained_model_path"] = str(Path(pointcloud_weights).resolve())
    cfg = OmegaConf.create({"model": model})
    OmegaConf.resolve(cfg)
    normalizer = OmegaConf.to_container(OmegaConf.load(root / "dataset/normalizer/base.yaml"), resolve=True)
    training = OmegaConf.to_container(OmegaConf.load(root / "train.yaml").train, resolve=True)
    return OmegaConf.to_container(cfg.model, resolve=True), normalizer, training


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def execution_precision(args):
    """Pin optional acceleration; preserve the pre-existing native FP32 path."""
    matmul = getattr(args, "float32_matmul_precision", "highest")
    parameter_dtype = getattr(args, "fsdp_parameter_dtype", "auto")
    if parameter_dtype != "auto" and getattr(args, "distributed_strategy", "ddp") != "fsdp":
        raise ValueError("FSDP parameter dtype requires FSDP training")
    if args.precision == "fp16":
        # This pre-existing native mode is not one of the audited acceleration
        # recipes. Do not silently change it while adding BF16/TF32 support.
        if matmul != "highest" or parameter_dtype != "auto":
            raise ValueError("Float32 matmul acceleration requires FP32 training")
        return None
    recipe = precision_recipe(args.precision, matmul, parameter_dtype)
    return None if (args.precision, matmul) == ("fp32", "highest") else recipe


def execution_sharding(args):
    """Record the opt-in communication change without altering old identities."""
    sharding = getattr(args, "fsdp_sharding_strategy", "FULL_SHARD")
    if sharding not in {"FULL_SHARD", "SHARD_GRAD_OP"}:
        raise ValueError("Unknown FSDP sharding strategy")
    if sharding == "FULL_SHARD":
        return None
    if getattr(args, "distributed_strategy", "ddp") != "fsdp":
        raise ValueError("FSDP sharding requires FSDP training")
    return {"sharding_strategy": sharding, "gradient_sync": "optimizer_boundary",
            "accumulation": "fsdp_no_sync"}


def training_identity(args, model_config, normalizer, sampling, provenance=None, *, selections, mixture=None):
    inputs = [{key: selection.get(key) for key in ("position", "version_id", "manifest_sha256")}
              for selection in selections]
    max_steps = getattr(args, "max_steps", None)
    precision = execution_precision(args)
    sharding = execution_sharding(args)
    return {"schema": RUN_SCHEMA, "dataset_format": FORMAT,
            **({"execution_precision": precision} if precision is not None else {}),
            **({"execution_sharding": sharding} if sharding is not None else {}),
            "datasets": inputs, "revision": REVISION,
            "pointcloud_recipe": collection_pointcloud_recipe(selections),
            "model_config": model_config, "normalizer": normalizer,
            "sampling": sampling, "mixture": mixture, "pretrained_assets": provenance,
            "distributed_strategy": getattr(args, "distributed_strategy", "ddp"),
            "budget": {"max_steps": max_steps, "max_epochs": None if max_steps is not None else getattr(args, "epochs", 32)},
            "training": {**{key: getattr(args, key) for key in (
                "batch_size", "learning_rate", "num_workers", "seed", "precision",
                "gpu_count", "gradient_accumulation")},
                # Existing checkpoints implicitly enabled validation. Preserve
                # that identity; disabling it is an explicit new run contract.
                **({"validation_enabled": False} if not getattr(args, "validation_enabled", True) else {})}}


def validate_resume(saved, identity):
    receipt = saved.get("skynet", {})
    if receipt.get("dataset_format") != FORMAT or receipt.get("schema") != RUN_SCHEMA:
        raise ValueError("Old or foreign checkpoints cannot resume the new UniDex recording dataset; use an explicit weights-only initialization")
    if receipt.get("identity_sha256") != stable_digest(identity):
        raise ValueError("Resume requires identical dataset, source, model, normalization and training configuration")
    if not saved.get("optimizer_states") or not saved.get("lr_schedulers"):
        raise ValueError("True resume requires optimizer and scheduler state; this is a weights-only checkpoint")
    loaders = ((saved.get("loops") or {}).get("fit_loop") or {}).get("state_dict", {}).get("combined_loader")
    if not isinstance(loaders, list) or len(loaders) != 1 or loaders[0].get("schema") != "skynet.unidex-loader-state/v1":
        raise ValueError("True resume requires the saved completed training data position")


def resume_initialization(checkpoint, identity, initialization_sha=None):
    """Inspect the receipt without retaining a full training state per rank."""
    import torch

    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    validate_resume(saved, identity)
    initialization = saved["skynet"]["initialization"]
    if initialization_sha and initialization.get("checkpoint_sha256") != initialization_sha:
        raise ValueError("Resume initialization differs from the checkpoint receipt")
    # Lightning restores the full model/optimizer/loader after distributed setup.
    # Loading it here as well would keep another complete copy on every rank.
    return copy.deepcopy(initialization)


def make_training_wrapper(policy, training_config, identity, initialization, sampler=None):
    import hydra
    import pytorch_lightning as lightning
    import torch
    from torch.optim.lr_scheduler import LambdaLR

    class CompletedTrainingBatches(lightning.Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if sampler is not None:
                sampler.mark_consumed(int(batch["action"].shape[0]))

    class NativeTraining(lightning.LightningModule):
        def __init__(self):
            super().__init__()
            self.policy = policy
            self.schedule = hydra.utils.instantiate(training_config["scheduler"])
            self._logged_step = 0
            self._step_loss_sum = None
            self._step_loss_count = 0
            self._last_validation_epoch = None

        def configure_callbacks(self):
            # Lightning orders checkpoint callbacks after ordinary callbacks.
            return [CompletedTrainingBatches()]

        def on_load_checkpoint(self, checkpoint):
            self._logged_step = int(checkpoint["global_step"])

        def training_step(self, batch, batch_idx):
            # Exactly the upstream flow-matching objective, all 82 dimensions.
            loss, metrics = self.policy(batch)
            detached = loss.detach()
            self._step_loss_sum = detached if self._step_loss_sum is None else self._step_loss_sum + detached
            self._step_loss_count += 1
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

        def on_validation_epoch_end(self):
            if not self.trainer.sanity_checking:
                self._last_validation_epoch = self.current_epoch

        def configure_optimizers(self):
            optimizer = hydra.utils.instantiate(training_config["optimizer"], params=self.policy.parameters())
            return {"optimizer": optimizer, "lr_scheduler": {
                "scheduler": LambdaLR(optimizer, lr_lambda=lambda step: self.schedule(step)),
                "interval": "step", "frequency": 1}}

        def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
            from torch.distributed.fsdp import FullyShardedDataParallel

            if isinstance(self.trainer.model, FullyShardedDataParallel) and gradient_clip_algorithm in (None, "norm"):
                # Lightning's generic FSDPPrecision cannot compute a global
                # norm from rank-local shards. FSDP's root method includes all
                # nested units and ranks, preserving upstream's L2 norm clip.
                if gradient_clip_val is not None and gradient_clip_val > 0:
                    self.trainer.model.clip_grad_norm_(gradient_clip_val, norm_type=2.0)
            else:
                super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

        def on_save_checkpoint(self, checkpoint):
            checkpoint["skynet"] = {**identity, "identity_sha256": stable_digest(identity),
                                    "initialization": initialization}

        def on_train_epoch_start(self):
            if sampler is not None:
                sampler.set_epoch(self.current_epoch)

        def on_train_batch_end(self, outputs, batch, batch_idx):
            # Lightning's outputs.loss is divided by gradient accumulation.
            # Report the mean original loss for this optimizer update instead.
            if self.global_step <= self._logged_step:
                return
            self._logged_step = self.global_step
            totals = torch.stack((self._step_loss_sum, self._step_loss_sum.new_tensor(self._step_loss_count)))
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(totals)
            loss = float((totals[0] / totals[1]).cpu())
            self._step_loss_sum, self._step_loss_count = None, 0
            if self.trainer.is_global_zero:
                with (Path(self.trainer.default_root_dir) / "logs.json.txt").open("a") as stream:
                    stream.write(json.dumps(dict(global_step=self.global_step, train_loss=loss), allow_nan=False) + "\n")

        def on_train_epoch_end(self):
            if not self.trainer.is_global_zero:
                return
            metrics = {key: float(value.detach().cpu()) for key, value in self.trainer.callback_metrics.items()
                       if hasattr(value, "numel") and value.numel() == 1}
            if self._last_validation_epoch != self.current_epoch:
                metrics.pop("val_loss", None)
            row = dict(epoch=self.current_epoch, global_step=self.global_step,
                       train_loss=metrics.get("train_loss"), val_loss=metrics.get("val_loss"), metrics=metrics)
            with (Path(self.trainer.default_root_dir) / "logs.json.txt").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

    return NativeTraining()


def make_training_checkpoint(output, *, save_every_steps=1000, keep_last=3):
    """Apply Skynet's latest-checkpoint contract through pinned Lightning.

    A validation-best callback can leave last.ckpt at an older step, including
    when max_steps ends a partial epoch. Step ranking keeps retention and final
    selection independent of validation cadence or improvements.
    """
    from pytorch_lightning.callbacks import ModelCheckpoint

    if save_every_steps < 1 or keep_last < 1:
        raise ValueError("Checkpoint interval and retention must be positive")

    class FinalStepCheckpoint(ModelCheckpoint):
        def on_train_start(self, trainer, pl_module):
            super().on_train_start(trainer, pl_module)
            # Lightning restores callback paths/rankings, but not this counter.
            # Until the next optimizer update, resumed accumulation microbatches
            # still have the saved global step and must not save partial state.
            self._last_global_step_saved = trainer.global_step

        def on_train_end(self, trainer, pl_module):
            if trainer.global_step <= self._last_global_step_saved:
                return
            # Reuse Lightning's distributed save, retention, callback state and
            # linking. All ranks participate in FSDP's full-state collection.
            candidates = self._monitor_candidates(trainer)
            self._save_topk_checkpoint(trainer, candidates)
            self._save_last_checkpoint(trainer, candidates)
            self._defer_save_until_validation = False

    return FinalStepCheckpoint(
        dirpath=str(Path(output) / "checkpoints"), filename="step-{step:08d}",
        auto_insert_metric_name=False, monitor="step", mode="max",
        every_n_train_steps=save_every_steps, every_n_epochs=0,
        save_on_train_epoch_end=False, save_top_k=keep_last, save_last="link",
    )


def parser():
    p = argparse.ArgumentParser()
    for name in ("repository", "output"):
        p.add_argument("--" + name, required=True)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset", help="Single immutable dataset, also used by conversion verification")
    inputs.add_argument("--data-spec", help="Frozen resolved-spec.json containing the selected training datasets")
    p.add_argument("--manifest-sha")
    p.add_argument("--base-weights")
    p.add_argument("--pointcloud-weights")
    p.add_argument("--weights-provenance")
    p.add_argument("--initialize-checkpoint")
    p.add_argument("--resume-checkpoint")
    p.add_argument("--initialize-checkpoint-sha")
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--max-steps", type=int, help="Fixed optimizer update budget; when set, replaces the epoch limit")
    p.add_argument("--checkpoint-every-steps", type=int, default=1000)
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--validation-enabled", action=argparse.BooleanOptionalAction, default=True,
                   help="Skynet execution option; disabling skips sanity checks and validation inference without changing the training split")
    p.add_argument("--mixing-policy", choices=UniDexMixtureSampler.POLICIES, default="window_proportional")
    p.add_argument("--unique-source-frames", type=int,
                   help="Skynet equal-hand budget of unique consumed raw frame indices; omitted uses all training data")
    p.add_argument("--data-selection-seed", type=int, default=20260920,
                   help="Skynet source subset seed, independent of model initialization")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--distributed-strategy", choices=["ddp", "fsdp"], default="ddp",
                   help="DDP follows upstream; Skynet FSDP shards training state across GPUs without changing the model")
    p.add_argument("--fsdp-sharding-strategy", choices=["FULL_SHARD", "SHARD_GRAD_OP"], default="FULL_SHARD",
                   help="Skynet option: SHARD_GRAD_OP retains parameters during accumulation and synchronizes gradients once per optimizer update")
    p.add_argument("--fsdp-parameter-dtype", choices=["auto", "float32"], default="auto",
                   help="Skynet BF16 option: float32 retains FP32 parameters and accumulated gradients, requiring more memory")
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="fp32")
    p.add_argument("--float32-matmul-precision", choices=["highest", "high"], default="highest",
                   help="Skynet FP32 Tensor Core acceleration; pointcloud geometry stays full FP32")
    p.add_argument("--control-hz", type=float, help="Training frequency; omitted uses the recording frequency")
    p.add_argument("--action-steps", type=int, default=30, help="Controller targets predicted per observation")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--config-only", action="store_true")
    return p


def load_training_selections(args):
    """Read the authoritative frozen bundle, then verify every actual manifest."""
    if args.data_spec:
        if args.manifest_sha:
            raise ValueError("A frozen data spec cannot be combined with a separate manifest checksum")
        selections = resolve_data_selections(json.loads(Path(args.data_spec).read_text()), role="training_data")
    else:
        if not args.manifest_sha:
            raise ValueError("A single dataset requires its exact manifest checksum")
        selections = [dict(position=0, version_id=None, path=args.dataset, manifest_sha256=args.manifest_sha)]
    verified = [{**selection, "metadata": validate_manifest(selection["path"], selection["manifest_sha256"])}
                for selection in selections]
    collection_pointcloud_recipe(verified)
    return verified


def training_limits(args):
    if (type(args.epochs) is not int or args.epochs < 1 or
            (args.max_steps is not None and (type(args.max_steps) is not int or args.max_steps < 1))):
        raise ValueError("UniDex training requires a positive epoch or optimizer-step budget")
    return {"max_epochs": -1 if args.max_steps is not None else args.epochs,
            "max_steps": args.max_steps if args.max_steps is not None else -1}


def validation_options(enabled):
    """Keep upstream epoch validation by default; explicitly skip all when off."""
    if type(enabled) is not bool:
        raise ValueError("Validation enabled must be a boolean")
    return {} if enabled else {"limit_val_batches": 0, "num_sanity_val_steps": 0}


def training_strategy(name, gpu_count, policy=None, *, execution_precision=None, sharding_strategy="FULL_SHARD"):
    if sharding_strategy not in {"FULL_SHARD", "SHARD_GRAD_OP"}:
        raise ValueError("Unknown FSDP sharding strategy")
    if name != "fsdp" and sharding_strategy != "FULL_SHARD":
        raise ValueError("FSDP sharding requires FSDP training")
    if name == "fsdp":
        if gpu_count < 2:
            raise ValueError("FSDP requires at least two GPUs")
        if policy is None:
            raise ValueError("FSDP requires the initialized native policy")
        from pytorch_lightning.strategies import FSDPStrategy
        # UniDex invokes decoder/attention custom methods directly, bypassing
        # those modules' forward hooks. Their MLPs and Uni3D visual blocks DO
        # execute forward, making them valid, bounded FSDP units. One giant
        # root unit otherwise allocates a model-sized gradient buffer at once.
        wrap_classes = {type(policy.embed_tokens)}
        wrap_classes.update(type(layer.mlp) for mixture in policy.joint_model.mixtures.values()
                            for layer in mixture.layers)
        wrap_classes.update(type(block) for block in policy.pointcloud_encoder.visual.blocks)
        # A full checkpoint preserves the existing portable policy/resume
        # contract. Original parameters support upstream's mixed frozen/trainable
        # modules. Lightning shards state and reconstructs it when saving.
        strategy_class = FSDPStrategy
        if sharding_strategy == "SHARD_GRAD_OP":
            class AccumulationFSDPStrategy(FSDPStrategy):
                @contextlib.contextmanager
                def block_backward_sync(self):
                    from torch.distributed.fsdp import FullyShardedDataParallel
                    if not isinstance(self.model, FullyShardedDataParallel):
                        raise TypeError("Gradient accumulation requires the root FSDP model")
                    # Lightning's default FSDP strategy does not defer sync.
                    # Its accumulation closure includes forward AND backward;
                    # the final microbatch exits this context and reduces once.
                    with self.model.no_sync():
                        yield None
            strategy_class = AccumulationFSDPStrategy
        return strategy_class(state_dict_type="full", use_orig_params=True,
                            sharding_strategy=sharding_strategy,
                            auto_wrap_policy=wrap_classes,
                            **(training_precision_kwargs(execution_precision) if execution_precision else {}))
    if name != "ddp":
        raise ValueError("Unknown distributed training strategy")
    return "ddp_find_unused_parameters_true" if gpu_count > 1 else "auto"


def main():
    args = parser().parse_args()
    verify_repository(args.repository)
    selections = load_training_selections(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        if args.data_spec:
            raise ValueError("Conversion verification accepts one dataset, not an experiment data spec")
        manifest = selections[0]["metadata"]
        validate_recorded_values(args.dataset, manifest)
        report = {"schema": "skynet.unidex-loader-validation/v1", "status": "PASSED",
                  "manifest_sha256": args.manifest_sha, "observation_mode": "pointcloud",
                  "validation_scope": "recorded_streams", "episodes": len(manifest["episodes"]),
                  "steps": sum(episode["steps"] for episode in manifest["episodes"]),
                  "pointcloud_recipe": collection_pointcloud_recipe(selections)}
        (output / "loader-validation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        return
    limits = training_limits(args)
    precision = execution_precision(args)
    execution_sharding(args)
    validation = validation_options(args.validation_enabled)
    sampling = resolve_collection_sampling(selections, control_hz=args.control_hz, action_steps=args.action_steps,
                                           window_policy="complete", require_validation=True)
    sampling = apply_frame_budget(selections, sampling, args.unique_source_frames,
                                  selection_seed=args.data_selection_seed)
    (output / "sampling.json").write_text(json.dumps(sampling, indent=2, allow_nan=False))
    print(json.dumps({"sampling": sampling}, allow_nan=False), flush=True)
    model_config, norm_config, train_config = load_native_config(
        args.repository, sampling["action_steps"], args.base_weights, args.pointcloud_weights)
    normalizer = NativeNormalizer(norm_config)
    datasets = {split: UniDexCollectionDataset(selections, split, normalizer, sampling)
                for split in (("train", "validation") if args.validation_enabled else ("train",))}
    samplers = {split: UniDexMixtureSampler(dataset.hands,
                args.mixing_policy if split == "train" else "window_proportional",
                seed=args.seed, shuffle=split == "train",
                batch_size=args.batch_size if split == "train" else 1,
                gradient_accumulation_steps=args.gradient_accumulation if split == "train" else 1)
                for split, dataset in datasets.items()}
    mixture = {**samplers["train"].identity(), "epoch_plan": samplers["train"].epoch_plan(args.gpu_count)}
    (output / "mixture.json").write_text(json.dumps(mixture, indent=2, allow_nan=False))
    for dataset in datasets.values():
        dataset[0]
    if args.config_only:
        (output / "native-config.json").write_text(json.dumps(model_config, indent=2))
        return
    if not all((args.base_weights, args.pointcloud_weights, args.weights_provenance)):
        raise ValueError("Training needs explicit local PaliGemma/Uni3D paths and their weight provenance; weights are never downloaded implicitly")
    if min(args.epochs, args.batch_size, args.gpu_count, args.gradient_accumulation,
           args.checkpoint_every_steps, args.keep_checkpoints) < 1 or args.num_workers < 0:
        raise ValueError("Invalid native training resource/count settings")
    if args.distributed_strategy == "fsdp" and args.gpu_count < 2:
        raise ValueError("FSDP requires at least two GPUs")
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
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf

    if not torch.cuda.is_available() or torch.cuda.device_count() < args.gpu_count:
        raise ValueError("Native UniDex training requires the requested CUDA runtime and GPUs")
    lightning.seed_everything(args.seed, workers=True)
    identity = training_identity(args, model_config, norm_config, sampling, provenance,
                                 selections=selections, mixture=mixture)
    initialization = None
    if args.resume_checkpoint:
        initialization = resume_initialization(
            args.resume_checkpoint, identity,
            args.initialize_checkpoint_sha if args.initialize_checkpoint else None)
    policy = hydra.utils.instantiate(OmegaConf.create(model_config))
    if initialization is None:
        initialization = {"weights": provenance}
        if args.initialize_checkpoint:
            initialization.update(load_policy_state(policy, args.initialize_checkpoint))
            initialization.update(mode="weights_only", checkpoint_sha256=args.initialize_checkpoint_sha)
        else:
            initialization.update(initialize_pretrained(policy, args.base_weights, args.pointcloud_weights))
            initialization["mode"] = "base_pretrained"
    if precision is not None:
        apply_policy_precision(policy, precision)
    strategy = training_strategy(args.distributed_strategy, args.gpu_count, policy,
                                 execution_precision=precision, sharding_strategy=args.fsdp_sharding_strategy)
    train_config["optimizer"]["lr"] = args.learning_rate
    wrapper = make_training_wrapper(policy, train_config, identity, initialization, samplers["train"])
    loaders = {"train": make_training_dataloader(datasets["train"], samplers["train"], batch_size=args.batch_size,
                                               num_workers=args.num_workers, seed=args.seed)}
    if args.validation_enabled:
        loaders["validation"] = DataLoader(datasets["validation"], batch_size=args.batch_size, sampler=samplers["validation"],
                                           num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                                           generator=torch.Generator().manual_seed(args.seed))
    checkpoint = make_training_checkpoint(output, save_every_steps=args.checkpoint_every_steps,
                                           keep_last=args.keep_checkpoints)
    # Slurm owns the allocation; Lightning launches per-GPU workers inside it.
    for key in list(os.environ):
        if key.startswith("SLURM_"):
            os.environ.pop(key)
    trainer = lightning.Trainer(accelerator="gpu", devices=args.gpu_count,
        strategy=strategy,
        **limits, **validation, precision={"bf16": "bf16-mixed", "fp16": "16-mixed", "fp32": "32-true"}[args.precision],
        accumulate_grad_batches=args.gradient_accumulation, gradient_clip_val=1.0, gradient_clip_algorithm="norm",
        use_distributed_sampler=False, logger=False, callbacks=[checkpoint], default_root_dir=str(output))
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        (output / "dataset-receipt.json").write_text(json.dumps({**identity, "initialization": initialization}, indent=2))
    trainer.fit(wrapper, loaders["train"], loaders.get("validation"), ckpt_path=args.resume_checkpoint)


def checkpoint_sampling(receipt):
    """Expose the learned physical time scale to the execution controller."""
    sampling = receipt.get("sampling", {})
    control_hz, horizon = sampling.get("control_hz"), sampling.get("action_steps")
    if (receipt.get("schema") not in {RUN_SCHEMA, "skynet.unidex-run/v2"} or type(control_hz) not in (int, float)
            or not math.isfinite(control_hz) or control_hz <= 0 or type(horizon) is not int or horizon < 1
            or receipt.get("model_config", {}).get("horizon_steps") != horizon):
        raise ValueError("UniDex checkpoint lacks valid experiment sampling and model horizon")
    return sampling


class UniDexPolicy:
    """Native tensor inference; returns physical FAAS chunks for a separate decoder."""
    def __init__(self, repository, checkpoint, checkpoint_sha, *, device="cuda", expected_datasets=None):
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
        self.pointcloud_recipe = checkpoint_pointcloud_recipe(receipt, expected_datasets)
        self.receipt = receipt
        self.sampling = checkpoint_sampling(receipt)
        self.control_hz = self.sampling["control_hz"]
        self.action_dt = 1.0 / self.control_hz
        self.horizon = self.sampling["action_steps"]
        cfg = copy.deepcopy(receipt["model_config"])
        cfg["_target_"] = INFERENCE_TARGET
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.model = hydra.utils.instantiate(OmegaConf.create(cfg))
        self.model.load_state_dict({key.removeprefix("policy."): value for key, value in saved["state_dict"].items()}, strict=True)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.normalizer = NativeNormalizer(receipt["normalizer"])

    def predict(self, pointcloud_ros, state_absolute, prompt):
        import numpy as np
        import torch
        cloud, state = np.asarray(pointcloud_ros, dtype=np.float32), np.asarray(state_absolute, dtype=np.float32)
        if (cloud.shape != (self.pointcloud_recipe["num_points"], 6) or state.shape != (82,)
                or not np.isfinite(cloud).all() or not np.isfinite(state).all()):
            raise ValueError("UniDex inference requires finite front-camera XYZRGB matching its checkpoint recipe and absolute FAAS82 state")
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
