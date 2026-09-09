"""Native robomimic BC training with DDP and native checkpoint serialization."""

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import tempfile

import torch
from training_parallel import LossModule, TrainingContext, first_tensor, launch_distributed


class RobomimicLoss(LossModule):
    def __init__(self, algorithm):
        super().__init__()
        self.nets = algorithm.nets
        self.algorithm = algorithm

    def compute(self, batch):
        predictions = self.algorithm._forward_training(batch)
        return self.algorithm._compute_losses(predictions, batch), {
            key: value.detach() for key, value in predictions.items()}


def shard_batch(value, rank, world, size):
    if isinstance(value, torch.Tensor):
        return (value[rank::world] if rank < size else value[:1]) if value.ndim and len(value) == size else value
    if isinstance(value, dict):
        return {k: shard_batch(v, rank, world, size) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(shard_batch(v, rank, world, size) for v in value)
    return value


def enable_distributed_training(context):
    import robomimic.algo as algorithms
    from robomimic.algo.bc import BC
    from robomimic.algo import PolicyAlgo

    factory, original_train = algorithms.algo_factory, BC.train_on_batch

    def distributed_factory(*args, **kwargs):
        algorithm = factory(*args, **kwargs)
        if not isinstance(algorithm, BC):
            raise ValueError("DexMimicGen distributed training requires a robomimic BC policy")
        algorithm._skynet_loss = RobomimicLoss(algorithm)
        algorithm._skynet_distributed = context.wrap(algorithm._skynet_loss)
        return algorithm

    def train_on_batch(self, batch, epoch, validate=False):
        distributed = getattr(self, "_skynet_distributed", None)
        if distributed is None:
            return original_train(self, batch, epoch, validate=validate)
        size = len(first_tensor(batch))
        count = len(range(context.rank, size, context.world_size))
        local = shard_batch(batch, context.rank, context.world_size, size)
        with torch.set_grad_enabled(not validate):
            info = PolicyAlgo.train_on_batch(self, local, epoch, validate=validate)
            losses, predictions = (self._skynet_loss if validate else distributed)(local)
            context.check_loss(losses["action_loss"], nonnegative=False)
            # Keep the native optimizer/gradient clipping and global mean loss.
            scaled = {k: v * count * context.world_size / size for k, v in losses.items()}
            recorded = {k: v.detach().new_tensor(context.mean(float(v.detach()) * count, count, size))
                for k, v in losses.items()}
            info.update(predictions=predictions, losses=recorded)
            if not validate:
                info.update(self._train_step(scaled))
        return info

    algorithms.algo_factory = distributed_factory
    BC.train_on_batch = train_on_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-count", type=int, required=True)
    # The adapter's separator separates wrapper settings from native settings.
    import sys
    args = sys.argv[1:]
    if "--" in args:
        args.remove("--")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--name")
    parsed = parser.parse_args(args)
    if launch_distributed(parsed.gpu_count):
        return
    context = TrainingContext(parsed.gpu_count)
    if context.world_size > 1:
        enable_distributed_training(context)
    from robomimic.config import config_factory
    import robomimic.scripts.train as native
    import robomimic.utils.train_utils as train_utils

    native_loader = native.DataLoader

    def data_loader(*args, **kwargs):
        if kwargs.get("num_workers", 0):
            kwargs.update(multiprocessing_context="spawn", persistent_workers=True)
        return native_loader(*args, **kwargs)

    native.DataLoader = data_loader

    document = json.loads(Path(parsed.config).read_text())
    config = config_factory(document["algo_name"])
    with config.values_unlocked():
        config.update(document)
    if parsed.dataset:
        config.train.data = parsed.dataset
    if parsed.name:
        config.experiment.name = parsed.name
    original_epoch = train_utils.run_epoch

    def run_epoch(*args, **kwargs):
        # Native loaders receive identical global batches before local slicing.
        # Use a dedicated epoch seed so rank-zero logging/rollouts cannot change their order.
        epoch = kwargs.get("epoch", 0)
        torch.manual_seed(config.train.seed + epoch)
        torch.cuda.manual_seed(config.train.seed + epoch + context.rank)
        return original_epoch(*args, **kwargs)

    train_utils.run_epoch = run_epoch
    # Non-primary native writers are isolated from registered run artifacts.
    with tempfile.TemporaryDirectory(prefix="skynet-ddp-worker-") as temporary:
        if not context.primary:
            with config.values_unlocked():
                config.train.output_dir = temporary
                config.experiment.save.enabled = False
                config.experiment.rollout.enabled = False
                config.experiment.logging.log_tb = False
                config.experiment.logging.log_wandb = False
                config.experiment.logging.terminal_output_to_txt = False
        config.lock()
        with open(os.devnull, "w") as discard, redirect_stdout(sys.stdout if context.primary else discard):
            native.train(config, device=context.device)
    context.close()


if __name__ == "__main__":
    main()
