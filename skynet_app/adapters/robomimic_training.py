"""Run native robomimic BC trainers with the shared multi-GPU loss reducer."""

import argparse
import copy
import runpy
import sys

from training_parallel import LossModule, ParallelLoss, validate_devices, first_tensor


class RobomimicLoss(LossModule):
    def __init__(self, algorithm):
        super().__init__()
        self.nets = algorithm.nets
        self.algorithm = algorithm

    def compute(self, batch):
        # Replicas own their network modules; mutable algorithm state is local to
        # this forward pass. Optimizers and checkpoint serialization stay native.
        algorithm = copy.copy(self.algorithm)
        algorithm.nets = self.nets
        algorithm.device = first_tensor(batch).device
        predictions = algorithm._forward_training(batch)
        losses = algorithm._compute_losses(predictions, batch)
        return losses, {
            key: value.detach().reshape(1) if value.ndim == 0 else value.detach()
            for key, value in predictions.items()
        }


def enable_parallel_training(devices):
    import torch
    import robomimic.algo as algorithms
    from robomimic.algo.bc import BC
    from robomimic.algo import PolicyAlgo

    factory, original_train = algorithms.algo_factory, BC.train_on_batch

    def parallel_factory(*args, **kwargs):
        algorithm = factory(*args, **kwargs)
        if not isinstance(algorithm, BC):
            raise ValueError(
                "DexMimicGen multi-GPU training requires a robomimic BC policy"
            )
        if algorithm.global_config.train.batch_size < len(devices):
            raise ValueError(
                "The robomimic training batch must contain at least one sample per GPU"
            )
        algorithm._skynet_parallel = ParallelLoss(RobomimicLoss(algorithm), devices)
        return algorithm

    def train_on_batch(self, batch, epoch, validate=False):
        parallel = getattr(self, "_skynet_parallel", None)
        if parallel is None:
            return original_train(self, batch, epoch, validate=validate)
        with torch.set_grad_enabled(not validate):
            info = PolicyAlgo.train_on_batch(self, batch, epoch, validate=validate)
            losses, predictions = parallel(batch)
            info.update(
                predictions=predictions,
                losses={k: v.detach() for k, v in losses.items()},
            )
            if not validate:
                info.update(self._train_step(losses))
        return info

    algorithms.algo_factory = parallel_factory
    BC.train_on_batch = train_on_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-count", type=int, required=True)
    args, native = parser.parse_known_args()
    if native[:1] == ["--"]:
        native = native[1:]
    devices = validate_devices(args.gpu_count)
    if len(devices) > 1:
        enable_parallel_training(devices)
    sys.argv = ["robomimic.scripts.train", *native]
    runpy.run_module("robomimic.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
