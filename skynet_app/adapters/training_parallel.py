"""Shared single-process multi-GPU loss reduction with unchanged model checkpoints."""

import copy
import json
import torch
from torch import nn


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    values = value.values() if isinstance(value, dict) else value
    for item in values:
        if isinstance(item, (torch.Tensor, dict, tuple, list)):
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def validate_devices(count):
    if count < 1 or torch.cuda.device_count() < count:
        raise ValueError(
            f"Requested {count} training GPUs, but only {torch.cuda.device_count()} are visible"
        )
    return list(range(count))


def training_batch_size(value, gpu_count, accumulation, semantics="per_device"):
    if semantics in {"per_device", "repository_native"}:
        return value * gpu_count
    if semantics == "global_before_accumulation":
        result = value
    elif semantics == "global_effective":
        if value % accumulation:
            raise ValueError(
                "Effective batch size must be divisible by gradient accumulation"
            )
        result = value // accumulation
    else:
        raise ValueError("Unknown batch semantics: " + semantics)
    if result < gpu_count:
        raise ValueError("The training batch must contain at least one sample per GPU")
    return result


class LossModule(nn.Module):
    """Each replica returns loss sums so uneven final batches remain correctly weighted."""

    def compute(self, batch):
        raise NotImplementedError

    def forward(self, batch):
        sample = first_tensor(batch)
        for module in self.modules():
            if isinstance(module.__dict__.get("device"), torch.device):
                module.device = sample.device
        losses, predictions = self.compute(batch)
        count = sample.shape[0]
        return {
            "loss_sums": {key: loss.reshape(1) * count for key, loss in losses.items()},
            "count": sample.new_tensor([count], dtype=torch.long),
            "device": sample.new_tensor([sample.device.index or 0], dtype=torch.long),
            "predictions": predictions,
        }


class PolicyLoss(LossModule):
    def __init__(self, policy, method=None):
        super().__init__()
        self.policy, self.method = policy, method

    def compute(self, batch):
        if self.method:
            if getattr(self, "_is_replica", False) and hasattr(
                self.policy, "noise_scheduler"
            ):
                self.policy.noise_scheduler = copy.deepcopy(self.policy.noise_scheduler)
            losses = getattr(self.policy, self.method)(batch)
        else:
            losses = self.policy(*batch)
        if isinstance(losses, torch.Tensor):
            losses = {"loss": losses}
        return losses, {}


class ParallelLoss:
    def __init__(self, module, devices):
        self.devices = devices
        self.module = module
        self.parallel = (
            nn.DataParallel(module, device_ids=devices) if len(devices) > 1 else None
        )
        self.reported_devices = set()

    def __call__(self, batch):
        if self.parallel is None:
            return self.module.compute(batch)
        result = self.parallel(batch)
        counts = result["count"]
        losses = {
            key: value.sum() / counts.sum()
            for key, value in result["loss_sums"].items()
        }
        used = set(result["device"].tolist())
        if not used <= self.reported_devices:
            print(
                json.dumps(
                    {
                        "event": "training_devices",
                        "devices": sorted(used),
                        "samples_per_device": counts.tolist(),
                    }
                ),
                flush=True,
            )
            self.reported_devices.update(used)
        return losses, result["predictions"]
