"""Shared DDP execution, exact dataset sharding, and checked loss accounting.

Each process owns one GPU. All inter-process tensors use the configured
collective transport; model outputs never use cross-GPU CUDA gather/copy.
"""

import datetime
import json
import math
import os
import subprocess
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler


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


def launch_distributed(count):
    """Launch workers once, retaining the selected runtime and exact arguments."""
    validate_devices(count)
    if count == 1 or "RANK" in os.environ:
        return False
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
         f"--nproc_per_node={count}", sys.argv[0], *sys.argv[1:]],
        check=True,
    )
    return True


def training_batch_size(value, gpu_count, accumulation, semantics="per_device"):
    if semantics in {"per_device", "repository_native"}:
        return value * gpu_count
    if semantics == "global_before_accumulation":
        result = value
    elif semantics == "global_effective":
        if value % accumulation:
            raise ValueError("Effective batch size must be divisible by gradient accumulation")
        result = value // accumulation
    else:
        raise ValueError("Unknown batch semantics: " + semantics)
    if result < gpu_count:
        raise ValueError("The training batch must contain at least one sample per GPU")
    return result


class ShardedBatchSampler(Sampler):
    """Every original sample contributes once, including uneven final batches.

    Empty ranks receive one zero-weight sample so all DDP workers participate
    in the same number of forwards/backwards. No padding sample affects loss.
    """
    def __init__(self, size, batch_size, rank=0, world_size=1, shuffle=False, seed=42):
        if size < 1 or batch_size < world_size or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed dataset or batch size")
        self.size, self.batch_size = size, batch_size
        self.rank, self.world_size = rank, world_size
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def global_count(self, index):
        return min(self.batch_size, self.size - index * self.batch_size)

    def valid_count(self, index):
        return len(range(self.rank, self.global_count(index), self.world_size))

    def __iter__(self):
        indices = (torch.randperm(self.size, generator=torch.Generator().manual_seed(
            self.seed + self.epoch)).tolist() if self.shuffle else list(range(self.size)))
        for start in range(0, self.size, self.batch_size):
            batch = indices[start:start + self.batch_size]
            yield batch[self.rank::self.world_size] or batch[:1]


class TrainingContext:
    def __init__(self, count):
        import torch.distributed as dist
        self.dist = dist
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world_size != count:
            raise ValueError("Training worker count differs from the allocated GPU count")
        validate_devices(count)
        torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", str(count))) // count))
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        self.primary = self.rank == 0
        if count > 1:
            dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=300), device_id=self.device)
            self.preflight()

    def preflight(self):
        # Exercise reduction and every broadcast root before building the policy.
        # Sentinel payloads include scalar metadata and longer gradient-like data.
        for size in (1, 17, 8192):
            expected = torch.arange(size, dtype=torch.float64, device=self.device)
            value = expected + self.rank * 17
            self.dist.all_reduce(value)
            target = expected * self.world_size + 17 * self.world_size * (self.world_size - 1) // 2
            if not torch.equal(value, target):
                raise RuntimeError("GPU communication check failed: incorrect reduction")
            for root in range(self.world_size):
                value = expected + self.rank * 13
                self.dist.broadcast(value, root)
                if not torch.equal(value, expected + root * 13):
                    raise RuntimeError("GPU communication check failed: incorrect broadcast")
        if self.primary:
            print(json.dumps(dict(event="training_devices", backend="ddp", devices=list(range(self.world_size)),
                communication="host_memory" if os.environ.get("NCCL_P2P_DISABLE") == "1" else "automatic",
                preflight="passed")), flush=True)

    def loader(self, dataset, batch_size, workers=0, shuffle=False, seed=42):
        sampler = ShardedBatchSampler(len(dataset), batch_size, self.rank, self.world_size, shuffle, seed)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
            pin_memory=True, **({"multiprocessing_context": "spawn", "persistent_workers": True} if workers else {}))

    def wrap(self, module):
        module.to(self.device)
        if self.world_size == 1:
            return module
        from torch.nn.parallel import DistributedDataParallel
        return DistributedDataParallel(module, device_ids=[self.local_rank], output_device=self.local_rank,
            find_unused_parameters=True)

    def total(self, values):
        result = torch.tensor(values, device=self.device, dtype=torch.float64)
        if self.world_size > 1:
            self.dist.all_reduce(result)
        if not torch.isfinite(result).all():
            raise ValueError("Training aggregation produced nonfinite values")
        return result.tolist()

    def mean(self, loss_sum, count, expected_count):
        summed, actual_count = self.total([loss_sum, count])
        if actual_count != expected_count or actual_count <= 0:
            raise ValueError(f"Training sample count mismatch: expected {expected_count}, got {actual_count}")
        return summed / actual_count

    def check_loss(self, loss, *, nonnegative=True):
        invalid = not bool(torch.isfinite(loss).all()) or (nonnegative and float(loss.detach()) < 0)
        if self.total([int(invalid)])[0]:
            raise ValueError("Training produced an invalid loss")

    def backward(self, loss, count):
        # DDP averages gradients; preserve sample-weighted sums until the update.
        (loss * count * self.world_size).backward()

    def normalize_gradients(self, parameters, local_count, expected_count):
        count = self.total([local_count])[0]
        if count != expected_count or count <= 0:
            raise ValueError(f"Training sample count mismatch: expected {expected_count}, got {count}")
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(count)

    def close(self):
        if self.world_size > 1 and self.dist.is_initialized():
            self.dist.destroy_process_group()


class LossModule(nn.Module):
    def forward(self, batch):
        return self.compute(batch)


class PolicyLoss(LossModule):
    def __init__(self, policy, method=None):
        super().__init__()
        self.policy, self.method = policy, method

    def compute(self, batch):
        losses = getattr(self.policy, self.method)(batch) if self.method else self.policy(*batch)
        if isinstance(losses, torch.Tensor):
            losses = {"loss": losses}
        return losses, {}
