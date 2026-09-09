"""Numerical checks for the shared training loss reducer; CUDA coverage runs on the cluster."""

import copy
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "skynet_app/adapters"))
from training_parallel import (
    ParallelLoss,
    PolicyLoss,
    training_batch_size,
    validate_devices,
)


class Regression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def compute_loss(self, batch):
        return (self.linear(batch["x"]) - batch["y"]).square().mean()


def test_batch_sizes_keep_the_declared_semantics():
    assert training_batch_size(4, 3, 2) == 12
    assert training_batch_size(12, 3, 2, "global_before_accumulation") == 12
    assert training_batch_size(24, 3, 2, "global_effective") == 12
    with pytest.raises(ValueError, match="divisible"):
        training_batch_size(25, 3, 2, "global_effective")
    with pytest.raises(ValueError, match="one sample"):
        training_batch_size(2, 3, 1, "global_before_accumulation")


def test_uneven_replica_batches_match_full_batch_loss_and_gradients():
    torch.manual_seed(1)
    model = Regression()
    reference = copy.deepcopy(model)
    batch = {"x": torch.randn(5, 3), "y": torch.randn(5, 2)}
    expected = reference.compute_loss(batch)
    expected.backward()
    module = PolicyLoss(model, "compute_loss")
    outputs = [
        module({k: v[a:b] for k, v in batch.items()}) for a, b in [(0, 3), (3, 5)]
    ]
    actual = sum(x["loss_sums"]["loss"].sum() for x in outputs) / sum(
        x["count"].sum() for x in outputs
    )
    actual.backward()
    torch.testing.assert_close(actual, expected)
    for parameter, target in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, target.grad)
    assert set(model.state_dict()) == set(
        reference.state_dict()
    ), "checkpoints retain native parameter names"


def test_one_gpu_path_keeps_original_policy_and_loss():
    model = Regression()
    loss = ParallelLoss(PolicyLoss(model, "compute_loss"), [0])
    batch = {"x": torch.randn(3, 3), "y": torch.randn(3, 2)}
    torch.testing.assert_close(loss(batch)[0]["loss"], model.compute_loss(batch))
    assert loss.module.policy is model


def test_missing_allocated_gpus_fails_instead_of_silently_using_one(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="only 1"):
        validate_devices(2)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two allocated CUDA GPUs"
)
def test_two_real_gpus_compute_and_reduce_the_same_gradient():
    torch.manual_seed(2)
    model = Regression().cuda(0)
    reference = copy.deepcopy(model)
    batch = {"x": torch.randn(7, 3), "y": torch.randn(7, 2)}
    expected = reference.compute_loss({k: v.cuda(0) for k, v in batch.items()})
    expected.backward()
    loss = ParallelLoss(PolicyLoss(model, "compute_loss"), validate_devices(2))
    actual = loss(batch)[0]["loss"]
    actual.backward()
    torch.testing.assert_close(actual, expected)
    for parameter, target in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, target.grad)
    assert loss.reported_devices == {0, 1}
