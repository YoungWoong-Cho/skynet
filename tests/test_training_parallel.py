"""Exact sample accounting and failure detection for shared distributed training."""
import math
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "skynet_app/adapters"))
from training_parallel import ShardedBatchSampler, TrainingContext, training_batch_size


def test_batch_sizes_keep_declared_semantics():
    assert training_batch_size(8, 4, 2) == 32
    assert training_batch_size(32, 4, 2, "global_before_accumulation") == 32
    assert training_batch_size(64, 4, 2, "global_effective") == 32
    with pytest.raises(ValueError, match="divisible"):
        training_batch_size(25, 4, 2, "global_effective")
    with pytest.raises(ValueError, match="one sample"):
        training_batch_size(2, 4, 1, "global_before_accumulation")


@pytest.mark.parametrize("size", [1, 2, 4, 5, 26, 29, 32])
@pytest.mark.parametrize("shuffle", [False, True])
def test_each_sample_contributes_exactly_once_with_uneven_and_empty_ranks(size, shuffle):
    samplers = [ShardedBatchSampler(size, 12, rank, 4, shuffle, 47) for rank in range(4)]
    all_indices=[]
    for epoch in [0, 1]:
        for sampler in samplers: sampler.set_epoch(epoch)
        batches = [list(sampler) for sampler in samplers]
        assert all(len(b) == math.ceil(size / 12) for b in batches)
        actual=[]
        for step in range(len(batches[0])):
            assert sum(s.valid_count(step) for s in samplers)==samplers[0].global_count(step)
            for rank, sampler in enumerate(samplers):
                assert batches[rank][step]  # zero-weight padding participates in DDP
                actual.extend(batches[rank][step][:sampler.valid_count(step)])
        assert sorted(actual)==list(range(size))
        all_indices.append(actual)
    if shuffle and size > 4: assert all_indices[0]!=all_indices[1]


def cpu_context():
    context=object.__new__(TrainingContext)
    context.world_size=1
    context.device=torch.device("cpu")
    return context


def test_corrupt_sample_counts_fail_instead_of_generating_tiny_loss():
    context=cpu_context()
    with pytest.raises(ValueError, match="sample count mismatch"):
        context.mean(40.0, 4519575824443980770, 32)
    assert context.mean(40.0, 32, 32)==1.25
    for bad in [float('nan'), float('inf'), -1.0]:
        with pytest.raises(ValueError, match="invalid loss"):
            context.check_loss(torch.tensor(bad))


def test_accumulated_gradients_are_normalized_by_real_samples():
    context=cpu_context()
    parameter=torch.nn.Parameter(torch.tensor([2.0]))
    (parameter.square()*3).backward()
    context.normalize_gradients([parameter], 3, 3)
    torch.testing.assert_close(parameter.grad, torch.tensor([4.0]))
    with pytest.raises(ValueError, match="sample count mismatch"):
        context.normalize_gradients([parameter], 3, 4)
