"""Check native batch sharding without duplicating observations or goals."""
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "skynet_app/adapters"))
from robomimic_training import shard_batch


@pytest.mark.parametrize("size", [1, 2, 5, 8])
def test_native_batches_preserve_observation_action_alignment(size):
    batch = {"obs": {"state": torch.arange(size).reshape(size, 1)},
             "actions": torch.arange(size).reshape(size, 1), "goal_obs": None}
    seen = []
    for rank in range(4):
        local = shard_batch(batch, rank, 4, size)
        assert local["goal_obs"] is None
        assert torch.equal(local["obs"]["state"], local["actions"])
        count = len(range(rank, size, 4))
        assert len(local["actions"]) == max(1, count)
        seen.extend(local["actions"][:count].flatten().tolist())
    assert sorted(seen) == list(range(size))
