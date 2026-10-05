"""Deterministic hand mixture and resumable data position used by HAT training."""
from collections import Counter

import pytest

from skynet_app.adapters.hat_data import HandMixtureSampler


def test_sampler_distinguishes_window_weighting_from_hand_balance_and_replays_ddp():
    hands = ["leap"] * 5 + ["shadow"] * 96
    native = HandMixtureSampler(hands, seed=8)
    assert sorted(native.global_indices()) == list(range(101))
    assert Counter(hands[index] for index in native.global_indices()) == {"leap": 5, "shadow": 96}
    balanced = HandMixtureSampler(hands, "hand_balanced", seed=8)
    counts = Counter(hands[index] for index in balanced.global_indices())
    assert sorted(counts.values()) == [50, 51]
    for epoch in (0, 1, 12):
        balanced.set_epoch(epoch)
        global_order = balanced.global_indices()
        shards = []
        for rank in range(3):
            replica = HandMixtureSampler(hands, "hand_balanced", seed=8, rank=rank, replicas=3)
            replica.set_epoch(epoch)
            shards.append(list(replica))
            assert len(replica) == 34
        interleaved = [value for row in zip(*shards) for value in row]
        assert interleaved[:101] == global_order and interleaved[101:] == global_order[:1]
        resumed = HandMixtureSampler(hands, "hand_balanced", seed=8)
        resumed.set_epoch(epoch)
        assert resumed.global_indices() == global_order
    balanced.set_epoch(0)
    assert balanced.global_indices() != global_order
    assert native.identity()["policy"] == "window_proportional"


@pytest.mark.parametrize("policy", HandMixtureSampler.POLICIES)
def test_sampler_resume_tracks_completed_samples_independently_of_prefetch(policy):
    hands = ["shadow"] * 7 + ["leap"] * 3
    first = HandMixtureSampler(hands, policy, seed=17)
    first.set_epoch(3)
    expected = list(first)
    # A worker can request every index before the first batch completes.
    prefetched = iter(first)
    assert list(prefetched) == expected
    assert first.state_dict()["consumed_samples"] == 0
    first.mark_consumed(2)
    first.mark_consumed(2)
    resumed = HandMixtureSampler(hands, policy, seed=17)
    resumed.load_state_dict(first.state_dict())
    resumed.set_epoch(3)
    assert len(resumed) == len(hands)
    assert list(resumed) == expected[4:]
    resumed.mark_consumed(6)
    assert list(resumed) == []
    resumed.set_epoch(4)
    assert resumed.consumed_samples == 0
    assert len(list(resumed)) == len(hands)
    assert list(resumed) != expected


def test_distributed_resume_uses_same_completed_count_for_each_rank_sequence():
    hands = ["shadow"] * 7 + ["leap"] * 4
    ranks = [HandMixtureSampler(hands, "hand_balanced", seed=19, rank=i, replicas=2) for i in range(2)]
    for sampler in ranks:
        sampler.set_epoch(2)
    expected = [list(sampler) for sampler in ranks]
    ranks[0].mark_consumed(4)
    saved = ranks[0].state_dict()
    for rank in range(2):
        resumed = HandMixtureSampler(hands, "hand_balanced", seed=19, rank=rank, replicas=2)
        resumed.load_state_dict(saved)
        assert list(resumed) == expected[rank][4:]
        assert len(resumed) == 6
    with pytest.raises(ValueError, match="world size"):
        HandMixtureSampler(hands, "hand_balanced", seed=19).load_state_dict(saved)
    with pytest.raises(ValueError, match="identity"):
        HandMixtureSampler(hands, "window_proportional", seed=19, rank=0, replicas=2).load_state_dict(saved)
    with pytest.raises(ValueError, match="position"):
        ranks[1].load_state_dict({**saved, "consumed_samples": 99})
    with pytest.raises(ValueError, match="sample count"):
        ranks[0].mark_consumed(3)


def test_sampler_rejects_incomplete_epoch_reset_instead_of_replaying_or_skipping_data():
    sampler = HandMixtureSampler(["shadow"] * 8, seed=42)
    sampler.mark_consumed(2)
    with pytest.raises(ValueError, match="incomplete data epoch"):
        sampler.set_epoch(1)
    assert sampler.epoch == 0 and sampler.consumed_samples == 2


@pytest.mark.parametrize("mixing", HandMixtureSampler.POLICIES)
def test_every_accumulated_update_has_global_batch_128_including_epoch_tail(mixing):
    replicas, microbatch, accumulation, windows = 4, 4, 8, 877
    samplers = [HandMixtureSampler(["wuji2"] * windows, mixing, rank=rank, replicas=replicas,
                batch_size=microbatch, gradient_accumulation_steps=accumulation)
                for rank in range(replicas)]
    shards = [list(sampler) for sampler in samplers]
    assert {len(shard) for shard in shards} == {224}
    global_order = [value for row in zip(*shards) for value in row]
    assert len(global_order) == 896
    if mixing == "window_proportional":
        assert sorted(global_order[:windows]) == list(range(windows))
    for start in range(0, len(shards[0]), microbatch * accumulation):
        assert sum(len(shard[start:start + microbatch * accumulation]) for shard in shards) == 128
    identity = samplers[0].identity()
    assert identity["complete_batch_size"] == 4
    assert identity["gradient_accumulation_steps"] == 8
    assert identity["optimizer_step_sample_alignment_per_rank"] == 32
    assert samplers[0].epoch_plan(4) == dict(replicas=4, draws_before_padding=877,
        draws_after_padding=896, padding_draws=19, samples_per_rank=224, microbatches_per_rank=56,
        optimizer_updates=7, global_batch_size=128)
    for _ in range(8):
        samplers[0].mark_consumed(4)
    saved = samplers[0].state_dict()
    resumed = HandMixtureSampler(["wuji2"] * windows, mixing, rank=0, replicas=4,
                                  batch_size=4, gradient_accumulation_steps=8)
    resumed.load_state_dict(saved)
    assert list(resumed) == shards[0][32:]
    changed = HandMixtureSampler(["wuji2"] * windows, mixing, rank=0, replicas=4, batch_size=4)
    with pytest.raises(ValueError, match="identity"):
        changed.load_state_dict(saved)


def test_distributed_fixed_budget_updates_always_consume_a_full_effective_batch():
    hands = ['a'] * 23 + ['b'] * 24
    samplers = [HandMixtureSampler(hands, 'hand_balanced', seed=1701, rank=rank, replicas=4, batch_size=4)
                for rank in range(4)]
    assert {len(sampler) for sampler in samplers} == {12}
    rows = [list(sampler) for sampler in samplers]
    assert all(len(row) % 4 == 0 for row in rows)
    assert sum(len(row) for row in rows) == 48
    assert all(0 <= index < len(hands) for row in rows for index in row)
    for sampler in samplers:
        for _ in range(3): sampler.mark_consumed(4)
        state = sampler.state_dict()
        sampler.set_epoch(1)
        sampler.load_state_dict(state)
        assert list(sampler) == []
    assert samplers[0].identity()['complete_batch_size'] == 4
