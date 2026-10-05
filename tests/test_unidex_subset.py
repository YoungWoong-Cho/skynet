"""Exact source-data budgets and rank-level sample counts, with real shared reads."""
from copy import deepcopy
import hashlib

import pytest

from skynet_app.adapters.recording_time import resolve_collection_sampling
from skynet_app.adapters.unidex_data import UniDexMixtureSampler, UniDexDataset
from skynet_app.adapters.unidex_subset import apply_frame_budget, subset_windows
from skynet_app.adapters.unidex_input import default_pointcloud_recipe


def selections(hands=2, counts=(100, 103, 120, 90)):
    result = []
    for hand in range(hands):
        episodes = [dict(index=i, id=f'{hand}-{i}', steps=count, hand_id=f'hand-{hand}',
                    streams={'scene_front_pointcloud': {'recipe': default_pointcloud_recipe()}},
                    capture={'step_dt': 1 / 60}, source={'sha256': hashlib.sha256(f'{hand}-{i}'.encode()).hexdigest()})
                    for i, count in enumerate(counts)]
        result.append(dict(position=hand, version_id=str(hand), manifest_sha256=str(hand) * 64,
                           metadata={'contract': 'skynet.unidex-pointcloud-faas/v1',
                                     'episodes': episodes, 'split': {'train': [0, 1, 2], 'validation': [3]}}))
    return result


def frame_union(rows, plan):
    frames, windows = set(), []
    for row, sampled in zip(rows, plan['datasets']):
        selected = subset_windows(row['metadata'], sampled['sampling'])
        for index, start in selected:
            episode = row['metadata']['episodes'][index]
            stride = sampled['sampling']['episodes'][index]['stride']
            for frame in range(start, start + plan['action_steps'] * stride, stride):
                frames.add((episode['source']['sha256'], frame))
            windows.append((row['version_id'], index, start))
    return frames, windows


def test_exact_budget_preserves_validation_and_never_counts_interpolated_frames():
    rows = selections()
    full = resolve_collection_sampling(rows, 30, 30, require_validation=True)
    before = deepcopy(full)
    limited = apply_frame_budget(rows, full, 180, selection_seed=123)
    frames, windows = frame_union(rows, limited)
    assert len(frames) == 180
    assert all(frame % 2 == 0 for _, frame in frames)
    assert len(windows) == limited['splits']['train']['windows']
    assert full == before
    assert limited['splits']['validation'] == full['splits']['validation']
    for row in limited['source_subset']['per_hand'].values():
        assert row['unique_source_frames'] == 90
    assert limited == apply_frame_budget(rows, full, 180, selection_seed=123)
    assert limited['source_subset']['source_frame_union_sha256'] != apply_frame_budget(rows, full, 180, selection_seed=124)['source_subset']['source_frame_union_sha256']


def test_data_selection_does_not_depend_on_input_order():
    rows = selections()
    first = apply_frame_budget(rows, resolve_collection_sampling(rows, 30, 30), 180)
    reversed_rows = list(reversed(rows))
    for i, row in enumerate(reversed_rows):
        row['position'] = i
    second = apply_frame_budget(reversed_rows, resolve_collection_sampling(reversed_rows, 30, 30), 180)
    assert first['source_subset']['source_frame_union_sha256'] == second['source_subset']['source_frame_union_sha256']


def test_a_short_recording_hand_cannot_silently_disappear_from_the_budget():
    rows = selections()
    for episode in rows[1]['metadata']['episodes']:
        episode['steps'] = 10
    full = resolve_collection_sampling(rows,30,30,require_validation=True)
    with pytest.raises(ValueError,match='Every selected training hand'):
        apply_frame_budget(rows,full,180)


@pytest.mark.parametrize('budget,match', [(181, 'divide equally'), (20, 'complete action chunk'), (1000, 'exceeds'), (True, 'integer')])
def test_bad_budget_fails_before_loading_a_model(budget, match):
    rows = selections()
    with pytest.raises(ValueError, match=match):
        apply_frame_budget(rows, resolve_collection_sampling(rows, 30, 30), budget)


@pytest.mark.parametrize('change', ['validation', 'bounds', 'source', 'stride'])
def test_frozen_view_rejects_validation_leakage_or_changed_source(change):
    rows = selections()
    plan = apply_frame_budget(rows, resolve_collection_sampling(rows, 30, 30), 180)
    sampled = plan['datasets'][0]['sampling']
    segment = sampled['training_subset']['segments'][0]
    if change == 'validation': segment['episode_index'] = 3
    elif change == 'bounds': segment['start'] = 1000
    elif change == 'source': segment['source_sha256'] = 'f' * 64
    elif change == 'stride': segment['stride'] = 1
    with pytest.raises(ValueError):
        subset_windows(rows[0]['metadata'], sampled)


def test_distributed_fixed_budget_updates_always_consume_a_full_effective_batch():
    hands = ['a'] * 23 + ['b'] * 24
    samplers = [UniDexMixtureSampler(hands, 'hand_balanced', seed=1701, rank=rank, replicas=4, batch_size=4)
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


def test_subset_reader_uses_the_existing_physical_streams(tmp_path):
    from test_unidex_runtime import recording_fixture, normalizer
    from skynet_app.adapters.recording_dataset import close_handles, digest
    root, metadata = recording_fixture(tmp_path, count=128)
    rows = [dict(position=0, version_id='original', path=str(root),
                 manifest_sha256=digest(root / 'manifest.json'), metadata=metadata)]
    plan = apply_frame_budget(rows, resolve_collection_sampling(rows, 30, 30), 40)
    original = {str(path): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    dataset = UniDexDataset(root, metadata, 'train', normalizer(), sampling=plan['datasets'][0]['sampling'])
    valid = UniDexDataset(root, metadata, 'validation', normalizer(), sampling=plan['datasets'][0]['sampling'])
    assert len(dataset) == 11 and len(valid) == 35
    assert dataset[0]['action'].shape == (30, 82)
    assert {str(path): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()} == original
    close_handles()


def test_submission_and_worker_resolve_the_identical_fixed_data_view():
    from skynet_app.recording_sampling import experiment_sampling
    from skynet_app.adapters.dataset_inputs import resolve_data_selections
    from skynet_app.adapters.unidex_manifest import manifest
    rows = selections()
    assignments = []
    for row in rows:
        row['metadata']['format'] = 'skynet.recording-dataset/v1'
        row['metadata']['registered_version_id'] = row['version_id']
        assignments.append(dict(role='training_data',position=row['position'],
            version=dict(path='/original/'+row['version_id'],manifest_sha256=row['manifest_sha256'],metadata=row['metadata']),
            config=dict(location=dict(path='/original/'+row['version_id'],kind='cluster',status='AVAILABLE',
                                      manifest_sha256=row['manifest_sha256']))))
    document = dict(data=dict(bundle=dict(assignments=assignments)),
        native=dict(config=dict(control_hz=30,action_steps=30,unique_source_frames=180,data_selection_seed=123)))
    planned = experiment_sampling(document, manifest())
    resolved = resolve_data_selections(document)
    actual = apply_frame_budget(resolved,resolve_collection_sampling(resolved,30,30,require_validation=True),180,selection_seed=123)
    assert planned == actual
    document['native']['config']['unique_source_frames'] = 181
    with pytest.raises(ValueError,match='divide equally'):
        experiment_sampling(document,manifest())
