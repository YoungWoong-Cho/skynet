"""ACT consumes a selected frame grid while shared source recordings stay intact."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_recording_dataset import source, request, prepare
from recording_dataset import RecordingDataset, digest
from recording_time import resolve_sampling
from act_training import RecordedACTDataset, read_sample
from act_native_data import NativeACTDataset, CAMERAS, prepare_recorded_act, validate_recorded_act
from act_native_checkpoint import identity
from xpolicy_runtime import checkpoint_sampling
from skynet_app.adapters.policy_contract import recorded_contract, simulation_stride


@pytest.fixture
def recordings(tmp_path):
    sources = [source(tmp_path, index=i, images=True)[0] for i in range(2)]
    spec = request(tmp_path, sources, rgb=True)
    manifest = prepare(spec)
    return Path(spec['output']), manifest


def files(root):
    return {str(p): digest(p) for p in root.rglob('*') if p.is_file()}


def test_bridge_samples_states_rgb_commands_and_normalization_on_one_grid(recordings):
    root, manifest = recordings
    before = files(root.parent)
    data = RecordingDataset(root, control_hz=30)
    stats = data.normalization()
    raw = RecordingDataset(root)
    np.testing.assert_allclose(stats['action_mean'], raw.episode(0).joint('action', [0, 2]).mean(axis=0))
    dataset = RecordedACTDataset(data, manifest, stats, 'train', chunk=3)
    assert dataset.samples == [(0, 0), (0, 1)]
    qpos, images, commands, mask = read_sample(data, 0, 1, stats, 3)
    np.testing.assert_allclose(qpos, (raw.episode(0).joint('state', 2) - stats['state_mean']) / stats['state_std'])
    assert images[:, 0, 0, 0].tolist() == [12, 22, 32]
    np.testing.assert_allclose(commands[0], (raw.episode(0).joint('action', 2) - stats['action_mean']) / stats['action_std'])
    np.testing.assert_array_equal(commands[1:], 0)
    assert mask.tolist() == [False, True, True]
    assert data.manifest == manifest and files(root.parent) == before


def test_native_alignment_uses_previous_sampled_command_and_masks_short_sequences(recordings, monkeypatch):
    root, manifest = recordings
    data = RecordingDataset(root, control_hz=30)
    stats = data.normalization()
    stats['qpos_mean'], stats['qpos_std'] = stats.pop('state_mean'), stats.pop('state_std')
    dataset = NativeACTDataset(data, [0], CAMERAS, stats, action_steps=50)
    monkeypatch.setattr(np.random, 'randint', lambda size: size - 1)
    images, state, actions, mask = dataset.sample(0)
    assert images[:, 0, 0, 0].tolist() == [12, 32, 22]
    # Native real-recording ACT starts at max(0, sampled observation step - 1).
    expected = (RecordingDataset(root).episode(0).joint('action', [0, 2]) - stats['action_mean']) / stats['action_std']
    np.testing.assert_allclose(actions[:2], expected)
    assert actions.shape == (50, 7) and mask.tolist() == [False, False] + [True] * 48
    assert data.episode_steps(0) == 2 and manifest['episodes'][0]['steps'] == 4


def test_native_private_mapping_pins_sampling_without_copying_or_mutating_recordings(recordings, tmp_path):
    root, manifest = recordings
    before = files(root)
    workspace = tmp_path / 'run'
    directory = workspace / 'XPolicyLab/policy/ACT'
    directory.mkdir(parents=True)
    (directory / 'utils.py').write_text('def load_data(*args): pass\n')
    prepare_recorded_act(root, manifest, workspace, validate_recorded_act(root, manifest), control_hz=30, action_steps=3)
    mapping = json.loads((directory / 'skynet-recording-mapping.json').read_text())
    assert mapping['recording_sampling'] == resolve_sampling(manifest, 30, 3, window_policy='pad', require_validation=True)
    namespace = {}
    exec((directory / 'utils.py').read_text(), namespace)
    assert namespace['load_data'].keywords['control_hz'] == 30
    assert namespace['load_data'].keywords['action_steps'] == 3
    assert not list(workspace.rglob('*.hdf5')) and files(root) == before


def test_conversion_structure_accepts_one_episode_but_native_training_requires_validation(tmp_path):
    src, _, _ = source(tmp_path, images=True)
    spec = request(tmp_path, [src], rgb=True)
    manifest = prepare(spec)
    root = Path(spec['output'])
    dimensions = validate_recorded_act(root, manifest)
    directory = tmp_path / 'verify/XPolicyLab/policy/ACT'
    directory.mkdir(parents=True)
    (directory / 'utils.py').write_text('def load_data(*args): pass\n')
    prepare_recorded_act(root, manifest, tmp_path / 'verify', dimensions, action_steps=1,
                         structural_only=True)
    assert json.loads((directory / 'skynet-recording-mapping.json').read_text())['recording_sampling'] is None
    with pytest.raises(ValueError, match='validation'):
        resolve_sampling(manifest, 30, 50, window_policy='pad', require_validation=True)


def test_checkpoint_sampling_rejects_frequency_and_chunk_changes(recordings, monkeypatch):
    _, manifest = recordings
    config = {'control_hz': 30, 'action_steps': 3}
    receipt = resolve_sampling(manifest, 30, 3, window_policy='pad', require_validation=True)
    assert checkpoint_sampling(manifest, config, receipt, require_validation=True) == receipt
    for change in ({'control_hz': 15}, {'action_steps': 4}):
        with pytest.raises(ValueError, match='sampling differs'):
            checkpoint_sampling(manifest, {**config, **change}, receipt, require_validation=True)
    monkeypatch.setenv('SKYNET_ACT_REVISION', 'source')
    monkeypatch.setenv('SKYNET_ACT_MANIFEST_SHA', 'manifest')
    native = {'native_args': {'chunk_size': 3}, 'policy_config': {'num_queries': 3}, 'state_dim': 7}
    train = SimpleNamespace(dataset=SimpleNamespace(episode_ids=[0], recording_sampling=receipt))
    val = SimpleNamespace(dataset=SimpleNamespace(episode_ids=[1], recording_sampling=receipt))
    assert identity(native, train, val)['recording_sampling'] == receipt
    val.dataset.recording_sampling = resolve_sampling(manifest, 15, 3, window_policy='pad', require_validation=True)
    with pytest.raises(ValueError, match='sampling must agree'):
        identity(native, train, val)


def test_selected_control_period_preserves_source_simulator_period(recordings):
    _, manifest = recordings
    before = deepcopy(manifest)
    contract = recorded_contract(manifest, control_hz=30)
    assert contract['source_step_dt'] == 1 / 60 and contract['step_dt'] == 1 / 30
    assert simulation_stride(contract, 1 / 60) == 2
    assert manifest == before
    with pytest.raises(ValueError, match='differs from the recording'):
        simulation_stride(contract, 1 / 30)
    for hz in (120, 24):
        with pytest.raises(ValueError, match='divide'):
            recorded_contract(manifest, control_hz=hz)
