"""Optional Skynet validation control leaves data splits and old defaults intact."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from skynet_app.adapters import _apply_parameter_flags
from skynet_app.adapters.unidex_manifest import manifest
from skynet_app.adapters import unidex_runtime as runtime
from test_unidex_runtime import normalizer, recording_fixture


@pytest.mark.parametrize('enabled', [None, True, False])
def test_validation_native_field_uses_shared_boolean_binding(enabled):
    declared = manifest()
    path = 'native.config.validation_enabled'
    field = next(field for field in declared.train.input_fields if field.path == path)
    assert field.kind == 'boolean' and field.default is True
    config = {} if enabled is None else {'validation_enabled': enabled}
    flags, blockers = [], []
    _apply_parameter_flags(flags, {'native': {'config': config}},
                           {path: declared.train.parameter_flags[path]}, blockers)
    assert not blockers
    expected = [] if enabled is None else ['--validation-enabled' if enabled else '--no-validation-enabled']
    assert flags == expected
    args = runtime.parser().parse_args(['--repository', '/repo', '--dataset', '/dataset', '--output', '/output', *flags])
    assert args.validation_enabled is (True if enabled is None else enabled)
    assert runtime.validation_options(args.validation_enabled) == (
        {'limit_val_batches': 0, 'num_sanity_val_steps': 0} if enabled is False else {})


def test_validation_requires_a_real_boolean():
    binding = manifest().train.parameter_flags['native.config.validation_enabled']
    for value in (0, 1, 'false', None):
        with pytest.raises(ValueError, match='boolean'):
            runtime.validation_options(value)
    flags, blockers = [], []
    _apply_parameter_flags(flags, {'native': {'config': {'validation_enabled': 'false'}}},
                           {'native.config.validation_enabled': binding}, blockers)
    assert not flags and any('boolean' in message for message in blockers)


def test_disabling_validation_changes_resume_identity_but_default_does_not(tmp_path):
    _, metadata = recording_fixture(tmp_path, count=2)
    selections = [dict(position=0, version_id='version', manifest_sha256='a' * 64, metadata=metadata)]
    args = SimpleNamespace(batch_size=4, learning_rate=.0001, num_workers=1, seed=42,
                           precision='fp32', gpu_count=4, gradient_accumulation=8, epochs=32, max_steps=30000)
    legacy = runtime.training_identity(args, {}, {}, {}, selections=selections)
    args.validation_enabled = True
    default = runtime.training_identity(args, {}, {}, {}, selections=selections)
    assert legacy == default and 'validation_enabled' not in default['training']
    args.validation_enabled = False
    disabled = runtime.training_identity(args, {}, {}, {}, selections=selections)
    assert disabled['training']['validation_enabled'] is False
    saved = {'skynet': {**legacy, 'identity_sha256': runtime.stable_digest(legacy)},
             'optimizer_states': [{}], 'lr_schedulers': [{}],
             'loops': {'fit_loop': {'state_dict': {'combined_loader': [{'schema': 'skynet.unidex-loader-state/v1'}]}}}}
    runtime.validate_resume(saved, default)
    with pytest.raises(ValueError, match='identical'):
        runtime.validate_resume(saved, disabled)


@pytest.mark.parametrize('enabled', [True, False])
def test_main_reads_only_training_split_when_validation_disabled(tmp_path, monkeypatch, enabled):
    root, metadata = recording_fixture(tmp_path, count=64)
    original_split = deepcopy(metadata['split'])
    selections = [dict(position=0, path=str(root), version_id='version', manifest_sha256='a' * 64, metadata=metadata)]
    dataset_class = runtime.UniDexCollectionDataset
    opened = []
    def dataset(*args):
        opened.append(args[1])
        return dataset_class(*args)
    monkeypatch.setattr(runtime, 'UniDexCollectionDataset', dataset)
    monkeypatch.setattr(runtime, 'verify_repository', lambda _: None)
    monkeypatch.setattr(runtime, 'load_training_selections', lambda _: selections)
    monkeypatch.setattr(runtime, 'load_native_config', lambda *args: ({'horizon_steps': 30}, normalizer().config, {}))
    output = tmp_path / 'output'
    argv = ['unidex_runtime.py', '--repository', str(tmp_path), '--dataset', str(root), '--output', str(output),
            '--config-only', '--validation-enabled' if enabled else '--no-validation-enabled']
    monkeypatch.setattr(runtime.sys, 'argv', argv)
    runtime.main()
    assert opened == (['train', 'validation'] if enabled else ['train'])
    assert metadata['split'] == original_split
    mixture = json.loads((output / 'mixture.json').read_text())
    assert mixture['epoch_windows'] == sum(metadata['episodes'][i]['steps'] - 30 + 1 for i in original_split['train'])


def test_real_lightning_disabled_validation_never_reads_or_predicts_holdout(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    lightning = pytest.importorskip('pytorch_lightning')
    pytest.importorskip('hydra')
    from test_unidex_runtime_checkpoint import training_parts
    wrapper, loader, sampler, identity = training_parts(max_steps=4)
    identity['training'] = {'validation_enabled': False}
    def unexpected(*args, **kwargs):
        pytest.fail('Disabled validation executed a holdout read or model inference')
    monkeypatch.setattr(wrapper.policy, 'infer_action', unexpected)
    class UnreadHoldout(torch.utils.data.Dataset):
        def __len__(self):
            return 2
        __getitem__ = unexpected
    trainer = lightning.Trainer(accelerator='cpu', devices=1, max_steps=4, max_epochs=-1,
        accumulate_grad_batches=2, precision='32-true', logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, default_root_dir=str(tmp_path),
        **runtime.validation_options(False))
    trainer.fit(wrapper, loader, torch.utils.data.DataLoader(UnreadHoldout()))
    assert trainer.global_step == 4 and wrapper.policy.seen
    rows = [json.loads(line) for line in (tmp_path / 'logs.json.txt').read_text().splitlines()]
    assert all(row.get('val_loss') is None for row in rows)
    assert all('val_loss' not in row.get('metrics', {}) for row in rows)
