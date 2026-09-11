import json

import pytest

from skynet_app.adapters.egoverse_runtime import joint_data, validate_episode_split
from skynet_app.adapters.egoverse_splits import validate_split
from skynet_app.policy_exports_api import ExportRequest
from test_policy_exports import setup, digest


def test_single_episode_export_preserves_source_and_separate_dataset(setup):
    service, session, source = setup
    full = service.create(session['id'], 'egoverse', 'All recordings')
    before = {str(p): digest(p) for p in source.rglob('*') if p.is_file()}
    job = service.create(session['id'], 'egoverse', 'Episode 2 overfit', overfit_episode=1)
    assert job['resource_id'] != full['resource_id']
    assert len(job['sources']) == 1 and job['sources'][0]['index'] == 1
    assert service.create(session['id'], 'egoverse', 'Other name', overfit_episode=1)['id'] == job['id']
    service.prepare(job['id'])
    result = service.get(job['id'])
    assert result['state'] == 'READY', result
    manifest = json.loads(service.artifact(job['id'], 'manifest.json').read_text())
    assert len(manifest['episodes']) == 1
    assert manifest['episodes'][0]['source_index'] == 1
    assert manifest['split'] == {'mode': 'single_episode_overfit', 'train': [0], 'validation': [0]}
    assert 'disjoint_episode_split' not in manifest['validation']['checks']
    root = service.root / job['id'] / 'output'
    assert len(list(root.rglob('*.zarr'))) == 1, 'no duplicated validation recording'
    validate_episode_split(root, manifest)
    assert manifest['episodes'][0]['steps'] == 3
    data = joint_data(root, 1, 0, overfit=True)
    folders = [data[key]['skynet_joints']['resolver']['folder_path'] for key in ('train_datasets', 'valid_datasets')]
    assert folders == [str(root / 'dataset/train')] * 2
    assert {str(p): digest(p) for p in source.rglob('*') if p.is_file()} == before
    assert service.database.get_data_resource(full['resource_id'])['metadata']['display_name'] == 'All recordings'


@pytest.mark.parametrize('changes', [
    {'format': 'dp'}, {'overfit_episode': 99}, {'overfit_episode': True},
    {'resource_id': 'existing'}, {'selections': [{'session_id': 'session-1', 'indices': [0]}]},
])
def test_invalid_overfit_is_rejected_before_registration(setup, changes):
    service, session, _ = setup
    kwargs = dict(session_id=session['id'], format='egoverse', name='Overfit', overfit_episode=0)
    kwargs.update(changes)
    before = len(service.list())
    with pytest.raises(ValueError):
        service.create(**kwargs)
    assert len(service.list()) == before


def test_overlap_is_only_allowed_for_explicit_single_episode_mode():
    with pytest.raises(ValueError, match='exactly one'):
        validate_split({'train': [0], 'validation': [0]}, 1)
    valid = {'mode': 'single_episode_overfit', 'train': [0], 'validation': [0]}
    validate_split(valid, 1)
    for split, count in [(valid, 2), ({**valid, 'validation': [1]}, 2), ({**valid, 'train': [False]}, 1)]:
        with pytest.raises(ValueError):
            validate_split(split, count)
    assert ExportRequest(session_id='source', name='One', format='egoverse', overfit_episode=0).overfit_episode == 0


def test_held_out_suite_rejects_overfit_before_submission():
    from skynet_app.evaluation_contracts import bind_suite_to_dataset
    suite = {'config_json': {'dataset_episode_binding': {'role': 'training_data', 'metadata_path': 'split.validation'}}}
    spec = {'data': {'bundle': {'assignments': [{'role': 'training_data', 'version': {'metadata': {'split': {'mode': 'single_episode_overfit', 'train': [0], 'validation': [0]}}}}]}}}
    with pytest.raises(ValueError, match='no held-out episodes'):
        bind_suite_to_dataset(suite, spec)
