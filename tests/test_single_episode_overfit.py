import json

import pytest

from skynet_app.adapters.egoverse_runtime import joint_data, validate_episode_split
from skynet_app.adapters.egoverse_splits import validate_split
from skynet_app.policy_exports_api import ExportRequest
from test_policy_exports import setup, digest, create, manifest_path


def test_single_episode_export_preserves_source_and_separate_dataset(setup):
    service, session, source = setup
    full = create(service, session['id'], 'egoverse', 'All recordings')
    before = {str(p): digest(p) for p in source.rglob('*') if p.is_file()}
    job = create(service, session['id'], 'egoverse', 'Episode 2 overfit', overfit_episode=1)
    assert job['resource_id'] != full['resource_id']
    assert len(job['sources']) == 1 and job['sources'][0]['index'] == 1
    assert create(service, session['id'], 'egoverse', 'Other name', overfit_episode=1)['id'] == job['id']
    service.prepare(job['id'])
    result = service.get(job['id'])
    assert result['state'] == 'READY', result
    manifest = json.loads(manifest_path(service,job).read_text())
    assert len(manifest['episodes']) == 1
    assert manifest['episodes'][0]['source_index'] == 1
    assert manifest['split']['mode'] == 'training_only'
    assert manifest['split']['train'] == [0] and manifest['split']['validation'] == []
    root = service.root / job['id'] / 'output'
    assert not list(root.rglob('*.zarr')), 'shared manifest stores no duplicate episode payload'
    validate_episode_split(root, manifest)
    assert manifest['episodes'][0]['steps'] == 3
    assert [p.name for p in root.iterdir()] == ['manifest.json']
    assert {str(p): digest(p) for p in source.rglob('*') if p.is_file()} == before
    assert service.database.get_data_resource(full['resource_id'])['display_name'] == 'All recordings'


@pytest.mark.parametrize('changes', [
    {'overfit_episode': 99}, {'overfit_episode': True},
    {'selections': [{'session_id': 'session-1', 'indices': [0]}]},
])
def test_invalid_overfit_is_rejected_before_registration(setup, changes):
    service, session, _ = setup
    kwargs = dict(session_id=session['id'], kind='egoverse', name='Overfit', overfit_episode=0)
    kwargs.update(changes)
    before = len(service.list())
    with pytest.raises(ValueError):
        create(service, **kwargs)
    assert len(service.list()) == before


def test_overlap_is_only_allowed_for_explicit_single_episode_mode():
    with pytest.raises(ValueError, match='exactly one'):
        validate_split({'train': [0], 'validation': [0]}, 1)
    valid = {'mode': 'single_episode_overfit', 'train': [0], 'validation': [0]}
    validate_split(valid, 1)
    for split, count in [(valid, 2), ({**valid, 'validation': [1]}, 2), ({**valid, 'train': [False]}, 1)]:
        with pytest.raises(ValueError):
            validate_split(split, count)
    assert ExportRequest(session_id='source', name='One', adapter_id='adapter', adapter_version_id='version', overfit_episode=0).overfit_episode == 0


def test_held_out_suite_rejects_overfit_before_submission():
    from skynet_app.evaluation_contracts import bind_suite_to_dataset
    suite = {'config_json': {'dataset_episode_binding': {'role': 'training_data', 'metadata_path': 'split.validation'}}}
    spec = {'data': {'bundle': {'assignments': [{'role': 'training_data', 'version': {'metadata': {'split': {'mode': 'single_episode_overfit', 'train': [0], 'validation': [0]}}}}]}}}
    with pytest.raises(ValueError, match='no held-out episodes'):
        bind_suite_to_dataset(suite, spec)


def test_recording_links_follow_exact_result_and_copied_episode_ownership(setup, monkeypatch):
    import copy
    service, original, _ = setup
    full = create(service, original['id'], 'egoverse', 'All recordings')
    subset = create(service, original['id'], 'egoverse', 'Episode 1', overfit_episode=0)
    service.prepare(full['id'])
    service.prepare(subset['id'])
    full, subset = service.get(full['id']), service.get(subset['id'])
    assert full['state'] == subset['state'] == 'READY'
    before = {row['id']: row['recording_ids'] for row in service.database.list_datasets()}
    assert before[full['version_id']] == [original['id']]
    assert before[subset['version_id']] == [original['id']]
    derived = copy.deepcopy(original)
    derived.update(id='derived', recordings=original['recordings'][:1])
    resource = service.database.get_data_resource(subset['resource_id'])
    service.database.update_data_resource(resource['id'], metadata={**resource['metadata'], 'recording_session_id': derived['id']})
    monkeypatch.setattr(service.live, 'list', lambda **_: [derived, original])
    monkeypatch.setattr(service.live, 'get', lambda identifier: derived if identifier == derived['id'] else original)
    links = {row['id']: row['recording_ids'] for row in service.database.list_datasets()}
    assert links == {full['version_id']: [original['id']], subset['version_id']: [derived['id']]}
    assert service.dataset(derived, create=False)['id'] == subset['resource_id']
    assert service.get(subset['id'])['sources'][0]['session_id'] == original['id'], 'immutable source lineage is unchanged'


def test_overfit_of_a_one_episode_recording_keeps_its_dataset_link(setup):
    service, session, _ = setup
    session['recordings'] = session['recordings'][:1]
    job = create(service, session['id'], 'egoverse', 'One episode', overfit_episode=0)
    assert service.dataset(session, create=False)['id'] == job['resource_id']
    service.prepare(job['id'])
    job = service.get(job['id'])
    assert job['state'] == 'READY'
    assert service.database.get_dataset(job['version_id'])['recording_ids'] == [session['id']]


@pytest.mark.parametrize("format", ["egoverse", "act", "fixture-rgb"])
def test_one_recording_prepares_without_a_validation_split(setup, format):
    service, session, _ = setup
    session["recordings"] = session["recordings"][:1]
    job = create(service, session["id"], format, "One recording")
    assert job["split"]["train"] == [0]
    assert job["split"]["validation"] == []
    assert job["split"]["mode"] == "training_only"
    service.prepare(job["id"])
    result = service.get(job["id"])
    assert result["state"] == "READY", result
    manifest = json.loads(manifest_path(service,job).read_text())
    assert manifest["split"]["validation"] == []
    from skynet_app.adapters.recording_dataset import verify_dataset
    verify_dataset(service.root / job['id'] / 'output')
