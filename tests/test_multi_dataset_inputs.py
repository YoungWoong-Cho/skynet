"""Multiple prepared results remain independent immutable experiment inputs."""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from skynet_app.adapters import (
    AdapterInputField, DataBundleInputBinding, TrainingProgressContract,
    TrainingProgressJsonlSource, canonical_adapter_manifest,
)
from skynet_app.adapters.dataset_inputs import resolve_data_selections
from skynet_app.adapters.recording_time import resolve_collection_sampling
from skynet_app.adapters.unidex_manifest import manifest as unidex_manifest
from skynet_app.adapters.unidex_input import default_pointcloud_recipe
from skynet_app.data_selection import snapshot
from skynet_app.database import Database
from skynet_app.model_io import resolve_model_io, adapter_io_contract
from skynet_app.pipeline_api import (
    PipelineService, parse_declared_training_progress, resolve_training_progress_contract,
    _training_progress_contract, training_progress_summary,
)
from skynet_app.recording_sampling import experiment_sampling
from test_experiments import make_spec


def metadata(index=0, hz=60, lengths=(121, 121), validation=True):
    return dict(format='skynet.recording-dataset/v1', contract='skynet.unidex-pointcloud-faas/v1',
                validation={'status': 'PASSED'}, registered_version_id=f'version-{index}',
                capture={'action_joint_names': ['a', 'b', 'c'], 'robot_joint_names': ['a', 'b', 'c'], 'cameras': {key: {'height': 224, 'width': 224} for key in ('scene_front', 'scene_left', 'scene_right')}},
                episodes=[dict(index=i, id=f'episode-{i}', steps=steps, capture={'step_dt': 1/hz},
                               source={'sha256': f'{index * 100 + i:064x}'},
                               streams={'scene_front_pointcloud': {'recipe': default_pointcloud_recipe()}})
                          for i, steps in enumerate(lengths)],
                split=dict(train=[0] if validation else list(range(len(lengths))),
                           validation=list(range(1, len(lengths))) if validation else []))


def assignment(index=0, **kwargs):
    sha = f'{index + 1000:064x}'
    return dict(role='training_data', position=index,
                resource=dict(provider='collection', namespace='datasets', name=f'hand-{index}', kind='dataset'),
                version=dict(format='skynet.recording-dataset/v1', revision='1', status='READY',
                             path=f'/dataset/{index}', manifest_sha256=sha, metadata=metadata(index, **kwargs)),
                config={'location': dict(kind='cluster', status='AVAILABLE', path=f'/dataset/{index}', manifest_sha256=sha)})


def document(count=2):
    return dict(native={'config': {'control_hz': 15, 'action_steps': 30}},
                data={'bundle': dict(id='selection:test', name='Inputs', version='experiment-inputs',
                     manifest_sha256='b'*64, assignments=[assignment(i) for i in range(count)])})


def test_many_binding_materializes_ordered_composites_and_validates_every_result():
    declared = unidex_manifest()
    doc = document(7)
    doc['data']['bundle']['assignments'].reverse()
    PipelineService._apply_manifest_data_bindings(doc, declared)
    inputs = doc['native']['config']['datasets']
    assert len(inputs) == 7
    assert [item['position'] for item in inputs] == list(range(7))
    assert inputs == resolve_data_selections(doc, runtime=True)
    assert inputs[6]['path'] == '/dataset/6' and inputs[6]['metadata']['registered_version_id'] == 'version-6'
    assert PipelineService._apply_manifest_data_bindings(doc, declared) == doc
    for mutation, message in [
        (lambda a: a['version']['metadata'].update(contract='unsupported'), 'data contract'),
        (lambda a: a['version'].update(format='unrelated'), 'incompatible'),
        (lambda a: a['config']['location'].update(manifest_sha256='f'*64), 'verified'),
        (lambda a: a['config']['location'].update(status='MISSING'), 'verified'),
    ]:
        broken = document()
        mutation(broken['data']['bundle']['assignments'][1])
        with pytest.raises(ValueError, match=message):
            PipelineService._apply_manifest_data_bindings(broken, declared)


def test_runtime_selections_preserve_sampling_and_provenance_without_copying_render_receipts():
    from skynet_app.adapters.dataset_inputs import validate_selection_sources
    from skynet_app.evaluation_targets import embodiment_ids

    doc = document(7)
    for index, item in enumerate(doc['data']['bundle']['assignments']):
        data = item['version']['metadata']
        data['capture'].update(robot=f'hand-{index}', step_dt=1/60)
        data['shared_artifacts'] = [{'render_receipt': 'x' * 40000}]
        for episode in data['episodes']:
            episode.update(hand_id=f'hand-{index}', streams={'state': {'render_receipt': 'x' * 40000}})
            episode['capture'].update(data['capture'])
            episode['source']['path'] = '/original/episode.pkl'
    before = deepcopy(doc)
    full = resolve_data_selections(doc)
    small = resolve_data_selections(doc, runtime=True)
    assert doc == before
    assert len(json.dumps(small)) < len(json.dumps(full)) / 20
    assert resolve_collection_sampling(small, 15, 30) == resolve_collection_sampling(full, 15, 30)
    assert [embodiment_ids(item['metadata']) for item in small] == [{f'hand-{i}'} for i in range(7)]
    for complete, projected in zip(full, small):
        assert {key: value for key, value in complete.items() if key != 'metadata'} == {
            key: value for key, value in projected.items() if key != 'metadata'}
        assert 'shared_artifacts' not in projected['metadata']
        assert 'streams' not in projected['metadata']['episodes'][0]
        assert 'cameras' not in projected['metadata']['capture']
    target = deepcopy(small[-1])
    target.update(version_id='target', manifest_sha256='f'*64)
    with pytest.raises(ValueError, match='Duplicate source recording'):
        validate_selection_sources([small[-1], target])
    target['metadata']['episodes'][0]['source'] = small[-1]['metadata']['episodes'][1]['source']
    with pytest.raises(ValueError, match='both training and validation'):
        validate_selection_sources([small[-1], target])
    small[0]['metadata']['episodes'][0]['source']['sha256'] = 'e'*64
    assert doc == before


def test_runtime_projection_keeps_unknown_contracts_and_full_evaluation_target_receipts():
    doc = document(1)
    assignment = doc['data']['bundle']['assignments'][0]
    assignment['role'] = 'evaluation_target'
    assignment['version']['metadata']['episodes'][0]['streams'] = {'scene_front_pointcloud': {'recipe': 'frozen geometry'}}
    assert resolve_data_selections(doc, 'evaluation_target')[0]['metadata'] == assignment['version']['metadata']
    assignment['role'] = 'training_data'
    assignment['version']['metadata']['contract'] = 'custom-adapter/needs-complete-metadata'
    assert resolve_data_selections(doc, runtime=True) == resolve_data_selections(doc)
    assignment['version']['metadata']['contract'] = 'skynet.unidex-pointcloud-faas/v1'
    assignment['version']['metadata']['validation']['status'] = 'PENDING'
    assert resolve_data_selections(doc, runtime=True) == resolve_data_selections(doc)


def test_old_full_derived_selection_is_compacted_but_real_overrides_are_rejected():
    doc = document(2)
    full = resolve_data_selections(doc)
    doc['native']['config']['datasets'] = deepcopy(full)
    PipelineService._apply_manifest_data_bindings(doc, unidex_manifest())
    assert doc['native']['config']['datasets'] == resolve_data_selections(doc, runtime=True)
    assert resolve_data_selections(doc) == full
    for key, value in [('path', '/changed'), ('manifest_sha256', 'e' * 64)]:
        broken = document(2)
        broken['native']['config']['datasets'] = resolve_data_selections(broken)
        broken['native']['config']['datasets'][0][key] = value
        with pytest.raises(ValueError, match='conflicts'):
            PipelineService._apply_manifest_data_bindings(broken, unidex_manifest())


def test_one_binding_still_rejects_extra_positions_and_many_declaration_is_unambiguous():
    declared = unidex_manifest().model_copy(deep=True)
    declared.train.input_fields = [AdapterInputField(path='native.config.dataset_path', label='Dataset', kind='string',
        data_binding=DataBundleInputBinding(role='training_data', value_path='location.path'))]
    with pytest.raises(ValueError, match='cannot consume training_data at position 1'):
        PipelineService._apply_manifest_data_bindings(document(), declared)
    one = document(1)
    PipelineService._apply_manifest_data_bindings(one, declared)
    assert one['native']['config']['dataset_path'] == '/dataset/0'
    with pytest.raises(ValidationError, match='complete role'):
        DataBundleInputBinding(role='training_data', cardinality='many', position=1, value_path='selection')
    with pytest.raises(ValidationError, match='kind json'):
        AdapterInputField(path='native.config.inputs', label='Datasets', kind='string',
                          data_binding=DataBundleInputBinding(role='training_data', cardinality='many', value_path='selection'))
    broken = document()
    broken['data']['bundle']['assignments'][1]['position'] = 3
    with pytest.raises(ValueError, match='consecutive'):
        PipelineService._apply_manifest_data_bindings(broken, unidex_manifest())


def test_frozen_composites_cannot_be_overridden_or_repeated():
    doc = document()
    doc['native']['config']['datasets'] = [dict(path='/different')]
    with pytest.raises(ValueError, match='conflicts'):
        PipelineService._apply_manifest_data_bindings(doc, unidex_manifest())
    for key in ('manifest_sha256', 'registered_version_id'):
        doc = document()
        first, second = [item['version'] for item in doc['data']['bundle']['assignments']]
        if key == 'manifest_sha256':
            second[key] = first[key]
            doc['data']['bundle']['assignments'][1]['config']['location'][key] = first[key]
        else:
            second['metadata'][key] = first['metadata'][key]
        with pytest.raises(ValueError, match='Duplicate dataset'):
            resolve_data_selections(doc)
    doc = document()
    first, second = [item['version']['metadata'] for item in doc['data']['bundle']['assignments']]
    second['episodes'][0]['source'] = first['episodes'][0]['source']
    with pytest.raises(ValueError, match='Duplicate source recording'):
        resolve_data_selections(doc)
    second['episodes'][0]['source'] = first['episodes'][1]['source']
    with pytest.raises(ValueError, match='both training and validation'):
        resolve_data_selections(doc)


def test_collection_sampling_uses_every_source_rate_and_preserves_episode_boundaries():
    doc = document()
    doc['data']['bundle']['assignments'][1] = assignment(1, hz=30, lengths=(61, 61))
    selected = resolve_data_selections(doc)
    before = deepcopy(doc)
    with pytest.raises(ValueError, match='common control frequency'):
        resolve_collection_sampling(selected, action_steps=30)
    plan = resolve_collection_sampling(selected, 15, 30, require_validation=True)
    assert [item['sampling']['episodes'][0]['stride'] for item in plan['datasets']] == [4, 2]
    assert plan['splits']['train']['windows'] == 4
    assert plan['splits']['validation']['windows'] == 4
    assert doc == before
    assert experiment_sampling(doc, unidex_manifest()) == plan
    with pytest.raises(ValueError, match='exactly'):
        resolve_collection_sampling(selected, 24, 30)


def test_submission_rejects_mixed_frozen_point_counts_without_rewriting_selected_data():
    doc = document()
    for episode in doc['data']['bundle']['assignments'][1]['version']['metadata']['episodes']:
        episode['streams']['scene_front_pointcloud']['recipe']['num_points'] = 1024
    before = deepcopy(doc)
    with pytest.raises(ValueError, match='different point-cloud recipes'):
        experiment_sampling(doc, unidex_manifest())
    assert doc == before
    for episode in doc['data']['bundle']['assignments'][0]['version']['metadata']['episodes']:
        episode['streams']['scene_front_pointcloud']['recipe']['num_points'] = 1024
    assert experiment_sampling(doc, unidex_manifest())['splits']['train']['windows'] == 4


def test_empty_individual_splits_and_short_datasets_use_collection_eligibility():
    doc = document(3)
    doc['data']['bundle']['assignments'][0] = assignment(0, lengths=(1,), validation=False)
    doc['data']['bundle']['assignments'][1] = assignment(1, lengths=(121,), validation=False)
    plan = resolve_collection_sampling(resolve_data_selections(doc), 15, 30, require_validation=True)
    assert plan['datasets'][0]['sampling']['splits']['train']['windows'] == 0
    assert plan['datasets'][1]['sampling']['splits']['validation']['windows'] == 0
    assert plan['splits']['train']['windows'] == 4
    assert plan['splits']['validation']['windows'] == 2
    assert plan['splits']['train']['excluded_episodes'] == ['version-0/episode-0']
    with pytest.raises(ValueError, match='No usable validation'):
        resolve_collection_sampling(resolve_data_selections(doc)[:2], 15, 30, require_validation=True)


def test_sweeps_recheck_all_selected_inputs_and_sampling(monkeypatch):
    doc = document()
    doc['data']['bundle']['assignments'][1] = assignment(1, hz=30, lengths=(61, 61))
    declared = unidex_manifest()
    spec = make_spec(**doc, sweep={'strategy': 'grid', 'axes': {'native.config.control_hz': [15, 20]}, 'seeds': [1]})
    monkeypatch.setattr(PipelineService, '_validate_manifest_input_fields', lambda *a: None)
    with pytest.raises(ValueError, match='exactly'):
        PipelineService._validate_sweep_inputs(spec, declared)
    bound = document()
    PipelineService._apply_manifest_data_bindings(bound, declared)
    spec = make_spec(**bound, sweep={'strategy': 'grid', 'axes': {'native.config.datasets': [[{'path': '/tampered'}]]}, 'seeds': [1]})
    with pytest.raises(ValueError, match='conflicts'):
        PipelineService._validate_sweep_inputs(spec, declared)


def test_model_io_requires_dimensions_from_every_dataset_to_agree():
    manifest = {'train': {'model_io': adapter_io_contract('egoverse-hpt-joints').model_dump(mode='json')}}
    doc = document()
    same = resolve_model_io(manifest, doc)
    assert same['compatible'] and same['resolved'] and same['dataset_count'] == 2
    doc['data']['bundle']['assignments'][1]['version']['metadata']['capture']['action_joint_names'].append('d')
    different = resolve_model_io(manifest, doc)
    assert not different['compatible'] and not different['resolved']
    assert dict(different['entries'])['Output · Joint commands'] == 'Varies across selected datasets'


def test_multiple_versions_snapshot_without_creating_or_changing_datasets(tmp_path):
    db = Database(tmp_path / 'multi-inputs')
    selections = []
    for index in range(2):
        resource = db.create_data_resource(category='dataset', provider='collection', namespace='data', source_key=f'hand-{index}', kind='dataset')
        version = db.create_data_resource_version(resource['id'], revision='1', format='skynet.recording-dataset/v1',
            path=f'/dataset/{index}', manifest_sha256=f'{index+1000:064x}', metadata=metadata(index))
        location = db.record_data_location(version['id'], kind='cluster', host='sky2', path=version['path'], manifest_sha256=version['manifest_sha256'])
        selections.append(dict(version_id=version['id'], location_id=location['id'], role='training_data', position=index))
    frozen = snapshot(db, selections)
    assert len(resolve_data_selections({'data': {'bundle': frozen}})) == 2
    assert db.list_data_bundles() == []
    assert len(db.list_datasets()) == 2
    db.update_dataset(selections[1]['version_id'], display_name='Renamed', archived=True)
    assert frozen['assignments'][1]['version']['metadata']['display_name'] != 'Renamed'
    with pytest.raises(ValueError, match='archived'):
        snapshot(db, selections)


def test_optional_contract_fields_do_not_change_old_adapter_hashes():
    declared = unidex_manifest().model_dump(mode='json')
    field = declared['train']['input_fields'][0]
    field['kind'] = 'string'
    field['data_binding'].update(cardinality='one', value_path='location.path')
    explicit = canonical_adapter_manifest(declared)
    del field['data_binding']['cardinality']
    assert canonical_adapter_manifest(declared) == explicit
    progress = TrainingProgressContract(source=TrainingProgressJsonlSource(path='log.jsonl', completed_key='step', required_key='step'))
    assert 'step_source' not in progress.model_dump(mode='json')


def test_progress_step_budget_uses_shared_reader_in_parser_and_run_summary():
    progress = TrainingProgressContract(unit='epoch', total_path='native.config.epochs',
        source=TrainingProgressJsonlSource(path='log.jsonl', completed_key='epoch', completed_offset=1, required_key='epoch'),
        step_source=TrainingProgressJsonlSource(path='log.jsonl', completed_key='global_step', required_key='global_step'))
    spec = dict(train={'max_steps': 50}, native={'config': {'epochs': 32}},
                source={'adapter_manifest': {'train': {'progress': progress.model_dump(mode='json')}}})
    content = json.dumps(dict(epoch=0, global_step=7)) + '\n'
    assert parse_declared_training_progress(content, progress, resolved_spec=spec)[0]['total'] == 50
    assert parse_declared_training_progress(content, progress, resolved_spec=spec)[0]['completed'] == 7
    contract, _ = _training_progress_contract({'resolved_spec_json': spec})
    assert contract.unit == 'step' and contract.source.completed_key == 'global_step'
    summary = training_progress_summary({'resolved_spec_json': spec})
    assert summary['unit'] == 'step' and summary['total'] == 50
    spec['train'] = {}
    epoch = resolve_training_progress_contract(progress, spec)
    assert epoch.unit == 'epoch' and epoch.source.completed_key == 'epoch'
    records = parse_declared_training_progress(content, progress, resolved_spec=spec)
    assert records[0]['completed'] == 1 and records[0]['total'] == 32
