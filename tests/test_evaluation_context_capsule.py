"""Large immutable dataset provenance must not be duplicated into worker input."""
from copy import deepcopy
import json

import pytest

from skynet_app.adapters import resolve_adapter_evaluation_plan
from skynet_app.adapters.dataset_inputs import (
    evaluation_context_json, resolve_data_selections, runtime_evaluation_context,
    validate_selection_sources,
)
from skynet_app.database import content_sha256
from skynet_app.evaluation_compatibility import compose_evaluator
from skynet_app.evaluation_targets import embodiment_ids
from skynet_app.adapters.recording_time import resolve_collection_sampling
from test_evaluation_compatibility import setup_case, suite
from test_evaluation_placement import evaluation_service
from test_multi_dataset_inputs import document


def large_context():
    doc = document(7)
    for index, assignment in enumerate(doc['data']['bundle']['assignments']):
        meta = assignment['version']['metadata']
        meta['capture'].update(robot=f'hand-{index}', step_dt=1 / 60)
        for episode in meta['episodes']:
            episode.update(hand_id=f'hand-{index}', session_id=f'session-{index}',
                           streams={'state': {'render_receipt': 'x' * 100_000}})
            episode['capture'].update(meta['capture'])
    inputs = resolve_data_selections(doc)
    target = deepcopy(inputs.pop())
    config = dict(initial_state='fresh_simulator_reset', target_dataset=target)
    return dict(
        compatibility=dict(policy_loader='unidex_faas', io_contract={'robot': 'hand-6', 'step_dt': 1 / 30}),
        policy=dict(native_config=dict(datasets=inputs, control_hz=30, action_steps=30)),
        target_dataset=target, unseen_embodiment=True,
        target_simulation_profile={'robot': 'hand-6', 'source_revision': 'a' * 40, 'asset_inventory_sha256': 'b' * 64},
        suite=dict(name='dexverse_recorded', config=config, config_sha256=content_sha256(config)),
        tasks=['Dexverse-PickCube-v0'], seeds=[42], episodes_per_task=20, parallelism=1,
    )


def test_large_context_keeps_full_planning_and_checksums_but_emits_small_worker_input():
    context = large_context()
    before = deepcopy(context)
    assert len(json.dumps(context)) > 1_000_000
    runtime = runtime_evaluation_context(context)
    payload = evaluation_context_json(context)
    assert len(payload.encode()) < 100_000
    assert context == before
    assert json.loads(payload) == runtime
    assert runtime_evaluation_context(runtime) == runtime
    assert runtime['suite']['planning_config_sha256'] == context['suite']['config_sha256']
    assert runtime['suite']['config_sha256'] == content_sha256(runtime['suite']['config'])
    for field in ('compatibility', 'target_simulation_profile', 'unseen_embodiment', 'tasks', 'seeds'):
        assert runtime[field] == context[field]
    assert runtime['target_dataset'] == runtime['suite']['config']['target_dataset']
    for full, small in zip([*context['policy']['native_config']['datasets'], context['target_dataset']],
                           [*runtime['policy']['native_config']['datasets'], runtime['target_dataset']]):
        assert {k: v for k, v in small.items() if k != 'metadata'} == {k: v for k, v in full.items() if k != 'metadata'}
        assert embodiment_ids(small['metadata']) == embodiment_ids(full['metadata'])
        assert 'streams' not in small['metadata']['episodes'][0]
    assert resolve_collection_sampling(runtime['policy']['native_config']['datasets'], 30, 30) == resolve_collection_sampling(context['policy']['native_config']['datasets'], 30, 30)


def test_compact_training_provenance_still_rejects_cross_split_source_leakage():
    runtime = runtime_evaluation_context(large_context())
    selected = runtime['policy']['native_config']['datasets']
    target = runtime['target_dataset']
    target['metadata']['episodes'][0]['source'] = selected[0]['metadata']['episodes'][1]['source']
    with pytest.raises(ValueError, match='both training and validation'):
        validate_selection_sources([*selected, target])


def test_other_evaluators_and_unknown_dataset_contracts_keep_custom_metadata():
    context = large_context()
    context['compatibility']['policy_loader'] = 'custom'
    assert runtime_evaluation_context(context) == context
    context['compatibility']['policy_loader'] = 'unidex_faas'
    context['target_dataset']['metadata']['contract'] = 'custom/manifest'
    assert runtime_evaluation_context(context)['target_dataset'] == context['target_dataset']


def test_common_evaluation_planner_compiles_large_context_without_losing_saved_snapshot():
    spec, document, manifest = setup_case()
    executor = compose_evaluator(document, manifest, suite())
    context = large_context()
    plan = resolve_adapter_evaluation_plan(spec, environment='isaac_lab', suite='dexverse_recorded',
        context=context, manifest=executor)
    assert plan.native_config['canonical_evaluation'] == context
    assert plan.capsule_files['adapter-support/evaluation-context.json'] == evaluation_context_json(context)
    assert len(plan.capsule_files['adapter-support/evaluation-context.json'].encode()) < 1_000_000
    # Retry dispatch adds identifiers but must use the same projection.
    dispatched = {**context, 'evaluation_id': 'evaluation', 'stage_id': 'stage'}
    from skynet_app.pipeline_api import evaluation_context_json as dispatch_json
    actual = json.loads(dispatch_json(dispatched))
    assert actual.pop('evaluation_id') == 'evaluation'
    assert actual.pop('stage_id') == 'stage'
    assert actual == json.loads(plan.capsule_files['adapter-support/evaluation-context.json'])


def test_dispatch_writes_identifiers_to_the_one_context_read_by_evaluators(evaluation_service, monkeypatch):
    from test_evaluation_placement import request_for
    from test_evaluation_dataset_lifecycle import dataset

    service, cluster, run, registered_suite = evaluation_service
    _, registered_target = dataset(service.database)
    resolve = service._resolve_evaluation_implementation
    original = {}

    def with_large_context(*args, **kwargs):
        spec, plan, context, *identities = resolve(*args, **kwargs)
        extra = large_context()
        target = extra['target_dataset']
        target.update(version_id=registered_target['id'], path=registered_target['path'],
                      manifest_sha256=registered_target['manifest_sha256'])
        for key in ('policy', 'compatibility', 'target_dataset', 'target_simulation_profile', 'unseen_embodiment'):
            context[key] = extra[key]
        context['suite']['config']['target_dataset'] = target
        context['suite']['config_sha256'] = content_sha256(context['suite']['config'])
        plan.native_config['canonical_evaluation'] = deepcopy(context)
        plan.capsule_files['adapter-support/evaluation-context.json'] = evaluation_context_json(context)
        original.update(deepcopy(context))
        return spec, plan, context, *identities

    monkeypatch.setattr(service, '_resolve_evaluation_implementation', with_large_context)
    result = service.create_evaluation(request_for(run, registered_suite))
    assert cluster.submit_count == 2  # Training fixture, then this evaluation.
    contexts = {name: payload for name, payload in cluster.capsule_files.items()
                if name.endswith('/evaluation-context.json')}
    assert len(contexts) == 1
    name, payload = next(iter(contexts.items()))
    assert name.endswith('/adapter-support/evaluation-context.json')
    emitted = json.loads(payload)
    assert emitted['evaluation_id'] == result['id']
    assert emitted['stage_id'] == result['stage_id']
    expected = runtime_evaluation_context({**original, 'evaluation_id': result['id'], 'stage_id': result['stage_id']})
    assert emitted == expected
    stage = next(s for s in service.database.get_run(run['id'])['stages'] if s['id'] == result['stage_id'])
    assert stage['resolved_config_json']['context'] == original
    assert len(json.dumps(original)) > 1_000_000
    assert len(payload.encode()) < 100_000
