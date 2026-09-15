import sys

import numpy as np
import pytest

from skynet_app.adapters.egoverse_manifest import manifests
from skynet_app.adapters import resolve_adapter_evaluation_plan
from skynet_app.adapters.policy_contract import recorded_contract, contract_issues, environment_mapping
from skynet_app.evaluation_compatibility import inspect_compatibility, compose_evaluator
from skynet_app.experiments import get_evaluation_catalog


def metadata():
    return {
        'contract': 'skynet.egoverse-rgb-joints/v1', 'policy_to_source_indices': [1, 0],
        'episodes': [{'source_index': 0}], 'split': {'train': [0], 'validation': []},
        'capture': {'robot': 'floating_shadow_right', 'hand': 'right', 'source_revision': 'a'*40,
                    'task': 'Dexverse-PickCube-v0', 'action_joint_names': ['wrist', 'finger'],
                    'action_scale': [1., 2.], 'action_offset': [0., .5], 'step_dt': 1/60,
                    'action_semantics': 'raw_joint_position_command; target = action * scale + offset',
                    'color_space': 'RGB', 'cameras': {name: {'width': 2, 'height': 2}
                        for name in ('scene_front', 'scene_left', 'scene_right')}}}


def suite(name='dexverse_recorded'):
    config = next(s.model_dump(mode='json') for s in get_evaluation_catalog() if s.suite == name)
    return {'name': name, 'evaluator_adapter': config['evaluator'], 'config_json': config}


def recorded_bundle():
    return {'id': 'bundle', 'name': 'Recorded', 'version': 'v1', 'manifest_sha256': 'a'*64,
            'assignments': [{'role': 'training_data', 'position': 0,
                'resource': {'provider': 'fixture', 'namespace': 'test', 'name': 'one', 'kind': 'dataset'},
                'version': {'revision': 'v1', 'format': 'egoverse-episodes-zarr/v1', 'path': '/prepared',
                            'manifest_sha256': 'a'*64, 'status': 'READY', 'metadata': metadata()}}]}


def setup_case():
    from test_egoverse_native_algorithms import native_spec
    manifest = next(m for m in manifests() if m.slug == 'egoverse-hpt')
    spec = native_spec(manifest, 'hpt_joints')
    document = spec.model_dump(mode='json', by_alias=True)
    document['data'] = {'bundle': {'assignments': [{'role': 'training_data', 'version': {'metadata': metadata()}}]}}
    return spec, document, manifest


def test_compose_hpt_with_suite_not_declared_by_training_adapter():
    spec, document, manifest = setup_case()
    assert all('dexverse_recorded' not in e.suites for e in manifest.evaluations)
    report, _ = inspect_compatibility(document, manifest, suite())
    assert report['ready'], report
    assert report['runtime_verification'] == 'pending'
    executor = compose_evaluator(document, manifest, suite())
    plan = resolve_adapter_evaluation_plan(spec, environment='isaac_lab', suite='dexverse_recorded',
        context={'compatibility': report}, manifest=executor)
    assert plan.argv[1].endswith('evaluation_workers.py')
    for name in ('policy_loading.py', 'recorded_policy_evaluation.py', 'policy_contract.py', 'evaluation_preflight.py'):
        assert 'adapter-support/'+name in plan.capsule_files
    assert all('dexverse_recorded' not in e.suites for e in manifest.evaluations), 'training manifest remains immutable'


def test_same_joint_count_is_not_a_matching_embodiment():
    contract = recorded_contract(metadata())
    assert environment_mapping(contract, ['finger', 'wrist'], [2.,1.], [.5,0.], 1/60) == [0,1]
    with pytest.raises(ValueError, match='identities'):
        environment_mapping(contract, ['thumb', 'arm'], [1.,2.], [0.,.5], 1/60)
    with pytest.raises(ValueError, match='scale or offset'):
        environment_mapping(contract, ['wrist','finger'], [1.,1.], [0.,.5], 1/60)
    with pytest.raises(ValueError, match='frequency'):
        environment_mapping(contract, ['wrist','finger'], [1.,2.], [0.,.5], 1/30)


def test_missing_evidence_and_unsupported_suite_remain_visible_but_not_ready():
    _, document, manifest = setup_case()
    report, _ = inspect_compatibility(document, manifest, suite('libero_10'))
    assert report['status'] == 'mapping_required'
    assert not report['ready'] and report['messages']
    document['data']['bundle']['assignments'][0]['version']['metadata']['capture'].pop('action_scale')
    report, _ = inspect_compatibility(document, manifest, suite())
    assert report['status'] == 'unknown'
    assert any('action_scale' in msg for msg in report['messages'])


def test_dataset_specific_rules_and_checkpoint_availability_are_enforced():
    _, document, manifest = setup_case()
    meta = document['data']['bundle']['assignments'][0]['version']['metadata']
    meta['episodes'].append({'source_index': 1})
    report, _ = inspect_compatibility(document, manifest, suite('dexverse_training_episode'))
    assert report['status'] == 'incompatible'
    assert any('single training episode' in msg for msg in report['messages'])
    report, _ = inspect_compatibility(document, manifest, suite(), {'status': 'AVAILABLE', 'sha256': 'a'*64, 'pruned_at': 'now'})
    assert report['status'] == 'incompatible'


def test_unsupported_submission_creates_no_stage_or_job():
    from types import SimpleNamespace
    from skynet_app.pipeline_api import PipelineService, EvaluationRequest
    _, document, manifest = setup_case()
    document['data'] = {'bundle': recorded_bundle()}
    target_suite = {**suite('libero_10'), 'id': 'suite'}
    run = {'id': 'run', 'resolved_spec_json': document, 'status': 'SUCCEEDED', 'stages': [], 'attempts': [], 'evaluations': []}
    service = PipelineService.__new__(PipelineService)
    service._resolve_evaluation_target = lambda *_: ({'valid': True, 'run_valid': True, 'checkpoint_valid': True, 'errors': {}}, run, {'id': 'cp', 'path': '/cp', 'sha256': 'a'*64, 'status': 'AVAILABLE'})
    service._resolve_evaluation_suite_selection = lambda *_: (target_suite, 'libero', ['task'], {})
    service._evaluation_source = lambda *_: ({}, manifest)
    def forbidden(*_, **__): raise AssertionError('No stage or job may be created for an incompatible selection')
    service.database = SimpleNamespace(create_stage=forbidden)
    service._submit_stage = forbidden
    service._resolve_evaluation_implementation = forbidden
    validation = service.validate_evaluation_target('run', '/cp', suite_id='suite', tasks=['task'])
    assert not validation['valid']
    assert validation['compatibility']['status'] == 'mapping_required'
    assert validation['plan_message'].startswith('No model loader')
    with pytest.raises(ValueError, match='No model loader'):
        service.create_evaluation(EvaluationRequest(run_id='run', suite_id='suite'))


@pytest.mark.parametrize('field,value', [('action_scale',[float('nan'),1.]), ('policy_to_source_indices',[1,1]), ('step_dt',0)])
def test_invalid_numeric_contracts_are_never_compatible(field,value):
    contract = recorded_contract(metadata()); contract[field] = value
    assert any(i['status'] == 'incompatible' for i in contract_issues(contract))


def test_real_probe_sequence_is_unscored_and_resets_before_scoring(monkeypatch):
    import skynet_app.adapters.policy_contract as contract_module
    monkeypatch.setitem(sys.modules, 'policy_contract', contract_module)
    from skynet_app.adapters.evaluation_preflight import verify_cycle
    contract = recorded_contract(metadata()); calls = []; reports = []
    observation = {'state': np.zeros(2), 'images': {name: np.zeros((2,2,3), dtype=np.uint8) for name in contract['cameras']}}
    def reset(): calls.append('reset')
    def observe(): calls.append('observe'); return observation
    def predict(obs): calls.append('predict'); return np.zeros((3,2))
    def advance(action): calls.append('advance'); assert action.shape == (2,)
    verify_cycle(contract, reset=reset, observe=observe, predict=predict, advance=advance, report=reports.append)
    assert calls == ['reset', 'observe', 'predict', 'advance', 'reset']
    assert reports[-1]['status'] == 'PASSED'
    calls.clear(); reports.clear()
    with pytest.raises(ValueError, match='output'):
        verify_cycle(contract, reset=reset, observe=observe, predict=lambda _: np.zeros((1,3)), advance=advance, report=reports.append)
    assert 'advance' not in calls and reports[-1]['status'] == 'FAILED'
    observation['images']['scene_front'] = np.zeros((3,2,3),dtype=np.uint8)
    with pytest.raises(ValueError, match='Camera'):
        verify_cycle(contract, reset=reset, observe=observe, predict=predict, advance=advance, report=reports.append)


def test_native_act_composes_the_native_loader_and_blocks_foreign_inputs():
    from skynet_app.adapters.xpolicy_native_manifest import manifests as native_manifests
    _, document, _ = setup_case()
    manifest = next(m for m in native_manifests() if m.slug == "xpolicylab-act-native")
    document["source"]["adapter"] = manifest.slug
    meta = document["data"]["bundle"]["assignments"][0]["version"]["metadata"]
    report, _ = inspect_compatibility(document, manifest, suite())
    assert report["status"] == "incompatible"
    meta["contract"] = "skynet.act-rgb-joints/v1"
    report, _ = inspect_compatibility(document, manifest, suite())
    assert report["ready"], report
    composed = compose_evaluator(document, manifest, suite())
    files = composed.evaluations[0].command.capsule_files
    for name in ("act_native_checkpoint.py", "act_native_evaluation.py", "act_native_data.py"):
        assert "adapter-support/" + name in files
    assert "xpolicylab-act-native" in files["adapter-support/xpolicy_evaluation.py"]
