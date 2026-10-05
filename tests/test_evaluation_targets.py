from copy import deepcopy

import pytest

from skynet_app.evaluation_contracts import bind_suite_to_dataset
from skynet_app.evaluation_targets import validate_unidex_target, attach_evaluation_target, UNIDEX_CONTRACT


def selection(hand, digest="a"):
    capture = dict(robot=hand, hand="right", task="Dexverse-PickCube-v0", source_revision="b"*40,
                   action_joint_names=["joint"], action_scale=[1.], action_offset=[0.])
    return dict(position=0, version_id=hand, path="/datasets/"+hand, manifest_sha256=digest*64,
                metadata=dict(contract=UNIDEX_CONTRACT, format="skynet.recording-dataset/v1", validation={"status":"PASSED"}, capture=capture,
                              episodes=[dict(hand_id=hand, capture=deepcopy(capture))]))


def training(*hands):
    assignments=[]
    for i, hand in enumerate(hands):
        item=selection(hand, chr(ord('a')+i))
        assignments.append(dict(role="training_data", position=i,
            version=dict(format="skynet.recording-dataset/v1", manifest_sha256=item['manifest_sha256'], metadata=item['metadata']),
            config=dict(location=dict(kind="cluster", status="AVAILABLE", path=item['path'], manifest_sha256=item['manifest_sha256']))))
    return dict(source=dict(adapter="unidex"), data=dict(bundle=dict(assignments=assignments)))


def suite(target=None):
    config=dict(tasks=["Dexverse-PickCube-v0", "Dexverse-StackCube-v0"], task_selection_mode="subset",
                dataset_task_binding=dict(role="training_data", metadata_path="capture.task"))
    if target:
        config['target_dataset']=target
    return dict(config_json=config)


def test_unseen_target_is_independent_and_does_not_mutate_training_inputs():
    spec=training('shadow', 'leap'); before=deepcopy(spec)
    target=selection('wuji')
    assert validate_unidex_target(spec, target, True)=='wuji'
    bound=bind_suite_to_dataset(suite(target), spec)
    assert bound['config_json']['tasks']==['Dexverse-PickCube-v0']
    assert bound['config_json']['task_source']=='evaluation_dataset'
    assert spec==before
    assert suite(target)['config_json']['tasks']==['Dexverse-PickCube-v0','Dexverse-StackCube-v0']


def test_seen_hand_requires_explicit_seen_evaluation_even_if_another_dataset():
    spec=training('shadow', 'leap')
    with pytest.raises(ValueError, match="training inputs"):
        validate_unidex_target(spec, selection('leap', 'f'), True)
    assert validate_unidex_target(spec, selection('leap', 'f'), False)=='leap'
    target=selection('wuji'); target['metadata']['episodes'].append(selection('leap')['metadata']['episodes'][0])
    with pytest.raises(ValueError, match="exactly one hand"):
        validate_unidex_target(spec, target, True)


def test_target_requires_consistent_task_format_and_fresh_scene_reset():
    spec=training('shadow'); target=selection('wuji')
    target['metadata']['contract']='another'
    with pytest.raises(ValueError, match="prepared UniDex"):
        validate_unidex_target(spec, target, True)
    target=selection('wuji'); target['metadata']['episodes'][0]['capture']['task']='wrong'
    with pytest.raises(ValueError, match="different task"):
        validate_unidex_target(spec, target, True)
    target_suite=suite(selection('wuji')); target_suite['config_json']['initial_state']='single_training_episode'
    with pytest.raises(ValueError, match="fresh simulator reset"):
        bind_suite_to_dataset(target_suite,spec)
    with pytest.raises(ValueError, match="Choose an evaluation dataset"):
        validate_unidex_target(spec,None,True)


def test_non_unidex_target_cannot_silently_change_native_policy_hand():
    spec=training('shadow'); spec['source']['adapter']='egoverse-hpt'
    with pytest.raises(ValueError, match="UniDex bridge"):
        attach_evaluation_target(None,suite(),spec,'wuji')


def test_registered_target_freezes_exact_version_and_survives_display_rename(tmp_path):
    from skynet_app.database import Database
    db=Database(tmp_path/'evaluation-target')
    target=selection('wuji','f')
    resource=db.create_data_resource(category='dataset', provider='collection', namespace='data', source_key='wuji', kind='dataset')
    version=db.create_data_resource_version(resource['id'], revision='1', format='skynet.recording-dataset/v1',
        path=target['path'], manifest_sha256=target['manifest_sha256'], metadata=target['metadata'])
    db.record_data_location(version['id'],kind='cluster',host='sky2',path=target['path'],manifest_sha256=target['manifest_sha256'])
    spec=training('shadow'); before=deepcopy(spec)
    frozen=attach_evaluation_target(db,suite(),spec,version['id'],True)['config_json']
    assert frozen['unseen_embodiment'] is True
    assert frozen['target_dataset']['version_id']==version['id']
    assert frozen['target_dataset']['manifest_sha256']==target['manifest_sha256']
    db.update_dataset(version['id'],display_name='Renamed',archived=True)
    assert frozen['target_dataset']['metadata']['display_name']!='Renamed'
    assert spec==before
    with pytest.raises(ValueError,match='archived'):
        attach_evaluation_target(db,suite(),spec,version['id'],True)


def test_rollout_viewer_uses_target_hand_and_never_attaches_a_training_demo(monkeypatch):
    import json
    from types import SimpleNamespace
    from skynet_app import pipeline_api, rollout_preview
    episode=dict(id='episode',task='Dexverse-PickCube-v0')
    evaluation=dict(run_id='run',stage_id='stage',episodes=[episode],result_path='/evaluation/result.json')
    run=dict(resolved_spec_json=training('shadow'),stages=[dict(id='stage',resolved_config_json={
        'context':{'target_dataset':selection('wuji')}})])
    database=SimpleNamespace(get_evaluation=lambda _:evaluation,get_run=lambda _, **kwargs:run)
    cluster=SimpleNamespace(run_with_fallback=lambda *a,**k:('sky2',json.dumps({'schema':'skynet.episode-viewer/v1'})))
    monkeypatch.setattr(pipeline_api,'service',SimpleNamespace(database=database,cluster=cluster))
    missing=pipeline_api.get_evaluation_episode_viewer('evaluation','episode')
    assert missing['robot']=='wuji'
    def forbidden(*a,**k):
        raise AssertionError('An unseen rollout must not attach a training-hand demonstration')
    monkeypatch.setattr(rollout_preview,'enrich_demonstration',forbidden)
    episode['video_path']='/evaluation/videos/episode.mp4'
    assert pipeline_api.get_evaluation_episode_viewer('evaluation','episode')['state']=='READY'


@pytest.mark.parametrize("adapter,contract", [("unidex", UNIDEX_CONTRACT), ("human-policy-hat", "skynet.hat-rgb-fingertips/v1"), ("egoverse-hpt", None)])
def test_browser_and_validation_share_target_contract(adapter, contract):
    from skynet_app.evaluation_targets import evaluation_target_contract
    assert evaluation_target_contract({"source": {"adapter": adapter}}) == contract


def test_hat_rejects_unidex_targets_but_accepts_held_out_rgb_hand():
    spec=training('shadow');spec['source']['adapter']='human-policy-hat'
    target=selection('wuji')
    with pytest.raises(ValueError,match='HAT RGB'):
        validate_unidex_target(spec,target,True)
    target['metadata']['contract']='skynet.hat-rgb-fingertips/v1'
    assert validate_unidex_target(spec,target,True)=='wuji'
