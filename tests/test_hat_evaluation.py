from copy import deepcopy
import numpy as np
import pytest
from skynet_app.adapters.hat_evaluation import hat_contract
from skynet_app.adapters.hat_manifest import manifest
from skynet_app.evaluation_compatibility import compose_evaluator, inspect_compatibility
from skynet_app.adapters.dataset_inputs import runtime_data_selection
from ops.datasets.action_codecs.unidex import load_codec


def metadata(hand="skynet_wuji_2_right"):
    codec=load_codec(hand);capture=codec.spec.capture
    capture['color_space']='RGB'
    capture['cameras']={name:dict(sensor=name,mount='fixed_scene',width=256,height=256,intrinsic_matrix=np.eye(3).tolist(),position_world=[0,0,0],quaternion_world_ros=[1,0,0,0]) for name in ['scene_front','scene_left','scene_right']}
    return dict(format='skynet.recording-dataset/v1',contract='skynet.hat-rgb-fingertips/v1',validation={'status':'PASSED'},capture=capture,policy_to_source_indices=list(range(codec.action_dim)),episodes=[dict(capture=capture,action_representation=dict(codec_sha256=codec.digest,frame='camera_opengl'),streams={'scene_front':{'path':'large-recording'}})])


def test_hat_rollout_contract_rejects_camera_or_hand_changes():
    m=metadata();contract=hat_contract(m)
    assert contract['retargeting']['scale']==1
    assert contract['action_representation']=='hat128_to_native_position_optimizer'
    wrong=deepcopy(m);wrong['capture']['robot']='floating_shadow_right'
    with pytest.raises(ValueError,match='Capture contract'):hat_contract(wrong)
    wrong=deepcopy(m);wrong['episodes'][0]['capture']=deepcopy(m['capture']);wrong['episodes'][0]['capture']['cameras']['scene_front']['width']=128
    with pytest.raises(ValueError,match='cameras'):hat_contract(wrong)


def test_hat_composes_existing_simulator_with_dedicated_cartesian_bridge():
    m=metadata();spec=dict(source={'adapter':'human-policy-hat'},native={'config':dict(dataset_path='/prepared',dataset_manifest_sha256='a'*64)},data={'bundle':{'assignments':[dict(role='training_data',version={'metadata':m})]}})
    suite=dict(evaluator_adapter='isaac_lab',name='dexverse_recorded',config_json={})
    compatibility,_=inspect_compatibility(spec,manifest(),suite)
    assert compatibility['ready'] and compatibility['policy_loader']=='hat_cartesian'
    entry=compose_evaluator(spec,manifest(),suite).evaluations[0]
    files=entry.command.capsule_files
    for name in ['hat_evaluation.py','hat_runtime.py','hat_data.py','observation_render.py','dexverse_evaluation.py','evaluation_workers.py','policy_transport.py']:
        assert 'adapter-support/'+name in files
    assert entry.runtime_profile_id=='isaacsim-5.1.0_isaaclab-2.3.2_py311'


def test_hat_context_projects_large_recordings_but_keeps_verified_path():
    selection=dict(path='/prepared',manifest_sha256='b'*64,metadata=metadata())
    value=runtime_data_selection(selection)
    assert value['path']==selection['path'] and value['manifest_sha256']==selection['manifest_sha256']
    assert 'streams' not in value['metadata']['episodes'][0]
    assert selection['metadata']['episodes'][0]['streams']


def test_hat_evaluation_capsule_projects_both_copies_of_target():
    from skynet_app.adapters.dataset_inputs import runtime_evaluation_context
    target=dict(path='/prepared',manifest_sha256='c'*64,metadata=metadata())
    original=dict(compatibility={'policy_loader':'hat_cartesian'},target_dataset=target,suite={'config':{'target_dataset':target},'config_sha256':'d'*64})
    result=runtime_evaluation_context(original)
    for selected in (result['target_dataset'],result['suite']['config']['target_dataset']):
        assert 'streams' not in selected['metadata']['episodes'][0]
    assert result['suite']['planning_config_sha256']=='d'*64
    assert original['target_dataset']['metadata']['episodes'][0]['streams']


def test_hat_reconstructs_all_three_measured_camera_calibrations(monkeypatch):
    import sys
    from ops.datasets import observation_render
    from skynet_app.adapters.hat_evaluation import configure_scene
    capture = metadata()['capture']
    # Actual measured WUJI2 recording calibrations, including the side sensors.
    poses = [
        ('third_person_camera', [-1.5, 0., 1.5], [-.4048279821872711, .5797538161277771, -.5797535181045532, .4048279821872711]),
        ('third_person_camera_left', [0., 1.5, 1.5], [8.873286105881562e-9, 4.691093025144255e-8, .870199978351593, -.4926987588405609]),
        ('third_person_camera_right', [0., -1.5, 1.5], [-.4926987290382385, .870199978351593, -3.834902173593946e-8, 1.6812473191407662e-8]),
    ]
    for camera, (sensor, pos, quat) in zip(capture['cameras'].values(), poses):
        camera.update(sensor=sensor, position_world=pos, quaternion_world_ros=quat,
                      intrinsic_matrix=[[293.19970703125, 0., 128.], [0., 293.19970703125, 128.], [0., 0., 1.]])
    jobs = []
    def capture_jobs(cfg, values, *, replay):
        assert replay is False
        jobs.extend(values)
    monkeypatch.setitem(sys.modules, 'observation_render', observation_render)
    monkeypatch.setattr(observation_render, 'configure_cameras', capture_jobs)
    sensors = configure_scene(None, {'cameras': capture['cameras'], 'source_revision': capture['source_revision']})
    assert len(jobs) == 3
    assert [job['recipe']['camera']['sensor'] for job in jobs] == [pose[0] for pose in poses]
    assert list(sensors.values()) == [pose[0] for pose in poses]
    for job, (_, pos, quat) in zip(jobs, poses):
        # The measured intrinsics and pose become the rendered camera, not the pinned defaults.
        camera = job['recipe']['camera']
        assert camera['projection']['horizontal_aperture'] * 293.19970703125 == pytest.approx(256 * camera['projection']['focal_length'])
        assert camera['offset'] == {'pos': pos, 'rot': quat, 'convention': 'ros'}


def test_hat_observation_validation_refuses_bad_state_images_and_poses():
    from skynet_app.adapters.policy_contract import validate_observation
    contract = hat_contract(metadata())
    def observation(**changes):
        value = dict(
            state=np.zeros(len(contract['joint_names'])),
            images={name: np.zeros((camera['height'], camera['width'], 3), dtype=np.uint8)
                    for name, camera in contract['cameras'].items()},
            world_from_camera=np.eye(4), world_from_root=np.eye(4))
        value.update(changes)
        return value
    validate_observation(contract, observation())
    bad_state = observation(); bad_state['state'][0] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        validate_observation(contract, bad_state)
    name = next(iter(contract['cameras']))
    small = observation(); small['images'][name] = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=f'Camera {name}'):
        validate_observation(contract, small)
    for pose in ('world_from_camera', 'world_from_root'):
        with pytest.raises(ValueError, match='rigid'):
            validate_observation(contract, observation(**{pose: np.zeros((4, 4))}))


@pytest.mark.parametrize("hand", __import__("ops.datasets.action_codecs.unidex", fromlist=["supported_robots"]).supported_robots())
def test_hat_uses_each_targets_own_controller_and_asset(hand):
    from skynet_app.adapters.hat_data import encode_human_state, fingertip_slots
    codec = load_codec(hand)
    contract = hat_contract(metadata(hand), control_hz=30)
    assert contract["robot"] == hand
    assert contract["codec_sha256"] == codec.digest
    q = np.clip(np.zeros(codec.action_dim),codec.spec.effective_lower,codec.spec.effective_upper)
    state = encode_human_state(codec,q,codec.encode_state(q)[0])
    assert state.shape == (128,) and np.isfinite(state).all()
    slots = fingertip_slots(codec)
    assert len(slots) == len(codec.spec.tip_links)
    absent = sorted(set(range(5))-set(slots))
    assert (state[43:58].reshape(5,3)[absent] == 0).all()


def test_hat_zero_shot_excludes_source_hands_and_keeps_checkpoint_inputs():
    from test_evaluation_targets import training, selection
    from skynet_app.evaluation_targets import validate_evaluation_target
    from skynet_app.adapters.hat_evaluation import checkpoint_inputs_match
    spec=training("shadow","leap")
    spec["source"]["adapter"]="human-policy-hat"
    for item in spec["data"]["bundle"]["assignments"]:
        item["version"]["metadata"]["contract"]="skynet.hat-rgb-fingertips/v1"
    target=selection("wuji")
    target["metadata"]["contract"]="skynet.hat-rgb-fingertips/v1"
    assert validate_evaluation_target(spec,target,True)=="wuji"
    seen=selection("leap");seen["metadata"]["contract"]=target["metadata"]["contract"]
    with pytest.raises(ValueError,match="training inputs"):
        validate_evaluation_target(spec,seen,True)
    source=dict(position=0,version_id="source",manifest_sha256="a"*64)
    saved={"training_inputs":[source]}
    context={"policy":{"native_config":{"datasets":[source]}},"target_dataset":target}
    assert checkpoint_inputs_match(saved,context)
    context["policy"]["native_config"]["datasets"]=[target]
    assert not checkpoint_inputs_match(saved,context)


def test_rollout_control_step_keeps_joint_calls_and_holds_cartesian_targets(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "skynet_app/adapters"))
    from dexverse_evaluation import policy_control_step
    obs = dict(state=np.zeros(3), images={"scene_front": np.zeros((2, 2, 3), dtype="u1")})
    calls = []
    result = policy_control_step(None, obs, [np.zeros(3)], calls.append, include_images=False)
    assert len(calls) == 1 and calls[0]["predict"] is False and calls[0]["observation"]["images"] == {}
    np.testing.assert_array_equal(result, np.zeros(3))
    calls.clear()
    held = policy_control_step(None, obs, [np.ones(3)], calls.append, cartesian=True)
    assert calls == []
    np.testing.assert_array_equal(held, np.ones(3))
