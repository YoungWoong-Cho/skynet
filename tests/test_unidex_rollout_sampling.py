"""The cheaper Skynet rollout must feed the unchanged UniDex model identical inputs."""
import copy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from ops.datasets import observation_render as render
from ops.datasets.observation_geometry import pose_from_ros
from skynet_app.adapters import unidex_evaluation as bridge
from skynet_app.adapters.policy_contract import validate_observation, unidex_contract
from tests.test_unidex_evaluation import metadata


def test_chunk_optimization_preserves_1200_step_inputs_actions_video_and_seed(monkeypatch):
    monkeypatch.setitem(sys.modules, 'observation_render', render)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'skynet_app/adapters'))
    from dexverse_evaluation import policy_control_step
    contract = unidex_contract(metadata(), control_hz=30)
    camera = contract['cameras']['scene_front']
    pose = pose_from_ros(camera['position_world'], camera['quaternion_world_ros'])
    n = len(contract['joint_names'])
    depth = np.zeros((256, 256), dtype='f4'); depth[100:103, 127:130] = 1.2
    original_cloud = bridge.point_cloud

    def run(optimized):
        state = np.linspace(.01, .1, n)[None]
        robot = NS(data=NS(joint_pos=state, root_pos_w=np.array([[.1,.2,.3]]), root_quat_w=np.array([[1.,0,0,0]])))
        env = NS(scene={'third_person_camera':object(), 'robot':robot}, sim=NS(render_mode='test', render=lambda:None))
        frame_id, tick = [0], [-1]
        result = {key:[] for key in ('captures','reads','renders','fps','calls','model_inputs','video','commands')}
        monkeypatch.setattr(render, 'capture_current_frame', lambda value: result['captures'].append((tick[0], value is env)))
        def read_camera(*args, **kwargs):
            result['reads'].append(tick[0])
            return dict(depth=depth.copy(), rgb=np.full((256,256,3), tick[0] % 256, dtype='u1'),
                        intrinsics=np.array(camera['intrinsic_matrix']), world_from_camera=pose.copy())
        monkeypatch.setattr(render, 'read_camera', read_camera)
        def cloud(*args, **kwargs):
            result['fps'].append(kwargs['frame_id'])
            return original_cloud(*args, **kwargs)
        monkeypatch.setattr(bridge, 'point_cloud', cloud)
        def observe(include_pointcloud=True):
            result['renders'].append(tick[0])
            value = bridge.observation_from_sensors(env, contract, list(range(n)), identity=['same-evaluation','episode-0'],
                                                   frame_id=frame_id[0], include_pointcloud=include_pointcloud)
            frame_id[0] += 1
            return value
        def request(value):
            result['calls'].append(value['predict'])
            if not value['predict']: return None
            result['model_inputs'].append(copy.deepcopy(value['observation']))
            return state.copy() + np.arange(30)[:,None] * .001
        observe(include_pointcloud=not optimized)
        pending = []
        for step in range(1200):
            tick[0] = step
            obs = observe(include_pointcloud=not optimized or (step % 2 == 0 and not pending))
            if step % 2 == 0:
                result['video'].append(hashlib.sha256(obs['images']['scene_front'].tobytes()).hexdigest())
                if optimized:
                    packed = policy_control_step(contract, obs, pending, request, unidex=True)
                else:
                    validate_observation(contract, obs)
                    predicted = request(dict(command='step', observation=obs, predict=not pending))
                    if not pending: pending = list(predicted)
                    packed = pending.pop(0)
            result['commands'].append(packed.copy())
            state[0] += .01 * (packed - state[0])
        return dict(result, frame_count=frame_id[0])

    eager, optimized = run(False), run(True)
    assert optimized['frame_count'] == eager['frame_count'] == 1201
    assert optimized['captures'] == eager['captures'] and len(optimized['captures']) == 1201
    assert optimized['reads'] == eager['reads'] and optimized['renders'] == eager['renders']
    assert optimized['fps'] == list(range(1, 1200, 60)) and len(eager['fps']) == 1201
    assert len(eager['calls']) == 600 and len(optimized['calls']) == 20
    assert optimized['video'] == eager['video'] and len(optimized['video']) == 600
    np.testing.assert_array_equal(optimized['commands'], eager['commands'])
    assert len(optimized['model_inputs']) == len(eager['model_inputs']) == 20
    for before, after in zip(eager['model_inputs'], optimized['model_inputs']):
        for key in ('state', 'pointcloud', 'world_from_camera', 'world_from_root'):
            np.testing.assert_array_equal(before[key], after[key])
        np.testing.assert_array_equal(before['images']['scene_front'], after['images']['scene_front'])


def test_sensor_validation_stays_strict_and_other_adapters_keep_calls(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'skynet_app/adapters'))
    from dexverse_evaluation import policy_control_step
    contract = unidex_contract(metadata(), control_hz=30)
    n = len(contract['joint_names'])
    obs = dict(state=np.zeros(n), images={'scene_front':np.zeros((256,256,3),dtype='u1')},
               world_from_camera=np.eye(4), world_from_root=np.eye(4))
    validate_observation(contract, obs, require_pointcloud=False)
    with pytest.raises(ValueError, match='XYZRGB1024'): validate_observation(contract, obs)
    pending = [np.zeros(n)]
    with pytest.raises(ValueError, match='XYZRGB1024'):
        policy_control_step(contract, obs, [], lambda _:pytest.fail('No inference on missing input'), unidex=True)
    for field, value, message in [('state',np.full(n,np.nan),'non-finite'),
                                  ('images',{'scene_front':np.zeros((1,1,3),dtype='u1')},'Camera'),
                                  ('world_from_camera',np.zeros((4,4)),'rigid'),
                                  ('world_from_root',np.zeros((4,4)),'rigid')]:
        with pytest.raises(ValueError, match=message):
            policy_control_step(contract, dict(obs, **{field:value}), pending.copy(), lambda _:None, unidex=True)
    calls = []
    result = policy_control_step(None, obs, pending.copy(), lambda value:calls.append(value), unidex=False, include_images=False)
    assert len(calls) == 1 and calls[0]['predict'] is False and calls[0]['observation']['images'] == {}
    np.testing.assert_array_equal(result, np.zeros(n))


def test_observation_sampling_is_common_across_models_budgets_and_worker_partitions(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'skynet_app/adapters'))
    from dexverse_evaluation import identity as progress_identity
    from ops.datasets.observation_geometry import point_cloud, keyed_seed
    contract = unidex_contract(metadata(), control_hz=30)
    task, seed, episode_index, frame_id = 'Dexverse-PickCube-v0', 42, 3, 61
    contexts = [dict(run_id=f'run-k{k}', checkpoint={'sha256':str(k)*64},
                     policy={'native_config':{'datasets':list(range(k))}},
                     episodes_per_task=budget, episode_assignments=assignments, unseen_embodiment=unseen)
                for k, budget, assignments, unseen in [(1,20,[[42,3]],True),(6,100,[[42,2],[42,3]],True),(3,4,None,False)]]
    assert len({progress_identity(context) for context in contexts}) == 3
    identities = [bridge.observation_sampling_identity(contract, task, seed, episode_index) for _ in contexts]
    assert identities[0] == identities[1] == identities[2]
    recipe = contract['pointcloud_recipe']
    view = dict(camera_id='scene_front', depth=np.ones((3,3),dtype='f4'),
                rgb=np.arange(27,dtype='u1').reshape(3,3,3), intrinsics=np.eye(3), world_from_camera=np.eye(4))
    clouds = [point_cloud([view], recipe, identity=value, frame_id=frame_id)[0] for value in identities]
    np.testing.assert_array_equal(clouds[0], clouds[1]); np.testing.assert_array_equal(clouds[0], clouds[2])
    common_seed = keyed_seed(identities[0], frame_id, recipe['seed'])
    alternatives = [bridge.observation_sampling_identity(contract, task, seed+1, episode_index),
                    bridge.observation_sampling_identity(contract, task, seed, episode_index+1),
                    bridge.observation_sampling_identity(dict(contract, robot='other-hand', codec_sha256='f'*64), task, seed, episode_index)]
    for value in alternatives:
        assert keyed_seed(value, frame_id, recipe['seed']) != common_seed
        assert not np.array_equal(point_cloud([view], recipe, identity=value, frame_id=frame_id)[0], clouds[0])
    changed_recipe = dict(recipe, seed=recipe['seed'] + 1)
    assert keyed_seed(identities[0], frame_id, changed_recipe['seed']) != common_seed
    assert not np.array_equal(point_cloud([view], changed_recipe, identity=identities[0], frame_id=frame_id)[0], clouds[0])
    assert keyed_seed(identities[0], frame_id+1, recipe['seed']) != common_seed
