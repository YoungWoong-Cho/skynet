"""Physical stream sharing, immutable verification and adapter sampling checks."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skynet_app/adapters'))
sys.path.insert(0, str(ROOT / 'ops/datasets'))
from recording_dataset import (FORMAT, RecordingDataset, canonical, close_handles,
                               digest, stream_reference, verify_dataset)
# Exercise the same standalone restricted unpickler that capsules freeze, without
# pulling web/database dependencies into the pure CPU reader runtime.
import ast
import types
if 'arrays' not in sys.modules:
    module=types.ModuleType('arrays')
    module.__dict__.update(pickle=pickle,np=np)
    source_text=(ROOT/'skynet_app/live_xr_review.py').read_text()
    definition=next(node for node in ast.parse(source_text).body if isinstance(node,ast.ClassDef) and node.name=='ArrayUnpickler')
    exec(ast.get_source_segment(source_text,definition),module.__dict__)
    sys.modules['arrays']=module
from recording_prepare import prepare, inspect_images, write_streams
from act_training import read_sample


@pytest.fixture(autouse=True)
def handles():
    yield
    close_handles()


def source(tmp_path, index=0, images=False):
    names = [f'{axis}_translation_joint' for axis in 'xyz'] + [f'{axis}_rotation_joint' for axis in 'xyz'] + ['finger']
    metadata = dict(robot='test_hand', task='test-task', hand='right', source_revision='source-v1',
        action_joint_names=names, robot_joint_names=names[::-1],
        groups=[dict(side='right', wrist_indices=list(range(6)), finger_indices=[6])],
        action_scale=[1]*7, action_offset=[0]*7, step_dt=1/60,
        action_semantics='raw_joint_position_command; target = action * scale + offset')
    actions = np.arange(28, dtype='f4').reshape(4, 7) + index*100
    states = actions + 10
    raw = dict(actions=actions, states=[dict(articulation={'robot':dict(
        joint_position=row[::-1][None], root_pose=np.array([[0,0,0,1,0,0,0]], dtype='f4'))})
        for row in [*states, states[-1]+1]], num_steps=4, success=True)
    payload = dict(format='dexverse_trajectory', schema_version=3, num_episodes=1,
        task=metadata['task'], robot_type=metadata['robot'], skynet_state_metadata=metadata, episodes=[raw])
    result = dict(index=index, recording_id=f'recording-{index}', session_id='session')
    if images:
        metadata.update(color_space='RGB', cameras={name:dict(mount='fixed_scene', width=256, height=256,
            intrinsic_matrix=np.eye(3).tolist(), world_from_camera=np.eye(4).tolist())
            for name in ('scene_front','scene_left','scene_right')})
        path = tmp_path / f'images-{index}.hdf5'
        with h5py.File(path, 'w') as file:
            file.attrs.update(schema='skynet.rgb-trajectory/v1', complete=True, metadata=json.dumps(metadata))
            file['state'] = states
            file['action'] = actions
            file['timestamps'] = np.arange(4)/60
            file['wall_times'] = 1000+np.arange(4)/60
            for camera, base in [('scene_front',10),('scene_left',20),('scene_right',30)]:
                file['images/'+camera] = np.broadcast_to((np.arange(4,dtype='u1')+base)[:,None,None,None],(4,256,256,3))
        image_sha = digest(path)
        raw['skynet_images'] = dict(sha256=image_sha)
        result.update(images=str(path), image_sha256=image_sha)
    path = tmp_path / f'episode-{index}.pkl'
    path.write_bytes(pickle.dumps(payload))
    result.update(recording=str(path), sha256=digest(path))
    return result, actions, states


def request(tmp_path, sources, name='a', rgb=False):
    streams = [dict(name=camera,modality='rgb',camera_ids=[camera],width=256,height=256,dtype='uint8',color_space='RGB')
               for camera in ('scene_front','scene_left','scene_right')] if rgb else []
    return dict(output=str(tmp_path/name), recording_root=str(tmp_path/'recordings'), sources=sources,
        adapter={'id':'adapter-'+name}, adapter_data_preset='rgb' if rgb else 'state',
        contract='skynet.act-rgb-joints/v1' if rgb else 'test.recording-state/v1',
        split={'train':[0], 'validation':list(range(1,len(sources)))}, observations=['state','rgb'] if rgb else ['state'],
        observation_requirements={'streams':streams}, source_revision='test-v1')


def test_two_adapter_manifests_share_exact_state_file_and_no_payload_copy(tmp_path):
    src, actions, states = source(tmp_path)
    first = prepare(request(tmp_path,[src]))
    before = {str(p):p.stat().st_mtime_ns for p in (tmp_path/'recordings').rglob('*') if p.is_file()}
    second = prepare(request(tmp_path,[src],name='b'))
    assert first['episodes'][0]['streams'] == second['episodes'][0]['streams']
    assert len(list((tmp_path/'recordings').rglob('values.hdf5'))) == 1
    assert before == {str(p):p.stat().st_mtime_ns for p in (tmp_path/'recordings').rglob('*') if p.is_file()}
    for folder in ('a','b'):
        assert [p.name for p in (tmp_path/folder).iterdir()] == ['manifest.json']
    artifact = first['shared_artifacts'][0]
    assert artifact['path'].endswith('values.hdf5') and artifact['spec']['schema']=='skynet.recording-file/v1'
    assert artifact['manifest_sha256'] == digest(artifact['path'])
    assert artifact['artifact_key'] == __import__('hashlib').sha256(canonical(artifact['spec'])).hexdigest()
    data=RecordingDataset(tmp_path/'a', digest(tmp_path/'a/manifest.json'), verify_files=True)
    np.testing.assert_array_equal(data.read(0,'state'),states)
    np.testing.assert_array_equal(data.read(0,'action'),actions)


def test_archive_rgb_is_referenced_in_place_and_state_is_not_copied(tmp_path):
    src, actions, states=source(tmp_path,images=True)
    before={p.name:digest(p) for p in tmp_path.iterdir() if p.is_file()}
    manifest=prepare(request(tmp_path,[src],rgb=True))
    streams=manifest['episodes'][0]['streams']
    assert set(ref['path'] for ref in streams.values()) == {src['images']}
    assert streams['scene_front']['dataset']=='images/scene_front'
    assert not (tmp_path/'recordings').exists()
    assert len(manifest['shared_artifacts'])==1 and manifest['shared_artifacts'][0]['owned'] is False
    assert before=={p.name:digest(p) for p in tmp_path.iterdir() if p.is_file()}
    probe=inspect_images(src,request(tmp_path,[src],rgb=True)['observation_requirements'])
    assert len(probe['streams'])==3


def test_dynamic_new_camera_choices_do_not_change_native_state_identity(tmp_path):
    src,_,_=source(tmp_path)
    first=prepare(request(tmp_path,[src]))
    # Render metadata must not change native joint stream identity.
    import recording_prepare
    original=recording_prepare.source_values
    def observed(*args):
        payload,raw,capture,arrays=original(*args)
        capture['cameras']={'scene_front':{'width':256,'height':256}}
        capture['color_space']='RGB'
        return payload,raw,capture,arrays
    from unittest.mock import patch
    with patch.object(recording_prepare,'source_values',observed):
        second=prepare(request(tmp_path,[src],name='b'))
    assert first['episodes'][0]['streams']['state']==second['episodes'][0]['streams']['state']


def test_shared_file_tampering_and_invalid_splits_are_rejected(tmp_path):
    src,_,_=source(tmp_path)
    manifest=prepare(request(tmp_path,[src]))
    path=Path(manifest['episodes'][0]['streams']['state']['path'])
    with path.open('ab') as file:file.write(b'tampered')
    with pytest.raises(ValueError,match='checksum'):
        verify_dataset(tmp_path/'a')
    broken=deepcopy(manifest);broken['split']['validation']=[0]
    (tmp_path/'a/manifest.json').write_bytes(canonical(broken))
    with pytest.raises(ValueError,match='disjoint'):
        verify_dataset(tmp_path/'a')


def test_codec_identity_cannot_silently_overwrite_values(tmp_path):
    kwargs=dict(recording_root=tmp_path,source_sha256='a'*64,codec_spec={'id':'test/v1'},metadata={})
    write_streams(**kwargs,arrays={'state':np.ones((3,2),dtype='f4')})
    with pytest.raises(ValueError,match='changed'):
        write_streams(**kwargs,arrays={'state':np.zeros((3,2),dtype='f4')})
    assert len(list(tmp_path.rglob('values.hdf5')))==1


def test_duplicate_random_frame_reads_and_joint_order_only_in_memory(tmp_path):
    src,actions,_=source(tmp_path)
    manifest=prepare(request(tmp_path,[src]))
    manifest['policy_to_source_indices']=list(reversed(range(7)))
    manifest['episodes'][0]['policy_to_source_indices']=manifest['policy_to_source_indices']
    data=RecordingDataset(tmp_path/'a',manifest=manifest)
    np.testing.assert_array_equal(data.episode(0).joint('action',[3,0,3]),actions[[3,0,3],::-1])
    np.testing.assert_array_equal(data.episode(0).read('action',[3,0,3]),actions[[3,0,3]])
    with pytest.raises(IndexError):data.read(0,'action',[4])


def test_act_chunk_mask_and_training_only_statistics(tmp_path):
    sources=[source(tmp_path,i,images=True)[0] for i in range(2)]
    manifest=prepare(request(tmp_path,sources,rgb=True))
    data=RecordingDataset(tmp_path/'a')
    stats=data.normalization()
    np.testing.assert_allclose(stats['action_mean'],data.read(0,'action').mean(axis=0))
    qpos,images,actions,mask=read_sample(data,0,3,stats,3)
    assert images.shape==(3,480,640,3) and images[:,0,0,0].tolist()==[13,23,33]
    np.testing.assert_allclose(actions[0],(data.read(0,'action',3)-stats['action_mean'])/stats['action_std'])
    np.testing.assert_array_equal(actions[1:],0)
    assert mask.tolist()==[False,True,True]


def test_external_hdf5_links_cannot_bypass_stream_checksum(tmp_path):
    original=tmp_path/'original.hdf5'
    with h5py.File(original,'w') as file:file['state']=np.zeros((3,2),dtype='f4')
    alias=tmp_path/'alias.hdf5'
    with h5py.File(alias,'w') as file:file['state']=h5py.ExternalLink(str(original),'state')
    with pytest.raises(ValueError,match='external'):
        stream_reference(alias,'state')


def test_symlink_ancestor_cannot_redirect_shared_payload(tmp_path):
    src,_,_=source(tmp_path)
    manifest=prepare(request(tmp_path,[src]))
    (tmp_path/'alias').symlink_to(tmp_path/'recordings',target_is_directory=True)
    ref=deepcopy(manifest['episodes'][0]['streams']['state'])
    ref['path']=str(tmp_path/'alias'/Path(ref['path']).relative_to(tmp_path/'recordings'))
    from recording_dataset import verify_reference
    with pytest.raises(ValueError,match='symbolic'):
        verify_reference(ref)
    with pytest.raises(ValueError,match='symbolic'):
        write_streams(tmp_path/'alias','a'*64,{'id':'bad/v1'},{'x':np.ones((2,1))},{})
    outside=tmp_path/'outside';outside.mkdir()
    (tmp_path/'recordings'/('b'*64)).symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='symbolic'):
        write_streams(tmp_path/'recordings','b'*64,{'id':'bad/v1'},{'x':np.ones((2,1))},{})
    assert not list(outside.iterdir()), 'Reject redirected source directories before creating the state directory'
    relative=deepcopy(manifest['episodes'][0]['streams']['state']);relative['path']='state.hdf5'
    with pytest.raises(ValueError,match='absolute paths'):
        verify_reference(relative)


def test_native_act_bridge_reads_shared_payload_and_preserves_native_alignment(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    from act_native_data import NativeACTDataset,prepare_recorded_act,validate_recorded_act,CAMERAS
    sources=[source(tmp_path,i,images=True)[0] for i in range(2)]
    spec=request(tmp_path,sources,rgb=True)
    spec['contract']='skynet.act-rgb-joints/v1'
    manifest=prepare(spec)
    data=RecordingDataset(tmp_path/'a')
    stats=data.normalization()
    stats['qpos_mean']=stats.pop('state_mean');stats['qpos_std']=stats.pop('state_std')
    dataset=NativeACTDataset(data,[0],CAMERAS,stats)
    monkeypatch.setattr(np.random,'randint',lambda *_:3)
    images,state,actions,mask=dataset[0]
    assert tuple(images.shape)==(3,3,480,640)
    np.testing.assert_allclose(images[:,0,0,0].numpy(),np.array([13,33,23])/255)
    expected=(data.read(0,'action',slice(2,None))-stats['action_mean'])/stats['action_std']
    np.testing.assert_allclose(actions[:2].numpy(),expected)
    assert mask.tolist()==[False,False]+[True]*48
    workspace=tmp_path/'native'
    directory=workspace/'XPolicyLab/policy/ACT'
    directory.mkdir(parents=True)
    (directory/'utils.py').write_text('def load_data(*args):\n    raise RuntimeError("old loader called")\n')
    prepare_recorded_act(tmp_path/'a',manifest,workspace,validate_recorded_act(tmp_path/'a',manifest))
    namespace={};exec((directory/'utils.py').read_text(),namespace)
    from act_native_data import load_data
    assert namespace['load_data'].func is load_data
    assert namespace['load_data'].keywords['action_steps'] == 50
    assert not list(workspace.rglob('*.hdf5'))
    config=json.loads((directory/'TASK_CONFIGS.json').read_text())
    assert next(iter(config.values()))['dataset_dir']==str(tmp_path/'a')


def test_joint_representation_string_and_cpu_preflight_require_no_gpu(tmp_path):
    from recording_prepare import preflight_source
    src,_,_=source(tmp_path)
    spec=request(tmp_path,[src])
    spec['action_representation']='raw_joint_position_command'
    probe=preflight_source(src,spec['observation_requirements'],spec['action_representation'])
    assert probe['streams']=={} and probe['steps']==4 and probe['capture']['robot']=='test_hand'
    assert prepare(spec)['action_representation']=='raw_joint_position_command'


def test_native_loader_verification_runs_as_portable_cpu_subprocess(tmp_path):
    pytest.importorskip('torch')
    from act_native_data import prepare_recorded_act,validate_recorded_act
    import os
    import subprocess
    sources=[source(tmp_path,i,images=True)[0] for i in range(2)]
    spec=request(tmp_path,sources,rgb=True);spec['contract']='skynet.act-rgb-joints/v1'
    manifest=prepare(spec)
    workspace=tmp_path/'native'
    directory=workspace/'XPolicyLab/policy/ACT';directory.mkdir(parents=True)
    (directory/'utils.py').write_text('def load_data(*args):\n    raise RuntimeError("old reader")\n')
    prepare_recorded_act(tmp_path/'a',manifest,workspace,validate_recorded_act(tmp_path/'a',manifest))
    script=tmp_path/'verify_reader.py'
    script.write_text('from pathlib import Path\nfrom act_native_data import verify_loader\n'
        'import json\nif __name__ == "__main__":\n'
        f'    print(json.dumps(verify_loader(Path({str(directory)!r}),{digest(tmp_path/"a/manifest.json")!r},42)))\n')
    # Native shared-memory helpers may inherit pipe descriptors. Match the
    # production worker's file-backed diagnostics and wait for the actual child.
    log_path=tmp_path/'loader.log'
    with log_path.open('w') as log:
        result=subprocess.run([sys.executable,str(script)],stdout=log,stderr=log,timeout=45,
            env={**os.environ,'PYTHONPATH':str(ROOT/'skynet_app/adapters'),'TMPDIR':'/tmp'})
    assert result.returncode==0,log_path.read_text()
    receipt=json.loads(log_path.read_text().strip().splitlines()[-1])
    assert receipt['schema']=='skynet.act-native-loader-validation/v2'
    assert receipt['split']=={'train':[0],'validation':[1]}
    assert receipt['batch_shapes']['train']['images']==[1,3,3,480,640]


@pytest.mark.parametrize('entrypoint', ['recorded_policy_evaluation', 'xpolicy_evaluation'])
def test_evaluation_entrypoints_verify_reference_only_manifest_before_policy_loading(tmp_path, monkeypatch, entrypoint):
    import importlib
    from types import SimpleNamespace
    from policy_contract import recorded_contract
    module=importlib.import_module(entrypoint)
    src,_,_=source(tmp_path,images=True)
    manifest=prepare(request(tmp_path,[src],rgb=True))
    context={'result_path':str(tmp_path/'evaluation/result.json'),
        'policy':{'adapter':'xpolicylab-act','native_config':{
            'dataset_path':str(tmp_path/'a'), 'dataset_manifest_sha256':digest(tmp_path/'a/manifest.json')}},
        'compatibility':{'io_contract':recorded_contract(manifest), 'policy_loader':'xpolicy_joints'}}
    path=tmp_path/'context.json';path.write_text(json.dumps(context))
    monkeypatch.setattr(sys,'argv',[entrypoint,'--context',str(path),'--source-dir',str(tmp_path/'native')])
    calls=[]
    def load(*args):
        assert args[-1]==manifest and 'files' not in args[-1]
        calls.append('policy')
        return SimpleNamespace(mode='rgb')
    monkeypatch.setattr(module,'load_policy' if entrypoint=='recorded_policy_evaluation' else 'RecordedPolicy',load)
    monkeypatch.setattr(module,'validate_simulation',lambda *_:calls.append('validate_simulation'))
    monkeypatch.setattr(module,'run_simulator',lambda *_:calls.append('simulate'))
    module.main()
    assert calls[-1]=='simulate' and calls.count('policy')==1
    calls.clear()
    close_handles()
    with h5py.File(src['images'],'r+') as file:
        file['action'][0,0] += 1
    with pytest.raises(ValueError,match='checksum'):
        module.main()
    assert not calls, 'Tampered shared payload must fail before loading a checkpoint or simulator'


def test_joint_runtime_rejects_old_dataset_formats_and_mixed_episode_joint_mapping(tmp_path):
    from xpolicy_runtime import validate_manifest
    sources=[source(tmp_path,i)[0] for i in range(2)]
    manifest=prepare(request(tmp_path,sources))
    assert validate_manifest(manifest) is manifest
    old=deepcopy(manifest);old['format']='xpolicylab-demonstrations/v1'
    with pytest.raises(ValueError,match='old exported formats'):
        validate_manifest(old)
    mixed=deepcopy(manifest);mixed['episodes'][1]['policy_to_source_indices'].reverse()
    with pytest.raises(ValueError,match='same joint mapping'):
        validate_manifest(mixed)
    mixed=deepcopy(manifest);mixed['episodes'][1]['capture']['action_scale'][0]=2
    with pytest.raises(ValueError,match='same capture contract'):
        validate_manifest(mixed)


def test_recording_reader_enforces_pinned_dataset_manifest_fingerprint(tmp_path):
    src,_,_=source(tmp_path)
    prepare(request(tmp_path,[src]))
    path=tmp_path/'a/manifest.json'
    data=RecordingDataset(path,sha=digest(path))
    assert len(data)==1 and data.episode(0).steps==4
    with pytest.raises(ValueError,match='pinned checksum'):
        RecordingDataset(path,sha='0'*64)


def test_unidex_manifest_preserves_human_task_instruction_and_verified_codec_identity(tmp_path, monkeypatch):
    from recording_prepare import recording_prompt
    src,_,_=source(tmp_path)
    src.update(session_profile={'instructions':'  Pick up the cube.  ','task_name':'Pick up cube'},
               hand_id='unverified-hand',action_representation={'id':'unverified-codec'})
    calibration=tmp_path/'calibration.hdf5'
    with h5py.File(calibration,'w') as file:file['poses']=np.broadcast_to(np.eye(4),(4,4,4))
    src['streams']={'scene_front_world_from_camera':stream_reference(calibration,'poses')}
    codec=types.ModuleType('action_codecs.unidex')
    codec.encode_recording=lambda *_:({'faas_state_absolute':np.zeros((4,82),dtype='f4'),
        'faas_action_absolute':np.ones((4,82),dtype='f4')},
        {'id':'skynet.unidex-faas/v1','codec_sha256':'b'*64,'frame':'camera_opengl'})
    monkeypatch.setitem(sys.modules,'action_codecs.unidex',codec)
    spec=request(tmp_path,[src]);spec['action_representation']={'id':'skynet.unidex-faas/v1'}
    episode=prepare(spec)['episodes'][0]
    assert episode['prompt']=='Pick up the cube.' and episode['hand_id']=='test_hand'
    assert episode['action_representation']['codec_sha256']=='b'*64
    capture={'task':'Dexverse-PickCube-v0'}
    assert recording_prompt(dict(src,prompt='Use explicit instruction'),capture)=='Use explicit instruction'
    assert recording_prompt({'session_profile':{'instructions':'  ','task_name':'Pick up cube'}},capture)=='Pick up cube'
    assert recording_prompt({},capture)=='Dexverse-PickCube-v0'


@pytest.mark.parametrize("dynamic", [False, True])
def test_camera_frame_codec_reuses_only_explicitly_fixed_archived_rgb_pose(tmp_path, monkeypatch, dynamic):
    src, _, _ = source(tmp_path, images=True)
    # The archived receipt, not a simulator default, determines the frame.
    with h5py.File(src['images'], 'r+') as file:
        metadata = json.loads(file.attrs['metadata'])
        metadata['cameras']['scene_front']['world_from_camera'][0][3] = 1.25
        metadata['cameras']['scene_front']['dynamic'] = dynamic
        file.attrs['metadata'] = json.dumps(metadata)
    src['image_sha256'] = digest(src['images'])
    payload = pickle.loads(Path(src['recording']).read_bytes())
    payload['episodes'][0]['skynet_images']['sha256'] = src['image_sha256']
    Path(src['recording']).write_bytes(pickle.dumps(payload))
    src['sha256'] = digest(src['recording'])
    codec = types.ModuleType('action_codecs.unidex')
    seen = []
    def encode(raw, capture, poses):
        seen.append(poses.copy())
        return {'faas_state_absolute': np.zeros((4,82), dtype='f4'),
                'faas_action_absolute': np.zeros((4,82), dtype='f4')}, {
                'id':'skynet.unidex-faas/v1','codec_sha256':'b'*64,'frame':'camera_opengl'}
    codec.encode_recording = encode
    monkeypatch.setitem(sys.modules,'action_codecs.unidex',codec)
    spec = request(tmp_path,[src],rgb=True)
    spec['action_representation'] = {'id':'skynet.unidex-faas/v1'}
    if dynamic:
        with pytest.raises(ValueError, match='per-frame'):
            prepare(spec)
        assert not seen
    else:
        result = prepare(spec)
        assert seen[0].shape == (4,4,4)
        np.testing.assert_array_equal(seen[0][:,0,3],1.25)
        assert result['episodes'][0]['streams']['scene_front']['path'] == src['images']
