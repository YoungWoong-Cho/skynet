"""Recover only verified, task-matched demonstrations; never invent actual poses."""
import copy
import hashlib
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
from test_episode_geometry import XML, ROOT
from skynet_app.rollout_preview import attach_demonstration, enrich_demonstration
from skynet_app.adapters.episode_geometry import replay_urdf, HandKinematics


def request(tmp_path):
    states=[{'articulation':{'robot':{'joint_position':np.array([[i,0.]]),'root_pose':np.asarray(ROOT)}}} for i in range(3)]
    payload={'format':'dexverse_trajectory','schema_version':3,'task':'Cube','robot_type':'hand','episodes':[{'states':states}]}
    path=tmp_path/'demo.pkl';path.write_bytes(pickle.dumps(payload))
    source={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'session_id':'source','source_index':0,'steps':2}
    capture={'task':'Cube','robot':'hand','hand':'right','step_dt':.1,'action_joint_names':['bend','move'],'robot_joint_names':['move','bend'],
             'kinematics_urdf':XML,'kinematics_sha256':hashlib.sha256(XML.encode()).hexdigest()}
    return dict(source=source,capture=capture,repository='/missing',viewer={'frames':[{'time':0,'actual':[[9,9,9]]},{'time':.1},{'time':.3}], 'warnings':[]})


def test_recovery_preserves_actual_and_expires_demonstration(tmp_path):
    data=request(tmp_path);result=attach_demonstration(data)
    assert result['frames'][0]['actual']==[[9,9,9]]
    assert result['frames'][1]['hand_poses']['demonstration']['joints']==[0,1]
    assert 'actual' not in result['frames'][1]
    assert 'demonstration' not in result['frames'][2]
    assert result['demonstration']['session_id']=='source'
    model=HandKinematics(result['kinematics_urdf'],result['joint_names'])
    np.testing.assert_allclose(result['frames'][1]['demonstration'],model.points([0,1],ROOT))


def test_checksum_scene_and_layout_are_checked(tmp_path):
    data=request(tmp_path)
    with pytest.raises(ValueError,match='checksum'):
        attach_demonstration({**data,'source':{**data['source'],'sha256':'0'*64}})
    with pytest.raises(ValueError,match='different scene'):
        attach_demonstration({**data,'capture':{**data['capture'],'task':'Stick'}})
    with pytest.raises(ValueError,match='different hand layouts'):
        attach_demonstration({**data,'viewer':{'point_names':['not-the-same-hand'],'frames':[]}})
    spec={'data':{'bundle':{'assignments':[{'role':'training_data','version':{'metadata':{'capture':data['capture']}}}]}}}
    # A different task must not even request the original recording or open a cluster connection.
    assert enrich_demonstration(None,None,spec,data['viewer'],'sky2',task='Stick') is data['viewer']


def test_missing_trace_does_not_claim_actual_keypoints_exist(tmp_path):
    data = request(tmp_path)
    result = attach_demonstration({**data, 'viewer': None, 'duration': .2})
    assert all('actual' not in frame for frame in result['frames'])
    assert not any('saved Actual keypoints' in warning for warning in result['warnings'])


def test_replay_tree_excludes_cluster_assets_but_keeps_joint_transforms():
    xml=XML.replace('<link name="finger"/>','<link name="finger"><visual><geometry><mesh filename="/cluster/secret.stl"/></geometry></visual></link>')
    clean=replay_urdf(xml)
    assert '/cluster' not in clean and '<mesh' not in clean
    np.testing.assert_allclose(HandKinematics(xml,['bend','move']).points([.4,.1],ROOT),HandKinematics(clean,['bend','move']).points([.4,.1],ROOT))
