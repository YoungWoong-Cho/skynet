"""Skynet Sharpa extension: independent URDF motion checks, not upstream support.

Reference vectors come from pinned UniDex 97d869e0 Wuji/Shadow URDFs and their
right-hand HAND_TRANSFORMS/FAAS signs; no expected vector uses the Sharpa map.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ops.datasets.action_codecs.unidex import load_codec, preflight

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'config/action_representations/unidex-faas-v1/skynet_sharpa_right.json'
URDF = SPEC.parent / 'assets/skynet_sharpa_right.urdf'
# Canonical positive FAAS axes and normalized fingertip velocities at URDF q=0.
THUMB_REFERENCES = [
    ([.422924620,-.056212869,-.904419637],[.889590537,-.164263276,.426199780]),
    ([.923775270,-.122783249,.362716866],[-.382934863,-.295263764,.875317200]),
    ([.667567135,.264381426,-.696029153],[.743543452,-.188142039,.641674145]),
    ([0.,-.707106781,.707106781],[-1.,0.,0.]),
    ([.667567135,.264381426,-.696029153],[.742736287,-.171264376,.647310839]),
]


def rotation(axis, angle):
    """Independent Rodrigues formula; does not use production FK."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x,y,z = axis
    skew = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.cos(angle)*np.eye(3)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*skew


def geometry(coordinates=None):
    coordinates = coordinates or {}
    links = {'h_right_hand_C_MC':np.eye(4)}
    joints = {}
    pending = list(ET.parse(URDF).getroot().findall('joint'))
    while True:
        ready = [j for j in pending if j.find('parent').get('link') in links]
        if not ready:
            break
        for j in ready:
            origin = j.find('origin')
            r,p,y = np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
            t = np.eye(4)
            t[:3,:3] = rotation([0,0,1],y) @ rotation([0,1,0],p) @ rotation([1,0,0],r)
            t[:3,3] = np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            t = links[j.find('parent').get('link')] @ t
            if j.get('type') != 'fixed':
                axis = np.fromstring(j.find('axis').get('xyz'),sep=' ')
                joints[j.get('name')] = t[:3,:3] @ axis
                t[:3,:3] = t[:3,:3] @ rotation(axis,coordinates.get(j.get('name'),0.))
            links[j.find('child').get('link')] = t
            pending.remove(j)
    return links,joints


@pytest.fixture
def codec():
    return load_codec('skynet_sharpa_right')


def test_exact_asset_bound_extension_covers_every_control(codec):
    data = json.loads(SPEC.read_text())
    body = {k:v for k,v in data.items() if k != 'spec_hash'}
    digest = hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert data['spec_hash'] == digest == codec.digest
    assert data['mapping_validation']['kind'] == 'anatomical_extension'
    assert data['mapping_validation']['exact_upstream_asset_equivalence'] is False
    assert preflight(codec.robot_id)['upstream_hand'] is None
    assert len(codec.mapping) == len(codec.independent) == 22
    assert {m['joint'] for m in codec.mapping} == set(codec.action_names[6:])
    assert len({m['slot'] for m in codec.mapping}) == 22
    assert [m['slot'] for m in codec.mapping] == [1,2,3,26,4,7,6,8,9,12,11,13,14,17,16,18,19,25,22,21,23,24]
    assert all(m['source_joint'] == m['joint'].removeprefix('h_') for m in codec.mapping)


def test_own_palm_frame_and_explicit_native_rest(codec):
    frame = np.asarray(codec.spec.data['link_to_frame'])
    np.testing.assert_array_equal(frame,np.eye(4))
    links,_ = geometry()
    tips = [links[name][:3,3] for name in codec.spec.tip_links]
    assert all(t[2] > .10 for t in tips)
    assert all(tips[i][1] > tips[i+1][1] for i in range(4))
    assert codec.spec.data['mapping_validation']['rest_reference'] == 'native_urdf_zero'
    assert all(m['offset'] == 0. for m in codec.mapping)
    value,_ = codec.encode_state(np.zeros(codec.action_dim))
    np.testing.assert_array_equal(value[18:],0.)
    # Exact registered skynet_alignment; no upstream wrist offset is borrowed.
    pose = codec.wrist_pose(np.zeros(codec.action_dim))
    np.testing.assert_allclose(pose[:3,:3],[[0,0,1],[0,1,0],[-1,0,0]],atol=1e-12)
    np.testing.assert_array_equal(pose[:3,3],0.)


@pytest.mark.parametrize('index',range(22))
def test_every_positive_slot_moves_its_actual_finger_in_reference_direction(codec,index):
    mapping = codec.mapping[index]
    links,axes = geometry()
    tip = next(t for t in codec.spec.tip_links if mapping['joint'].split('_')[2] in t)
    advanced,_ = geometry({mapping['joint']:.001/mapping['scale']})
    axis = axes[mapping['joint']]/mapping['scale']
    tangent = advanced[tip][:3,3]-links[tip][:3,3]
    tangent /= np.linalg.norm(tangent)
    if index < 5:
        reference_axis,reference_tangent = np.asarray(THUMB_REFERENCES[index])
        assert axis @ reference_axis > .85
        assert tangent @ reference_tangent > .80
    elif mapping['slot'] == 25:
        # Both are proximal pinky cupping DOFs. Sharpa's rotary metacarpal has
        # lateral travel, unlike Shadow's oblique hinge: do not claim equivalence.
        assert axis @ [0.,.573601902,.819134212] > .81
        assert tangent @ [1.,0.,0.] > .49
        assert tangent[1] < -.86
    elif mapping['slot'] in {6,11,16,21}:
        assert axis @ [-1.,0.,0.] > .9999
        assert tangent @ [0.,1.,0.] > .9999
    else:
        assert axis @ [0.,1.,0.] > .9999
        assert tangent @ [1.,0.,0.] > .9999
    for other in set(codec.spec.tip_links)-{tip}:
        np.testing.assert_array_equal(advanced[other],links[other])


def test_native_command_and_wrist_roundtrip(codec):
    rng = np.random.default_rng(20260920)
    low,high = codec.spec.effective_lower.copy(),codec.spec.effective_upper.copy()
    low[:6],high[:6] = [-.2,-.2,.1,-.6,-.6,-.6],[.2,.2,.5,.6,.6,.6]
    targets = low+(high-low)*rng.uniform(.05,.95,(64,codec.action_dim))
    targets[0,6:] = 0.
    actions = (targets-codec.action_offset)/codec.action_scale
    encoded,mask = codec.encode_actions(actions,targets[0])
    np.testing.assert_allclose(codec.decode_actions(encoded,targets[0]),actions,atol=1e-9,rtol=0)
    assert np.count_nonzero(mask) == 31
    for q in targets[::8]:
        fk = codec.spec.forward_kinematics(q)[codec.spec.frame_link] @ np.asarray(codec.spec.data['link_to_frame'])
        np.testing.assert_allclose(codec.wrist_pose(q),fk,atol=1e-12,rtol=0)
    # Conversion preserves recorded targets even when outside execution limits.
    actions[0,11] = high[11]+.25
    encoded,_ = codec.encode_actions(actions,targets[0])
    np.testing.assert_allclose(codec.decode_actions(encoded,targets[0]),actions,atol=1e-9,rtol=0)
