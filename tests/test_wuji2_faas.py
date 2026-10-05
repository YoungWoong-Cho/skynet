"""WUJI2 geometry independently checked against pinned Wuji functional motion."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ops.datasets.action_codecs.unidex import load_codec
from wuji_faas_reference import UPSTREAM, assert_checksum_and_explicit_anatomical_extension, fixed_transform, geometry as urdf_geometry

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'config/action_representations/unidex-faas-v1/skynet_wuji_2_right.json'
URDF = SPEC.parent / 'assets/skynet_wuji_2_right.urdf'


def geometry(coordinates=None):
    return urdf_geometry(URDF, 'h_r_wrist', coordinates)


@pytest.fixture
def codec():
    return load_codec('skynet_wuji_2_right')


def test_checksum_and_explicit_anatomical_extension(codec):
    assert_checksum_and_explicit_anatomical_extension(SPEC, codec)


def test_own_wrist_origin_and_right_handed_anatomical_frame(codec):
    frame = np.asarray(codec.spec.data['link_to_frame'])
    np.testing.assert_array_equal(frame[:3,3],0.)
    assert np.linalg.det(frame[:3,:3]) == pytest.approx(1.)
    links, _ = geometry()
    tips = [frame[:3,:3].T @ links[name][:3,3] for name in codec.spec.tip_links]
    assert all(tip[2] > .07 for tip in tips)
    assert tips[0][1] > tips[1][1] > tips[2][1] > tips[3][1] > tips[4][1]
    pose = codec.wrist_pose(np.zeros(codec.action_dim))
    np.testing.assert_allclose(pose[:3,:3],[[0,0,1],[0,1,0],[-1,0,0]],atol=1e-12)
    np.testing.assert_allclose(pose[:3,3],0.,atol=1e-12)


@pytest.mark.parametrize('index',range(20))
def test_each_positive_joint_motion_matches_functional_reference(codec,index):
    mapping = codec.mapping[index]
    links, joints = geometry()
    axis, _ = joints[mapping['joint']]
    frame = np.asarray(codec.spec.data['link_to_frame'])[:3,:3]
    reference_axis, reference_tangent = np.asarray(UPSTREAM[index])
    positive_axis = frame.T @ axis / mapping['scale']
    assert positive_axis @ reference_axis > (.88 if index < 4 else .99)
    tip = codec.spec.tip_links[index//4]
    advanced, _ = geometry({mapping['joint']:.001/mapping['scale']})
    tangent = frame.T @ (advanced[tip][:3,3]-links[tip][:3,3])
    tangent /= np.linalg.norm(tangent)
    assert tangent @ reference_tangent > (.80 if index < 4 else .98)
    if index >= 4:
        assert tangent[1 if index%4==1 else 0] > (.95 if index%4==1 else .98)
    for other in set(codec.spec.tip_links)-{tip}:
        np.testing.assert_array_equal(advanced[other],links[other])


def test_explicit_native_zero_and_thumb_projected_flexion(codec):
    assert codec.spec.data['mapping_validation']['rest_reference'] == 'native_urdf_zero'
    assert all(m['offset'] == 0. for m in codec.mapping)
    values, _ = codec.encode_state(np.zeros(codec.action_dim))
    np.testing.assert_array_equal(values[18:],0.)
    joints = {j.get('name'):j for j in ET.parse(URDF).getroot().findall('joint')}
    for name in ('h_r_thumb_mcp','h_r_thumb_ip'):
        joint = joints[name]
        transform = fixed_transform(joint)[:3,:3]
        axis = transform @ np.fromstring(joint.find('axis').get('xyz'),sep=' ')
        before, after = np.array([0.,0.,-1.]), transform @ np.array([0.,0.,-1.])
        before -= axis*(before @ axis)
        after -= axis*(after @ axis)
        angle = np.arctan2(axis @ np.cross(before,after),before @ after)
        assert abs(angle) < 1e-12


def test_native_commands_roundtrip_without_target_clipping(codec):
    rng = np.random.default_rng(2202)
    low, high = codec.spec.effective_lower.copy(),codec.spec.effective_upper.copy()
    low[:6], high[:6] = [-.2,-.2,.1,-.6,-.6,-.6],[.2,.2,.5,.6,.6,.6]
    targets = low+(high-low)*rng.uniform(.05,.95,(32,codec.action_dim))
    targets[0,6:] = 0.
    actions = (targets-codec.action_offset)/codec.action_scale
    encoded, mask = codec.encode_actions(actions,targets[0])
    np.testing.assert_allclose(codec.decode_actions(encoded,targets[0]),actions,atol=1e-9)
    assert np.count_nonzero(mask) == 29
    actions[0,10] = high[10]+.25
    encoded, _ = codec.encode_actions(actions,targets[0])
    np.testing.assert_allclose(codec.decode_actions(encoded,targets[0]),actions,atol=1e-9)
