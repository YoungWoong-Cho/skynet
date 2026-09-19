"""WUJI1 geometry independently checked against pinned Wuji functional motion.

Reference: UniDex 97d869e0; Wuji main.urdf SHA256
94c06247d1b0250e9c12c43ed9957233c414c765903f08a00c8a883186fd7c66.
Axes/tip tangents are at q=0 in official wrist coordinates after FAAS signs.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ops.datasets.action_codecs.unidex import load_codec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'config/action_representations/unidex-faas-v1/skynet_wuji_1_right.json'
URDF = SPEC.parent / 'assets/skynet_wuji_1_right.urdf'
UPSTREAM = [
    ([.422924620,-.056212869,-.904419637],[.889590537,-.164263276,.426199780]),
    ([.923775270,-.122783249,.362716866],[-.382934863,-.295263764,.875317200]),
    ([.667567135,.264381426,-.696029153],[.743543452,-.188142039,.641674145]),
    ([.667567135,.264381426,-.696029153],[.742736287,-.171264376,.647310839]),
    ([.222838335,.968606315,-.110203822],[.971577276,-.211402392,.106520538]),
    ([-.973813961,.215949321,-.071079257],[.222857829,.968552873,-.110633272]),
    ([.222857829,.968552873,-.110633271],[.972793730,-.213581841,.089750522]),
    ([.223119383,.968494835,-.110614175],[.971235544,-.211186483,.110008121]),
    ([-.000579437,.999999831,-.000050694],[.992436262,.000581277,.122759632]),
    ([-.996194528,-.000581650,-.087155743],[-.000541680,.999999737,-.000482247]),
    ([-.000541681,.999999737,-.000482246],[.994374896,.000589711,.105916095]),
    ([-.000274108,.999999857,-.000458837],[.992000397,.000329836,.126234318]),
    ([-.054430156,.994458905,.089937999],[.991972843,.043557855,.118712224]),
    ([-.995414918,-.046945442,-.083338268],[-.054396080,.994499557,.089508086]),
    ([-.054396078,.994499560,.089508060],[.993771978,.045188661,.101858921]),
    ([-.054128718,.994512133,.089530440],[.991611175,.042997246,.121895503]),
    ([-.171574322,.966581576,.190479159],[.980611072,.148976190,.127310721]),
    ([-.983371755,-.156348797,-.092385307],[-.171548523,.966669495,.190055757]),
    ([-.171548523,.966669495,.190055758],[.982073932,.152499795,.110790816]),
    ([-.171284388,.966711455,.190080566],[.980316597,.148001705,.130670827]),
]


def rotation(axis, angle):
    """Independent test-only Rodrigues formula; no production codec FK."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.cos(angle)*np.eye(3)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*skew


def fixed_transform(joint):
    origin = joint.find('origin')
    r, p, y = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
    t = np.eye(4)
    t[:3,:3] = rotation([0,0,1],y) @ rotation([0,1,0],p) @ rotation([1,0,0],r)
    t[:3,3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    return t


def geometry(coordinates=None):
    """Direct URDF FK from physical wrist; floating ancestors are excluded."""
    coordinates = coordinates or {}
    pending = list(ET.parse(URDF).getroot().findall('joint'))
    links = {'h_right_palm_link':np.eye(4)}
    joints = {}
    while True:
        ready = [j for j in pending if j.find('parent').get('link') in links]
        if not ready:
            break
        for j in ready:
            t = links[j.find('parent').get('link')] @ fixed_transform(j)
            if j.get('type') != 'fixed':
                a = np.fromstring(j.find('axis').get('xyz'), sep=' ')
                joints[j.get('name')] = (t[:3,:3] @ a, t[:3,3].copy())
                t[:3,:3] = t[:3,:3] @ rotation(a, coordinates.get(j.get('name'),0.))
            links[j.find('child').get('link')] = t
            pending.remove(j)
    return links, joints


@pytest.fixture
def codec():
    return load_codec('skynet_wuji_1_right', spec_root=SPEC.parent)


def test_checksum_and_explicit_anatomical_extension(codec):
    data = json.loads(SPEC.read_text())
    body = {k:v for k,v in data.items() if k != 'spec_hash'}
    checksum = hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert data['spec_hash'] == checksum == codec.digest
    assert data['mapping_validation']['kind'] == 'anatomical_extension'
    assert data['mapping_validation']['exact_upstream_asset_equivalence'] is False
    assert len(codec.mapping) == len(codec.independent) == 20
    assert {m['joint'] for m in codec.mapping} == set(codec.action_names[6:])
    assert len({m['slot'] for m in codec.mapping}) == 20
    assert all(m['source_joint'] == m['joint'].removeprefix('h_') for m in codec.mapping)
    assert [m['slot'] for m in codec.mapping] == [1,2,3,4,7,6,8,9,12,11,13,14,17,16,18,19,22,21,23,24]


def test_own_wrist_origin_and_right_handed_anatomical_frame(codec):
    frame = np.asarray(codec.spec.data['link_to_frame'])
    np.testing.assert_array_equal(frame[:3,3],0.)
    assert np.linalg.det(frame[:3,:3]) == pytest.approx(1.)
    links, _ = geometry()
    tips = [frame[:3,:3].T @ links[name][:3,3] for name in codec.spec.tip_links]
    assert all(tip[2] > .06 for tip in tips)
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
    assert positive_axis @ reference_axis > .99998
    tip = codec.spec.tip_links[index//4]
    advanced, _ = geometry({mapping['joint']:.001/mapping['scale']})
    tangent = frame.T @ (advanced[tip][:3,3]-links[tip][:3,3])
    tangent /= np.linalg.norm(tangent)
    assert tangent @ reference_tangent > .9998
    if index >= 4:
        assert tangent[1 if index%4==1 else 0] > .95
    for other in set(codec.spec.tip_links)-{tip}:
        np.testing.assert_array_equal(advanced[other],links[other])


def test_explicit_native_zero_and_no_upstream_sign_copy(codec):
    assert codec.spec.data['mapping_validation']['rest_reference'] == 'native_urdf_zero'
    assert all(m['offset'] == 0. and m['scale'] == 1. for m in codec.mapping)
    values, _ = codec.encode_state(np.zeros(codec.action_dim))
    np.testing.assert_array_equal(values[18:],0.)
    # This is a coordinate reference, not a pose command: the thumb CMC lower
    # limit is positive. Conversion preserves this asset's actual joint values.
    assert codec.spec.effective_lower[6] == pytest.approx(.0475)
    q = np.zeros(codec.action_dim)
    q[10] = .1
    state, _ = codec.encode_state(q)
    assert state[18 + 7] == pytest.approx(.1)


def test_native_commands_roundtrip_without_target_clipping(codec):
    rng = np.random.default_rng(1101)
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
