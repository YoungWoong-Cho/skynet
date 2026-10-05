"""Independent Wuji geometry and pinned functional motion shared by the WUJI1 and WUJI2 tests.

Reference: UniDex 97d869e0; Wuji main.urdf SHA256
94c06247d1b0250e9c12c43ed9957233c414c765903f08a00c8a883186fd7c66.
Axes/tip tangents are at q=0 in official wrist coordinates after FAAS signs.
"""
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np

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


def geometry(urdf, wrist, coordinates=None):
    """Direct URDF FK from the physical wrist link; floating ancestors are excluded."""
    coordinates = coordinates or {}
    pending = list(ET.parse(urdf).getroot().findall('joint'))
    links = {wrist:np.eye(4)}
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


def assert_checksum_and_explicit_anatomical_extension(spec, codec):
    data = json.loads(spec.read_text())
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
