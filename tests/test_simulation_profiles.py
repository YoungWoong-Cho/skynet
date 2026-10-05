"""Evaluation reuses pinned collection assets without preparing observations."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skynet_app.cluster_runtime import DEFAULT_GATEWAY, WORK_ROOT
from skynet_app.database import Database, canonical_json
from skynet_app.observation_preparation import ObservationPreparation
from skynet_app.simulation_profiles import frozen_cluster_profile, freeze_target_simulation_profile


class Cluster:
    def __init__(self, files):
        self.files, self.reads = files, []

    def read_file(self, path, gateway, *, max_bytes):
        self.reads.append((path, gateway, max_bytes))
        return gateway, self.files[path]


@pytest.fixture
def context(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    (root / 'config').mkdir(parents=True)
    base = dict(repository='/pinned/dexverse', source_revision='a'*40, runtime='/pinned/runtime', asset_bundle='/pinned/assets')
    (root / 'config/live_video.json').write_text(json.dumps(base))
    robot, hand_digest = 'skynet_wuji_2_right', 'b'*64
    hand_root = f'{WORK_ROOT}/hands/{robot}/{hand_digest}'
    hand_raw = canonical_json(dict(schema='skynet.simulation-hand/v1', robot=robot, digest=hand_digest))
    assets = canonical_json(dict(schema_version='dataset-bundle/v1', integrity={'inventory_sha256': 'e'*64}))
    cluster = Cluster({hand_root + '/manifest.json': hand_raw, '/pinned/assets/manifest.json': assets})
    def unavailable(*a, **kw):
        raise ValueError('No local bundle')
    monkeypatch.setattr('skynet_app.simulation_hands.find_bundle', unavailable)
    database = Database(tmp_path / 'profile-db')
    profile = dict(robot=robot, hand='right', task='Dexverse-PickCube-v0', source_revision='a'*40,
                   recording_schema_version=3, hand_bundle=dict(digest=hand_digest, manifest_sha256=hashlib.sha256(hand_raw.encode()).hexdigest()))
    def add(identifier, digest, *, task=None):
        current = deepcopy(profile)
        if task:
            current['task'] = task
        path = 'recordings/live/episode-000001.pkl'
        session = dict(id=identifier, profile=current, recordings=[path], recording_checksums={path: digest})
        with database.transaction() as connection:
            connection.execute('INSERT INTO live_xr_sessions VALUES (?,?)', (identifier, canonical_json(session)))
        capture = {key: current[key] for key in ('robot', 'hand', 'task', 'source_revision')}
        capture['hand_adapter_digest'] = hand_digest
        episode = dict(id=digest, session_id=identifier, source={'sha256': digest, 'path': '/archived/' + path},
                       hand_id=robot, capture=capture)
        return session, episode
    session, episode = add('session-a', 'c'*64)
    return SimpleNamespace(root=root, db=database, cluster=cluster, session=session, episode=episode, add=add,
                           hand_root=hand_root, hand_raw=hand_raw, assets=assets)


def test_target_profile_reuses_observation_pin_and_reads_each_session_once(context):
    c = context
    target = {'metadata': {'episodes': [c.episode, deepcopy(c.episode)]}}
    result = freeze_target_simulation_profile(c.db, c.cluster, target, c.root)
    assert result['repository'] == '/pinned/dexverse' and result['source_revision'] == 'a'*40
    assert result['hand_bundle']['root'] == c.hand_root
    assert result['hand_bundle']['manifest_sha256'] == hashlib.sha256(c.hand_raw.encode()).hexdigest()
    assert result['asset_bundle_manifest_sha256'] == hashlib.sha256(c.assets.encode()).hexdigest()
    assert len(c.cluster.reads) == 2
    # Shared-storage manifests must not depend on one login host being up.
    assert {gateway for _, gateway, _ in c.cluster.reads} == {DEFAULT_GATEWAY}
    service = SimpleNamespace(database=c.db, cluster=c.cluster, live=SimpleNamespace(root=c.root))
    assert ObservationPreparation(service).profile(c.session) == result
    assert c.session['profile']['hand_bundle'].get('root') is None


def test_multiple_sessions_require_one_identical_environment(context):
    c = context
    _, second = c.add('session-b', 'd'*64)
    result = freeze_target_simulation_profile(c.db, c.cluster, {'metadata': {'episodes': [c.episode, second]}}, c.root)
    assert result['robot'] == c.episode['hand_id']
    _, other = c.add('different-task', 'f'*64, task='Dexverse-PushCube-v0')
    with pytest.raises(ValueError, match='one identical'):
        freeze_target_simulation_profile(c.db, c.cluster, {'metadata': {'episodes': [c.episode, other]}}, c.root)


@pytest.mark.parametrize('mutation,message', [
    (lambda e: e.update(session_id='missing'), 'unavailable'),
    (lambda e: e['source'].update(sha256='f'*64), 'does not belong'),
    (lambda e: e['capture'].update(robot='wrong'), 'robot differs'),
    (lambda e: e['capture'].update(source_revision='f'*40), 'source_revision differs'),
    (lambda e: e['capture'].update(hand_adapter_digest='f'*64), 'hand asset differs'),
])
def test_target_provenance_failures_do_not_read_cluster_or_schedule_work(context, mutation, message):
    episode = deepcopy(context.episode)
    mutation(episode)
    with pytest.raises(ValueError, match=message):
        freeze_target_simulation_profile(context.db, context.cluster, {'metadata': {'episodes': [episode]}}, context.root)
    assert context.cluster.reads == []


def test_changed_hand_manifest_and_unverified_asset_inventory_are_rejected(context):
    c = context
    c.cluster.files[c.hand_root + '/manifest.json'] += '\n'
    with pytest.raises(ValueError, match='checksum changed'):
        frozen_cluster_profile(c.root, c.cluster, c.session)
    c.cluster.files[c.hand_root + '/manifest.json'] = c.hand_raw
    c.cluster.files['/pinned/assets/manifest.json'] = json.dumps(dict(schema_version='dataset-bundle/v1', integrity={}))
    with pytest.raises(ValueError, match='verified inventory'):
        frozen_cluster_profile(c.root, c.cluster, c.session)


def test_frozen_profile_content_and_digest_do_not_depend_on_the_route_that_read_it(context):
    c = context
    config = c.root / 'config/live_video.json'
    # A "gateway" saved in the video profile is hashed content, never a route.
    config.write_text(json.dumps(dict(json.loads(config.read_text()), gateway='retired-host')))
    session = deepcopy(c.session)
    del session['profile']['hand_bundle']
    profile = frozen_cluster_profile(c.root, c.cluster, session)
    assert [gateway for _, gateway, _ in c.cluster.reads] == [DEFAULT_GATEWAY]
    assert profile == dict(
        gateway='retired-host', repository='/pinned/dexverse', runtime='/pinned/runtime', asset_bundle='/pinned/assets',
        source_revision='a'*40, task_version=0, recording_schema_version=3, robot='skynet_wuji_2_right', hand='right',
        task='Dexverse-PickCube-v0', asset_bundle_manifest_sha256=hashlib.sha256(c.assets.encode()).hexdigest(),
        asset_inventory_sha256='e'*64, device='cuda:0')
    # Evaluation identities hash this profile, so its digest is pinned.
    assert (hashlib.sha256(canonical_json(profile).encode()).hexdigest()
            == 'cf14cea8f205fb91f15c3225cbf2119b0695514787fbf0b3e3724610c0412aae')
    assert frozen_cluster_profile(c.root, c.cluster, session, gateway='another-route') == profile
    assert c.cluster.reads[-1][1] == 'another-route'
    with_hand = frozen_cluster_profile(c.root, c.cluster, c.session)
    assert with_hand == dict(profile, hand_bundle=dict(c.session['profile']['hand_bundle'], root=c.hand_root))


def test_recording_identity_fallback_requires_an_existing_session_and_source_checksum(context):
    episode = deepcopy(context.episode)
    episode['recording_id'] = episode.pop('session_id')
    assert freeze_target_simulation_profile(context.db, context.cluster, {'metadata': {'episodes': [episode]}}, context.root)['robot'] == episode['hand_id']
    episode['recording_id'] = 'unrelated-recording-id'
    with pytest.raises(ValueError, match='unavailable'):
        freeze_target_simulation_profile(context.db, context.cluster, {'metadata': {'episodes': [episode]}}, context.root)
