"""Saved capture provenance survives independent downstream recipe changes."""
from copy import deepcopy
import hashlib
import json

import pytest

from skynet_app.observation_contracts import (
    content_digest, plan_artifacts, reuse_recorded_captures,
)
from test_observation_contracts import cloud, identity
from test_observation_preparation import context, cloud_contract
from test_policy_exports import create, set_requirements
from skynet_app.cluster_config import CLUSTER


def capture_identity(revision='1' * 64):
    return dict(identity(), renderer_revision=revision, task='PickCube', robot='wuji2',
                runtime='isaac', camera_policy='scene-cameras/v1')


def nodes(contract=None, scene=None, source='a' * 64, episode=0):
    return plan_artifacts(source, {'episode_index': episode}, contract or cloud(), scene or capture_identity())


def ready(planned):
    return [dict(artifact_key=node['artifact_key'], spec=deepcopy(node['spec']), state='READY',
                 path='/captures/' + node['artifact_key'], manifest_sha256='f' * 64)
            for node in planned if node['modality'] in {'rgb', 'depth'}]


def test_new_point_count_reuses_actual_old_capture_identity():
    original = nodes()
    requirement = cloud()
    requirement['streams'][0]['num_points'] = 10000
    requested = nodes(requirement, capture_identity('2' * 64))
    result = reuse_recorded_captures(requested, ready(original))
    assert [n['artifact_key'] for n in result[:-1]] == [n['artifact_key'] for n in original[:-1]]
    assert result[-1]['recipe']['num_points'] == 10000
    assert result[-1]['artifact_key'] not in {original[-1]['artifact_key'], requested[-1]['artifact_key']}
    assert result[-1]['dependencies'] == [n['artifact_key'] for n in original[:-1]]
    assert all(n['spec']['render_identity']['renderer_revision'] == '1' * 64 for n in result)
    assert result[-1]['spec']['derivation_revision'] == '2' * 64
    assert all(n['artifact_key'] == content_digest(n['spec']) for n in result)
    assert requested[-1]['spec']['render_identity']['renderer_revision'] == '2' * 64


def test_geometry_revision_cannot_reuse_an_older_derived_cloud():
    captures = ready(nodes())
    first = reuse_recorded_captures(nodes(scene=capture_identity('2' * 64)), captures)
    later = reuse_recorded_captures(nodes(scene=capture_identity('3' * 64)), captures)
    assert first[:-1] == later[:-1]
    assert first[-1]['dependencies'] == later[-1]['dependencies']
    assert first[-1]['spec']['render_identity'] == later[-1]['spec']['render_identity']
    assert first[-1]['artifact_key'] != later[-1]['artifact_key']
    assert first[-1]['spec']['derivation_revision'] == '2' * 64
    assert later[-1]['spec']['derivation_revision'] == '3' * 64


@pytest.mark.parametrize('change', [
    'source', 'episode', 'width', 'height', 'pose', 'projection', 'camera_policy',
    'task', 'robot', 'runtime', 'source_revision', 'assets', 'timing', 'missing',
    'not_ready', 'bad_manifest_hash', 'bad_artifact_hash',
])
def test_incompatible_or_unverified_capture_is_not_reused(change):
    original = nodes()
    scene, requirement, source, episode = capture_identity('2' * 64), cloud(), 'a' * 64, 0
    candidates = ready(original)
    if change == 'source':
        source = 'b' * 64
    elif change == 'episode':
        episode = 1
    elif change in {'width', 'height'}:
        requirement['streams'][0][change] += 1
    elif change in {'pose', 'projection'}:
        scene['cameras']['front'][change] = 'changed'
    elif change in {'camera_policy', 'task', 'robot', 'runtime', 'source_revision'}:
        scene[change] += '-changed'
    elif change == 'assets':
        scene['asset_sha256'] = 'different-assets'
    elif change == 'timing':
        for artifact in candidates:
            artifact['spec']['timing']['alignment'] = 'post_action_state'
            artifact['artifact_key'] = content_digest(artifact['spec'])
    elif change == 'missing':
        candidates.pop()
    elif change == 'not_ready':
        candidates[0]['state'] = 'FAILED'
    elif change == 'bad_manifest_hash':
        candidates[0]['manifest_sha256'] = 'bad'
    elif change == 'bad_artifact_hash':
        candidates[0]['artifact_key'] = '0' * 64
    requested = nodes(requirement, scene, source, episode)
    assert reuse_recorded_captures(requested, candidates) == requested


def test_cloud_cannot_mix_capture_families_and_prefers_complete_current_family():
    old, current = nodes(), nodes(scene=capture_identity('2' * 64))
    mixed = ready(old)[:2] + ready(current)[2:]
    assert reuse_recorded_captures(current, mixed) == current
    assert reuse_recorded_captures(current, ready(old) + ready(current)) == current


def test_adding_independent_view_reuses_front_without_forging_missing_capture():
    requirement = cloud()
    front = deepcopy(requirement['streams'][0])
    front['camera_ids'] = ['front']
    front['name'] = 'front_cloud'
    side = dict(front, camera_ids=['side'], name='side_cloud')
    saved = nodes(dict(requirement, streams=[front]))
    requested = nodes(dict(requirement, streams=[front, side]), capture_identity('2' * 64))
    result = reuse_recorded_captures(requested, ready(saved))
    assert result[:2] == saved[:2]
    assert result[2]['spec']['render_identity'] == saved[2]['spec']['render_identity']
    assert result[2]['spec']['derivation_revision'] == '2' * 64
    assert result[3:] == requested[3:]


def test_new_conversion_uses_cpu_derive_with_preserved_old_renderer(context, monkeypatch):
    service, cluster = context.service, context.cluster
    set_requirements(service, 'test-recording-inputs', cloud_contract())
    original = create(service, 'first', 'fixture-rgb', 'Saved captures')
    service.prepare(original['id'])
    producer = service.observations.store.producers()[0]
    assert producer['request']['mode'] == 'render'
    cluster.publish(producer)
    service.observations.tick()
    saved = service.observations.store.for_job(original['id'])
    capture_keys = {key for key, value in saved.items() if value['state'] == 'READY'}
    original_revision = next(value['spec']['render_identity']['renderer_revision']
                             for value in saved.values())
    # A worker lifecycle-only edit changes the entire frozen capsule digest.
    original_files = service.observations.frozen_files()
    monkeypatch.setattr(service.observations, 'frozen_files', lambda: dict(original_files,
        **{'observation_supervisor.py': original_files['observation_supervisor.py'] + '\n# lifecycle revision\n'}))
    requirement = cloud_contract()
    requirement['streams'][1]['num_points'] = 10000
    set_requirements(service, 'test-recording-inputs', requirement)
    converted = create(service, 'first', 'fixture-rgb', 'Ten thousand points')
    assert converted['observation_worker_sha256'] != original['observation_worker_sha256']
    service.prepare(converted['id'])
    current = service.get(converted['id'])
    assert current['state'] == 'QUEUED', current
    assert len(cluster.requests('render')) == 1
    derived = cluster.requests('derive')
    assert len(derived) == 1
    request = derived[0]['requests'][0]
    assert request['recipe']['num_points'] == 10000
    assert {d['artifact_key'] for d in request['dependencies']} == capture_keys
    assert request['spec']['render_identity']['renderer_revision'] == original_revision
    for dependency in request['dependencies']:
        assert dependency == {key: saved[dependency['artifact_key']][key]
                              for key in ('artifact_key', 'path', 'manifest_sha256')}
    # The source profile stays pinned to its recording. The derive worker uses
    # the saved capture specification, not the latest renderer recipe.
    assert derived[0]['sources'][0]['profile'] == producer['request']['sources'][0]['profile']
    script = cluster.submissions[-1]['script']
    assert '#SBATCH --gres' not in script
    derive = CLUSTER.defaults.background_jobs.observation_derive
    assert f'#SBATCH --cpus-per-task={derive.cpus_per_task}' in script and f'#SBATCH --time={derive.time_limit}' in script
    raw = (service.root / converted['id'] / 'observation-plan.json').read_text()
    assert hashlib.sha256(raw.encode()).hexdigest() == current['observation_plan_sha256']
    assert {node['artifact_key'] for node in json.loads(raw)['nodes'] if node['modality'] != 'point_cloud'} == capture_keys
