"""Real PostgreSQL lifecycle checks with an explicit fake cluster boundary.

The planner, immutable capsules, producer claims, leases, manifest verification,
conversion staging and version references are real. Slurm and GPU rendering are
represented by the fake cluster; numerical artifact tests live separately.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from skynet_app.cluster_runtime import ClusterClient, ClusterError, WORK_ROOT
from skynet_app.database import Database, canonical_json
from test_policy_exports import create, install_adapters, set_requirements
from skynet_app.live_xr_review import LiveReviewService
from skynet_app.observation_contracts import ARTIFACT_SCHEMA, PREPARE_SCHEMA, rgb_requirements
from skynet_app.policy_exports import PolicyExportService
from test_policy_exports_cluster import receipt as conversion_receipt

ROOT = Path(__file__).resolve().parents[1]
REVISION = '30cc673e27684b9f10186fa6bea731aed246bc9f'


class Cluster:
    def __init__(self):
        self.files, self.submissions, self.states, self.tokens = {}, [], {}, {}
        self.lock = threading.Lock()
        self.status_calls = []
        self.base = json.loads((ROOT / 'config/live_video.json').read_text())
        self.files[self.base['asset_bundle'] + '/manifest.json'] = canonical_json(dict(
            schema_version='dataset-bundle/v1', integrity=dict(inventory_sha256='e' * 64)))

    def candidates(self, gateway):
        return ['sky2']

    def write_capsule_files(self, identifier, files, gateway):
        assert gateway == 'sky2'
        paths = {}
        with self.lock:
            for relative, value in files.items():
                path = f'{WORK_ROOT}/jobs/runs/{identifier}/{relative}'
                assert path not in self.files or self.files[path] == value
                self.files[path] = value
                paths[relative] = path
        return gateway, paths

    def submit_script(self, script, identifier, gateway, *, submission_key):
        assert gateway == 'sky2'
        ClusterClient._run_id(identifier)
        ClusterClient._submission_token(submission_key)
        with self.lock:
            if submission_key in self.tokens:
                return SimpleNamespace(job_id=self.tokens[submission_key])
            job_id = str(len(self.submissions) + 100)
            self.tokens[submission_key] = job_id
            self.submissions.append(dict(script=script, id=identifier, key=submission_key, job_id=job_id))
            self.states[job_id] = 'PENDING'
        return SimpleNamespace(job_id=job_id)

    def job_statuses(self, identifiers, gateway):
        assert gateway == 'sky2'
        self.status_calls.append(list(identifiers))
        return gateway, {identifier: dict(State=self.states[identifier]) for identifier in identifiers}

    def read_file(self, path, gateway, *, max_bytes):
        assert gateway == 'sky2'
        if path not in self.files:
            raise ClusterError('No file: ' + path)
        value = self.files[path]
        assert len(value.encode()) <= max_bytes
        return gateway, value

    def read_optional_file(self, path, gateway, *, max_bytes):
        assert gateway == 'sky2'
        return self.read_file(path, gateway, max_bytes=max_bytes) if path in self.files else (gateway, None)

    def read_log(self, path, gateway, *, lines=500, max_bytes=1_000_000):
        assert gateway == 'sky2'
        if path not in self.files:
            raise ClusterError('No log: ' + path)
        tail = ''.join(self.files[path].splitlines(keepends=True)[-lines:])
        return gateway, tail.encode()[-max_bytes:].decode(errors='replace')

    def requests(self, mode):
        return [json.loads(raw) for path, raw in self.files.items()
                if path.endswith('/request.json') and json.loads(raw).get('mode') == mode]

    def publish(self, producer, *, corrupt=None):
        """Publish explicit synthetic worker receipts, never execute a simulator."""
        artifacts = []
        for node in producer['request']['requests']:
            manifest = dict(schema=ARTIFACT_SCHEMA, artifact_key=node['artifact_key'], spec=node['spec'],
                            validation=dict(status='PASSED'))
            if corrupt == 'spec':
                manifest['spec'] = dict(manifest['spec'], source_sha256='f' * 64)
            raw = canonical_json(manifest)
            self.files[node['output_dir'] + '/manifest.json'] = raw
            artifacts.append(dict(artifact_key=node['artifact_key'], path=node['output_dir'],
                                  manifest_sha256=hashlib.sha256(raw.encode()).hexdigest(), status='READY'))
        result = dict(schema=PREPARE_SCHEMA, request_id=producer['id'], attempt_token=producer['attempt_token'],
                      state='READY', artifacts=artifacts)
        if corrupt == 'attempt':
            result['attempt_token'] = 'another-attempt'
        if corrupt == 'partial':
            result['artifacts'] = artifacts[:-1]
        self.files[producer['root'] + '/result.json'] = canonical_json(result)
        self.states[producer['cluster_job_id']] = 'COMPLETED'


@pytest.fixture
def context(tmp_path, monkeypatch):
    db = Database(tmp_path / 'observation-integration.store')
    cluster = Cluster()
    sessions = {}
    def add_session(identifier='first', robot='floating_shadow_hand', checksum='a' * 64):
        path = 'recordings/live/episode-000000.pkl'
        session = dict(id=identifier, state='STOPPED', gateway='sky2', root='/collection/' + identifier,
                       created_at='2026-09-15T00:00:00Z', profile=dict(display_name=identifier, source_revision=REVISION,
                       task='Dexverse-PickCube-v0', robot=robot, hand='right', recording_schema_version=3),
                       recordings=[path], recording_checksums={path: checksum}, recording_images={},
                       archive=dict(state='READY', root=f'{WORK_ROOT}/datasets/raw/{identifier}/output'))
        sessions[identifier] = session
        with db.transaction() as connection:
            connection.execute('INSERT INTO live_xr_sessions VALUES (?,?)', (identifier, canonical_json(session)))
        return session
    add_session()
    def resolve(session, relative):
        return cluster, 'sky2', session['archive']['root'] + '/' + relative
    live = SimpleNamespace(root=ROOT, database=db, get=sessions.__getitem__, list=lambda **kw: list(sessions.values()),
                           archive=SimpleNamespace(resolve=resolve, ensure=lambda _: pytest.fail('archive already ready')))
    service = PolicyExportService(LiveReviewService(live, root=tmp_path / 'reviews'), root=tmp_path / 'exports', cluster=cluster)
    install_adapters(service)
    monkeypatch.setattr(PolicyExportService, '_preflight_sources', lambda self, job, sources: sources)
    monkeypatch.setattr(service, 'dispatch', lambda _: None)
    yield SimpleNamespace(service=service, cluster=cluster, sessions=sessions, add_session=add_session, db=db)
    service.stop()


def cloud_contract():
    result = rgb_requirements(['scene_front'])
    result['streams'].append(dict(name='cloud', modality='point_cloud', camera_ids=['scene_front'], width=256, height=256,
                                 channels='XYZRGB', coordinate_frame='world', num_points=128, sampling='farthest_point', seed=7,
                                 processing_order=['unproject', 'world_transform', 'merge', 'crop', 'sample'],
                                 insufficient_points='repeat_with_mask', crop=None, depth_range=[0, 2.5]))
    return result


def test_state_only_conversion_has_no_observation_producer(context):
    service, cluster = context.service, context.cluster
    job = create(service, 'first', 'fixture-state', 'State only')
    assert job['observation_contract']['streams'] == []
    service.prepare(job['id'])
    current = service.get(job['id'])
    assert current['state'] == 'PENDING', current.get('error') or current
    assert not service.observations.store.producers()
    assert not service.observations.store.for_job(job['id'])
    assert len(cluster.submissions) == 1
    assert '#SBATCH --gres' not in cluster.submissions[0]['script']
    request = json.loads(cluster.files[current['cluster_root'] + '/request.json'])
    assert request['sources'][0]['images'] is None
    assert 'observation_artifacts' not in request['sources'][0]


def test_plan_claim_render_derive_then_cpu_and_version_refs(context, monkeypatch):
    service, cluster = context.service, context.cluster
    # Exercise the generic declared-requirements path without inventing a new
    # production adapter or changing the existing converter contract.
    set_requirements(service, 'test-recording-inputs', cloud_contract())
    job = create(service, 'first', 'fixture-rgb', 'RGB and cloud requirements')
    assert not cluster.submissions  # clicking Convert only freezes requirements
    service.prepare(job['id'])
    current = service.get(job['id'])
    assert current['state'] == 'QUEUED' and current['stage'] == 'OBSERVATIONS', current
    render = service.observations.store.producers()[0]
    assert render['request']['mode'] == 'render'
    assert {node['modality'] for node in render['request']['requests']} == {'rgb', 'depth'}
    assert len(cluster.submissions) == 1 and '#SBATCH --gres=gpu:' in render['script']
    assert not cluster.requests('derive')
    submitted = cluster.requests('render')[0]
    assert {source['sha256'] for source in submitted['sources']} == {'a' * 64}
    assert all('dependency_keys' not in node for node in submitted['requests'])
    assert submitted['staging_root'] == render['root'] + '/staging'
    profile = submitted['sources'][0]['profile']
    assert 'export OMNI_KIT_ACCEPT_EULA=YES' in render['script']
    assert 'export PYTHONPATH=' + profile['repository'] + '/source/dexverse:' in render['script']
    assert 'cd ' + profile['repository'] in render['script']
    cluster.publish(render)
    service.observations.tick()
    service.prepare(job['id'])
    derive = service.observations.store.producers()[0]
    assert derive['request']['mode'] == 'derive'
    assert '#SBATCH --gres' not in derive['script'] and 'export CUDA_VISIBLE_DEVICES=' in derive['script']
    derive_request = cluster.requests('derive')[0]
    assert derive_request['staging_root'] == derive['root'] + '/staging'
    assert 'export OMNI_KIT_ACCEPT_EULA' not in derive['script']
    assert 'export PYTHONPATH=' not in derive['script']
    request = derive_request['requests'][0]
    assert request['modality'] == 'point_cloud' and len(request['dependencies']) == 2
    assert all(ref['manifest_sha256'] and ref['path'].startswith(f'{WORK_ROOT}/datasets/recordings/') for ref in request['dependencies'])
    assert len(cluster.submissions) == 2
    cluster.publish(derive)
    service.observations.tick()
    service.prepare(job['id'])
    pending = service.get(job['id'])
    assert pending['state'] == 'PENDING', pending
    assert len(cluster.submissions) == 3
    assert '#SBATCH --cpus-per-task=4' in pending['cluster_script'] and '#SBATCH --gres' not in pending['cluster_script']
    cpu_request = json.loads(cluster.files[pending['cluster_root'] + '/request.json'])
    source = cpu_request['sources'][0]
    assert source['images'] is None
    assert set(source['observation_artifacts']) == {'scene_front'}
    assert set(source['observation_artifacts']['scene_front']) == {'rgb', 'depth', 'point_cloud'}
    assert source['observation_contract'] == job['observation_contract']
    conversion_receipt(service, pending)
    cluster.states[pending['cluster_job_id']] = 'COMPLETED'
    service.prepare(job['id'])
    ready = service.get(job['id'])
    assert ready['state'] == 'READY', ready
    keys = set(service.observations.store.for_job(job['id']))
    with context.db.connection() as connection:
        refs = {row['artifact_key'] for row in connection.execute('SELECT * FROM observation_version_inputs WHERE version_id=?', (ready['version_id'],))}
    assert refs == keys and len(refs) == 3
    assert not list(service.root.rglob('*.pkl')) and not list(service.root.rglob('*.hdf5'))


def test_concurrent_formats_share_one_rgb_producer(context):
    service, cluster = context.service, context.cluster
    first = create(service, 'first', 'fixture-rgb', 'Fixture RGB')
    second = create(service, 'first', 'act', 'ACT')
    # Independent monitors use separate store owners against the same real DB.
    other = PolicyExportService(service.reviews, root=service.root, cluster=cluster)
    barrier = threading.Barrier(2)
    def prepare(instance, job):
        barrier.wait(timeout=5)
        instance.prepare(job['id'])
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(prepare, instance, job) for instance, job in [(service, first), (other, second)]]
            for future in futures:
                future.result(timeout=30)
        first_refs = service.observations.store.for_job(first['id'])
        second_refs = other.observations.store.for_job(second['id'])
        assert set(first_refs) == set(second_refs) and len(first_refs) == 3
        producers = service.observations.store.producers()
        assert len(producers) == 1 and len(cluster.submissions) == 1
        cluster.publish(producers[0])
        other.observations.tick()
        for instance, job in [(service, first), (other, second)]:
            instance.prepare(job['id'])
            assert instance.get(job['id'])['state'] == 'PENDING', instance.get(job['id'])
        assert len(cluster.requests('render')) == 1 and len(cluster.submissions) == 3
    finally:
        other.stop()


def test_multi_hand_selection_keeps_simulator_batches_isolated(context):
    service, cluster = context.service, context.cluster
    context.add_session('second', robot='skynet_wuji_2_right', checksum='b' * 64)
    job = create(service, 'first', 'fixture-rgb', 'Two hands', selections=[dict(session_id='first', indices=None), dict(session_id='second', indices=None)])
    service.prepare(job['id'])
    current = service.get(job['id'])
    assert current['stage'] == 'OBSERVATIONS', current
    producers = service.observations.store.producers()
    assert len(producers) == 2
    for producer in producers:
        request = producer['request']
        assert request['mode'] == 'render' and len(request['sources']) == 1
        source = request['sources'][0]
        assert {node['source_sha256'] for node in request['requests']} == {source['sha256']}
        assert {node['spec']['render_identity']['robot'] for node in request['requests']} == {source['profile']['robot']}
    assert {p['request']['sources'][0]['profile']['robot'] for p in producers} == {'floating_shadow_hand', 'skynet_wuji_2_right'}
    assert len(service.observations.store.for_job(job['id'])) == 6
    # One hand finishing cannot make an incomplete multi-hand conversion runnable.
    cluster.publish(producers[0])
    service.observations.tick()
    service.prepare(job['id'])
    assert service.get(job['id'])['stage'] == 'OBSERVATIONS'
    assert len(cluster.submissions) == 2
    cluster.publish(producers[1])
    service.observations.tick()
    service.prepare(job['id'])
    assert service.get(job['id'])['state'] == 'PENDING'
    assert len(cluster.submissions) == 3


@pytest.mark.parametrize('corrupt', ['attempt', 'spec', 'partial'])
def test_unverified_observation_receipt_never_submits_converter(context, corrupt):
    service, cluster = context.service, context.cluster
    job = create(service, 'first', 'fixture-rgb', 'Invalid result')
    service.prepare(job['id'])
    producer = service.observations.store.producers()[0]
    cluster.publish(producer, corrupt=corrupt)
    service.observations.tick()
    service.prepare(job['id'])
    failed = service.get(job['id'])
    assert failed['state'] == 'FAILED', failed
    assert not failed.get('version_id') and len(cluster.submissions) == 1
    assert all(row['state'] == 'FAILED' for row in service.observations.store.for_job(job['id']).values())


def test_changed_frozen_plan_is_rejected_before_cpu_submission(context):
    service, cluster = context.service, context.cluster
    job = create(service, 'first', 'fixture-rgb', 'Frozen plan')
    service.prepare(job['id'])
    plan = service.root / job['id'] / 'observation-plan.json'
    plan.write_text(plan.read_text() + ' ')
    service.prepare(job['id'])
    failed = service.get(job['id'])
    assert failed['state'] == 'FAILED' and 'Frozen observation plan changed' in failed['error']
    assert len(cluster.submissions) == 1


@pytest.mark.parametrize('log_prefix', [None, '', 'Simulator initialization\n' * 20000], ids=['missing-log', 'short-log', 'large-log'])
def test_terminal_scheduler_failure_without_receipt_does_not_wait_forever(context, log_prefix):
    service, cluster = context.service, context.cluster
    job = create(service, 'first', 'fixture-rgb', 'Failed bootstrap')
    service.prepare(job['id'])
    producer = service.observations.store.producers()[0]
    cluster.states[producer['cluster_job_id']] = 'FAILED'
    if log_prefix is not None:
        cluster.files[producer['root'] + '/observations.log'] = log_prefix + 'ModuleNotFoundError: simulator dependency'
    service.observations.tick()
    service.prepare(job['id'])
    failed = service.get(job['id'])
    assert failed['state'] == 'FAILED' and 'Observation job ended as FAILED' in failed['error']
    assert ('ModuleNotFoundError' in failed['error']) == (log_prefix is not None)
    assert not service.observations.store.producers()
    assert len(cluster.submissions) == 1
    assert all(row['state'] == 'FAILED' for row in service.observations.store.for_job(job['id']).values())


def test_unknown_scheduler_status_keeps_same_attempt_until_recovery(context, monkeypatch):
    service, cluster = context.service, context.cluster
    job = create(service, 'first', 'fixture-rgb', 'Reconnect')
    service.prepare(job['id'])
    producer = service.observations.store.producers()[0]
    statuses = cluster.job_statuses
    def unavailable(*args, **kwargs):
        raise ClusterError('gateway is temporarily unavailable')
    monkeypatch.setattr(cluster, 'job_statuses', unavailable)
    service.observations.tick()
    waiting = service.observations.store.producers()[0]
    assert waiting['id'] == producer['id'] and waiting['attempt_token'] == producer['attempt_token']
    assert waiting['state'] == 'PENDING' and len(cluster.submissions) == 1
    monkeypatch.setattr(cluster, 'job_statuses', statuses)
    cluster.publish(waiting)
    service.observations.tick()
    service.prepare(job['id'])
    assert service.get(job['id'])['state'] == 'PENDING'
    assert len(cluster.requests('render')) == 1 and len(cluster.submissions) == 2


def test_long_selection_is_partitioned_into_bounded_complete_render_requests(context, monkeypatch):
    from skynet_app import observation_preparation as module
    service, cluster = context.service, context.cluster
    monkeypatch.setattr(module, 'MAX_BATCH_SOURCES', 2)
    for index in range(1, 5):
        context.add_session(f'episode-{index}', checksum=f'{index:064x}')
    job = create(service, 'first', 'fixture-rgb', 'Bounded selection', selections=[
        dict(session_id=identifier, indices=None) for identifier in context.sessions])
    service.prepare(job['id'])
    assert service.get(job['id'])['stage'] == 'OBSERVATIONS'
    requests = cluster.requests('render')
    assert len(requests) == 3 and all(len(request['sources']) <= 2 for request in requests)
    assert all(len(canonical_json(request).encode()) < module.MAX_REQUEST_BYTES for request in requests)
    keys = [node['artifact_key'] for request in requests for node in request['requests']]
    assert len(keys) == len(set(keys)) == 15
    assert set(keys) == set(service.observations.store.for_job(job['id']))
    assert {source['sha256'] for request in requests for source in request['sources']} == {
        sha for session in context.sessions.values() for sha in session['recording_checksums'].values()}


def test_batch_payload_size_limit_is_independent_of_source_count(monkeypatch):
    from skynet_app import observation_preparation as module
    monkeypatch.setattr(module, 'MAX_BATCH_BYTES', 5000)
    sources = [dict(sha256=str(i), profile=dict(payload='x' * 1000)) for i in range(3)]
    nodes = [dict(source_sha256=str(i), spec=dict(payload='y' * 1200)) for i in range(3)]
    batches = list(module.request_batches(nodes, sources))
    assert len(batches) == 3
    assert [batch[0] for batch, _ in batches] == nodes
    for batch, selected in batches:
        assert len(canonical_json(dict(requests=batch, sources=selected)).encode()) < module.MAX_BATCH_BYTES
    nodes[0]['spec']['payload'] = 'z' * 6000
    with pytest.raises(ValueError, match='One recording observation request exceeds'):
        list(module.request_batches(nodes, sources))


def test_observation_transport_tracks_service_injection(context):
    replacement = Cluster()
    context.service.cluster = replacement
    assert context.service.observations.cluster is replacement


@pytest.mark.parametrize('scheduler_state', ['REQUEUED', 'RESIZING', 'UNKNOWN', 'SPECIAL_EXIT', 'PREEMPTED', 'REVOKED'])
def test_scheduler_transition_and_unknown_state_preserve_observation_attempt(context, scheduler_state):
    service, cluster = context.service, context.cluster
    job = create(service, 'first', 'fixture-rgb', 'Keep producer attempt')
    service.prepare(job['id'])
    original = service.observations.store.producers()[0]
    keys = set(service.observations.store.for_job(job['id']))
    cluster.states[original['cluster_job_id']] = scheduler_state
    service.observations.tick()
    service.prepare(job['id'])
    service.retry(job['id'])
    service.prepare(job['id'])
    current = service.observations.store.producers()[0]
    assert current['id'] == original['id'] and current['attempt_token'] == original['attempt_token']
    assert current['cluster_job_id'] == original['cluster_job_id'] and current['script'] == original['script']
    assert current['state'] == ('RUNNING' if scheduler_state == 'RESIZING' else 'PENDING')
    assert set(service.observations.store.for_job(job['id'])) == keys
    assert len(cluster.submissions) == 1 and service.get(job['id'])['state'] != 'FAILED'
    cluster.publish(current)
    service.observations.tick()
    service.prepare(job['id'])
    assert service.get(job['id'])['state'] == 'PENDING'
    assert len(cluster.submissions) == 2 and len(cluster.requests('render')) == 1


def completed_without_receipt(context, monkeypatch):
    from skynet_app import observation_preparation as module
    clock = [1000.0]
    monkeypatch.setattr(module, 'time', lambda: clock[0])
    service, cluster = context.service, context.cluster
    job = create(service, 'first', 'fixture-rgb', 'Completed without receipt')
    service.prepare(job['id'])
    producer = service.observations.store.producers()[0]
    cluster.states[producer['cluster_job_id']] = 'COMPLETED'
    return service, cluster, job, producer, clock


@pytest.mark.parametrize('has_log', [False, True])
def test_completed_missing_receipt_fails_after_durable_visibility_grace(context, monkeypatch, has_log):
    from skynet_app.observation_preparation import ObservationPreparation, COMPLETED_RESULT_GRACE_SECONDS
    service, cluster, job, producer, clock = completed_without_receipt(context, monkeypatch)
    keys = set(service.observations.store.for_job(job['id']))
    if has_log:
        cluster.files[producer['root'] + '/observations.log'] = 'Simulator startup\n' * 20000 + 'Exited before writing result'
    service.observations.tick()
    first = service.observations.store.producers()[0]
    assert first['result_missing_since'] == clock[0]
    clock[0] += COMPLETED_RESULT_GRACE_SECONDS - 1
    service.observations.tick()
    assert len(service.observations.store.producers()) == 1
    # A restarted monitor reads the durable first-confirmed-missing timestamp.
    service.observations = ObservationPreparation(service)
    clock[0] += 2
    service.observations.tick()
    service.prepare(job['id'])
    failed = service.get(job['id'])
    assert failed['state'] == 'FAILED'
    assert 'completed without a result file' in failed['error']
    assert ('Exited before writing result' in failed['error']) == has_log
    assert not service.observations.store.producers()
    assert len(cluster.submissions) == 1
    assert all(row['state'] == 'FAILED' for row in service.observations.store.for_job(job['id']).values())
    service.retry(job['id'])
    service.prepare(job['id'])
    retried = service.observations.store.producers()[0]
    assert retried['id'] != producer['id'] and retried['attempt_token'] != producer['attempt_token']
    assert set(service.observations.store.for_job(job['id'])) == keys
    assert len(cluster.submissions) == 2


def test_completed_receipt_visible_during_grace_uses_original_attempt(context, monkeypatch):
    service, cluster, job, producer, clock = completed_without_receipt(context, monkeypatch)
    service.observations.tick()
    clock[0] += 30
    cluster.publish(producer)
    service.observations.tick()
    assert not service.observations.store.producers()
    assert all(row['state'] == 'READY' for row in service.observations.store.for_job(job['id']).values())
    service.prepare(job['id'])
    assert service.get(job['id'])['state'] == 'PENDING'
    assert len(cluster.requests('render')) == 1 and len(cluster.submissions) == 2


def test_completed_receipt_transport_failure_never_confirms_missing(context, monkeypatch):
    service, cluster, job, producer, clock = completed_without_receipt(context, monkeypatch)
    read = cluster.read_optional_file
    def offline(*args, **kwargs):
        raise ClusterError('connection interrupted')
    monkeypatch.setattr(cluster, 'read_optional_file', offline)
    service.observations.tick()
    clock[0] += 120
    service.observations.tick()
    current = service.observations.store.producers()[0]
    assert current.get('result_missing_since') is None
    monkeypatch.setattr(cluster, 'read_optional_file', read)
    service.observations.tick()
    since = service.observations.store.producers()[0]['result_missing_since']
    monkeypatch.setattr(cluster, 'read_optional_file', offline)
    clock[0] += 120
    service.observations.tick()
    current = service.observations.store.producers()[0]
    assert current['result_missing_since'] == since
    assert current['attempt_token'] == producer['attempt_token']
    assert current['cluster_job_id'] == producer['cluster_job_id'] and len(cluster.submissions) == 1
    # A result already present while disconnected is still accepted after grace.
    cluster.publish(producer)
    monkeypatch.setattr(cluster, 'read_optional_file', read)
    service.observations.tick()
    assert not service.observations.store.producers()
    assert all(row['state'] == 'READY' for row in service.observations.store.for_job(job['id']).values())


def test_requeued_producer_resets_completed_result_visibility_grace(context, monkeypatch):
    service, cluster, job, producer, clock = completed_without_receipt(context, monkeypatch)
    service.observations.tick()
    clock[0] += 120
    cluster.states[producer['cluster_job_id']] = 'REQUEUED'
    service.observations.tick()
    assert service.observations.store.producers()[0]['result_missing_since'] is None
    cluster.states[producer['cluster_job_id']] = 'COMPLETED'
    service.observations.tick()
    assert service.observations.store.producers()[0]['result_missing_since'] == clock[0]
    assert len(cluster.submissions) == 1
    cluster.publish(producer)
    service.observations.tick()
    assert all(row['state'] == 'READY' for row in service.observations.store.for_job(job['id']).values())


def test_observation_tick_batches_statuses_and_preserves_attempts_on_outage(context, monkeypatch):
    service, cluster = context.service, context.cluster
    context.add_session('second', robot='skynet_wuji_2_right', checksum='b' * 64)
    job = create(service, 'first', 'fixture-rgb', 'Batched scheduler status',
        selections=[dict(session_id='first', indices=None), dict(session_id='second', indices=None)])
    service.prepare(job['id'])
    producers = service.observations.store.producers()
    assert len(producers) == 2 and len(cluster.submissions) == 2
    assert cluster.status_calls == [], 'New submissions wait until the next shared read'
    job_ids = {producer['cluster_job_id'] for producer in producers}
    for identifier in job_ids:
        cluster.states[identifier] = 'RUNNING'
    service.observations.tick()
    assert len(cluster.status_calls) == 1 and set(cluster.status_calls[0]) == job_ids
    assert all(producer['state'] == 'RUNNING' for producer in service.observations.store.producers())
    original_statuses = cluster.job_statuses
    failures = []
    def unavailable(identifiers, gateway):
        failures.append(list(identifiers))
        raise ClusterError('gateway unavailable')
    monkeypatch.setattr(cluster, 'job_statuses', unavailable)
    service.observations.tick()
    assert len(failures) == 1 and set(failures[0]) == job_ids
    unchanged = service.observations.store.producers()
    assert {(p['id'], p['attempt_token'], p['cluster_job_id']) for p in unchanged} == {
        (p['id'], p['attempt_token'], p['cluster_job_id']) for p in producers}
    assert all(p['state'] == 'RUNNING' for p in unchanged) and len(cluster.submissions) == 2
    monkeypatch.setattr(cluster, 'job_statuses', original_statuses)
    for producer in producers:
        cluster.publish(producer)
    cluster.status_calls.clear()
    service.observations.tick()
    assert len(cluster.status_calls) == 1 and set(cluster.status_calls[0]) == job_ids
    assert not service.observations.store.producers()
    assert all(row['state'] == 'READY' for row in service.observations.store.for_job(job['id']).values())
    assert len(cluster.submissions) == 2
