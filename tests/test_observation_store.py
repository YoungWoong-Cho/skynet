import time

import pytest
from skynet_app.database import Database
from skynet_app.observation_contracts import plan_artifacts, rgb_requirements
from skynet_app.observation_store import ObservationStore


@pytest.fixture
def context(tmp_path):
    db = Database(tmp_path / 'observations.store')
    with db.transaction() as c:
        c.execute("INSERT INTO live_xr_sessions VALUES ('session','{}')")
        for job in ('first', 'second'):
            c.execute('INSERT INTO policy_exports VALUES (?,?)', (job, '{}'))
    store = ObservationStore(db)
    source = dict(session_id='session', path='recordings/a.pkl', sha256='a'*64)
    nodes = plan_artifacts('a'*64, {}, rgb_requirements(['front']), dict(renderer='fixed', cameras={'front': {'sensor': 'fixed'}}))
    for job in ('first', 'second'):
        store.attach(job, nodes, [source])
    return db, store, source, nodes


def test_competing_converts_claim_one_producer_and_share_artifacts(context):
    db, store, source, nodes = context
    first = store.claim(nodes, {'mode': 'render'})
    assert first
    other = ObservationStore(db)
    assert other.claim(nodes, {'mode': 'render'}) is None
    assert len(other.producers()) == 1
    assert set(store.for_job('first')) == set(other.for_job('second'))
    owned = store.acquire(first['id'])
    key = nodes[0]['artifact_key']
    store.finish(owned, artifacts=[dict(artifact_key=key, path='/observations/'+key, manifest_sha256='b'*64)])
    assert store.for_job('second')[key]['state'] == 'READY'


def test_delete_initiating_convert_does_not_delete_producer(context):
    db, store, _, nodes = context
    producer = store.claim(nodes, {'mode': 'render'})
    with db.transaction() as c:
        c.execute("DELETE FROM policy_exports WHERE id='first'")
    assert store.producers()[0]['id'] == producer['id']
    assert store.for_job('second')[nodes[0]['artifact_key']]['state'] == 'QUEUED'


def test_expired_owner_cannot_publish_after_recovery(context):
    db, first, _, nodes = context
    producer = first.claim(nodes, {'mode': 'render'})
    old = first.acquire(producer['id'])
    second = ObservationStore(db)
    assert second.acquire(producer['id']) is None
    with db.transaction() as c:
        c.execute('UPDATE observation_producers SET lease_until=0 WHERE id=?', (producer['id'],))
    recovered = second.acquire(producer['id'])
    assert recovered['attempt_token'] == old['attempt_token']
    with pytest.raises(ValueError, match='stale'):
        first.finish(old, error='old worker')
    second.finish(recovered, error='actual terminal failure')
    first.retry('first')
    replacement = first.claim(nodes, {'mode': 'render'})
    assert replacement['id'] != old['id']
    assert replacement['attempt_token'] != old['attempt_token']


def test_retry_cannot_change_published_artifact(context):
    db, store, _, nodes = context
    producer = store.acquire(store.claim(nodes, {'mode': 'render'})['id'])
    key = nodes[0]['artifact_key']
    store.finish(producer, artifacts=[dict(artifact_key=key, path='/observations/'+key, manifest_sha256='b'*64)])
    store.retry('first')
    assert store.claim(nodes, {}) is None
    with pytest.raises(Exception, match='immutable'):
        with db.transaction() as c:
            c.execute("UPDATE observation_artifacts SET path='elsewhere' WHERE artifact_key=?", (key,))


def test_partial_receipt_cannot_publish(context):
    db, store, _, nodes = context
    producer = store.acquire(store.claim(nodes, {})['id'])
    with pytest.raises(ValueError, match='every claimed artifact'):
        store.finish(producer, artifacts=[])
    assert store.for_job('first')[nodes[0]['artifact_key']]['state'] == 'QUEUED'


def test_same_monitor_cannot_acquire_a_second_live_lease(context):
    _, store, _, nodes = context
    producer = store.claim(nodes, {})
    assert store.acquire(producer['id'])
    assert store.acquire(producer['id']) is None


def test_expired_same_store_lease_cannot_update_finish_or_release_successor(context):
    db, store, _, nodes = context
    producer = store.claim(nodes, {})
    old = store.acquire(producer['id'])
    with db.transaction() as c:
        c.execute('UPDATE observation_producers SET lease_until=0 WHERE id=?', (producer['id'],))
    new = store.acquire(producer['id'])
    assert old['lease_token'] != new['lease_token']
    assert old['attempt_token'] == new['attempt_token']
    with pytest.raises(ValueError, match='stale'):
        store.update(old, state='RUNNING')
    with pytest.raises(ValueError, match='stale'):
        store.finish(old, error='expired worker')
    store.release(old)
    current = store.update(new, state='RUNNING')
    assert current['lease_token'] == new['lease_token']
    store.finish(current, error='current worker failed')
    assert store.for_job('first')[nodes[0]['artifact_key']]['error'] == 'current worker failed'
