"""Durable ownership and references for immutable, shared observations.

Transactions claim missing artifacts before submission. A producer belongs to
its artifacts, not the Convert that happened to request them first. Recovery
continues the same submission token; only an explicitly failed attempt is retried.
"""
import json
import re
import time
from uuid import uuid4

from .database import canonical_json, utc_now
from .observation_contracts import content_digest


class ObservationStore:
    def __init__(self, database):
        self.database = database

    @staticmethod
    def _artifact(row):
        item = dict(row)
        item['spec'] = json.loads(item.pop('spec_json'))
        return item

    @staticmethod
    def _guard_sources(connection, session_ids):
        for identifier in set(session_ids):
            if connection.execute(
                "SELECT 1 FROM maintenance_operations WHERE target_kind IN ('recording','recording-file') AND plan_json::jsonb->'records'->'live_xr_sessions' @> jsonb_build_array(?::text)",
                (identifier,),
            ).fetchone():
                raise ValueError('This recording is being deleted. Finish its pending deletion first.')

    def attach(self, job_id, nodes, sources):
        by_key = {source['sha256']: source for source in sources}
        with self.database.transaction() as c:
            self._guard_sources(c, [source['session_id'] for source in sources])
            for node in nodes:
                key, spec = node['artifact_key'], node['spec']
                if content_digest(spec) != key:
                    raise ValueError('Observation identity differs from its immutable specification')
                now = utc_now()
                c.execute('''INSERT INTO observation_artifacts
                    (artifact_key,spec_json,state,created_at,updated_at) VALUES (?,?,'MISSING',?,?)
                    ON CONFLICT(artifact_key) DO NOTHING''', (key, canonical_json(spec), now, now))
                existing = c.execute('SELECT spec_json FROM observation_artifacts WHERE artifact_key=?', (key,)).fetchone()
                if json.loads(existing[0]) != spec:
                    raise ValueError('Observation key collision')
                source = by_key[node['source_sha256']]
                c.execute('INSERT INTO observation_sources VALUES (?,?,?) ON CONFLICT DO NOTHING',
                          (key, source['session_id'], source['path']))
                c.execute('INSERT INTO observation_job_inputs VALUES (?,?) ON CONFLICT DO NOTHING', (job_id, key))
                for parent in node['dependencies']:
                    c.execute('INSERT INTO observation_artifact_inputs VALUES (?,?) ON CONFLICT DO NOTHING', (key, parent))

    def for_job(self, job_id):
        with self.database.connection() as c:
            return {r['artifact_key']: self._artifact(r) for r in c.execute('''SELECT a.* FROM observation_artifacts a
                JOIN observation_job_inputs i ON i.artifact_key=a.artifact_key WHERE i.job_id=?''', (job_id,))}

    def progress(self, job_id):
        with self.database.connection() as c:
            return [dict(row) for row in c.execute("""SELECT DISTINCT p.id,p.state,
                p.payload_json::jsonb->>'cluster_job_id' AS cluster_job_id,
                p.payload_json::jsonb->>'reason' AS reason,
                p.payload_json::jsonb->'request'->>'mode' AS mode
                FROM observation_job_inputs i JOIN observation_artifacts a USING(artifact_key)
                JOIN observation_producers p ON p.id=a.producer_id WHERE i.job_id=?""", (job_id,))]

    def claim(self, nodes, request):
        """Atomically claim an eligible batch; competing Converts share the winner."""
        with self.database.transaction() as c:
            claimed = []
            for node in nodes:
                self._guard_sources(c, [r[0] for r in c.execute('SELECT session_id FROM observation_sources WHERE artifact_key=?', (node['artifact_key'],))])
                row = c.execute('SELECT * FROM observation_artifacts WHERE artifact_key=?', (node['artifact_key'],)).fetchone()
                if not row or row['state'] != 'MISSING':
                    continue
                if any(c.execute('SELECT state FROM observation_artifacts WHERE artifact_key=?', (key,)).fetchone()[0] != 'READY'
                       for key in node.get('dependency_keys', node['dependencies'])):
                    continue
                claimed.append(node)
            if not claimed:
                return None
            identifier, token, now = str(uuid4()), str(uuid4()), utc_now()
            request = dict(request, request_id=identifier, attempt_token=token, requests=claimed)
            payload = dict(id=identifier, attempt_token=token, request=request)
            c.execute('''INSERT INTO observation_producers
                (id,attempt_token,state,payload_json,created_at,updated_at) VALUES (?,?,'QUEUED',?,?,?)''',
                (identifier, token, canonical_json(payload), now, now))
            for node in claimed:
                c.execute("UPDATE observation_artifacts SET state='QUEUED',producer_id=?,error=NULL,updated_at=? WHERE artifact_key=?",
                          (identifier, now, node['artifact_key']))
            return dict(payload, state='QUEUED')

    def producers(self):
        with self.database.connection() as c:
            return [dict(json.loads(r['payload_json']), state=r['state']) for r in c.execute(
                "SELECT * FROM observation_producers WHERE state NOT IN ('READY','FAILED') ORDER BY created_at")]

    def acquire(self, identifier, *, now=None, lease_seconds=600):
        now = time.time() if now is None else now
        lease_token = str(uuid4())
        with self.database.transaction() as c:
            row = c.execute('''UPDATE observation_producers SET lease_owner=?,lease_until=?
                WHERE id=? AND lease_until<? AND state NOT IN ('READY','FAILED') RETURNING *''',
                (lease_token, now + lease_seconds, identifier, now)).fetchone()
            return dict(json.loads(row['payload_json']), state=row['state'], lease_token=lease_token) if row else None

    def update(self, producer, *, state=None, **changes):
        with self.database.transaction() as c:
            row = c.execute('SELECT * FROM observation_producers WHERE id=?', (producer['id'],)).fetchone()
            if (not row or row['attempt_token'] != producer['attempt_token'] or not producer.get('lease_token') or row['lease_owner'] != producer['lease_token']
                    or row['lease_until'] < time.time() or row['state'] in {'READY', 'FAILED'}):
                raise ValueError('Observation producer lease changed; stale update rejected')
            payload = json.loads(row['payload_json'])
            payload.update(changes)
            state = state or row['state']
            c.execute('UPDATE observation_producers SET state=?,payload_json=?,updated_at=? WHERE id=?',
                      (state, canonical_json(payload), utc_now(), producer['id']))
            return dict(payload, state=state, lease_token=producer['lease_token'])

    def release(self, producer):
        with self.database.transaction() as c:
            c.execute('UPDATE observation_producers SET lease_until=0,lease_owner=NULL WHERE id=? AND lease_owner=?',
                      (producer['id'], producer.get('lease_token')))

    def finish(self, producer, artifacts=None, error=None):
        artifacts = artifacts or []
        with self.database.transaction() as c:
            row = c.execute('SELECT * FROM observation_producers WHERE id=?', (producer['id'],)).fetchone()
            if (not row or row['attempt_token'] != producer['attempt_token'] or not producer.get('lease_token') or row['lease_owner'] != producer['lease_token']
                    or row['lease_until'] < time.time() or row['state'] in {'READY', 'FAILED'}):
                raise ValueError('Observation producer lease changed; stale publication rejected')
            owned = {r[0] for r in c.execute('SELECT artifact_key FROM observation_artifacts WHERE producer_id=?', (producer['id'],))}
            if not error and ({a['artifact_key'] for a in artifacts} != owned or len(artifacts) != len(owned)):
                raise ValueError('Observation receipt does not contain every claimed artifact exactly once')
            now = utc_now()
            if error:
                c.execute("UPDATE observation_artifacts SET state='FAILED',error=?,updated_at=? WHERE producer_id=?",
                          (str(error), now, producer['id']))
            else:
                for artifact in artifacts:
                    if not re.fullmatch('[a-f0-9]{64}', artifact.get('manifest_sha256', '')):
                        raise ValueError('Observation manifest checksum is missing')
                    c.execute("UPDATE observation_artifacts SET state='READY',path=?,manifest_sha256=?,error=NULL,updated_at=? WHERE artifact_key=? AND producer_id=?",
                              (artifact['path'], artifact['manifest_sha256'], now, artifact['artifact_key'], producer['id']))
            payload = json.loads(row['payload_json'])
            payload['error'] = str(error) if error else None
            c.execute('UPDATE observation_producers SET state=?,payload_json=?,lease_until=0,lease_owner=NULL,updated_at=? WHERE id=?',
                      ('FAILED' if error else 'READY', canonical_json(payload), now, producer['id']))

    def retry(self, job_id):
        with self.database.transaction() as c:
            c.execute("""UPDATE observation_artifacts SET state='MISSING',producer_id=NULL,error=NULL,updated_at=?
                WHERE state='FAILED' AND artifact_key IN (SELECT artifact_key FROM observation_job_inputs WHERE job_id=?)""", (utc_now(), job_id))

    @staticmethod
    def bind_version(connection, job_id, version_id):
        missing = connection.execute('''SELECT 1 FROM observation_job_inputs i JOIN observation_artifacts a USING(artifact_key)
            WHERE i.job_id=? AND a.state<>'READY' LIMIT 1''', (job_id,)).fetchone()
        if missing:
            raise ValueError('Dataset cannot be published before its observations are verified')
        connection.execute('''INSERT INTO observation_version_inputs SELECT ?,artifact_key FROM observation_job_inputs
            WHERE job_id=? ON CONFLICT DO NOTHING''', (version_id, job_id))
