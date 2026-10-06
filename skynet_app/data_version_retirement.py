"""Explicit migration retirement of obsolete recording payload copies.

This is separate from ordinary dataset deletion: all immutable versions,
revisions, bundles, runs and checkpoints survive. Only verified obsolete
preparation output copies are removed after new experiment heads are rebound.
"""
import hashlib
import json
import re

from .database import canonical_json, utc_now
from .training_contracts import RECORDING_DATASET_FORMAT

TERMINAL = {'SUCCEEDED','COMPLETED','FAILED','CANCELLED','CANCELED','TIMEOUT','SKIPPED','BLOCKED','SUBMISSION_FAILED','PREEMPTED','NODE_FAIL','OUT_OF_MEMORY'}
OLD_RECORDING_FORMATS = {'egoverse-episodes-zarr/v1', 'xpolicylab-act-hdf5/v1',
                         'xpolicylab-demonstrations/v1'}


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def references(value, needles):
    if isinstance(value, str):
        return value in needles or any(value.startswith(path.rstrip('/') + '/') for path in needles if path.startswith('/'))
    if isinstance(value, dict):
        return any(references(item, needles) for item in value.values())
    return isinstance(value, (list, tuple)) and any(references(item, needles) for item in value)


def episode_hashes(metadata):
    result = [item.get('source', {}).get('sha256') or item.get('sha256') for item in metadata.get('episodes', [])]
    if not result or any(not isinstance(sha,str) or not re.fullmatch('[a-f0-9]{64}',sha) for sha in result) or len(result)!=len(set(result)):
        raise ValueError('Migration needs the complete unique source recording checksums')
    return sorted(result)


class DataVersionRetirement:
    pointcloud_upgrade = False

    def __init__(self, database, preparation_service):
        if database.workspace_id is not None:
            raise ValueError('Dataset migration requires the system database to inspect every workspace dependency')
        self.db, self.preparation = database, preparation_service

    def _record(self, connection, old, new):
        rows = connection.execute('SELECT * FROM data_resource_versions WHERE id=ANY(?)', ([old,new],)).fetchall()
        versions = {row['id']:dict(row) for row in rows}
        if set(versions)!={old,new} or old==new:
            raise ValueError('Select distinct registered original and replacement dataset versions')
        before, after = versions[old], versions[new]
        old_meta, new_meta = _json(before['metadata_json']), _json(after['metadata_json'])
        allowed_old = {RECORDING_DATASET_FORMAT} if self.pointcloud_upgrade else OLD_RECORDING_FORMATS
        if before['format'] not in allowed_old or after['format']!=RECORDING_DATASET_FORMAT:
            raise ValueError('Only obsolete recording exports can migrate to the shared recording format')
        if before['resource_id']!=after['resource_id']:
            raise ValueError('Replacement must be an immutable new version of the same dataset resource')
        resource = connection.execute('SELECT * FROM data_resources WHERE id=?',(before['resource_id'],)).fetchone()
        if (resource['provider'],resource['namespace'])!=('collection','datasets'):
            raise ValueError('Only managed collection preparation output can be retired')
        if episode_hashes(old_meta)!=episode_hashes(new_meta):
            raise ValueError('Replacement must preserve every original recording checksum exactly once')
        if old_meta.get('contract')!=new_meta.get('contract'):
            raise ValueError('Replacement policy data contract differs from the old experiment dataset')
        if new_meta.get('validation',{}).get('status')!='PASSED':
            raise ValueError('Replacement data has not passed conversion validation')
        if before['manifest_sha256']==after['manifest_sha256']:
            raise ValueError('Replacement must have a new immutable manifest')
        all_jobs=[_json(row[0]) for row in connection.execute('SELECT payload_json FROM policy_exports').fetchall()]
        jobs=[job for job in all_jobs if job.get('version_id')==old]
        replacements=[job for job in all_jobs if job.get('version_id')==new]
        if (not jobs or old_meta.get('export_id') not in {job['id'] for job in jobs}
                or not replacements or new_meta.get('export_id') not in {job['id'] for job in replacements}):
            raise ValueError('Both dataset versions require their original preparation ownership records')
        for job in jobs:
            if job.get('state') not in {'READY','FAILED','DELETE_FAILED'} or job['id'] in self.preparation.active:
                raise ValueError('Wait for original dataset preparation to stop')
            if old_meta.get('converter_sha256') and old_meta['converter_sha256']!=job.get('converter_sha256'):
                raise ValueError('Original converter identity differs from its preparation record')
            if sorted(item['sha256'] for item in job.get('sources',[]))!=episode_hashes(old_meta):
                raise ValueError('Original source checksums differ from the preparation record')
        for job in replacements:
            if (job.get('state')!='READY' or not job.get('training_ready')
                    or job.get('manifest_sha256')!=after['manifest_sha256']
                    or job.get('loader_validation',{}).get('manifest_sha256')!=after['manifest_sha256']):
                raise ValueError('Replacement must pass its real adapter reader and reach READY')
            if sorted(item['sha256'] for item in job.get('sources',[]))!=episode_hashes(new_meta):
                raise ValueError('Replacement source checksums differ from its preparation record')
        if connection.execute('SELECT 1 FROM data_version_retirements WHERE version_id=?',(new,)).fetchone():
            raise ValueError('Replacement dataset is already being retired')
        locations=[dict(row) for row in connection.execute('SELECT * FROM data_locations WHERE version_id=ANY(?)',([old,new],)).fetchall()]
        new_locations=[row for row in locations if row['version_id']==new and row['status']=='AVAILABLE' and row['manifest_sha256']==after['manifest_sha256']]
        if not any(row['kind']=='cluster' for row in new_locations):
            raise ValueError('Replacement needs a verified available shared cluster location')
        old_locations=[row for row in locations if row['version_id']==old]
        if connection.execute('SELECT 1 FROM data_resource_versions WHERE id<>? AND manifest_sha256=?',(old,before['manifest_sha256'])).fetchone():
            raise ValueError('Original payload is shared with another dataset version')
        if connection.execute('SELECT 1 FROM data_derivation_inputs i JOIN data_derivations d ON d.id=i.derivation_id WHERE i.input_version_id=? AND d.output_version_id<>?',(old,old)).fetchone():
            raise ValueError('Another dataset still derives from the old converted payload')
        return before,after,jobs,old_locations,new_locations

    @staticmethod
    def _needles(connection, version, locations):
        result={version['id'],version['manifest_sha256'],version['path']}
        result.update(row['path'] for row in locations)
        for row in connection.execute('SELECT b.id,b.manifest_sha256 FROM data_bundles b JOIN data_bundle_assignments a ON a.bundle_id=b.id WHERE a.version_id=?',(version['id'],)).fetchall():
            result.update([row['id'],row['manifest_sha256']])
        return result

    def _plan(self, connection, old, new, *, lock=False):
        before,after,jobs,locations,new_locations=self._record(connection,old,new)
        active=[row['id'] for row in connection.execute('SELECT id,status FROM job_attempts').fetchall() if row['status'] not in TERMINAL]
        if active:
            raise ValueError('Wait for all active or queued jobs to stop before retiring dataset payloads')
        old_refs=self._needles(connection,before,locations)
        new_refs=self._needles(connection,after,new_locations)
        # Evaluation targets are independent of experiment training revisions.
        # They cannot be rebound by creating a new training revision, and their
        # frozen stages/attempts may be resumed after completion or failure.
        if self.db._evaluation_target_references(connection, old_refs):
            raise ValueError('An evaluation still references the old dataset; remove that evaluation before retiring its payload')
        revisions=connection.execute('SELECT id,experiment_id,revision_number,requested_spec_json FROM experiment_revisions ORDER BY experiment_id,revision_number DESC').fetchall()
        heads={};affected=set()
        for row in revisions:
            heads.setdefault(row['experiment_id'],row)
            if references(_json(row['requested_spec_json']),old_refs):affected.add(row['experiment_id'])
        variants=connection.execute('SELECT v.resolved_spec_json,r.experiment_id,r.id AS revision_id FROM variants v JOIN experiment_revisions r ON r.id=v.experiment_revision_id').fetchall()
        for row in variants:
            if references(_json(row['resolved_spec_json']),old_refs):
                affected.add(row['experiment_id'])
                if heads[row['experiment_id']]['id']==row['revision_id']:
                    raise ValueError('Latest resolved experiment variants still reference the old dataset')
        head_receipts=[]
        for identifier in sorted(affected):
            head=heads[identifier];spec=_json(head['requested_spec_json'])
            if references(spec,old_refs) or not references(spec,new_refs):
                raise ValueError('Create a new experiment revision bound to the verified replacement first')
            head_receipts.append(dict(experiment_id=identifier,revision_id=head['id'],revision_number=head['revision_number'],spec_sha256=fingerprint(spec)))
        # Freeze only identity and owned paths; mutable status markers do not
        # prevent a safe retry after files were partially removed.
        plan=dict(schema='skynet.data-version-retirement/v1',version_id=old,replacement_version_id=new,
            original_manifest_sha256=before['manifest_sha256'],replacement_manifest_sha256=after['manifest_sha256'],
            source_sha256=episode_hashes(_json(before['metadata_json'])),experiment_heads=head_receipts,
            jobs=[dict(id=job['id'],version_id=old,target=job.get('target'),converter_sha256=job.get('converter_sha256')) for job in sorted(jobs,key=lambda job:job['id'])],
            locations=[{key:row[key] for key in ('id','version_id','kind','host','path','manifest_sha256')} for row in sorted(locations,key=lambda row:row['id'])],
            original_path=before['path'])
        plan['token']=fingerprint(plan)
        return plan,before,jobs,locations

    def preview(self, old_version_id, replacement_version_id):
        with self.db.operation_lock('pipeline'),self.preparation.lock,self.db.read_snapshot() as connection:
            return self._plan(connection,old_version_id,replacement_version_id)[0]

    def _delete_payloads(self, plan, before, jobs, locations):
        self.preparation._delete_dataset_copies(jobs, [before], locations)

    def _finish_payloads(self, connection, plan):
        pass

    def retire(self, old_version_id, replacement_version_id, plan_token):
        with self.db.operation_lock('pipeline'),self.preparation.lock:
            with self.db.transaction() as connection:
                existing=connection.execute('SELECT * FROM data_version_retirements WHERE version_id=?',(old_version_id,)).fetchone()
                if existing:
                    if existing['replacement_version_id']!=replacement_version_id:
                        raise ValueError('Retirement is already bound to another replacement')
                    if existing['state']=='RETIRED':
                        return _json(existing['receipt_json'])
                plan,before,jobs,locations=self._plan(connection,old_version_id,replacement_version_id,lock=True)
                if plan['token']!=plan_token:
                    raise ValueError('Migration dependencies changed; review the retirement plan again')
                now=utc_now()
                connection.execute('INSERT INTO data_version_retirements(version_id,replacement_version_id,state,plan_json,receipt_json,error,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(version_id) DO UPDATE SET state=excluded.state,plan_json=excluded.plan_json,error=NULL,updated_at=excluded.updated_at',
                    (old_version_id,replacement_version_id,'PENDING',canonical_json(plan),'{}',None,now,now))
            error=None
            try:
                # Existing cleanup enforces exact prepared roots and job UUIDs.
                # Neither source/raw versions nor replacement refs are passed.
                self._delete_payloads(plan,before,jobs,locations)
            except Exception as exc:
                error=exc
            receipt=dict(schema=plan['schema'],version_id=old_version_id,replacement_version_id=replacement_version_id,
                         state='FAILED' if error else 'RETIRED',plan_token=plan_token,
                         old_manifest_sha256=plan['original_manifest_sha256'],new_manifest_sha256=plan['replacement_manifest_sha256'],
                         history_preserved=True,original_recordings_preserved=True,completed_at=utc_now())
            if error:receipt['error']=str(error)
            with self.db.transaction() as connection:
                if not error:
                    self._finish_payloads(connection,plan)
                connection.execute('UPDATE data_version_retirements SET state=?,receipt_json=?,error=?,updated_at=? WHERE version_id=?',
                    (receipt['state'],canonical_json(receipt),str(error) if error else None,utc_now(),old_version_id))
                connection.execute("UPDATE data_locations SET status='REMOVED' WHERE version_id=?",(old_version_id,))
                for job in jobs:
                    job.update(retirement=receipt,training_ready=False,updated_at=utc_now())
                    connection.execute('UPDATE policy_exports SET payload_json=? WHERE id=?',(canonical_json(job),job['id']))
            if error:raise ValueError('Dataset retirement needs retry; original history is preserved: '+str(error)) from error
            return receipt
