"""Retiring old generated payloads preserves all experiment and raw history."""
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest

from skynet_app.database import Database, canonical_json
from skynet_app.data_version_retirement import DataVersionRetirement
from skynet_app.training_contracts import RECORDING_DATASET_FORMAT


@pytest.fixture
def migration(tmp_path):
    db=Database(tmp_path/'retirement.store')
    resource=db.create_data_resource(provider='collection',namespace='datasets',source_key='session',
        category='dataset',kind='demonstrations',display_name='Recordings',metadata={'managed_dataset':True})
    old_job,new_job=str(uuid4()),str(uuid4())
    source_sha='a'*64
    metadata=dict(contract='skynet.egoverse-rgb-joints/v1',episodes=[{'index':0,'steps':3,'sha256':source_sha}],
                  validation={'status':'PASSED'},converter_sha256='c'*64,export_id=old_job)
    old=db.create_data_resource_version(resource['id'],revision='old',format='egoverse-episodes-zarr/v1',
        path=str(tmp_path/'old-output'),manifest_sha256='1'*64,metadata=metadata)
    metadata=deepcopy(metadata);metadata.update(export_id=new_job,format=RECORDING_DATASET_FORMAT,
        episodes=[{'index':0,'steps':3,'id':'recording-0','source':{'sha256':source_sha}}])
    new=db.create_data_resource_version(resource['id'],revision='new',format=RECORDING_DATASET_FORMAT,
        path=str(tmp_path/'new-manifest'),manifest_sha256='2'*64,metadata=metadata)
    for version in (old,new):
        db.record_data_location(version['id'],kind='cluster',host='skynet',path=version['path'],manifest_sha256=version['manifest_sha256'])
    for identifier,version in [(old_job,old),(new_job,new)]:
        job=dict(id=identifier,version_id=version['id'],resource_id=resource['id'],state='READY',target='cluster',
            sources=[{'sha256':source_sha}],converter_sha256='c'*64,training_ready=True,
            manifest_sha256=version['manifest_sha256'],loader_validation={'manifest_sha256':version['manifest_sha256']})
        with db.transaction() as c:c.execute('INSERT INTO policy_exports VALUES (?,?)',(identifier,canonical_json(job)))
    experiment=db.create_experiment(name='Prior experiment',requested_spec={'native':{'config':{'dataset_manifest_sha256':old['manifest_sha256'],'dataset_path':old['path']}}})
    revision=experiment['latest_revision']
    variant=db.create_variant(revision['id'],name='variant',parameters={},resolved_spec={'version_id':old['id']})
    run=db.create_run(variant['id'],seed=42,adapter_name='EgoVerse',adapter_version='1',run_directory=str(tmp_path/'run'),status='COMPLETED')
    stage=db.create_stage(run['id'],stage_type='TRAIN',name='train',status='COMPLETED')
    attempt=db.create_job_attempt(stage['id'],status='COMPLETED')
    checkpoint=db.create_checkpoint(run['id'],checkpoint_type='last',path=str(tmp_path/'model.ckpt'),produced_by_attempt_id=attempt['id'])
    metric=db.record_metric(run['id'],name='loss',value=.12,scope='train',step=100)
    rebound=db.create_experiment_revision(experiment['id'],{'native':{'config':{'dataset_manifest_sha256':new['manifest_sha256'],'dataset_path':new['path']}}})
    calls=[]
    service=SimpleNamespace(active=set(),lock=db.operation_lock('dataset-preparation'),
        _delete_dataset_copies=lambda jobs,versions,locations:calls.append((jobs,versions,locations)))
    retirement=DataVersionRetirement(db,service)
    return SimpleNamespace(db=db,old=old,new=new,resource=resource,retirement=retirement,service=service,
        calls=calls,experiment=experiment,checkpoint=checkpoint,metric=metric,run=run,stage=stage,
        old_revision=revision,new_revision=rebound,old_job=old_job,new_job=new_job)


def test_retirement_keeps_versions_revisions_runs_checkpoints_and_metrics(migration):
    m=migration
    with m.db.connection() as c:
        before={table:[dict(row) for row in c.execute('SELECT * FROM '+table).fetchall()]
                for table in ('data_resource_versions','experiment_revisions','variants','runs','checkpoints','metrics')}
    plan=m.retirement.preview(m.old['id'],m.new['id'])
    result=m.retirement.retire(m.old['id'],m.new['id'],plan['token'])
    assert result['state']=='RETIRED' and result['original_recordings_preserved']
    assert len(m.calls)==1
    jobs,versions,locations=m.calls[0]
    assert [row['id'] for row in versions]==[m.old['id']]
    assert [row['id'] for row in jobs]==[m.old_job]
    assert all(row['version_id']==m.old['id'] for row in locations)
    with m.db.connection() as c:
        for table,expected in before.items():
            assert [dict(row) for row in c.execute('SELECT * FROM '+table).fetchall()]==expected
        assert c.execute('SELECT status FROM data_locations WHERE version_id=?',(m.old['id'],)).fetchone()[0]=='REMOVED'
        assert c.execute('SELECT status FROM data_locations WHERE version_id=?',(m.new['id'],)).fetchone()[0]=='AVAILABLE'
        assert c.execute('SELECT state FROM data_version_retirements WHERE version_id=?',(m.old['id'],)).fetchone()[0]=='RETIRED'
    assert m.retirement.retire(m.old['id'],m.new['id'],plan['token'])==result
    assert len(m.calls)==1


def test_partial_cleanup_failure_blocks_old_copies_and_retries_same_plan(migration):
    m=migration;plan=m.retirement.preview(m.old['id'],m.new['id'])
    def fail(*args):raise OSError('simulated storage outage')
    m.service._delete_dataset_copies=fail
    with pytest.raises(ValueError,match='needs retry'):
        m.retirement.retire(m.old['id'],m.new['id'],plan['token'])
    with m.db.connection() as c:
        row=c.execute('SELECT * FROM data_version_retirements WHERE version_id=?',(m.old['id'],)).fetchone()
        assert row['state']=='FAILED' and 'storage outage' in row['error']
        assert c.execute('SELECT status FROM data_locations WHERE version_id=?',(m.old['id'],)).fetchone()[0]=='REMOVED'
    retry=m.retirement.preview(m.old['id'],m.new['id'])
    assert retry['token']==plan['token']
    m.service._delete_dataset_copies=lambda *args:m.calls.append(args)
    assert m.retirement.retire(m.old['id'],m.new['id'],retry['token'])['state']=='RETIRED'


def test_latest_head_must_be_rebound_and_plan_invalidates_after_new_revision(migration):
    m=migration;plan=m.retirement.preview(m.old['id'],m.new['id'])
    m.db.create_experiment_revision(m.experiment['id'],{'version_id':m.new['id'],'changed':True})
    with pytest.raises(ValueError,match='changed'):
        m.retirement.retire(m.old['id'],m.new['id'],plan['token'])
    m.db.create_experiment_revision(m.experiment['id'],{'version_id':m.old['id']})
    with pytest.raises(ValueError,match='new experiment revision'):
        m.retirement.preview(m.old['id'],m.new['id'])
    assert not m.calls


def test_active_and_queued_jobs_block_retirement(migration):
    m=migration
    m.db.create_job_attempt(m.stage['id'],status='PENDING')
    with pytest.raises(ValueError,match='active or queued'):
        m.retirement.preview(m.old['id'],m.new['id'])
    assert not m.calls


def test_replacement_reader_receipt_is_required(migration):
    m=migration
    import json
    with m.db.transaction() as c:
        job=json.loads(c.execute('SELECT payload_json FROM policy_exports WHERE id=?',(m.new_job,)).fetchone()[0])
        job['loader_validation']={}
        c.execute('UPDATE policy_exports SET payload_json=? WHERE id=?',(canonical_json(job),m.new_job))
    with pytest.raises(ValueError,match='real adapter reader'):
        m.retirement.preview(m.old['id'],m.new['id'])


def test_ordinary_dataset_deletion_still_blocks_historical_experiment_references(migration):
    m=migration
    with pytest.raises(ValueError,match='experiment'):
        m.db.delete_prepared_dataset(m.resource['id'],lambda *args:pytest.fail('ordinary delete ran cleanup'),identifier=m.old_job)


def test_replacement_cannot_drop_or_change_original_recordings(migration):
    m=migration
    metadata=deepcopy(m.new['metadata'])
    metadata['episodes'][0]['source']['sha256']='b'*64
    bad=m.db.create_data_resource_version(m.resource['id'],revision='different-recording',format=RECORDING_DATASET_FORMAT,
        path=m.new['path']+'-different',manifest_sha256='3'*64,metadata=metadata)
    with pytest.raises(ValueError,match='every original recording checksum'):
        m.retirement.preview(m.old['id'],bad['id'])
    assert not m.calls


def test_completed_retirement_receipt_cannot_be_rewritten(migration):
    m=migration;plan=m.retirement.preview(m.old['id'],m.new['id'])
    m.retirement.retire(m.old['id'],m.new['id'],plan['token'])
    import psycopg
    with pytest.raises(psycopg.Error,match='immutable'):
        with m.db.transaction() as c:
            c.execute("UPDATE data_version_retirements SET state='FAILED' WHERE version_id=?",(m.old['id'],))


def insert_retirement(migration,state):
    m=migration
    from skynet_app.database import utc_now
    plan=m.retirement.preview(m.old['id'],m.new['id'])
    now=utc_now()
    with m.db.transaction() as c:
        c.execute('INSERT INTO data_version_retirements VALUES (?,?,?,?,?,?,?,?)',
            (m.old['id'],m.new['id'],state,canonical_json(plan),'{}',None,now,now))


@pytest.mark.parametrize('state',['PENDING','FAILED','RETIRED'])
def test_retirement_states_hide_active_choice_but_preserve_historical_lookup(migration,state):
    from skynet_app.data_selection import snapshot,assert_available
    m=migration;insert_retirement(m,state)
    historical=m.db.get_data_resource_version(m.old['id'])
    assert historical['retirement']['state']==state
    assert historical['id']==m.old['id'] and historical['metadata']['episodes']
    active=m.db.get_data_resource(m.resource['id'])
    assert [row['id'] for row in active['versions']]==[m.new['id']]
    assert active['version_count']==1
    assert m.db.get_run(m.run['id'])['resolved_spec_json']['version_id']==m.old['id']
    with pytest.raises(ValueError,match='retired'):
        snapshot(m.db,[{'version_id':m.old['id']}])
    for value in (m.old['id'],m.old['manifest_sha256'],m.old['path']+'/episode.hdf5'):
        with pytest.raises(ValueError,match='retired'):
            assert_available(m.db,{'nested':[{'old':value}]})
    assert_available(m.db,{'resource':m.resource['id'],'version_id':m.new['id'],'path':m.old['path']+'-different'})


@pytest.mark.parametrize('state',['PENDING','FAILED','RETIRED'])
def test_old_run_submission_resume_and_rerun_fail_before_any_mutation(migration,state,monkeypatch):
    from skynet_app.pipeline_api import PipelineService, EvaluationRequest
    m=migration;insert_retirement(m,state)
    service=object.__new__(PipelineService)
    service.database=m.db
    service._reconcile_lock=m.db.operation_lock('pipeline')
    service.run_manual_actions=lambda _:pytest.fail('retired run should reject before manual action processing')
    before=m.db.get_run(m.run['id'])
    for call in (
        lambda:service._submit_stage(m.run['id'],m.stage['id'],'auto'),
        lambda:service.retry_run(m.run['id'],'resume','auto'),
        lambda:service.rerun_run(m.run['id'],'auto'),
    ):
        with pytest.raises(ValueError,match='retired'):
            call()
    assert m.db.get_run(m.run['id'])==before
    service._resolve_evaluation_target=lambda *_:({'run_valid':True,'checkpoint_valid':True},before,m.checkpoint)
    service._resolve_evaluation_suite_selection=lambda *_:pytest.fail('evaluation resolved before retirement guard')
    with pytest.raises(ValueError,match='retired'):
        service.create_evaluation(EvaluationRequest(run_id=m.run['id'],suite_id='unused'))
    assert m.db.get_run(m.run['id'])==before
    # New submissions reject even before the orphan-repair status mutation.
    m.db.create_experiment_revision(m.experiment['id'],{'version_id':m.old['id']})
    monkeypatch.setattr(m.db,'repair_pre_submission_orphans',lambda *_:pytest.fail('repair ran before retirement guard'))
    with pytest.raises(ValueError,match='retired'):
        service.submit_experiment(m.experiment['id'])


def test_retirement_checks_use_constant_count_queries_for_many_versions(migration,monkeypatch):
    from contextlib import contextmanager
    from skynet_app.data_selection import assert_available
    m=migration;insert_retirement(m,'PENDING')
    # Metadata expansion stays batched, including unretired rows with no receipt.
    with m.db.connection() as connection:
        rows=connection.execute('SELECT * FROM data_resource_versions').fetchall()
        queries=[]
        class Counted:
            def execute(self,sql,*args):
                queries.append(sql)
                return connection.execute(sql,*args)
        values=m.db._data_version_payloads(Counted(),rows*5,include_resource=True)
        assert len(values)==len(rows)*5 and len(queries)==3
        assert sum('data_version_retirements' in sql for sql in queries)==1
    original=m.db.connection;queries=[]
    @contextmanager
    def counted_connection():
        with original() as connection:
            class Counted:
                def execute(self,sql,*args):
                    queries.append(sql)
                    return connection.execute(sql,*args)
            yield Counted()
    monkeypatch.setattr(m.db,'connection',counted_connection)
    assert_available(m.db,{'version_id':m.new['id']})
    assert len(queries)==3
    assert 'metadata_json' not in queries[0] and 'v.*' not in queries[0]
