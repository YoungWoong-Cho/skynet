"""Retired converted datasets stay readable as history but are rejected as inputs."""
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest

from skynet_app.database import Database, canonical_json
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
    db.record_metric(run['id'],name='loss',value=.12,scope='train',step=100)
    db.create_experiment_revision(experiment['id'],{'native':{'config':{'dataset_manifest_sha256':new['manifest_sha256'],'dataset_path':new['path']}}})
    return SimpleNamespace(db=db,old=old,new=new,resource=resource,experiment=experiment,checkpoint=checkpoint,
        run=run,stage=stage,old_job=old_job)


def insert_retirement(migration,state):
    """Record the old version as retired by the new one without touching payload copies."""
    m=migration
    from skynet_app.database import utc_now
    plan={'schema':'skynet.data-version-retirement/v1','version_id':m.old['id'],'replacement_version_id':m.new['id']}
    now=utc_now()
    with m.db.transaction() as c:
        c.execute('INSERT INTO data_version_retirements VALUES (?,?,?,?,?,?,?,?)',
            (m.old['id'],m.new['id'],state,canonical_json(plan),'{}',None,now,now))


def test_ordinary_dataset_deletion_still_blocks_historical_experiment_references(migration):
    m=migration
    with pytest.raises(ValueError,match='experiment'):
        m.db.delete_prepared_dataset(m.resource['id'],lambda *args:pytest.fail('ordinary delete ran cleanup'),identifier=m.old_job)


def test_completed_retirement_receipt_cannot_be_rewritten(migration):
    m=migration;insert_retirement(m,'RETIRED')
    import psycopg
    with pytest.raises(psycopg.Error,match='immutable'):
        with m.db.transaction() as c:
            c.execute("UPDATE data_version_retirements SET state='FAILED' WHERE version_id=?",(m.old['id'],))


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
