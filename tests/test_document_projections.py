import json
from pathlib import Path

import pytest

from skynet_app.database import Database
from skynet_app.maintenance import Maintenance, _REFERENCE_CACHE
from test_maintenance import history, LocalCluster


def test_source_projection_and_original_receipt_are_both_preserved(tmp_path):
    db=Database(tmp_path/'projection.db')
    project=db.create_project('project')
    spec={'source':{'revision':'abc','repository':'repo','project_subdirectory':'train'},
          'nested':['/shared/data','/shared/data',{'escaped':'/한글/"quoted"'}],
          'data':{'bundle':{'assignments':[{'resource':{'provider':'p','namespace':'n','name':'data'},
             'version':{'manifest_sha256':'digest','metadata':{'registered_version_id':'version'}}}]}},
          'padding':'x'*1_000_000}
    experiment=db.create_experiment(project_id=project['id'],name='test',requested_spec=spec)
    revision=experiment['latest_revision']['id']
    with db.connection() as c:
        p=c.execute("SELECT projection_json FROM document_projections WHERE table_name='experiment_revisions' AND record_id=?",(revision,)).fetchone()[0]
        original=c.execute('SELECT requested_spec_json FROM experiment_revisions WHERE id=?',(revision,)).fetchone()[0]
    assert json.loads(original)==spec
    assert p['source']=={'git_revision':'abc','repository':'repo','project_subdirectory':'train'}
    assert set(p['paths'])=={'/shared/data','/한글/"quoted"'}
    assert p['assignments']==[{'version_id':'version','digest':'digest','provider':'p','namespace':'n','source_key':'data'}]
    assert len(json.dumps(p))<1000
    assert db.list_experiments()[0]['git_revision']=='abc'
    db.create_experiment_revision(experiment['id'],requested_spec={'source':{'revision':'changed'}})
    assert db.list_experiments()[0]['git_revision']=='changed'


def test_projection_is_atomic_for_raw_sql_writes_and_deletes(tmp_path):
    db=Database(tmp_path/'projection.db')
    project=db.create_project('project')
    experiment=db.create_experiment(project_id=project['id'],name='test',requested_spec={})
    revision=experiment['latest_revision']['id']
    with pytest.raises(RuntimeError):
        with db.transaction() as c:
            # Draft immutable records may be deleted, but rollback must restore the projection.
            c.execute('DELETE FROM experiment_revisions WHERE id=?',(revision,))
            assert c.execute('SELECT count(*) FROM document_projections WHERE record_id=?',(revision,)).fetchone()[0]==0
            raise RuntimeError('rollback')
    with db.connection() as c:
        assert c.execute('SELECT count(*) FROM document_projections WHERE record_id=?',(revision,)).fetchone()[0]==1
    with db.transaction() as c:
        c.execute('DELETE FROM experiment_revisions WHERE id=?',(revision,))
        assert c.execute('SELECT count(*) FROM document_projections WHERE record_id=?',(revision,)).fetchone()[0]==0


def test_cold_reference_lookup_uses_projection_and_fails_closed(history, monkeypatch):
    manager, db, system, experiment, run, evaluation, root = history
    payload={'nested':['/protected/a',{'value':'/protected/b'}]}
    with db.transaction() as c:
        c.execute("INSERT INTO policy_exports(id,payload_json) VALUES ('projection-test',?)",(json.dumps(payload),))
    _REFERENCE_CACHE.clear()
    with db.connection() as c:
        original=c.execute
        def guarded(query,*args):
            assert 'jsonb_path_query_array' not in query and 'payload_json::jsonb' not in query
            return original(query,*args)
        monkeypatch.setattr(c,'execute',guarded)
        assert ('/protected/a','policy_exports','projection-test') in manager._references(c)
    with db.transaction() as c:
        c.execute("UPDATE policy_exports SET payload_json=? WHERE id='projection-test'",(json.dumps({'new':'/protected/c'}),))
    with db.connection() as c:
        refs=manager._references(c)
        assert ('/protected/c','policy_exports','projection-test') in refs
        assert ('/protected/a','policy_exports','projection-test') not in refs
    with db.transaction() as c:
        c.execute("DELETE FROM document_projections WHERE table_name='policy_exports' AND record_id='projection-test'")
    _REFERENCE_CACHE.clear()
    with db.connection() as c, pytest.raises(ValueError,match='incomplete'):
        manager._references(c)


def test_migration_backfills_existing_documents_and_rollback_keeps_paths(tmp_path):
    db=Database(tmp_path/'projection.db')
    with db.transaction() as c:
        c.execute('DROP FUNCTION skynet_refresh_document_projection() CASCADE')
        c.execute('DROP FUNCTION skynet_project_document(TEXT,TEXT)')
        c.execute('DROP TABLE document_projections')
        c.execute("INSERT INTO policy_exports(id,payload_json) VALUES ('old-record',?)",(json.dumps({'path':'/old/path'}),))
        for version in ('018','020','021','022','023'):
            c.executescript(next((Path(__file__).parents[1]/'skynet_app/migrations/postgresql').glob(f'{version}_*.sql')).read_text())
        assert c.execute("SELECT projection_json->'paths' FROM document_projections WHERE record_id='old-record'").fetchone()[0]==['/old/path']
    with pytest.raises(RuntimeError):
        with db.transaction() as c:
            c.execute("UPDATE policy_exports SET payload_json=? WHERE id='old-record'",(json.dumps({'path':'/new/path'}),))
            assert c.execute("SELECT projection_json->'paths' FROM document_projections WHERE record_id='old-record'").fetchone()[0]==['/new/path']
            raise RuntimeError('rollback')
    with db.connection() as c:
        assert c.execute("SELECT projection_json->'paths' FROM document_projections WHERE record_id='old-record'").fetchone()[0]==['/old/path']


MIGRATIONS = Path(__file__).parents[1] / 'skynet_app/migrations/postgresql'
OLD_DISPLAY_EXPRESSION = """
    SELECT jsonb_set(spec #- '{source,adapter_manifest,train,capsule_files}', '{data,bundle,assignments}',
        COALESCE((SELECT jsonb_agg(jsonb_set(item, '{version,metadata}',
            (COALESCE(item #> '{version,metadata}', '{}'::jsonb) - 'episodes' - 'shared_artifacts') || jsonb_build_object('episodes',
            CASE WHEN jsonb_typeof(item #> '{version,metadata,episodes}') = 'array'
                 THEN to_jsonb(jsonb_array_length(item #> '{version,metadata,episodes}'))
                 ELSE item #> '{version,metadata,episodes}' END)))
          FROM jsonb_array_elements(COALESCE(spec #> '{data,bundle,assignments}', '[]'::jsonb)) item), '[]'::jsonb))::text
    FROM (SELECT ?::jsonb AS spec) selected
"""


def reapply(c, versions):
    c.execute('DROP FUNCTION skynet_refresh_document_projection() CASCADE')
    c.execute('DROP FUNCTION skynet_project_document(TEXT,TEXT)')
    c.execute('DROP TABLE document_projections')
    for version in versions:
        c.executescript(next(MIGRATIONS.glob(f'{version}_*.sql')).read_text())


def test_020_keeps_earlier_projections_byte_identical_and_extends_variants(tmp_path):
    db=Database(tmp_path/'projection.db')
    project=db.create_project('project')
    spec={'source':{'revision':'abc','repository':'repo','project_subdirectory':'train','adapter_manifest':{'train':{'capsule_files':{'a.py':'x'*5000}}}},
          'tracking':{'providers':[{'provider':'wandb','enabled':True}]},
          'data':{'bundle':{'assignments':[{'resource':{'provider':'p','namespace':'n','name':'data'},
             'version':{'manifest_sha256':'digest','metadata':{'registered_version_id':'version','episodes':[1,2,3],'shared_artifacts':['/s']}}}]}}}
    experiment=db.create_experiment(project_id=project['id'],name='test',requested_spec=spec)
    variant=db.create_variant(experiment['latest_revision']['id'],name='one',parameters={},resolved_spec=spec)
    with db.transaction() as c:
        c.execute("INSERT INTO policy_exports(id,payload_json) VALUES ('export',?)",(json.dumps({'path':'/exported'}),))
        reapply(c, ['018'])
        before={row[0]:row[1] for row in c.execute("SELECT table_name||':'||record_id, projection_json::text FROM document_projections").fetchall()}
        c.executescript(next(MIGRATIONS.glob('020_*.sql')).read_text())
        after={row[0]:row[1] for row in c.execute("SELECT table_name||':'||record_id, projection_json::text FROM document_projections").fetchall()}
        display=c.execute(OLD_DISPLAY_EXPRESSION,(json.dumps(spec),)).fetchone()[0]
    for key in before:
        if key.startswith('variants:'):
            assert json.loads(after[key])=={**json.loads(before[key]),'tracking_providers':spec['tracking']['providers'],'display':json.loads(display)}
        else:
            assert after[key]==before[key], key
    assert json.loads(display)['data']['bundle']['assignments'][0]['version']['metadata']=={'registered_version_id':'version','episodes':3}
    assert 'capsule_files' not in json.dumps(json.loads(display)['source'])


LIST_SPEC = {'train':{'max_steps':8000,'other':1},'resources':{'gpu':{'count':2,'type':'any'},'partition':'p','account':'a'},
    'native':{'config':{'epochs':3,'datasets':[{'path':'/d'}]*200,'initial_checkpoint':'/saved.ckpt'}},
    'tracking':{'providers':[{'provider':'wandb','enabled':True}]},
    'source':{'revision':'abc','adapter_manifest':{'slug':'hat','train':{'capsule_files':{'big.py':'c'*500_000},'argv':['x']*500,'input_fields':[{'name':'f'}]*200,
        'progress':{'unit':'step','total_path':'train.max_steps','starts_at_zero':True,'source':{'kind':'jsonl','path':'progress.jsonl','completed_key':'step'}}}}},
    'data':{'bundle':{'name':'cube','assignments':[
        {'role':'training_data','resource':{'id':'r1','name':'Cube','provider':'p'},'version':{'format':'f','manifest_sha256':'d'*64,
         'metadata':{'display_name':'Cube','registered_version_id':'v1','episodes':list(range(50_000)),'shared_artifacts':['/a']*1000,'capture':{'x':'y'*5000}}}},
        {'role':'training_data','resource':{'id':'r2'},'version':{'metadata':{'num_episodes':None,'registered_version_id':'v2'}}},
        {'role':'training_data','resource':{'id':'r3'},'version':{'metadata':{'num_episodes':7,'episodes':[1,2]}}},
        {'role':'evaluation_target','version':{'metadata':{'episodes':[1]}}}]}}}


def list_views(spec):
    """Every value the run list derives from a spec; equal for the stored spec and its projection."""
    from skynet_app import data_selection
    from skynet_app.pipeline_api import _mapping_path, _resolved_training_total, _training_progress_contract, training_progress_summary
    run={'status':'RUNNING','resolved_spec_json':spec}
    attempts=[{'id':'a1','attempt_number':1,'status':'RUNNING','started_at':'2026-09-20T10:00:00Z'}]
    return dict(
        resources=spec.get('resources') or {},
        training_data=data_selection.describe(spec),
        contract=_training_progress_contract(run),
        totals={path:_resolved_training_total(spec,path) for path in ('train.max_steps','native.config.epochs')},
        pinned=bool(_mapping_path(spec,'native.config.initial_checkpoint')[1]),
        summary=training_progress_summary(run,attempts=attempts,checkpoints=[],metrics=[],
            progress_samples=[{'attempt_id':'a1','restart_count':0,'completed':100,'unit':'step','recorded_at':'2026-09-20T10:05:00Z'}],
            resolved_spec=spec,now='2026-09-20T10:10:00Z'))


def test_021_narrows_the_run_list_spec_and_keeps_other_projections_byte_identical(tmp_path):
    db=Database(tmp_path/'projection.db')
    project=db.create_project('project')
    experiment=db.create_experiment(project_id=project['id'],name='test',requested_spec=LIST_SPEC)
    variant=db.create_variant(experiment['latest_revision']['id'],name='one',parameters={},resolved_spec=LIST_SPEC)
    with db.transaction() as c:
        c.execute("INSERT INTO policy_exports(id,payload_json) VALUES ('export',?)",(json.dumps({'path':'/exported'}),))
        reapply(c, ['018','020'])
        before={row[0]:row[1] for row in c.execute("SELECT table_name||':'||record_id, projection_json::text FROM document_projections").fetchall()}
        c.executescript(next(MIGRATIONS.glob('021_*.sql')).read_text())
        after={row[0]:row[1] for row in c.execute("SELECT table_name||':'||record_id, projection_json::text FROM document_projections").fetchall()}
    for key in before:
        if key.startswith('variants:'):
            wide,narrow=json.loads(before[key]),json.loads(after[key])
            assert {k:v for k,v in narrow.items() if k!='display'}=={k:v for k,v in wide.items() if k!='display'}
            display=narrow['display']
        else:
            assert after[key]==before[key], key
    text=json.dumps(display)
    assert len(text)<3000 and not any(word in text for word in ('capsule_files','datasets','capture','shared_artifacts','argv'))
    assert list_views(display)==list_views(LIST_SPEC)
    metadata=[item['version']['metadata'] for item in display['data']['bundle']['assignments']]
    assert metadata[0]=={'display_name':'Cube','registered_version_id':'v1','episodes':50_000}
    assert metadata[1]=={'registered_version_id':'v2','num_episodes':None}, 'an explicit null num_episodes keeps its presence'
    assert metadata[2]=={'episodes':2,'num_episodes':7}
    # The run list reads that projection, not the stored specification.
    run=db.create_run(variant['id'],seed=1,adapter_name='x',adapter_version='1',run_directory='/run',status='SUCCEEDED')
    assert db.run_progress_evidence([run['id']])[run['id']]['resolved_spec_json']==display


def test_display_projection_keeps_every_run_list_view_on_large_and_empty_specs(tmp_path):
    db=Database(tmp_path/'projection.db')
    with db.connection() as c:
        for spec in (LIST_SPEC,{'train':{'max_steps':2}},{}):
            projected=c.execute("SELECT skynet_project_document(?, 'variants') -> 'display'",(json.dumps(spec),)).fetchone()[0]
            assert list_views(projected)==list_views(spec)
            assert len(json.dumps(projected))<3000


def test_offloaded_bodies_are_skipped_by_the_trigger_and_projected_by_their_writer(tmp_path):
    db=Database(tmp_path/'projection.db')
    experiment=db.create_experiment(name='stage',requested_spec={})
    variant=db.create_variant(experiment['latest_revision']['id'],name='run',parameters={},resolved_spec={})
    run=db.create_run(variant['id'],seed=1,adapter_name='x',adapter_version='1',run_directory='/run',status='COMPLETED')
    target={'version_id':'v','manifest_sha256':'d'*64,'path':'/target'}
    marker=json.dumps({'$skynet_object_v1':{'sha256':'0'*64,'path':'/objects/0','size':1}})
    with db.transaction() as c:
        c.execute("INSERT INTO workflow_stages(id,run_id,stage_type,name,status,resolved_config_json,created_at,updated_at) VALUES ('inline',?,'EVALUATE','e','COMPLETED',?,?,?)",
                  (run['id'],json.dumps({'context':{'target_dataset':target}}),'2026-01-01T00:00:00Z','2026-01-01T00:00:00Z'))
        c.execute("INSERT INTO workflow_stages(id,run_id,stage_type,name,status,resolved_config_json,created_at,updated_at) VALUES ('offloaded',?,'EVALUATE','f','COMPLETED',?,?,?)",
                  (run['id'],marker,'2026-01-01T00:00:00Z','2026-01-01T00:00:00Z'))
        rows={row[0]:row[1] for row in c.execute("SELECT record_id, projection_json FROM document_projections WHERE table_name='workflow_stages'").fetchall()}
        assert rows=={'inline':{'assignments':[{'role':'evaluation_target','version_id':'v','digest':'d'*64,'path':'/target'}]}}
        # A later marker write keeps whatever the writer projected; a delete removes it.
        c.execute("UPDATE workflow_stages SET resolved_config_json=? WHERE id='inline'",(marker,))
        assert c.execute("SELECT count(*) FROM document_projections WHERE record_id='inline'").fetchone()[0]==1
        c.execute("DELETE FROM workflow_stages WHERE id='inline'")
        assert c.execute("SELECT count(*) FROM document_projections WHERE record_id='inline'").fetchone()[0]==0
    with db.connection() as c, pytest.raises(ValueError,match='incomplete'):
        db._evaluation_target_references(c,['v'])


def test_022_projects_attempt_resume_pins_and_keeps_other_kinds_byte_identical(tmp_path):
    db=Database(tmp_path/'projection.db')
    experiment=db.create_experiment(name='pins',requested_spec=LIST_SPEC)
    variant=db.create_variant(experiment['latest_revision']['id'],name='run',parameters={},resolved_spec=LIST_SPEC)
    run=db.create_run(variant['id'],seed=1,adapter_name='x',adapter_version='1',run_directory='/run',status='RUNNING')
    stage=db.create_stage(run['id'],stage_type='TRAIN',name='train')
    snapshots={'pinned':{'plan':{'native_config':{'initial_checkpoint':'/saved.ckpt'}}},
               'provenance':{'migration_provenance':{'checkpoint':{'path':'/old.ckpt'}}},
               'spec':{'resolved_spec':{'native':{'config':{'initial_checkpoint':'/x'}}}},
               'empty':{'plan':{'native_config':{'initial_checkpoint':''}}},
               'null':{'plan':{'native_config':{'initial_checkpoint':None}}},
               'false':{'migration_provenance':{'checkpoint':False}},
               'missing':{'plan':{}}}
    attempts={name:db.create_job_attempt(stage['id'],status='FAILED',execution_snapshot_json=snapshot) for name,snapshot in snapshots.items()}
    with db.transaction() as c:
        c.execute("INSERT INTO policy_exports(id,payload_json) VALUES ('export',?)",(json.dumps({'path':'/exported'}),))
        reapply(c, ['018','020','021'])
        before={row[0]:row[1] for row in c.execute("SELECT table_name||':'||record_id, projection_json::text FROM document_projections").fetchall()}
        assert not any(key.startswith('job_attempts:') for key in before)
        c.executescript(next(MIGRATIONS.glob('022_*.sql')).read_text())
        after={row[0]:row[1] for row in c.execute("SELECT table_name||':'||record_id, projection_json::text FROM document_projections").fetchall()}
    for key,value in before.items():
        assert after[key]==value, key
    flags={name:json.loads(after['job_attempts:'+attempt['id']])['has_initial_checkpoint'] for name,attempt in attempts.items()}
    assert flags=={'pinned':True,'provenance':True,'spec':True,'empty':False,'null':False,'false':False,'missing':False}
    # An attempt without a snapshot has nothing to pin and needs no projection.
    attempts['bare']=db.create_job_attempt(stage['id'],status='PENDING'); flags['bare']=False
    evidence=db.run_progress_evidence([run['id']])[run['id']]
    assert {row['id']:row['has_initial_checkpoint'] for row in evidence['attempts']}=={attempts[name]['id']:flag for name,flag in flags.items()}
    # A new attempt projects through the trigger; a missing projection fails the list closed.
    later=db.create_job_attempt(stage['id'],status='RUNNING',execution_snapshot_json={'plan':{'native_config':{'initial_checkpoint':'/again'}}})
    assert db.run_progress_evidence([run['id']])[run['id']]['attempts'][-1]['has_initial_checkpoint'] is True
    with db.transaction() as c:
        c.execute("DELETE FROM document_projections WHERE table_name='job_attempts' AND record_id=?",(later['id'],))
    with pytest.raises(ValueError,match='incomplete'):
        db.run_progress_evidence([run['id']])
    assert db.repair_document_projections()==1
    assert db.run_progress_evidence([run['id']])[run['id']]['attempts'][-1]['has_initial_checkpoint'] is True


ADAPTER_MANIFEST = {
    'schema_version': 'skynet.adapter/v1', 'slug': 'compact', 'display_name': 'Compact', 'aliases': ['cmp'],
    'runtime': {'allowed_backends': ['uv']}, 'capabilities': {'version': 1},
    'defaults': {'resources': {'gpu_type': 'any'}},
    'train': {'capsule_files': {'train.py': 'print(1)' * 50}, 'model_io': {'inputs': ['rgb']},
              'input_fields': [{'path': 'native.config.epochs', 'label': 'Epochs'}], 'supported_canonical_fields': ['epochs']},
    'evaluations': [{'environment': 'sim', 'suites': ['s1'], 'command': {'capsule_files': {'eval.py': 'x' * 100}, 'model_io': {}, 'argv': ['python']}}, 'not-an-object'],
}


def compact(manifest):
    train = {k: v for k, v in manifest['train'].items() if k not in ('capsule_files', 'model_io')}
    evaluations = [({**e, 'command': {k: v for k, v in e['command'].items() if k not in ('capsule_files', 'model_io')}}
                    if isinstance(e, dict) else e) for e in manifest['evaluations']]
    return {**manifest, 'train': train, 'evaluations': evaluations}


def test_023_projects_compact_adapter_manifests_and_keeps_other_kinds_byte_identical(tmp_path):
    db=Database(tmp_path/'projection.db')
    experiment=db.create_experiment(name='pins',requested_spec=LIST_SPEC)
    db.create_variant(experiment['latest_revision']['id'],name='run',parameters={},resolved_spec=LIST_SPEC)
    adapter=db.create_adapter(name='Compact',manifest=ADAPTER_MANIFEST)
    scalar=db.create_adapter(name='Scalar',manifest={'slug':'scalar','train':3,'evaluations':'none'})
    with db.transaction() as c:
        reapply(c, ['018','020','021','022'])
        before={row[0]:row[1] for row in c.execute("SELECT table_name||':'||record_id, projection_json::text FROM document_projections").fetchall()}
        assert not any(key.startswith('adapters:') for key in before)
        c.executescript(next(MIGRATIONS.glob('023_*.sql')).read_text())
        after={row[0]:row[1] for row in c.execute("SELECT table_name||':'||record_id, projection_json::text FROM document_projections").fetchall()}
    for key,value in before.items():
        assert after[key]==value, key
    version=adapter['latest_version']['id']
    assert json.loads(after['adapters:'+version])=={'manifest':compact(ADAPTER_MANIFEST)}
    # Non-object subtrees are left alone rather than failing the projection.
    assert json.loads(after['adapters:'+scalar['latest_version']['id']])=={'manifest':{'slug':'scalar','train':3,'evaluations':'none'}}
    # The list serves the projection; the detail keeps the full manifest.
    listed=next(item for item in db.list_adapter_registry(manifests='projected') if item['id']==adapter['id'])
    assert listed['latest_version']['manifest']==compact(ADAPTER_MANIFEST)
    assert 'capsule_files' not in listed['latest_version']['manifest']['train']
    assert db.get_adapter(adapter['id'])['latest_version']['manifest']==ADAPTER_MANIFEST
    assert next(item for item in db.list_adapter_registry(manifests='none') if item['id']==adapter['id'])['latest_version']['manifest']=={}
