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
        for version in ('018','020'):
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
    # The run list reads that projection, not the stored specification.
    run=db.create_run(variant['id'],seed=1,adapter_name='x',adapter_version='1',run_directory='/run',status='SUCCEEDED')
    assert db.run_progress_evidence([run['id']])[run['id']]['resolved_spec_json']==json.loads(display)


def test_display_projection_matches_the_run_list_expression_on_large_specs(tmp_path):
    db=Database(tmp_path/'projection.db')
    specs=[{'train':{'max_steps':1},'source':{'adapter_manifest':{'train':{'capsule_files':{'big.py':'c'*500_000}}}},
            'data':{'bundle':{'assignments':[{'version':{'metadata':{'episodes':list(range(50_000)),'shared_artifacts':['/a']*1000,'display_name':'d'}}},
                                             {'version':{'metadata':{'episodes':12}}},{'version':{}}]}}},
           {'train':{'max_steps':2}},{}]
    with db.connection() as c:
        for spec in specs:
            projected=c.execute("SELECT skynet_project_document(?, 'variants') -> 'display'",(json.dumps(spec),)).fetchone()[0]
            expected=json.loads(c.execute(OLD_DISPLAY_EXPRESSION,(json.dumps(spec),)).fetchone()[0])
            assert projected==expected
            assert len(json.dumps(projected))<10_000


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
