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
        migration=Path(__file__).parents[1]/'skynet_app/migrations/postgresql/018_document_projections.sql'
        c.executescript(migration.read_text())
        assert c.execute("SELECT projection_json->'paths' FROM document_projections WHERE record_id='old-record'").fetchone()[0]==['/old/path']
    with pytest.raises(RuntimeError):
        with db.transaction() as c:
            c.execute("UPDATE policy_exports SET payload_json=? WHERE id='old-record'",(json.dumps({'path':'/new/path'}),))
            assert c.execute("SELECT projection_json->'paths' FROM document_projections WHERE record_id='old-record'").fetchone()[0]==['/new/path']
            raise RuntimeError('rollback')
    with db.connection() as c:
        assert c.execute("SELECT projection_json->'paths' FROM document_projections WHERE record_id='old-record'").fetchone()[0]==['/old/path']
