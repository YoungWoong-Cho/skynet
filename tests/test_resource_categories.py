from pathlib import Path

import pytest
from pydantic import ValidationError

from skynet_app.database import Database, canonical_json
from skynet_app.data_resource_policy import RESOURCE_TYPES
from skynet_app.db_backend import INTEGRITY_ERRORS
from skynet_app.pipeline_api import DataResourceCreateRequest


def test_allowed_types_match_request_and_database_constraints(tmp_path):
    db = Database(tmp_path / 'catalog')
    for category, types in RESOURCE_TYPES.items():
        for kind in types:
            payload = dict(category=category, kind=kind, provider='test', namespace='catalog', name=kind)
            request = DataResourceCreateRequest(**payload)
            result = db.create_data_resource(**request.model_dump())
            assert result['category'] == category
    for category, kind in [('dataset', 'model'), ('file', 'demonstrations'), ('dataset', 'qa_fixture'), ('dataset', 'typo'), ('other', 'dataset')]:
        payload = dict(category=category, kind=kind, provider='test', namespace='invalid', name=kind)
        with pytest.raises(ValidationError):
            DataResourceCreateRequest(**payload)
        with pytest.raises(ValueError):
            db.create_data_resource(**payload)
        with pytest.raises(INTEGRITY_ERRORS), db.transaction() as c:
            c.execute('INSERT INTO data_resources (id, provider, namespace, name, category, kind, metadata_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)',
                      (f'{category}-{kind}', 'test', 'invalid', kind, category, kind, '{}', '2026-01-01', '2026-01-01'))
    with pytest.raises(ValidationError):
        DataResourceCreateRequest(provider='test', namespace='missing', name='missing', kind='dataset')


def test_file_versions_cannot_hide_recording_sources(tmp_path):
    db = Database(tmp_path / 'catalog')
    resource = db.create_data_resource(category='file', kind='model', provider='test', namespace='files', name='model')
    metadata = {'sources': [{'session_id': 'recording-a'}]}
    with pytest.raises(ValueError, match='Files cannot be linked'):
        db.create_data_resource_version(resource['id'], revision='v1', format='pt', path='/cluster/model', manifest_sha256='a'*64, metadata=metadata)
    with pytest.raises(INTEGRITY_ERRORS), db.transaction() as c:
        c.execute('INSERT INTO data_resource_versions (id,resource_id,revision,format,path,manifest_sha256,status,metadata_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)',
                  ('v1',resource['id'],'v1','pt','/cluster/model','a'*64,'READY',canonical_json(metadata),'2026-01-01'))
    with pytest.raises(INTEGRITY_ERRORS), db.transaction() as c:
        c.execute("UPDATE data_resources SET category='dataset',kind='dataset' WHERE id=?", (resource['id'],))


def test_publishing_and_deleting_result_updates_parent(tmp_path):
    db = Database(tmp_path / 'catalog')
    resource = db.create_data_resource(category='dataset', kind='demonstrations', provider='collection', namespace='datasets', name='cube')
    with db.transaction() as c:
        c.execute("UPDATE data_resources SET updated_at='2026-01-01T00:00:00.000Z' WHERE id=?", (resource['id'],))
    version = db.create_data_resource_version(resource['id'], revision='v1', format='zarr', path='/cluster/cube', manifest_sha256='a'*64)
    assert db.get_data_resource(resource['id'])['updated_at'] > '2026-01-01'
    with db.transaction() as c:
        c.execute("UPDATE data_resources SET updated_at='2026-01-01T00:00:00.000Z' WHERE id=?", (resource['id'],))
        c.execute("SELECT set_config('skynet.allow_dataset_delete', 'on', true)")
        c.execute('DELETE FROM data_resource_versions WHERE id=?', (version['id'],))
    updated = db.get_data_resource(resource['id'])
    assert updated['updated_at'] > '2026-01-01'
    assert updated['versions'] == []


def test_legacy_classification_migration_preserves_payload_and_provenance(tmp_path):
    db = Database(tmp_path / 'legacy')
    migration = Path(__file__).resolve().parents[1] / 'skynet_app/migrations/postgresql/008_resource_categories.sql'
    with db.transaction() as c:
        c.executescript('''CREATE SCHEMA legacy_catalog;
            SET LOCAL search_path TO legacy_catalog;
            CREATE TABLE data_resources (id text PRIMARY KEY, kind text, metadata_json text, updated_at text, CONSTRAINT file_resources_no_recording_link CHECK (true));
            CREATE TABLE data_resource_versions (id text, resource_id text, metadata_json text, path text, manifest_sha256 text, created_at text);
        ''')
        for id, kind, metadata in [('qa', 'qa_fixture', {'license':'test'}), ('file','simulation-assets',{'license':'upstream'}), ('data','demonstrations', {'session_id':'source'})]:
            c.execute('INSERT INTO data_resources VALUES (?,?,?,?)', (id,kind,canonical_json(metadata),'2026-01-01'))
        c.execute('INSERT INTO data_resource_versions VALUES (?,?,?,?,?,?)', ('result','data','{"sources":[{"session_id":"source"}]}','/cluster/keep','a'*64,'2026-09-12'))
        c.executescript(migration.read_text())
        rows = {row['id']:dict(row) for row in c.execute('SELECT * FROM data_resources')}
        assert rows['qa']['category'] == 'dataset' and rows['qa']['kind'] == 'dataset'
        assert '"test_fixture": true' in rows['qa']['metadata_json'] and '"license": "test"' in rows['qa']['metadata_json']
        assert rows['file']['category'] == 'file' and rows['file']['kind'] == 'simulation_assets'
        assert rows['data']['metadata_json'] == canonical_json({'session_id':'source'})
        result = dict(c.execute('SELECT * FROM data_resource_versions').fetchone())
        assert result['path'] == '/cluster/keep' and result['manifest_sha256'] == 'a'*64
        assert rows['data']['updated_at'] == '2026-09-12'


def test_file_is_not_a_training_dataset_but_remains_an_asset_input(tmp_path):
    from skynet_app import data_selection
    db = Database(tmp_path / 'choices')
    resource = db.create_data_resource(category='file', kind='simulation_assets', provider='test', namespace='files', name='scene')
    version = db.create_data_resource_version(resource['id'], revision='v1', format='usd', path='/cluster/scene', manifest_sha256='a'*64)
    assert data_selection.choices(db) == []
    with pytest.raises(ValueError, match='Choose a dataset'):
        data_selection.snapshot(db, [{'version_id':version['id']}])
    asset = data_selection.snapshot(db, [{'role':'simulation_assets', 'version_id':version['id']}])
    assert asset['assignments'][0]['role'] == 'simulation_assets'
