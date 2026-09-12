from pathlib import Path

import pytest

from skynet_app.database import Database, canonical_json
from skynet_app.data_resource_policy import FILE_RESOURCE_KINDS
from skynet_app.db_backend import INTEGRITY_ERRORS


def test_file_recording_links_rejected_and_dataset_links_retained(tmp_path):
    db = Database(tmp_path / 'registry')
    for kind in sorted(FILE_RESOURCE_KINDS):
        resource = db.create_data_resource(category="file", provider='local', namespace='files', name=kind, kind=kind, metadata={'license': 'upstream'})
        for key in ('session_id', 'recording_session_id'):
            with pytest.raises(ValueError, match='Files cannot be linked'):
                db.create_data_resource(category="file", provider='local', namespace='files', name=kind + key, kind=kind, metadata={key: 'recording'})
            with pytest.raises(ValueError, match='Files cannot be linked'):
                db.update_data_resource(resource['id'], metadata={key: 'recording'})
            with pytest.raises(INTEGRITY_ERRORS):
                with db.transaction() as c:
                    c.execute('UPDATE data_resources SET metadata_json=? WHERE id=?', (canonical_json({key: 'recording'}), resource['id']))
        assert db.get_data_resource(resource['id'])['metadata'] == {'license': 'upstream'}
    dataset = db.create_data_resource(category="dataset", provider='collection', namespace='datasets', name='demo', kind='demonstrations', metadata={'session_id': 'original'})
    db.update_data_resource(dataset['id'], metadata={'session_id': 'original', 'recording_session_id': 'copy'})
    assert db.get_data_resource(dataset['id'])['metadata']['recording_session_id'] == 'copy'


def test_migration_removes_only_file_ownership_keys(tmp_path):
    db = Database(tmp_path / 'registry')
    metadata = {'session_id': 'original', 'recording_session_id': 'copy', 'license': 'upstream', 'hand': {'name': 'Shadow'}}
    dataset = db.create_data_resource(category="dataset", provider='collection', namespace='datasets', name='demo', kind='demonstrations', metadata=metadata)
    file = db.create_data_resource(category="file", provider='local', namespace='files', name='assets', kind='simulation_assets')
    migration = Path(__file__).resolve().parents[1] / 'skynet_app/migrations/postgresql/007_file_recording_separation.sql'
    with db.transaction() as c:
        c.execute('ALTER TABLE data_resources DROP CONSTRAINT file_resources_no_recording_link')
        c.execute('UPDATE data_resources SET metadata_json=? WHERE id=?', (canonical_json(metadata), file['id']))
        c.executescript(migration.read_text())
    assert db.get_data_resource(file['id'])['metadata'] == {'license': 'upstream', 'hand': {'name': 'Shadow'}}
    assert db.get_data_resource(dataset['id'])['metadata'] == metadata


def test_dataset_sources_are_distinct_and_exposed_in_list_and_detail(tmp_path):
    db = Database(tmp_path / 'registry')
    resource = db.create_data_resource(category="dataset", provider='collection', namespace='datasets', name='combined', kind='demonstrations', metadata={'session_id': 'a'})
    db.create_data_resource_version(resource['id'], revision='one', format='skynet.episodes/v1', path='/test/source', manifest_sha256='a'*64, metadata={'sources': [{'session_id':'a'}, {'session_id':'b'}, {'session_id':'b'}]})
    assert db.get_data_resource(resource['id'])['recording_ids'] == ['a', 'b']
    assert db.list_data_resources()[0]['recording_ids'] == ['a', 'b']
    db.update_data_resource(resource['id'], metadata={'session_id':'a', 'recording_session_id':'copy'})
    assert db.get_data_resource(resource['id'])['recording_ids'] == ['b', 'copy']
