"""Each published output owns its presentation and exact source membership."""
import json
from pathlib import Path

import pytest

from skynet_app import data_selection
from skynet_app.database import Database, canonical_json
from skynet_app.db_backend import INTEGRITY_ERRORS


def resource(db, **kwargs):
    return db.create_data_resource(category='dataset', provider='collection', namespace='datasets',
        source_key='session-a', kind='demonstrations', display_name='Shadow · Cube',
        description='Original description', metadata={'session_id': 'a'}, **kwargs)


def version(db, parent, name='one', **kwargs):
    return db.create_data_resource_version(parent['id'], revision=name, format='skynet.recording-dataset/v1',
        path='/datasets/'+name, manifest_sha256=(name.encode().hex()+'0'*64)[:64], **kwargs)


def test_flat_catalog_keeps_exact_sources_and_hides_raw_and_files(tmp_path):
    db=Database(tmp_path/'db');parent=resource(db)
    raw= db.create_data_resource_version(parent['id'], revision='raw', format='skynet.episodes/v1',
        path='/raw', manifest_sha256='e'*64, metadata={'sources':[{'session_id':'a'}, {'session_id':'b'}]})
    first=version(db,parent,metadata={'sources':[{'session_id':'a'}], 'adapter':{'name':'UniDex'}})
    second=version(db,parent,'two',metadata={'source_version_id':raw['id'], 'adapter':{'name':'HPT'}})
    file=db.create_data_resource(category='file',provider='local',namespace='assets',source_key='hand',kind='simulation_assets')
    version(db,file,'file')
    rows=db.list_datasets()
    assert [row['id'] for row in rows]==[second['id'],first['id']]
    assert rows[0]['recording_ids']==['a','b'] and rows[1]['recording_ids']==['a']
    assert rows[0]['display_name']=='Shadow · Cube · HPT'
    assert rows[1]['display_name']=='Shadow · Cube · UniDex'
    assert all(row['category']=='dataset' and 'versions' not in row for row in rows)
    assert db.get_dataset(raw['id']) is None and db.get_dataset(parent['id']) is None


def test_copied_single_episode_owner_overrides_parent_receipt(tmp_path):
    db=Database(tmp_path/'db');parent=resource(db)
    db.update_data_resource(parent['id'],metadata={'session_id':'a','recording_session_id':'copy'})
    one=version(db,parent,metadata={'sources':[{'session_id':'a'}]})
    two=version(db,parent,'two',metadata={'sources':[{'session_id':'b'}]})
    assert db.get_dataset(one['id'])['recording_ids']==['copy']
    assert db.get_dataset(two['id'])['recording_ids']==['b']


def test_edit_archive_isolated_and_does_not_change_manifests_or_snapshots(tmp_path):
    db=Database(tmp_path/'db');parent=resource(db)
    first=version(db,parent,metadata={'display_name':'First dataset','sources':[{'session_id':'a'}]})
    second=version(db,parent,'two',metadata={'display_name':'Second dataset'})
    snapshot=data_selection.snapshot(db,[{'version_id':first['id']}])
    before=db.get_data_resource_version(first['id'])
    db.update_dataset(first['id'],display_name='Renamed',description='Only this result',archived=True)
    after=db.get_data_resource_version(first['id'])
    assert before==after
    assert snapshot['assignments'][0]['version']['metadata']['display_name']=='First dataset'
    assert db.get_dataset(first['id'])['description']=='Only this result'
    assert db.get_dataset(second['id'])['display_name']=='Second dataset'
    assert [row['id'] for row in db.list_datasets()]==[second['id']]
    assert {row['id'] for row in db.list_datasets(include_archived=True)}=={first['id'],second['id']}
    assert [row['id'] for row in data_selection.choices(db)]==[second['id']]
    with pytest.raises(ValueError,match='archived'):
        data_selection.snapshot(db,[{'version_id':first['id']}])
    db.update_dataset(first['id'],archived=False)
    assert data_selection.snapshot(db,[{'version_id':first['id']}])['name']=='Renamed'
    assert len(data_selection.choices(db))==2
    with pytest.raises(INTEGRITY_ERRORS):
        with db.transaction() as c:c.execute('UPDATE data_resource_versions SET metadata_json=? WHERE id=?',(canonical_json({'changed':True}),first['id']))


def test_preset_links_identify_exact_result(tmp_path):
    db=Database(tmp_path/'db');parent=resource(db)
    first=version(db,parent);second=version(db,parent,'two')
    project=db.create_project('Test')
    experiment=db.create_experiment(project_id=project['id'],name='Uses first',requested_spec={'data':{'bundle':data_selection.snapshot(db,[{'version_id':first['id']}])}})
    assert db.get_dataset(first['id'])['experiment_presets'][0]['experiment_id']==experiment['id']
    assert db.get_dataset(second['id'])['experiment_presets']==[]


def test_migration_backfills_archive_and_names_without_rewriting_versions(tmp_path):
    db=Database(tmp_path/'db');parent=resource(db)
    first=version(db,parent,metadata={'display_name':'Shadow · Cube','adapter':{'name':'UniDex'}})
    second=version(db,parent,'two',metadata={'adapter':{'name':'HPT'}})
    before={item['id']:item for item in db.get_data_resource(parent['id'])['versions']}
    migration=Path(__file__).resolve().parents[1]/'skynet_app/migrations/postgresql/016_flat_dataset_catalog.sql'
    with db.transaction() as c:
        c.execute('DROP TRIGGER dataset_presentation_created ON data_resource_versions')
        c.execute('DROP FUNCTION initialize_dataset_presentation()')
        c.execute('DROP TABLE data_dataset_presentations')
        c.execute("UPDATE data_resources SET archived_at='2026-09-19' WHERE id=?",(parent['id'],))
        c.executescript(migration.read_text())
    assert db.get_data_resource(parent['id'])['archived_at'] is None
    assert db.list_datasets()==[]
    rows=db.list_datasets(include_archived=True)
    assert {row['display_name'] for row in rows}=={'Shadow · Cube · UniDex','Shadow · Cube · HPT'}
    assert all(row['archived_at']=='2026-09-19' for row in rows)
    assert before=={item['id']:item for item in db.get_data_resource(parent['id'])['versions']}
    third=version(db,parent,'three')
    assert db.get_dataset(third['id'])['archived_at'] is None


def test_deletion_requires_specific_result_and_keeps_external_sibling(tmp_path):
    db=Database(tmp_path/'db')
    parent=db.create_data_resource(category='dataset',provider='huggingface',namespace='demo',source_key='one',kind='dataset')
    first=version(db,parent);second=version(db,parent,'two')
    with pytest.raises(ValueError,match='exactly one'):
        db.delete_prepared_dataset(parent['id'],lambda *_:None)
    removed=[]
    result=db.delete_prepared_dataset(parent['id'],lambda jobs,versions,locations:removed.extend(versions),version_id=first['id'])
    assert result['version_id']==first['id'] and [v['id'] for v in removed]==[first['id']]
    assert db.get_data_resource_version(first['id']) is None
    assert db.get_data_resource_version(second['id'])
    assert db.get_data_resource(parent['id'])
    assert db.get_dataset(second['id'])


def test_dataset_api_accepts_only_presentation_and_filters_files(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from skynet_app import pipeline_api

    db=Database(tmp_path/'db');parent=resource(db)
    first=version(db,parent);second=version(db,parent,'two')
    file=db.create_data_resource(category='file',provider='local',namespace='files',source_key='weights',kind='model')
    app=FastAPI();app.include_router(pipeline_api.router)
    app.dependency_overrides[pipeline_api.require_workspace_records]=lambda:None
    monkeypatch.setattr(pipeline_api,'service',SimpleNamespace(database=db))
    with TestClient(app) as client:
        response=client.get('/api/data/datasets')
        assert response.status_code==200
        assert [row['id'] for row in response.json()['datasets']]==[second['id'],first['id']]
        assert client.get('/api/data/datasets/'+parent['id']).status_code==404
        assert client.patch('/api/data/resources/'+parent['id'],json={'archived':True}).status_code==422
        assert client.delete('/api/data/resources/'+parent['id']).status_code==422
        response=client.patch('/api/data/datasets/'+first['id'],json={'display_name':'Renamed','archived':True})
        assert response.status_code==200 and response.json()['dataset']['archived_at']
        assert response.json()['dataset']['created_at']==first['created_at']
        assert response.json()['dataset']['metadata']==first['metadata']
        assert response.json()['dataset']['description']=='Original description'
        assert len(client.get('/api/data/datasets').json()['datasets'])==1
        assert len(client.get('/api/data/datasets?include_archived=true').json()['datasets'])==2
        for field in ('metadata','resource_id','format','id'):
            assert client.patch('/api/data/datasets/'+first['id'],json={field:'changed'}).status_code==422
        assert client.patch('/api/data/datasets/'+first['id'],json={'display_name':'   '}).status_code==422
        assert client.patch('/api/data/datasets/absent',json={'archived':True}).status_code==404
        assert [row['id'] for row in client.get('/api/data/resources?category=file').json()['resources']]==[file['id']]
        assert 'dataset' in client.get('/api/data/resources?category=file').json()['resource_types']
