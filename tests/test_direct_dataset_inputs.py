"""Prepared data goes directly into experiment snapshots, with no extra registry row."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from skynet_app.database import Database
from skynet_app import data_selection, pipeline_api
from skynet_app.model_io import adapter_io_contract


def registered(tmp_path):
    db = Database(tmp_path / 'test-db')
    resource = db.create_data_resource(category="dataset", provider='collection', namespace='datasets', source_key='Cube', kind='dataset')
    version = db.create_data_resource_version(resource['id'], revision='capture-a', format='zarr',
        path='/cluster/prepared/a', manifest_sha256='a'*64,
        metadata={'episodes': 1, 'capture': {'action_joint_names': ['a','b','c'], 'robot_joint_names': ['a','b','c']}})
    location = db.record_data_location(version['id'], kind='cluster', host='sky2', path='/cluster/prepared/a', manifest_sha256='a'*64)
    return db, resource, version, location


def test_direct_selection_is_pinned_and_has_no_bundle_row(tmp_path):
    db, resource, version, location = registered(tmp_path)
    selection = dict(version_id=version['id'], location_id=location['id'])
    frozen = data_selection.snapshot(db, [selection])
    assert db.list_data_bundles() == []
    assert frozen['assignments'][0]['config']['location']['path'] == '/cluster/prepared/a'
    db.create_data_resource_version(resource['id'], revision='capture-b', format='hdf5', path='/cluster/prepared/b', manifest_sha256='b'*64)
    assert frozen['assignments'][0]['version']['manifest_sha256'] == 'a'*64
    assert data_selection.snapshot(db, [selection]) == frozen
    with pytest.raises(ValueError, match='unique'):
        data_selection.snapshot(db, [selection, selection])
    with pytest.raises((KeyError, ValueError)):
        data_selection.snapshot(db, [{**selection, 'location_id': 'not-registered'}])
    db.update_data_resource(resource['id'], archived=True)
    assert data_selection.choices(db) == []
    with pytest.raises(ValueError, match='archived'):
        data_selection.snapshot(db, [selection])


def test_source_references_are_not_trainable_results_and_input_roles_survive(tmp_path):
    db, resource, version, location = registered(tmp_path)
    source = db.create_data_resource_version(resource['id'], revision='source', format='skynet.episodes/v1', path='/cluster/original.json', manifest_sha256='c'*64)
    assert [r['id'] for r in data_selection.choices(db)] == [version['id']]
    with pytest.raises(ValueError, match='original recording'):
        data_selection.snapshot(db, [{'version_id': source['id']}])
    selected = [dict(version_id=version['id'], location_id=location['id'], role=role) for role in ['training_data', 'validation_data']]
    assert {a['role'] for a in data_selection.snapshot(db, selected)['assignments']} == {'training_data', 'validation_data'}


def test_model_dimensions_resolve_from_direct_selection(tmp_path, monkeypatch):
    db, resource, version, location = registered(tmp_path)
    monkeypatch.setattr(pipeline_api, 'service', SimpleNamespace(database=db))
    app = FastAPI(); app.include_router(pipeline_api.router)
    response = TestClient(app).post('/api/model-io/preview', json={
        'data_selections': [{'version_id': version['id'], 'location_id': location['id']}],
        'manifest': {'train': {'model_io': adapter_io_contract('egoverse-hpt-joints').model_dump(mode='json')}}})
    assert response.status_code == 200, response.text
    assert dict(response.json()['entries'])['Output · Joint commands'] == '100 × 3 (steps × values)'
    assert db.list_data_bundles() == []
