from types import SimpleNamespace
import pytest
from prepared_fixture import preparation, published
from skynet_app import prepared_deletion, data_selection


def test_preview_read_only_and_deletion_rechecks_dependencies(preparation,tmp_path,monkeypatch):
    service,job,path,capsule,raw,_=published(preparation,tmp_path,monkeypatch)
    workspace=SimpleNamespace(owns=lambda *_:True)
    plan=prepared_deletion.preview(service,workspace,'prepared',job['id'])
    assert not plan['blockers'] and path.exists() and capsule.exists()
    assert str(path) in {item['path'] for item in plan['files']}
    db=service.database;project=db.create_project('preview')
    spec={'data':{'bundle':data_selection.snapshot(db,[{'version_id':job['version_id']}])}}
    db.create_experiment(project_id=project['id'],name='Uses this data',requested_spec=spec)
    revised=prepared_deletion.preview(service,workspace,'prepared',job['id'])
    assert any(item['label']=='Uses this data' for item in revised['blockers'])
    with pytest.raises(ValueError,match='dependencies changed'):
        prepared_deletion.delete(service,workspace,'prepared',job['id'],plan['token'])
    assert path.exists() and raw.exists()


def test_preview_removes_only_selected_manifest(preparation,tmp_path,monkeypatch):
    service,job,path,capsule,raw,shared=published(preparation,tmp_path,monkeypatch)
    workspace=SimpleNamespace(owns=lambda *_:True)
    plan=prepared_deletion.preview(service,workspace,'prepared',job['id'])
    assert not plan['blockers']
    assert prepared_deletion.delete(service,workspace,'prepared',job['id'],plan['token'])['deleted']
    assert not path.exists() and not capsule.exists() and raw.exists() and shared.exists()


def test_removed_local_copy_flow_is_rejected(preparation):
    with pytest.raises(ValueError,match='Unknown'):
        prepared_deletion.preview(preparation[0],SimpleNamespace(),'local-copy','unused')
