"""Exact dataset deletion keeps raw recordings, shared streams and live users safe."""
from types import SimpleNamespace
import pytest
from prepared_fixture import preparation, published
from skynet_app import data_selection, dataset_cleanup, prepared_deletion
from skynet_app.cluster_runtime import ClusterError


def test_complete_delete_preserves_raw_and_shared(preparation,tmp_path,monkeypatch):
    service,job,path,capsule,raw,shared=published(preparation,tmp_path,monkeypatch)
    before=(raw.read_bytes(),shared.read_bytes())
    assert service.delete_dataset(job['resource_id'],job['id'])['deleted']
    assert not path.exists() and not capsule.exists() and not (service.root/job['id']).exists()
    assert service.database.get_data_resource_version(job['version_id']) is None
    assert service.database.get_data_resource_version(job['source_version_id'])
    assert before==(raw.read_bytes(),shared.read_bytes())


@pytest.mark.parametrize('resolved',[False,True])
def test_used_dataset_blocks_every_file_removal(preparation,tmp_path,monkeypatch,resolved):
    service,job,path,capsule,raw,shared=published(preparation,tmp_path,monkeypatch)
    db=service.database
    project=db.create_project('test')
    spec={'data':{'bundle':data_selection.snapshot(db,[{'version_id':job['version_id']}])}}
    experiment=db.create_experiment(project_id=project['id'],name='Uses dataset',requested_spec={} if resolved else spec)
    if resolved:
        db.create_variant(experiment['latest_revision']['id'],name='resolved',parameters={},resolved_spec={'native':{'path':str(path)}})
    with pytest.raises(ValueError,match='used by an experiment'):
        service.delete_dataset(job['resource_id'],job['id'])
    assert all(p.exists() for p in (path,capsule,raw,shared))


def test_cluster_failure_retains_metadata_for_safe_retry(preparation,tmp_path,monkeypatch):
    service,job,path,*_=published(preparation,tmp_path,monkeypatch)
    good=service.cluster.ssh
    def fail(*a,**kw): raise ClusterError('offline')
    monkeypatch.setattr(service.cluster,'ssh',fail)
    with pytest.raises(ValueError,match='Retry Delete dataset'):
        service.delete_dataset(job['resource_id'],job['id'])
    assert service.get(job['id'])['state']=='DELETE_FAILED'
    assert service.database.get_data_resource_version(job['version_id']) and path.exists()
    monkeypatch.setattr(service.cluster,'ssh',good)
    assert service.delete_dataset(job['resource_id'],job['id'])['deleted']
    assert not path.exists()


@pytest.mark.parametrize('gateway,failed_host,expected',[
    ('sky2',None,['sky2']),
    ('sky2','sky2',['sky2','sky1']),
    ('unconfigured-host',None,['sky1']),
])
def test_cleanup_prefers_recorded_configured_gateway_and_preserves_fallback(
        preparation,tmp_path,monkeypatch,gateway,failed_host,expected):
    service,job,path,capsule,raw,shared=published(preparation,tmp_path,monkeypatch)
    service.update(job['id'],gateway=gateway)
    cleanup=service.cluster.ssh
    attempts=[]
    service.cluster.candidates=lambda _:('sky1','sky2')

    def resolve(host):
        assert host in ('sky1','sky2'), 'Saved metadata cannot introduce an SSH host'
        return host

    def ssh(host,command,timeout):
        attempts.append(host)
        if host==failed_host:
            raise ClusterError('Confirmed gateway is unavailable')
        return cleanup(host,command,timeout)

    service.cluster.resolve_gateway=resolve
    service.cluster.ssh=ssh
    assert service.delete_dataset(job['resource_id'],job['id'])['deleted']
    assert attempts==expected
    assert not path.exists() and not capsule.exists()
    assert raw.exists() and shared.exists()


def test_unused_adapter_result_deleted_without_touching_used_result(preparation,tmp_path,monkeypatch):
    service,first,path,*_=published(preparation,tmp_path,monkeypatch)
    db=service.database
    project=db.create_project('test')
    db.create_experiment(project_id=project['id'],name='Keep this input',requested_spec={'data':{'bundle':data_selection.snapshot(db,[{'version_id':first['version_id']}])}})
    _,second,other,*_=published(preparation,tmp_path,monkeypatch,'rgb')
    assert second['resource_id']==first['resource_id']
    assert service.delete_dataset(second['resource_id'],second['id'])['deleted']
    assert path.exists() and not other.exists()
    assert db.get_data_resource_version(first['version_id'])
    assert db.get_data_resource_version(second['version_id']) is None


def test_manual_bundle_blocks_deletion(preparation,tmp_path,monkeypatch):
    service,job,path,*_=published(preparation,tmp_path,monkeypatch)
    service.database.create_data_bundle(name='manual',version='1',assignments=[{'role':'training_data','version_id':job['version_id']}])
    with pytest.raises(ValueError,match='bundle'):
        service.delete_dataset(job['resource_id'],job['id'])
    assert path.exists()


def test_cleanup_rejects_redirected_paths(tmp_path):
    root=tmp_path/'exports';root.mkdir()
    original=tmp_path/'recordings';original.mkdir()
    (original/'original.pkl').write_bytes(b'original')
    identifier='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
    (root/identifier).symlink_to(original,target_is_directory=True)
    with pytest.raises(ValueError,match='symbolic'):dataset_cleanup.cleanup(root,jobs=[identifier])
    with pytest.raises(ValueError):dataset_cleanup.cleanup(root,jobs=['../recordings'])
    assert (original/'original.pkl').read_bytes()==b'original'


def test_external_registration_preserves_source_and_blocks_active_import(preparation,tmp_path):
    service,*_=preparation;db=service.database
    resource=db.create_data_resource(category='dataset',provider='huggingface',namespace='test',source_key='External',kind='dataset')
    source=tmp_path/'external.zarr';source.write_bytes(b'Externally owned dataset')
    version=db.create_data_resource_version(resource['id'],revision='1',format='zarr',path=str(source),manifest_sha256='e'*64)
    db.record_data_location(version['id'],kind='cluster',host='test',path=str(source),manifest_sha256='e'*64)
    job=db.create_data_import(resource['id'],request={'revision':'1'})
    db.update_data_import(job['id'],version_id=version['id'])
    workspace=SimpleNamespace(owns=lambda *_:True)
    blocked=prepared_deletion.preview(service,workspace,'dataset',version['id'])
    assert 'import to finish' in blocked['blockers'][0]['reason']
    db.update_data_import(job['id'],state='FAILED',version_id=version['id'])
    plan=prepared_deletion.preview(service,workspace,'dataset',version['id'])
    assert not plan['blockers'] and not plan['files']
    with pytest.raises(ValueError,match='changed'):prepared_deletion.delete(service,workspace,'dataset',version['id'],blocked['token'])
    assert prepared_deletion.delete(service,workspace,'dataset',version['id'],plan['token'])['deleted']
    assert source.read_bytes()==b'Externally owned dataset'
