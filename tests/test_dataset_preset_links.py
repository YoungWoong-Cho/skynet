"""Dataset/preset navigation follows immutable inputs and workspace ownership."""
from skynet_app.database import Database
from skynet_app.workspaces import WorkspaceDirectory


def test_links_include_all_revisions_and_only_owned_dataset_inputs(tmp_path):
    system = Database(tmp_path / 'links')
    directory = WorkspaceDirectory(system)
    alice = system.for_workspace(directory.open('alice@example.com')[0]['id'])
    bob = system.for_workspace(directory.open('bob@example.com')[0]['id'])
    versions = []
    for index, category in enumerate(['dataset', 'dataset', 'file']):
        resource = system.create_data_resource(category=category, provider='test', namespace='inputs', name=f'input-{index}', kind='dataset' if category == 'dataset' else 'model')
        versions.append(system.create_data_resource_version(resource['id'], revision='1', format='zarr', path=f'/cluster/{index}', manifest_sha256=str(index+1)*64))
    def spec(version, direct=False):
        bundle = system.create_data_bundle(name=version['id'] + str(direct), version='1', assignments=[{'role':'training_data', 'version_id':version['id']}])
        snapshot = system.data_bundle_snapshot(bundle['id'])
        if direct:
            snapshot['assignments'][0]['version']['metadata']['registered_version_id'] = version['id']
            snapshot['assignments'][0]['resource']['name'] = 'former-name'
        return {'data':{'bundle':snapshot}}
    first = spec(versions[0], True)
    second = spec(versions[1])
    project = alice.create_project('Alice')
    preset = alice.create_experiment(project_id=project['id'], name='Cube', requested_spec=first)
    alice.create_experiment_revision(preset['id'], first)
    alice.create_experiment_revision(preset['id'], second)
    alice.create_experiment_revision(preset['id'], spec(versions[2]))
    alice.create_experiment_revision(preset['id'], {})
    private = bob.create_experiment(project_id=bob.create_project('Bob')['id'], name='Private', requested_spec=first)
    links = alice.dataset_preset_links()
    assert [(row['revision_number'], row['version_id']) for row in links] == [(1, versions[0]['id']), (2, versions[0]['id']), (3, versions[1]['id'])]
    assert {row['experiment_id'] for row in links} == {preset['id']}
    assert {row['experiment_id'] for row in bob.dataset_preset_links()} == {private['id']}
