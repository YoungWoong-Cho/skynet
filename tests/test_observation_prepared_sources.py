"""Named observation variants cannot be lost behind a camera/modality shortcut."""
from copy import deepcopy

from skynet_app.observation_contracts import plan_artifacts, rgb_requirements
from skynet_app.observation_preparation import prepared_source


def test_named_camera_variants_and_cloud_crops_survive_conversion_staging():
    contract = rgb_requirements(['scene_front'])
    contract['streams'][0]['name'] = 'front_large'
    small = deepcopy(contract['streams'][0])
    small.update(name='front_small', width=128, height=128)
    contract['streams'].append(small)
    for name, count in [('fine_cloud', 128), ('coarse_cloud', 32)]:
        contract['streams'].append(dict(name=name, modality='point_cloud', camera_ids=['scene_front'],
            width=256, height=256, channels='XYZ', coordinate_frame='world', num_points=count,
            sampling='farthest_point', seed=42, processing_order=['unproject', 'world_transform', 'merge', 'crop', 'sample'],
            insufficient_points='repeat_with_mask', crop=None))
    source = dict(sha256='a' * 64)
    nodes = plan_artifacts(source['sha256'], {}, contract, dict(source_revision='frozen', cameras={'scene_front': {'sensor': 'front'}}))
    existing = {n['artifact_key']: dict(artifact_key=n['artifact_key'], path='/observations/' + n['artifact_key'], manifest_sha256='b' * 64) for n in nodes}
    result = prepared_source(source, contract, nodes, existing)
    streams = result['observation_streams']
    assert set(streams) == {'front_large', 'front_small', 'fine_cloud', 'coarse_cloud'}
    assert len({ref['artifact_key'] for ref in streams.values()}) == 4
    # A consumer must choose an explicit variant instead of receiving whichever
    # incompatible size or point count happened to be planned last.
    assert set(result['observation_artifacts']['scene_front']) == {'depth'}
    assert prepared_source(source, contract, list(reversed(nodes)), existing) == result


from test_observation_preparation import context, cloud_contract


def test_point_cloud_sampling_is_scheduled_per_recording(context, monkeypatch):
    from skynet_app.dataset_formats import RECIPES
    from copy import deepcopy
    service = context.service
    context.add_session('second', checksum='b' * 64)
    recipe = deepcopy(RECIPES['dp'])
    recipe['observation_requirements'] = cloud_contract()
    monkeypatch.setitem(RECIPES, 'dp', recipe)
    job = service.create('first', 'dp', 'Separate CPU episodes', selections=[
        dict(session_id='first', indices=[0]), dict(session_id='second', indices=[0])])
    service.prepare(job['id'])
    renders = service.observations.store.producers()
    assert len(renders) == 1
    context.cluster.publish(renders[0])
    service.observations.tick()
    service.prepare(job['id'])
    derives = context.cluster.requests('derive')
    assert len(derives) == 2
    assert all(len(request['sources']) == 1 for request in derives)
    assert {request['sources'][0]['sha256'] for request in derives} == {'a' * 64, 'b' * 64}
