"""Conversion keeps every source frame; temporal eligibility belongs to experiments."""
from copy import deepcopy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ops.datasets.recording_probe import probe, validate_source_split
from skynet_app.adapters.recording_time import resolve_sampling, source_frequency
from skynet_app.recording_sampling import experiment_sampling
from skynet_app.adapters.hat_manifest import manifest as hat_manifest


def manifest(lengths=(90, 121), rates=(60, 60), split=None):
    return dict(format='skynet.recording-dataset/v1',
                contract='skynet.hat-rgb-fingertips/v1', validation={'status': 'PASSED'},
                episodes=[dict(index=i, id=f'episode-{i}', steps=n, capture={'step_dt':1/rates[i]})
                          for i, n in enumerate(lengths)],
                split=split or dict(train=[0], validation=[1]))


def test_conversion_accepts_single_short_source_without_training_split(monkeypatch):
    monkeypatch.setitem(sys.modules, 'recording_prepare', SimpleNamespace(
        preflight_source=lambda *a, **kw:dict(capture={'step_dt':1/60}, steps=1, streams={})))
    request=dict(job_id='test', attempt_id='one', sources=[dict(sha256='a'*64)],
                 split=dict(train=[0],validation=[]), requirements=dict(
                     contract='skynet.hat-rgb-fingertips/v1', observation_requirements={'streams':[]}))
    result=probe(request)
    assert result['verified'] and result['sources'][0]['steps']==1


def test_conversion_rejects_split_leakage_and_duplicate_sources():
    with pytest.raises(ValueError, match='disjoint'):
        validate_source_split(dict(train=[0], validation=[0]), [{'sha256':'a'}])
    with pytest.raises(ValueError, match='copies'):
        validate_source_split(dict(train=[0], validation=[1]), [{'sha256':'a'}, {'sha256':'a'}])


def test_short_recording_trainability_depends_on_experiment_not_conversion():
    data=manifest()
    before=deepcopy(data)
    plan=resolve_sampling(data, 30, 30, require_validation=True)
    assert plan['episodes'][0]['stride']==2
    assert plan['episodes'][0]['sampled_steps']==45
    assert plan['splits']['train']==dict(windows=16, episodes=1, excluded_episodes=[])
    with pytest.raises(ValueError, match='No usable train windows'):
        resolve_sampling(data, 15, 30, require_validation=True)
    assert data==before


def test_exclusion_does_not_discard_long_enough_episodes_or_cross_splits():
    data=manifest((90,117,121), (60,60,60), dict(train=[0,1], validation=[2]))
    plan=resolve_sampling(data,15,30,require_validation=True)
    assert plan['splits']['train']==dict(windows=1, episodes=1, excluded_episodes=['episode-0'])
    assert plan['splits']['validation']['windows']==2
    padded=resolve_sampling(data,15,30,window_policy='pad',require_validation=True)
    assert padded['splits']['train']['windows']==53


def test_mixed_source_rates_need_explicit_exact_common_frequency():
    data=manifest((121,61),(60,30))
    with pytest.raises(ValueError, match='different source frequencies'):
        resolve_sampling(data, action_steps=30)
    plan=resolve_sampling(data,15,30,require_validation=True)
    assert [e['stride'] for e in plan['episodes']]==[4,2]
    assert [e['sampled_steps'] for e in plan['episodes']]==[31,31]


@pytest.mark.parametrize('rate',[0,-1,True,60.1,120,24,7,float('nan'),float('inf'),5e-324])
def test_invalid_frequencies_are_validation_errors(rate):
    with pytest.raises(ValueError):
        resolve_sampling(manifest(),rate,1)


def test_tiny_source_period_is_validation_error():
    with pytest.raises(ValueError,match='finite'):
        source_frequency({'capture':{'step_dt':5e-324}})


def test_backend_reuses_worker_eligibility_and_manifest_default():
    declaration=hat_manifest()
    doc=dict(native={'config':{'control_hz':30,'window_policy':'complete'}},
             data={'bundle':{'assignments':[dict(role='training_data', position=0,
                 config={'location':dict(kind='cluster',status='AVAILABLE',path='/data',manifest_sha256='b'*64)},
                 version={'format':'skynet.recording-dataset/v1','manifest_sha256':'b'*64,'metadata':manifest(lengths=(190, 121))})]}})
    plan=experiment_sampling(doc,declaration)
    assert plan['action_steps']==50 and plan['control_hz']==30
    doc['native']['config']['control_hz']=15
    with pytest.raises(ValueError,match='No usable train'):
        experiment_sampling(doc,declaration)
    doc['native']['config']['action_steps']=20
    assert experiment_sampling(doc,declaration)['splits']['train']['windows']==29


def test_reader_samples_state_action_rgb_and_timestamps_without_changing_files(tmp_path):
    from test_recording_dataset import source, request
    from recording_prepare import prepare
    from recording_dataset import RecordingDataset, digest, close_handles
    src,actions,states=source(tmp_path,images=True)
    prepare(request(tmp_path,[src],rgb=True))
    root=tmp_path/'a'
    before={str(p):digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    data=RecordingDataset(root, control_hz=30)
    episode=data.episode(0)
    assert data.episode_steps(0)==2 and episode.source_steps==4
    np.testing.assert_array_equal(episode.read('state'), states[::2])
    np.testing.assert_array_equal(episode.read('action'), actions[::2])
    np.testing.assert_array_equal(episode.read('action',[1,0,1]), actions[[2,0,2]])
    np.testing.assert_array_equal(episode.read('action',slice(None,None,-1)), actions[[2,0]])
    np.testing.assert_array_equal(episode.read('action',-1), actions[2])
    np.testing.assert_array_equal(episode.read('scene_front')[:,0,0,0],[10,12])
    np.testing.assert_allclose(episode.read('timestamps'),[0,2/60])
    np.testing.assert_allclose(data.normalization()['action_mean'],actions[::2].mean(axis=0))
    for indices in ([0.5],[True], [2], [-1]):
        with pytest.raises(IndexError): episode.read('state',indices)
    close_handles()
    assert before=={str(p):digest(p) for p in tmp_path.rglob('*') if p.is_file()}


def test_sweep_validation_checks_each_frequency_before_launch(monkeypatch):
    from test_experiments import make_spec
    from skynet_app.pipeline_api import PipelineService
    declaration=hat_manifest()
    spec=make_spec(native={'config':{'control_hz':30,'action_steps':30,'window_policy':'complete'}},
        data={'bundle':dict(id='bundle',name='data',version='1',manifest_sha256='a'*64,
            assignments=[dict(role='training_data',position=0,
                config={'location':dict(kind='cluster',status='AVAILABLE',path='/data',manifest_sha256='b'*64)},
                resource=dict(provider='collection',namespace='datasets',name='test',kind='dataset'),
                version=dict(revision='1',format='skynet.recording-dataset/v1',path='/data',
                             manifest_sha256='b'*64,status='READY',metadata=manifest()))])},
        sweep={'strategy':'grid','axes':{'native.config.control_hz':[30,15]},'seeds':[1]})
    monkeypatch.setattr(PipelineService,'_validate_manifest_input_fields',lambda *a:None)
    with pytest.raises(ValueError,match='No usable train windows.*15 Hz'):
        PipelineService._validate_sweep_inputs(spec,declaration)


def test_timestamps_are_verified_against_source_period(tmp_path):
    import h5py
    import json
    from test_recording_dataset import source, request
    from recording_prepare import prepare
    from recording_dataset import verify_dataset, digest
    src,_,_=source(tmp_path,images=True)
    data=prepare(request(tmp_path,[src],rgb=True))
    with h5py.File(src['images'],'r+') as file:
        file['timestamps'][2]=0.025
    fingerprint=digest(src['images'])
    for reference in data['episodes'][0]['streams'].values():
        reference['sha256']=fingerprint
    path=tmp_path/'a/manifest.json';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='timestamps must be regular'):
        verify_dataset(path)
