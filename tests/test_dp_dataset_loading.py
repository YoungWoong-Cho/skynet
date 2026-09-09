"""Verify compact image reads preserve history padding and action alignment."""
from pathlib import Path
import sys

import numpy as np
import pytest

torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).parents[1]/'skynet_app/adapters'))
from dp_training import RecordedDataset, CAMERAS


class Array:
    def __init__(self):
        self.values=np.arange(10,dtype=np.uint8).reshape(10,1,1,1)
        self.reads=[]
    def __getitem__(self,selection):
        self.reads.append(selection)
        return self.values[selection]


@pytest.mark.parametrize('start,end,pad,expected',[(0,10,1,[0,0]),(0,10,0,[0,1]),(9,10,0,[9,9]),(5,10,0,[5,6])])
def test_only_required_camera_frames_are_loaded_with_matching_padding(start,end,pad,expected):
    dataset=object.__new__(RecordedDataset)
    dataset.n_obs_steps=2;dataset.horizon=16
    dataset.keys=['state','action',*CAMERAS.values()]
    class Sampler:
        indices=[(start,end,pad,17)]
        def sample_sequence(self,index):
            return {'state':np.zeros((17,2),np.float32),'action':np.arange(34,dtype=np.float32).reshape(17,2)}
    dataset.sampler=Sampler()
    dataset.replay_buffer={name:Array() for name in CAMERAS.values()}
    sample=dataset[0]
    for name in CAMERAS.values():
        assert sample[name].flatten().tolist()==expected
        assert all(s.stop-s.start<=2 for s in dataset.replay_buffer[name].reads)
    batch=dataset[np.array([0])]
    result=dataset.postprocess(batch,torch.device('cpu'))
    assert tuple(result['action'].shape)==(1,16,2)
    assert result['action'][0,0].tolist()==[2,3]
    assert tuple(result['obs']['agent_pos'].shape)==(1,2,2)
