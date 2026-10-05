import json
from pathlib import Path

import pytest

from skynet_app.slurm import RUNNER_SOURCE


def test_runner_records_actual_sidecar_step_not_filename(tmp_path):
    namespace = {'__name__': 'checkpoint_test'}
    exec(compile(RUNNER_SOURCE, 'runtime-wrapper.py', 'exec'), namespace)
    run = tmp_path/'run'; project = tmp_path/'project'; project.mkdir()
    folder = run/'checkpoints'; folder.mkdir(parents=True)
    checkpoint = folder/'step-999.ckpt'; checkpoint.write_bytes(b'weights')
    checkpoint.with_name(checkpoint.name+'.skynet-checkpoint.json').write_text(json.dumps(
        {'schema':'skynet.checkpoint-score/v1','global_step':42}))
    namespace['snapshot_checkpoints'](run, project, {'run_id':'run','checkpoint_globs':['checkpoints/*.ckpt']}, final=True)
    assert json.loads((folder/'selected-for-inference.json').read_text())['training_step']==42
    checkpoint.with_name(checkpoint.name+'.skynet-checkpoint.json').write_text('{"global_step":true}')
    with pytest.raises(RuntimeError, match='training step'):
        namespace['_checkpoint_training_step'](checkpoint)
