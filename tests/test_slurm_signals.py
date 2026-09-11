"""Exercise batch warnings with real shell, wrapper and trainer processes."""
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from skynet_app.slurm import BATCH_WARNING_HANDLER, RUNNER_SOURCE, _supervise_runtime


def wait_for(path, process):
    deadline = time.monotonic() + 10
    while not path.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"Process exited before {path.name}: {stdout}\n{stderr}")
        if time.monotonic() > deadline:
            pytest.fail(f"Timed out waiting for {path}")
        time.sleep(0.01)


TRAINER = '''import json, os, signal, sys, time
from pathlib import Path
root = Path(os.environ['SKYNET_RUN_DIR'])
checkpoint = root / 'artifacts' / 'last.ckpt'
checkpoint.parent.mkdir(exist_ok=True)
if '--checkpoint' in sys.argv:
    (root / 'resumed.json').write_text(json.dumps(sys.argv[sys.argv.index('--checkpoint')+1]))
    assert checkpoint.read_text() == 'saved-on-stop'
    raise SystemExit(0)
checkpoint.write_text('previous-save')
if 'graceful' in sys.argv:
    def stop(sig, frame):
        time.sleep(0.15)
        checkpoint.write_text('saved-on-stop')
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
(root / 'ready').touch()
while True: time.sleep(0.02)
'''


def launch(tmp_path, *, graceful=True, resume=False, warning_before_runtime=False):
    run = tmp_path / 'run'
    run.mkdir(exist_ok=True)
    attempt = run / 'attempts' / ('job2' if resume else 'job1')
    attempt.mkdir(parents=True, exist_ok=True)
    wrapper = tmp_path / 'wrapper.py'
    wrapper.write_text(RUNNER_SOURCE)
    trainer = tmp_path / 'trainer.py'
    trainer.write_text(TRAINER)
    execution = tmp_path / 'execution.json'
    execution.write_text(json.dumps({
        'run_id': 'run', 'stage': 'train', 'auto_resume': resume,
        'argv': [sys.executable, str(trainer), *(['graceful'] if graceful else [])],
        'resume_argv': ['--checkpoint', '{{SKYNET_RESUME_CHECKPOINT}}'],
        'preparation_steps': [], 'checkpoint_globs': ['artifacts/last.ckpt'],
        'checkpoint': {'keep_last': 3, 'final_selector': 'latest'},
    }))
    command = shlex.join([sys.executable, str(wrapper), str(execution)])
    before = ['touch "$SKYNET_RUN_DIR/batch-ready"', 'sleep 0.3'] if warning_before_runtime else []
    script = '\n'.join([
        'set -Eeuo pipefail', BATCH_WARNING_HANDLER,
        *(_supervise_runtime([*before, command])),
    ])
    env = dict(os.environ, SKYNET_RUN_DIR=str(run), SKYNET_SOURCE_DIR=str(tmp_path),
               SKYNET_PROJECT_DIR=str(tmp_path), SKYNET_CAPSULE_DIR=str(attempt),
               SKYNET_RUN_ID='run', SLURM_JOB_ID='job2' if resume else 'job1')
    process = subprocess.Popen(['bash', '-c', script], env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    return process, run, attempt


def cleanup(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


@pytest.mark.parametrize('graceful', [True, False])
def test_batch_warning_waits_for_trainer_and_preserves_resumable_checkpoint(tmp_path, graceful):
    process, run, attempt = launch(tmp_path, graceful=graceful)
    try:
        wait_for(run / 'ready', process)
        # Signal the batch shell, exactly as --signal=B:USR1 does.
        process.send_signal(signal.SIGUSR1)
        time.sleep(0.03)
        process.send_signal(signal.SIGUSR1)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 124, (stdout, stderr)
        receipt = json.loads((attempt / 'state/interruption.json').read_text())
        assert receipt == {'schema_version': 1, 'job_id': 'job1', 'run_id': 'run',
                           'reason': 'time_limit_warning', 'exit_code': 124}
        saved = json.loads((run / 'checkpoints/latest.json').read_text())
        assert saved['resumable'] is True and saved['final'] is False
        assert not (run / 'checkpoints/selected-for-inference.json').exists()
        assert Path(saved['path']).read_text() == ('saved-on-stop' if graceful else 'previous-save')
    finally:
        cleanup(process)
    if graceful:
        resumed, _, _ = launch(tmp_path, resume=True)
        try:
            stdout, stderr = resumed.communicate(timeout=10)
            assert resumed.returncode == 0, (stdout, stderr)
            assert json.loads((run / 'resumed.json').read_text()) == saved['path']
            assert json.loads((run / 'checkpoints/selected-for-inference.json').read_text())['final'] is True
        finally:
            cleanup(resumed)


def test_warning_during_runtime_setup_never_launches_training(tmp_path):
    process, run, attempt = launch(tmp_path, warning_before_runtime=True)
    try:
        wait_for(run / 'batch-ready', process)
        process.send_signal(signal.SIGUSR1)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 124, (stdout, stderr)
        assert not (run / 'ready').exists()
        assert (attempt / 'state/interruption.json').is_file()
    finally:
        cleanup(process)


@pytest.mark.parametrize('return_code', [0, 17, 124])
def test_runtime_supervisor_preserves_normal_exit_codes(return_code):
    script = '\n'.join(['set -Eeuo pipefail', *_supervise_runtime([f'exit {return_code}'])])
    result = subprocess.run(['bash', '-c', script], timeout=5)
    assert result.returncode == return_code
