import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from skynet_app.adapters import builtin_adapter_manifests
from skynet_app.adapters.egoverse_evaluation import policy_camera_views
from skynet_app.adapters.evaluation_video import compose_camera_views, rgb_frame
from skynet_app.adapters.openpi_libero_bridge import _FILTERED_CLIENT_SOURCE


def test_three_views_keep_all_pixels_and_rgb_order_without_mutating_inputs():
    frames = {name: np.full((1, 3, 16, 24), color, dtype=np.float32)
              for name, color in [('scene_front', 0.2), ('scene_left', 0.4), ('scene_right', 0.6)]}
    before = {key: value.copy() for key, value in frames.items()}
    video = compose_camera_views(frames, channel_first=True, captions=('Step 1',))
    assert video.shape == (80, 80, 3)
    for index, (key, frame) in enumerate(frames.items()):
        np.testing.assert_array_equal(video[32:48, index * 24:(index + 1) * 24],
                                      np.rint(frame[0].transpose(1, 2, 0) * 255).astype(np.uint8))
        np.testing.assert_array_equal(frame, before[key])
    red = np.zeros((16, 16, 3), dtype=np.uint8)
    red[..., 0] = 255
    np.testing.assert_array_equal(compose_camera_views({'red': red})[32:48, :16], red)


def test_four_views_wrap_and_different_dimensions_are_padded_not_stretched():
    views = {str(i): np.full((16, 24, 3), i + 1, dtype=np.uint8) for i in range(4)}
    views['1'] = np.full((8, 8, 3), 20, dtype=np.uint8)
    frame = compose_camera_views(views)
    assert frame.shape == (96, 80, 3)
    np.testing.assert_array_equal(frame[36:44, 32:40], views['1'])
    np.testing.assert_array_equal(frame[80:96, :24], views['3'])
    assert np.all(frame[80:96, 24:] == 24)
    with pytest.raises(ValueError, match='at least one'):
        compose_camera_views({})
    with pytest.raises(ValueError, match='RGB'):
        rgb_frame(np.zeros((5, 5)))


def test_camera_selection_uses_checkpoint_and_current_embodiment():
    sample = {'embodiment': np.array([100]), 'front': 1, 'wrist': 2, 'unused': 3}
    act = SimpleNamespace(camera_keys=['front', 'wrist'])
    assert policy_camera_views(act, sample) == {'front': 1, 'wrist': 2}
    hpt = SimpleNamespace(camera_keys={100: ['front', 'wrist', 'unused'], 101: ['elsewhere']},
                          encoders={'front': {}, 'wrist': {}})
    assert policy_camera_views(hpt, sample) == {'front': 1, 'wrist': 2}
    with pytest.raises(KeyError):
        policy_camera_views(SimpleNamespace(camera_keys=['missing']), sample)


def test_all_multi_view_evaluation_capsules_include_shared_renderer():
    for manifest in builtin_adapter_manifests():
        for evaluation in manifest.evaluations:
            if evaluation.command and (manifest.slug.startswith('egoverse-') or
                                      manifest.slug in {'xpolicylab-dp', 'xpolicylab-act'} or
                                      evaluation.environment == 'libero'):
                files = evaluation.command.capsule_files
                assert 'adapter-support/evaluation_video.py' in files, manifest.slug
                compile(files['adapter-support/evaluation_video.py'], 'evaluation_video.py', 'exec')


def test_openpi_replay_contains_both_views_without_changing_policy_observation(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'skynet_app/adapters'))
    source = tmp_path / 'upstream.py'
    source.write_text('''def eval_libero(args):
    img, wrist_img = args
    replay_images = []
    replay_images.append(img)
    element = {"observation/image": img, "observation/wrist_image": wrist_img}
    return replay_images, element
''')
    spec = importlib.util.spec_from_file_location('_video_test_upstream', source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    client = {'__name__': '_filtered_client_test'}
    exec(compile(_FILTERED_CLIENT_SOURCE, '<client>', 'exec'), client)
    client['include_policy_views'](module)
    front = np.full((16, 16, 3), 70, np.uint8)
    wrist = np.full((16, 16, 3), 130, np.uint8)
    frames, observation = module.eval_libero((front, wrist))
    assert observation['observation/image'] is front
    assert observation['observation/wrist_image'] is wrist
    np.testing.assert_array_equal(frames[0][32:48, :16], front)
    np.testing.assert_array_equal(frames[0][32:48, 16:32], wrist)
