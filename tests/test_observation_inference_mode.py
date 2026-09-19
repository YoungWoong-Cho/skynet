"""Exercise the native renderer lifecycle with real CPU inference tensors."""
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from ops.datasets import observation_render as render

torch = pytest.importorskip("torch")


@pytest.fixture
def renderer_case(tmp_path, monkeypatch):
    modes, closed = {}, []
    failures = set()
    scene = SimpleNamespace(joint_acc=torch.zeros(1))

    def close(name):
        modes[name] = torch.is_inference_mode_enabled()
        closed.append(name)
        if name in failures:
            raise RuntimeError(name + " failed")

    # Isaac Lab replaces its acceleration buffer during scene.update(), then
    # reset_to() writes to that same buffer at the next episode boundary.
    def update(**_):
        scene.joint_acc = torch.ones(1) + 1

    scene.update = update
    env = SimpleNamespace(scene=scene, reset=lambda: None, step_dt=1 / 60,
                          physics_dt=1 / 120, close=lambda: close("env"))
    app = SimpleNamespace(config={}, is_running=lambda: True,
                          is_exiting=lambda: False, close=lambda **_: close("app"))
    settings = SimpleNamespace(set_bool=lambda *_: None, set_float=lambda *_: None)

    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    module("carb", settings=SimpleNamespace(get_settings=lambda: settings))
    core = module("omni.replicator.core", orchestrator=SimpleNamespace(step=lambda **_: None))
    replicator = module("omni.replicator", core=core)
    timeline = module("omni.timeline", get_timeline_interface=lambda: SimpleNamespace(get_current_time=lambda: 0.0))
    module("omni", replicator=replicator, timeline=timeline)
    module("scene_restore", restore_state=lambda env, _: env.scene.joint_acc.zero_())
    module("trajectory", restore_episode_conditions=lambda *_: None)
    module("recording_metadata", array=np.asarray,
           state_metadata=lambda *_: ({}, {"action_joint_names": ["joint"]}))

    def create_environment(self):
        self.torch = torch
        modes["create"] = torch.is_inference_mode_enabled()
        self.env = env
        self.cameras = {}
        self.scene_configuration_digest = "scene"
        self.scene_asset_digests = {}
        self.hand_work = SimpleNamespace(cleanup=lambda: close("hand"))
        if "create" in failures:
            raise RuntimeError("create failed")

    monkeypatch.setattr(render.DexVerseRenderer, "_create_environment", create_environment)
    monkeypatch.setattr(render, "_verify_assets", lambda _: {})
    monkeypatch.setattr(render.runpy, "run_path", lambda *_args, **_kwargs: {"simulation_app": app})
    revision = "30cc673e27684b9f10186fa6bea731aed246bc9f"
    (tmp_path / ".skynet-source-revision").write_text(revision)
    profile = dict(repository=str(tmp_path), source_revision=revision, runtime="test",
                   task="Dexverse-PickCube-v0", robot="floating_shadow_right", hand="right")
    identity = render.render_identity(profile, "a" * 64)
    source = dict(episode_key="episode", profile=profile, sha256="b" * 64)
    job = dict(episode_key="episode", camera_id="scene_front",
               recipe={"camera": identity["cameras"]["scene_front"]},
               spec={"render_identity": {k: v for k, v in identity.items() if k != "cameras"}})
    renderer = render.DexVerseRenderer({"episode": source}, [job])
    return SimpleNamespace(renderer=renderer, source=source, scene=scene,
                           modes=modes, closed=closed, failures=failures)


def test_two_episodes_can_restore_buffers_created_during_capture(renderer_case):
    case = renderer_case
    previous_argv = list(sys.argv)
    previous_mode = torch.is_inference_mode_enabled()
    episode = {"states": [{}, {}], "actions": np.zeros((2, 1))}
    with case.renderer as renderer:
        for _ in range(2):
            renderer.begin_episode(case.source, {"schema_version": 2}, episode, 1 / 60)
            assert renderer.capture({}, []) == {}
            assert case.scene.joint_acc.is_inference()
    assert case.modes == {"create": True, "env": True, "app": True, "hand": True}
    assert torch.is_inference_mode_enabled() == previous_mode
    assert sys.argv == previous_argv


@pytest.mark.parametrize("failure", ["create", "body", "env", "app", "hand"])
def test_errors_restore_mode_and_close_all_resources(renderer_case, failure):
    case = renderer_case
    previous_argv = list(sys.argv)
    previous_mode = torch.is_inference_mode_enabled()
    case.failures.add(failure)
    with pytest.raises(RuntimeError, match=failure + " failed"):
        with case.renderer:
            if failure == "body":
                raise RuntimeError("body failed")
    assert case.closed == ["env", "app", "hand"]
    assert all(case.modes.values())
    assert torch.is_inference_mode_enabled() == previous_mode
    assert sys.argv == previous_argv


def test_outer_inference_mode_is_preserved(renderer_case):
    with torch.inference_mode():
        with renderer_case.renderer:
            assert torch.is_inference_mode_enabled()
        assert torch.is_inference_mode_enabled()
