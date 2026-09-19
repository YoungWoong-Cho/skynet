from types import SimpleNamespace

import pytest

from ops.datasets.observation_render import check_renderer_driver


@pytest.mark.parametrize("driver", ["595.84", "595.71.05"])
def test_evidenced_driver_incompatibility_has_actionable_error(monkeypatch, driver):
    calls = []
    def probe(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=driver + "\n")
    monkeypatch.setattr("ops.datasets.observation_render.subprocess.run", probe)
    with pytest.raises(RuntimeError, match="cluster administrator"):
        check_renderer_driver("/envs/isaacsim-5.1.0_isaaclab-2.3.2_py311")
    assert calls[0][0] == ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    assert calls[0][1]["timeout"] == 10


@pytest.mark.parametrize("driver", ["580.65.06", "595.90", "600.10"])
def test_other_driver_versions_proceed_to_actual_validation(monkeypatch, driver):
    monkeypatch.setattr("ops.datasets.observation_render.subprocess.run", lambda *a, **k: SimpleNamespace(stdout=driver+"\n"))
    check_renderer_driver("/envs/isaacsim-5.1.0_isaaclab-2.3.2_py311")


def test_other_isaac_versions_are_not_blocked_by_51_crash(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("This check is only for the evidenced Isaac 5.1 runtime")
    monkeypatch.setattr("ops.datasets.observation_render.subprocess.run", unexpected)
    check_renderer_driver("/envs/isaacsim-5.0.0_isaaclab-2.2.0_py311")
    check_renderer_driver("/envs/isaacsim-6.0.0_py311")
