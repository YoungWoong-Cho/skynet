"""Isaac render and frozen-hand helpers shared by recorded-simulator rollouts."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

from ops.datasets import observation_render as render


def test_shared_render_scheduler_does_not_advance_physics(monkeypatch):
    calls, time = [], [4.]
    orchestrator = NS(step=lambda **kw: calls.append(kw))
    timeline = NS(get_timeline_interface=lambda: NS(get_current_time=lambda: time[0]))
    replicator = NS(core=NS(orchestrator=orchestrator))
    monkeypatch.setitem(sys.modules, "omni", NS(replicator=replicator, timeline=timeline))
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", replicator.core)
    monkeypatch.setitem(sys.modules, "omni.timeline", timeline)
    env = NS(physics_dt=1 / 120, scene=NS(update=lambda **kw: calls.append(kw)))
    render.capture_current_frame(env)
    assert calls == [dict(rt_subframes=4, pause_timeline=False, delta_time=0., wait_for_render=True), dict(dt=1 / 120)]
    orchestrator.step = lambda **kw: time.__setitem__(0, time[0] + .01)
    with pytest.raises(RuntimeError, match="advanced the simulation timeline"):
        render.capture_current_frame(env)


def test_frozen_hand_install_uses_private_verified_copy(monkeypatch, tmp_path):
    original = tmp_path / "source"
    original.mkdir()
    contents = b"frozen asset"
    (original / "simulation.urdf").write_bytes(contents)
    manifest = dict(digest="hand-id", files={"simulation.urdf": dict(sha256=hashlib.sha256(contents).hexdigest(), size_bytes=len(contents))})
    (original / "manifest.json").write_text(json.dumps(manifest))
    profile = dict(hand_bundle=dict(root=str(original), digest="hand-id", manifest_sha256=hashlib.sha256((original / "manifest.json").read_bytes()).hexdigest()))
    installations = []
    monkeypatch.setitem(sys.modules, "runtime", NS(install=lambda p: installations.append(p) or manifest, validate_environment=lambda *_: None))
    receipts = {}
    work, installed, validate = render.install_frozen_hand(profile, staging_root=str(tmp_path / "private"), asset_receipts=receipts)
    try:
        assert installations == [Path(work.name)] and installed == manifest
        assert Path(work.name) != original and (Path(work.name) / "simulation.urdf").read_bytes() == contents
        (Path(work.name) / "generated.usd").write_bytes(b"generated")
        assert not (original / "generated.usd").exists()
        assert str(Path(work.name) / "simulation.urdf") in receipts
    finally:
        work.cleanup()
    (original / "simulation.urdf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen hand asset changed"):
        render.install_frozen_hand(profile, staging_root=str(tmp_path / "private"))
