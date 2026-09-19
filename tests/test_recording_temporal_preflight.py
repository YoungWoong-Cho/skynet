import copy
import sys
from types import SimpleNamespace

import pytest

from ops.datasets.recording_probe import probe, validate_native_split, validate_temporal

TEMPORAL = dict(source_fps=60, control_hz=15, frame_stride=4, action_horizon=30, execution_horizon=1)
CONTRACT = "skynet.unidex-pointcloud-faas/v1"


def inspected(count=117, dt=1/60):
    return dict(capture={"step_dt": dt}, steps=count, streams={})


def test_short_real_inspire_length_fails_before_camera_work_and_boundary_passes():
    source = dict(sha256="a"*64)
    with pytest.raises(ValueError, match="116 frames.*at least 117.*No camera work"):
        validate_temporal(TEMPORAL, source, inspected(116))
    validate_temporal(TEMPORAL, source, inspected(117))
    with pytest.raises(ValueError, match="recorded control timing"):
        validate_temporal(TEMPORAL, source, inspected(145, 1/30))
    validate_temporal(None, source, inspected(1, 1/30))


@pytest.mark.parametrize("split", [None, {"train":[0,1],"validation":[]}, {"train":[0],"validation":[0]}, {"train":[0],"validation":[2]}])
def test_unidex_episode_split_rejects_missing_empty_leaking_or_outside_members(split):
    with pytest.raises(ValueError, match="separate nonempty"):
        validate_native_split({"contract":CONTRACT}, split, [{"sha256":"a"*64},{"sha256":"b"*64}])


def test_split_guard_keeps_training_only_other_adapters_and_rejects_duplicate_sources():
    sources = [{"sha256":"a"*64},{"sha256":"b"*64}]
    split = dict(train=[0],validation=[1])
    validate_native_split({"contract":CONTRACT},split,sources)
    validate_native_split({"contract":"skynet.egoverse-rgb-joints/v1"},{"train":[0],"validation":[]},sources[:1])
    with pytest.raises(ValueError, match="copies"):
        validate_native_split({"contract":CONTRACT},split,[sources[0],sources[0]])


def test_cpu_probe_enforces_temporal_before_success_receipt(monkeypatch):
    monkeypatch.setitem(sys.modules,"recording_prepare",SimpleNamespace(preflight_source=lambda *a,**kw:inspected(116)))
    request=dict(job_id="test",attempt_id="one",sources=[dict(sha256="a"*64)],split=dict(train=[0],validation=[]),
                 requirements=dict(contract=CONTRACT,observation_requirements={"streams":[]},temporal=TEMPORAL))
    with pytest.raises(ValueError,match="at least 117"):
        probe(request)
    monkeypatch.setitem(sys.modules,"recording_prepare",SimpleNamespace(preflight_source=lambda *a,**kw:inspected(117)))
    with pytest.raises(ValueError,match="separate nonempty"):
        probe(request)
    request["sources"].append(dict(sha256="b"*64));request["split"]=dict(train=[0],validation=[1])
    assert probe(request)["verified"] is True
