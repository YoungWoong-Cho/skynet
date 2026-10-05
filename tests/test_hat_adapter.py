"""HAT integration: physical geometry, train-only statistics and native provenance."""
import hashlib
import json
from copy import deepcopy

import h5py
import numpy as np
import pytest

from skynet_app.adapters.hat_data import CAMERAS, CONTRACT, encode_human_state, prepare_data
from skynet_app.adapters.hat_manifest import manifest, REVISION
from ops.datasets.action_codecs.unidex import load_codec


def test_hat_tips_are_palm_relative_and_wrist_keeps_absolute_camera_pose():
    codec = load_codec("skynet_wuji_2_right")
    q = np.zeros(codec.action_dim)
    first, _ = codec.encode_state(q)
    a = encode_human_state(codec, q, first)
    q[:6] = [.5, -.2, .3, .2, -.3, .4]
    second, _ = codec.encode_state(q)
    b = encode_human_state(codec, q, second)
    np.testing.assert_allclose(a[43:58], b[43:58], atol=1e-7)
    np.testing.assert_allclose(b[30:39], second[:9])
    active = np.r_[30:39,43:58]
    assert np.all(b[np.setdiff1d(np.arange(128),active)] == 0)
    assert a[43:58].reshape(5,3).shape == (5,3)
    q[6] += .5
    third, _ = codec.encode_state(q)
    c = encode_human_state(codec,q,third)
    assert np.linalg.norm(c[43:46]-b[43:46]) > .01
    np.testing.assert_allclose(c[46:58],b[46:58],atol=1e-7)


def recording_fixture(tmp_path, *, validation_offset=10):
    codec = load_codec("skynet_wuji_2_right")
    episodes=[]
    for index, offset in enumerate([0, validation_offset]):
        state = np.zeros((3,codec.action_dim),np.float32)
        state[:,0]=np.array([0,.01,.02])+offset
        action = state.copy()
        absolute_state=np.stack([codec.encode_state(q)[0] for q in state]).astype(np.float32)
        targets=codec.targets(action)
        absolute_action=np.stack([codec.encode_state(q)[0] for q in targets]).astype(np.float32)
        arrays={"state":state,"action":action,"faas_state_absolute":absolute_state,"faas_action_absolute":absolute_action,
                "timestamps":np.arange(3,dtype=np.float64)/60}
        arrays.update({c:np.zeros((3,256,256,3),np.uint8) for c in CAMERAS})
        path=tmp_path / f"episode-{index}.hdf5"
        with h5py.File(path,"w") as file:
            for name,values in arrays.items():file[name]=values
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        streams={name:dict(path=str(path),dataset=name,sha256=sha,shape=list(value.shape),dtype=str(value.dtype)) for name,value in arrays.items()}
        episodes.append(dict(id=f"episode-{index}",index=index,steps=3,source={"sha256":str(index)*64},
            capture=codec.spec.capture,streams=streams,policy_to_source_indices=list(range(codec.action_dim)),
            action_representation=dict(id="skynet.unidex-faas/v1",codec_sha256=codec.digest,frame="camera_opengl",
                                       action_semantics="controller_targets",state_alignment="pre_action")))
    doc=dict(format="skynet.recording-dataset/v1",contract=CONTRACT,episodes=episodes,steps=6,split={"train":[0],"validation":[1]})
    path=tmp_path / "manifest.json";path.write_text(json.dumps(doc))
    return path,hashlib.sha256(path.read_bytes()).hexdigest(),doc


def test_hat_normalization_excludes_validation_and_actions_use_controller_targets(tmp_path):
    path,sha,_=recording_fixture(tmp_path)
    recordings,values,stats,sampling=prepare_data(path.parent,sha,action_steps=5)
    np.testing.assert_allclose(stats["state_mean"],values[0,"state"].mean(0),atol=1e-7)
    assert stats["state_mean"][30] < .1
    assert values[1,"state"][0,30] > 9
    np.testing.assert_allclose(values[0,"action"][:,30]-values[0,"state"][:,30],.5,atol=1e-6)
    assert sampling["control_hz"] == 60
    assert sampling["splits"]["train"]["windows"] == 3
    np.testing.assert_allclose(stats["state_std"][30],.01,atol=1e-7)
    assert np.all(stats["state_std"] >= .01)


def test_hat_rejects_stale_geometry_and_inexact_frequency(tmp_path):
    path,sha,doc=recording_fixture(tmp_path)
    with pytest.raises(ValueError,match="divide"):
        prepare_data(path.parent,sha,control_hz=17)
    doc["episodes"][0]["action_representation"]["codec_sha256"]="f"*64
    path.write_text(json.dumps(doc));sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match="checksum-pinned"):
        prepare_data(path.parent,sha)


def test_hat_manifest_reuses_shared_capsules_and_labels_native_vs_skynet():
    m=manifest()
    assert m.slug == "human-policy-hat"
    assert not m.capabilities.supports_resume
    assert not m.capabilities.supports_multi_gpu_single_node
    assert not m.evaluations
    assert m.train.progress.unit == "step"
    assert "train.max_steps" in m.train.supported_canonical_fields
    assert m.train.data_requirements.recording_conversion.presets[0].training_setup["revision"] == REVISION
    assert "Original" in m.description and "Skynet" in m.description
    for name in ("recording_dataset.py","recording_time.py","training_parallel.py","action_codecs/unidex.py","observation_geometry.py"):
        assert "adapter-support/"+name in m.train.capsule_files


def test_hat_best_checkpoint_is_accepted_by_existing_slurm_selector(tmp_path):
    from skynet_app.adapters.hat_runtime import write_checkpoint_score
    from skynet_app.slurm import RUNNER_SOURCE
    namespace = {"__name__": "skynet_runtime_test"}
    exec(compile(RUNNER_SOURCE, "runtime-wrapper.py", "exec"), namespace)
    best = tmp_path / "best.ckpt"
    best.write_bytes(b"retained-policy-weights")
    write_checkpoint_score(best, 200, {"val_loss": .25})
    selected, receipt = namespace["_select_checkpoint"]([best], "best", final=True)
    assert selected == best
    assert receipt["actual"] == "best"
    assert receipt["score"] == .25
    assert receipt["mode"] == "min"


def test_hat_model_config_names_an_official_config_and_defaults_to_hat_linear(monkeypatch):
    from skynet_app.adapters import hat_runtime
    assert hat_runtime.DEFAULT_MODEL_CONFIG == "hat_linear"
    assert hat_runtime.MODEL_CONFIGS == {"hat_linear": "hdt/configs/models/hat_linear.yaml",
                                         "act_resnet": "hdt/configs/models/act_resnet.yaml"}
    assert set(hat_runtime.ORIGINAL_MODELS) == set(hat_runtime.MODEL_CONFIGS)
    base = ["hat_runtime.py", "--repository", "source", "--output", "out", "--data-spec", "spec.json"]
    monkeypatch.setattr("sys.argv", base)
    assert hat_runtime.arguments().model_config == "hat_linear"
    monkeypatch.setattr("sys.argv", base + ["--model-config", "act_resnet"])
    assert hat_runtime.arguments().model_config == "act_resnet"
    monkeypatch.setattr("sys.argv", base + ["--model-config", "resnet50"])
    with pytest.raises(SystemExit):
        hat_runtime.arguments()
    with pytest.raises(ValueError, match="Unknown official HAT model config"):
        hat_runtime.build_policy("unused", 50, 1e-4, "resnet50")


def test_hat_resnet_preset_only_swaps_the_official_vision_config():
    from skynet_app.adapters import _apply_parameter_flags
    m = manifest()
    presets = {preset.id: preset for preset in m.train.presets}
    fit, resnet = presets["hat-fit/v1"], presets["hat-fit-resnet/v1"]
    assert fit.values["native.config.model_config"] == "hat_linear"
    assert presets["hat-seven-hands-main/v1"].values["native.config.model_config"] == "hat_linear"
    assert resnet.values["native.config.model_config"] == "act_resnet"
    unchanged = lambda values: {k: v for k, v in values.items() if k != "native.config.model_config"}
    assert unchanged(resnet.values) == unchanged(fit.values)
    assert resnet.source.endswith(f"/{REVISION}/hdt/configs/models/act_resnet.yaml")
    field = next(field for field in m.train.input_fields if field.path == "native.config.model_config")
    assert field.default == "hat_linear" and field.choices == ["hat_linear", "act_resnet"]
    for value in ("hat_linear", "act_resnet"):
        argv, blockers = [], []
        _apply_parameter_flags(argv, {"native": {"config": {"model_config": value}}},
                               {"native.config.model_config": m.train.parameter_flags["native.config.model_config"]}, blockers)
        assert argv == ["--model-config", value] and not blockers
