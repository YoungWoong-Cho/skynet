"""Recorded DP alignment, provenance and shared submission/evaluation integration."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from skynet_app.adapters.dp_data import CAMERAS, horizon_for, window_indices, validate_joint_manifest, training_normalizer
from skynet_app.adapters.dp_runtime import REVISION, SCHEMA, verify_checkpoint_identity
from skynet_app.adapters.dp_manifest import manifest
from skynet_app.adapters.policy_contract import JOINT_SEMANTICS


def joint_manifest():
    capture=dict(robot="wuji2",hand="right",source_revision="30cc673e27684b9f10186fa6bea731aed246bc9f",step_dt=1/60,
        action_joint_names=["wrist_x","finger"],action_scale=[1.,.5],action_offset=[0.,.2],
        action_semantics=JOINT_SEMANTICS,color_space="RGB",
        cameras={'scene_front': {'height': 256, 'intrinsic_matrix': [[293.19970703125, 0.0, 128.0], [0.0, 293.19970703125, 128.0], [0.0, 0.0, 1.0]], 'mount': 'fixed_scene', 'position_world': [-1.5, 0.0, 1.5], 'quaternion_world_ros': [-0.4048279821872711, 0.5797538161277771, -0.5797535181045532, 0.4048279821872711], 'sensor': 'third_person_camera', 'width': 256}, 'scene_left': {'height': 256, 'intrinsic_matrix': [[293.19970703125, 0.0, 128.0], [0.0, 293.19970703125, 128.0], [0.0, 0.0, 1.0]], 'mount': 'fixed_scene', 'position_world': [0.0, 1.5, 1.5], 'quaternion_world_ros': [8.873286105881562e-09, 4.691093025144255e-08, 0.870199978351593, -0.4926987588405609], 'sensor': 'third_person_camera_left', 'width': 256}, 'scene_right': {'height': 256, 'intrinsic_matrix': [[293.19970703125, 0.0, 128.0], [0.0, 293.19970703125, 128.0], [0.0, 0.0, 1.0]], 'mount': 'fixed_scene', 'position_world': [0.0, -1.5, 1.5], 'quaternion_world_ros': [-0.4926987290382385, 0.870199978351593, -3.834902173593946e-08, 1.6812473191407662e-08], 'sensor': 'third_person_camera_right', 'width': 256}})
    ep=dict(steps=3,capture=deepcopy(capture),policy_to_source_indices=[0,1],streams={
        **{s:dict(shape=[3,2],dtype="float32") for s in ("state","action")},
        **{c:dict(shape=[3,256,256,3],dtype="uint8") for c in CAMERAS}})
    return dict(contract="skynet.hat-rgb-fingertips/v1",capture=capture,policy_to_source_indices=[0,1],episodes=[ep])


@pytest.mark.parametrize("step",[0,1,4,9])
def test_policy_action_starts_at_current_frame_without_future_observations(step):
    horizon=horizon_for(50,2)
    obs,actions=window_indices(step,10,2,horizon)
    assert horizon==52 and len(obs)==2
    assert obs[-1]==step and max(obs)<=step
    assert actions[2-1]==step  # Official DP predict_action slicing offset.
    assert min(actions)>=0 and max(actions)<10
    np.testing.assert_array_equal(actions[1:51],np.minimum(np.arange(step,step+50),9))


def test_history_and_last_action_repeat_only_within_same_episode():
    obs,act=window_indices(0,1,3,horizon_for(8,3))
    assert obs.tolist()==[0,0,0] and act.tolist()==[0]*12
    with pytest.raises(ValueError):window_indices(10,10,2,52)
    with pytest.raises(ValueError):horizon_for(0,2)


@pytest.mark.parametrize("mutation",["hand","order","camera","dimension","dtype"])
def test_rejects_mixed_hand_or_corrupt_stream_contract(mutation):
    data=joint_manifest()
    assert validate_joint_manifest(data)==2
    ep=data["episodes"][0]
    if mutation=="hand":ep["capture"]["hand"]="left"
    if mutation=="order":ep["policy_to_source_indices"]=[1,0]
    if mutation=="camera":ep["capture"]["cameras"][CAMERAS[0]]["width"]=128
    if mutation=="dimension":ep["streams"]["action"]["shape"]=[3,3]
    if mutation=="dtype":ep["streams"][CAMERAS[0]]["dtype"]="float32"
    with pytest.raises(ValueError):validate_joint_manifest(data)


def test_normalizer_reads_training_episodes_only(monkeypatch):
    fitted={}
    class Normalizer(dict):
        def fit(self,data,**kwargs):fitted.update(data)
    fake=types.ModuleType("diffusion_policy.model.common.normalizer")
    fake.LinearNormalizer=Normalizer
    fake.SingleFieldLinearNormalizer=types.SimpleNamespace(create_identity=lambda: "identity")
    monkeypatch.setitem(sys.modules,fake.__name__,fake)
    class Recordings:
        manifest={"split":{"train":[0],"validation":[1]}}
        def episode(self,index):
            assert index==0,"Validation must never enter normalization"
            return types.SimpleNamespace(joint=lambda source:np.array([[1.,2.],[3.,4.]],np.float32))
    value=training_normalizer(Recordings())
    assert fitted["agent_pos"].max()==4 and fitted["action"].shape==(2,2)
    assert all(value[c]=="identity" for c in CAMERAS)


@pytest.mark.parametrize("key",["manifest_sha256","repository_revision","action_steps","observation_steps","control_hz"])
def test_checkpoint_cannot_silently_change_dataset_architecture_or_frequency(key):
    payload=dict(schema=SCHEMA,repository_revision=REVISION,manifest_sha256="a"*64,
        settings=dict(action_steps=50,observation_steps=2),recording_sampling=dict(control_hz=60))
    config=dict(action_steps=50,observation_steps=2,control_hz=60)
    verify_checkpoint_identity(payload,"a"*64,config)
    if key in payload:payload[key]="b"*64
    elif key in payload["settings"]:payload["settings"][key]+=1
    else:payload["recording_sampling"][key]=30
    with pytest.raises(ValueError):verify_checkpoint_identity(payload,"a"*64,config)


def test_canonical_manifest_supplies_training_and_rollout_capsules():
    from skynet_app.adapters import builtin_adapter_manifests
    from skynet_app.evaluation_compatibility import RECORDED_POLICY_MODELS
    m=manifest()
    assert m.slug=="diffusion-policy"
    assert any(a.slug==m.slug for a in builtin_adapter_manifests())
    assert RECORDED_POLICY_MODELS[m.slug][0]=="diffusion_policy_joints"
    assert m.train.strict_canonical_inputs and m.train.strict_native_config
    assert not m.capabilities.supports_resume
    assert m.train.progress.unit=="step"
    for name in ["dp_runtime.py","dp_data.py","dp_evaluation.py","recording_dataset.py","recording_time.py","policy_contract.py"]:
        compile(m.train.capsule_files["adapter-support/"+name],name,"exec")


def test_evaluation_uses_same_joint_contract_and_existing_simulator():
    from skynet_app.evaluation_compatibility import inspect_compatibility, compose_evaluator
    metadata=joint_manifest()
    metadata.update(format="skynet.recording-dataset/v1",validation={"status":"PASSED"})
    spec=dict(source={"adapter":"diffusion-policy"},native={"config":dict(dataset_path="/prepared",dataset_manifest_sha256="a"*64)},
        data={"bundle":{"assignments":[dict(role="training_data",version={"metadata":metadata})]}})
    suite=dict(evaluator_adapter="isaac_lab",name="dexverse_recorded",config_json={})
    report,_=inspect_compatibility(spec,manifest(),suite)
    assert report["ready"],report["messages"]
    assert report["policy_loader"]=="diffusion_policy_joints"
    assert report["io_contract"]["action_semantics"]==JOINT_SEMANTICS
    entry=compose_evaluator(spec,manifest(),suite).evaluations[0]
    assert entry.runtime_profile_id=="isaacsim-5.1.0_isaaclab-2.3.2_py311"
    for name in ("dp_evaluation.py","dp_runtime.py","recorded_policy_evaluation.py","policy_loading.py","dexverse_evaluation.py","policy_transport.py"):
        assert "adapter-support/"+name in entry.command.capsule_files


def test_inference_observation_history_updates_while_chunk_is_executing(monkeypatch):
    # Frozen worker modules are standalone; avoid importing Torch in API tests.
    from skynet_app.adapters import dp_data,dp_runtime,recording_dataset,recording_time
    for module in (dp_data,dp_runtime,recording_dataset,recording_time):
        monkeypatch.setitem(sys.modules,module.__name__.rsplit(".",1)[1],module)
    path=Path(dp_data.__file__).with_name("dp_evaluation.py")
    spec=importlib.util.spec_from_file_location("dp_eval_test",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from collections import deque
    policy=object.__new__(module.RecordedPolicy)
    policy.torch=None;policy.dimension=2;policy.observation_steps=2;policy.history=deque(maxlen=2)
    for step in range(3):
        obs=dict(state=np.array([step,0.]),images={c:np.zeros((256,256,3),np.uint8) for c in CAMERAS})
        assert policy.step(obs,False) is None
    assert [x["agent_pos"][0] for x in policy.history]==[1,2]


def frozen_dp_spec():
    metadata = joint_manifest()
    metadata.update(format="skynet.recording-dataset/v1", validation={"status":"PASSED"}, registered_version_id="wuji2-data")
    return dict(source={"adapter":"diffusion-policy"}, native={"config":dict(dataset_path="/prepared",dataset_manifest_sha256="a"*64,control_hz=60)},
        data={"bundle":{"assignments":[dict(role="training_data",position=0,
            version=dict(format="skynet.recording-dataset/v1",manifest_sha256="a"*64,metadata=metadata),
            config={"location":dict(kind="cluster",status="AVAILABLE",path="/prepared",manifest_sha256="a"*64)})]}})


def test_dp_freezes_its_training_hand_scene_without_mutating_input(monkeypatch):
    from skynet_app import evaluation_targets
    seen=[]
    monkeypatch.setattr(evaluation_targets.data_selection,"assert_available",lambda db,target:seen.append(target))
    spec=frozen_dp_spec(); before=deepcopy(spec)
    suite=dict(config_json={})
    result=evaluation_targets.attach_evaluation_target(None,suite,spec)
    assert result["config_json"]["target_dataset"]["version_id"]=="wuji2-data"
    assert result["config_json"]["target_dataset"]["manifest_sha256"]=="a"*64
    assert result["config_json"]["unseen_embodiment"] is False
    assert seen and spec==before and suite==dict(config_json={})
    for kwargs in (dict(unseen_embodiment=True),dict(target_dataset_id="different-hand")):
        with pytest.raises(ValueError):evaluation_targets.attach_evaluation_target(None,suite,spec,**kwargs)
    spec["native"]["config"]["dataset_manifest_sha256"]="b"*64
    with pytest.raises(ValueError,match="SHA256"):
        evaluation_targets.attach_evaluation_target(None,suite,spec)


def test_dp_reopens_checksum_pinned_target_and_rejects_changed_identity(monkeypatch):
    from skynet_app.adapters import dp_simulation
    from skynet_app.adapters.dataset_inputs import resolve_data_selections, runtime_evaluation_context
    spec=frozen_dp_spec();target=resolve_data_selections(spec)[0]
    context=dict(policy={"native_config":spec["native"]["config"]},target_dataset=target,
                 compatibility={"policy_loader":"diffusion_policy_joints"})
    calls=[]
    def verify(path,digest):
        calls.append((path,digest));return target["metadata"]
    monkeypatch.setattr(dp_simulation,"verify_dataset",verify)
    projected=runtime_evaluation_context(context)
    assert "streams" not in projected["target_dataset"]["metadata"]["episodes"][0]
    assert dp_simulation.load_target(projected)==target["metadata"]
    assert calls==[("/prepared","a"*64)]
    for key,value in (("path","/other"),("manifest_sha256","b"*64)):
        changed=deepcopy(projected);changed["target_dataset"][key]=value
        with pytest.raises(ValueError,match="SHA256"):dp_simulation.load_target(changed)
    assert len(calls)==1  # Reject a different target before reading it.


def test_dp_capsule_restores_assets_without_cartesian_decoder():
    from skynet_app.evaluation_compatibility import compose_evaluator
    entry=compose_evaluator(frozen_dp_spec(),manifest(),dict(evaluator_adapter="isaac_lab",name="dexverse_recorded",config_json={})).evaluations[0]
    files=entry.command.capsule_files
    for name in ("dp_simulation.py","observation_render.py","observation_geometry.py"):
        compile(files["adapter-support/"+name],name,"exec")
    assert "adapter-support/hat_evaluation.py" not in files
    assert "adapter-support/unidex_evaluation.py" not in files
    assert "evaluation.target_dataset.manifest_sha256" in entry.command.required_values


def test_dp_camera_configuration_and_live_mismatch_are_checked(monkeypatch):
    from skynet_app.adapters import dp_simulation
    from skynet_app.adapters.policy_contract import recorded_contract
    from ops.datasets import observation_render, observation_geometry
    monkeypatch.setitem(sys.modules,"observation_render",observation_render)
    monkeypatch.setitem(sys.modules,"observation_geometry",observation_geometry)
    contract=recorded_contract(joint_manifest())
    jobs=[]
    monkeypatch.setattr(observation_render,"configure_cameras",lambda cfg,values,**kw:jobs.extend(values))
    assert set(dp_simulation.configure_scene(None,contract))==set(CAMERAS)
    assert len(jobs)==3 and all(j["modality"]=="rgb" for j in jobs)
    monkeypatch.setattr(observation_render,"capture_current_frame",lambda env:None)
    frame=dict(world_from_camera=np.eye(4),intrinsics=np.eye(3),rgb=np.zeros((256,256,3),np.uint8))
    monkeypatch.setattr(observation_render,"read_camera",lambda *a,**kw:frame)
    env=types.SimpleNamespace(scene={c["sensor"]:object() for c in contract["cameras"].values()},sim=types.SimpleNamespace(render=lambda:None,render_mode=0))
    with pytest.raises(ValueError,match="camera pose"):
        dp_simulation.observation_from_sensors(env,contract,[0,1])
    frame["world_from_camera"]=observation_geometry.pose_from_ros(contract["cameras"]["scene_front"]["position_world"],contract["cameras"]["scene_front"]["quaternion_world_ros"])
    with pytest.raises(ValueError,match="intrinsics"):
        dp_simulation.observation_from_sensors(env,contract,[0,1])
