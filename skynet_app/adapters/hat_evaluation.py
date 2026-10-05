"""Skynet HAT rollout bridge; original HAT inference, existing Dex retargeting.

The trained 128-D Cartesian policy stays unchanged. Skynet uses the exact
target hand asset, recorded camera frame, metric fingertips, and native joint limits.
The existing dex_retargeting PositionOptimizer is used at scale 1, without a
human hand scale or temporal regularizer. Whole saved chunks execute at the
saved control frequency; no temporal ensemble or chunk interpolation is added.
"""
import hashlib
import math
from pathlib import Path
import random
import numpy as np

try:
    from .hat_data import CAMERAS, CONTRACT, encode_human_state, fingertip_slots
    from .policy_contract import recorded_contract, validate_observation
    from ops.datasets.action_codecs.unidex import load_codec, encode_absolute, anchor_faas_actions
    from ops.datasets.action_codecs.geometry import inverse_transform
    from ops.datasets.observation_geometry import pose_from_ros
except ImportError:
    from hat_data import CAMERAS, CONTRACT, encode_human_state, fingertip_slots
    from policy_contract import recorded_contract, validate_observation
    from action_codecs.unidex import load_codec, encode_absolute, anchor_faas_actions
    from action_codecs.geometry import inverse_transform
    from observation_geometry import pose_from_ros


def hat_contract(manifest, *, control_hz=None):
    if manifest.get('contract') != CONTRACT:
        raise ValueError('HAT rollout requires its frozen RGB/fingertip dataset')
    capture = manifest['capture']
    codec = load_codec(capture['robot'], capture)
    fingertip_slots(codec)
    contract = recorded_contract(manifest, control_hz=control_hz)
    if contract['policy_to_source_indices'] != list(range(codec.action_dim)):
        raise ValueError('HAT rollout requires the verified target hand native order')
    for episode in manifest['episodes']:
        load_codec(codec.robot_id, episode['capture'])
        if episode['capture'].get('cameras') != capture['cameras']:
            raise ValueError('HAT rollout cameras must be identical across episodes')
        rep = episode['action_representation']
        if rep.get('codec_sha256') != codec.digest or rep.get('frame') != 'camera_opengl':
            raise ValueError('HAT rollout requires the trained camera-frame geometry')
    for camera in contract['cameras'].values():
        if camera.get('mount') != 'fixed_scene' or (camera['height'],camera['width']) != (256,256):
            raise ValueError('HAT rollout requires the three recorded fixed RGB cameras')
    contract.update(observation_mode='rgb', codec_sha256=codec.digest,
        state_representation='hat128_camera_opengl_wrist_metric_local_tips',
        action_representation='hat128_to_native_position_optimizer',
        retargeting=dict(package='dex-retargeting',version='0.4.6',optimizer='PositionOptimizer',
            scale=1.0,norm_delta=0.0,huber_delta=0.02,ftol_abs=1e-5,limit_epsilon=0.0))
    return contract


def camera_from_root(observation):
    return np.diag([1.,-1.,-1.,1.]) @ inverse_transform(observation['world_from_camera']) @ observation['world_from_root']


class RecordedPolicy:
    mode = 'rgb'

    def __init__(self, context, source_dir, manifest):
        import torch
        try:
            from .hat_runtime import build_policy, REVISION, DINO_REVISION, DEFAULT_MODEL_CONFIG
        except ImportError:
            from hat_runtime import build_policy, REVISION, DINO_REVISION, DEFAULT_MODEL_CONFIG
        expected = context['compatibility']['io_contract']
        self.contract = hat_contract(manifest, control_hz=1 / expected['step_dt'])
        if expected != self.contract:
            raise ValueError('HAT rollout contract changed')
        path = Path(context['checkpoint']['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != context['checkpoint']['sha256']:
            raise ValueError('HAT checkpoint checksum changed')
        saved = torch.load(path,map_location='cpu',weights_only=False)
        config = context['policy']['native_config']
        if (saved.get('schema') != 'skynet.hat-checkpoint/v1'
                or saved.get('repository_revision') != REVISION or saved.get('dinov2_revision') != DINO_REVISION
                or not checkpoint_inputs_match(saved, context)):
            raise ValueError('HAT checkpoint source or training data differs')
        self.horizon = int(saved['settings']['action_steps'])
        self.control_hz = float(saved['recording_sampling']['control_hz'])
        if self.horizon != config['action_steps'] or not math.isclose(self.control_hz,1/expected['step_dt']):
            raise ValueError('HAT saved sampling differs from rollout')
        model_config = saved['settings'].get('model_config', DEFAULT_MODEL_CONFIG)
        self.policy, settings = build_policy(source_dir,self.horizon,saved['settings']['learning_rate'],model_config)
        if settings != saved['native_model_config']:
            raise ValueError('HAT model settings differ from the saved checkpoint')
        self.policy.load_state_dict(saved['model'],strict=True)
        self.policy.cuda().eval()
        self.stats = {k:np.asarray(v,dtype=np.float32) for k,v in saved['normalization'].items()}
        for k in ('state_mean','state_std','action_mean','action_std'):
            v=self.stats[k]
            if v.shape!=(128,) or not np.isfinite(v).all() or (k.endswith('std') and (v<=0).any()):
                raise ValueError('Invalid HAT normalization')
        self.codec=load_codec(self.contract['robot'],manifest['capture'])

    def reset(self,seed):
        import torch
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

    def step(self,observation,predict):
        if not predict:return None
        import cv2
        import torch
        validate_observation(self.contract,observation)
        q=np.asarray(observation['state'])
        faas=encode_absolute(self.codec,q,camera_from_root=camera_from_root(observation))
        state=encode_human_state(self.codec,q,faas)
        state=(state-self.stats['state_mean'])/(self.stats['state_std']+1e-6)
        images=np.stack([cv2.resize(observation['images'][c],(320,240),interpolation=cv2.INTER_LINEAR).transpose(2,0,1) for c in CAMERAS])
        with torch.inference_mode():
            action=self.policy(torch.from_numpy(images.copy()).float().cuda()[None]/255,
                torch.from_numpy(state).cuda()[None])[0].cpu().numpy()
        if action.shape!=(self.horizon,128) or not np.isfinite(action).all():
            raise ValueError('Invalid HAT prior action chunk')
        return action*(self.stats['action_std']+1e-6)+self.stats['action_mean']


class ActionDecoder:
    """Use installed Dex retargeting with the exact recorded target hand kinematics."""
    def __init__(self,capture,urdf_path):
        from importlib.metadata import version
        from dex_retargeting.robot_wrapper import RobotWrapper
        from dex_retargeting.optimizer import PositionOptimizer
        if version('dex-retargeting')!='0.4.6':raise ValueError('Unverified dex-retargeting version')
        self.codec=load_codec(capture['robot'],capture)
        self.codec.spec.verify_runtime_asset(urdf_path)
        self.robot=RobotWrapper(str(urdf_path))
        mimic = {name: joint.mimic for name, joint in self.codec.spec.joints.items() if joint.mimic}
        if set(self.robot.dof_joint_names)!=set(self.codec.action_names) | set(mimic):
            raise ValueError('Retargeting URDF joint identities differ')
        names=[name for name in self.codec.action_names if name not in self.codec.spec.data['wrist_names']]
        self.tip_slots=fingertip_slots(self.codec)
        self.finger_indices=[self.codec.action_names.index(n) for n in names]
        self.solver=PositionOptimizer(self.robot,names,list(self.codec.spec.tip_links),np.arange(len(self.tip_slots)),norm_delta=0.0)
        if mimic:
            from dex_retargeting.kinematics_adaptor import MimicJointKinematicAdaptor
            sources, multipliers, offsets = [], [], []
            for name in mimic:
                source, multiplier, offset = name, 1., 0.
                while source in mimic:
                    relation = mimic[source]
                    offset += multiplier * relation['offset']
                    multiplier *= relation['multiplier']
                    source = relation['joint']
                if source not in names:
                    raise ValueError('Mimic joint must derive from a controlled finger joint')
                sources.append(source); multipliers.append(multiplier); offsets.append(offset)
            self.solver.set_kinematic_adaptor(MimicJointKinematicAdaptor(
                self.robot, names, sources, list(mimic), multipliers, offsets))
        limits=np.stack([self.codec.spec.effective_lower,self.codec.spec.effective_upper],axis=1)[self.finger_indices]
        self.solver.set_joint_limit(limits,epsilon=0.0)
        self.limits=limits
        self.zero_wrist=self.codec.wrist_pose(np.zeros(self.codec.action_dim))
        self.receipt=dict(optimizer='dex_retargeting.PositionOptimizer',version=version('dex-retargeting'),
            codec_sha256=self.codec.digest,scale=1.0,norm_delta=0.0,limit_epsilon=0.0)
        # Independently compare installed Pinocchio FK against Skynet's verified FK.
        probe=np.clip(np.zeros(self.codec.action_dim),self.codec.spec.effective_lower,self.codec.spec.effective_upper)
        expanded=self.codec.spec.expand(probe)
        self.robot.compute_forward_kinematics(np.array([expanded[n] for n in self.robot.dof_joint_names]))
        expected=self.codec.spec.forward_kinematics(probe)
        for name in self.codec.spec.tip_links:
            if not np.allclose(self.robot.get_link_pose(self.robot.get_link_index(name)),expected[name],atol=1e-6):
                raise ValueError('Installed retargeting FK differs from the verified target hand asset')

    def decode(self,chunk,observation):
        import torch
        values=np.asarray(chunk)
        if values.ndim!=2 or values.shape[1]!=128 or not np.isfinite(values).all():
            raise ValueError('Expected finite HAT128 action chunk')
        measured=np.asarray(observation['state'],dtype=float)
        last=np.clip(measured[self.finger_indices],self.limits[:,0],self.limits[:,1])
        targets=[]
        for value in values:
            local=value[43:58].reshape(5,3)[self.tip_slots]
            root_tips=local @ self.zero_wrist[:3,:3].T+self.zero_wrist[:3,3]
            # Wrist joints remain fixed at zero for the local finger solve.
            objective=self.solver.get_objective_function(root_tips,np.zeros(len(self.solver.idx_pin2fixed)),last)
            self.solver.opt.set_min_objective(objective)
            with torch.enable_grad():
                last=np.asarray(self.solver.opt.optimize(last),dtype=float)
            if not np.isfinite(last).all():raise ValueError('Nonfinite finger retargeting result')
            native=np.zeros(self.codec.action_dim);native[self.finger_indices]=last
            encoded=encode_absolute(self.codec,native,camera_from_root=camera_from_root(observation))
            encoded[:9]=value[30:39]
            targets.append(encoded)
        anchor=encode_absolute(self.codec,measured,camera_from_root=camera_from_root(observation))
        # Reuse existing row-6D, camera/root and controller scale/offset decoding.
        relative=anchor_faas_actions(anchor,np.asarray(targets))
        return self.codec.decode_actions(relative,measured,clamp=True).astype(np.float32)


def configure_scene(cfg,contract):
    from observation_render import configure_cameras,camera_recipe_from_capture
    jobs=[]
    for name,capture in contract['cameras'].items():
        camera=camera_recipe_from_capture(capture,contract['source_revision'],camera_id=name)
        jobs.append(dict(camera_id=name,modality='rgb',recipe=dict(camera=camera,width=camera['width'],height=camera['height'])))
    configure_cameras(cfg,jobs,replay=False)
    return {name:c['sensor'] for name,c in contract['cameras'].items()}


def observation_from_sensors(env,contract,joint_ids):
    from observation_render import read_camera,capture_current_frame
    capture_current_frame(env)
    images={};front=None
    for name,capture in contract['cameras'].items():
        view=read_camera(env.scene[capture['sensor']],env.sim.render,('rgb',),render_mode=env.sim.render_mode)
        if not np.allclose(view['world_from_camera'],pose_from_ros(capture['position_world'],capture['quaternion_world_ros']),atol=1e-5):
            raise ValueError('Live HAT camera pose differs from the recording')
        if not np.allclose(view['intrinsics'],capture['intrinsic_matrix'],atol=1e-5):
            raise ValueError('Live HAT camera intrinsics differ from the recording')
        images[name]=view['rgb']
        if name=='scene_front':front=view['world_from_camera']
    robot=env.scene['robot']
    return dict(state=robot.data.joint_pos[0,joint_ids].detach().cpu().numpy().copy(),images=images,
        world_from_camera=front,world_from_root=pose_from_ros(robot.data.root_pos_w[0].detach().cpu().numpy(),robot.data.root_quat_w[0].detach().cpu().numpy()))


def checkpoint_inputs_match(saved, context):
    """The evaluation target must never substitute for checkpoint training inputs."""
    config = context['policy']['native_config']
    if saved.get('training_inputs') is None:
        return saved.get('manifest_sha256') == config.get('dataset_manifest_sha256')
    expected = config.get('datasets')
    if not isinstance(expected, list) or not expected:
        return False
    identity = [{k: row.get(k) for k in ('position', 'version_id', 'manifest_sha256')} for row in expected]
    return saved['training_inputs'] == identity
