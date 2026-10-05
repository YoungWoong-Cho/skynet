"""Official DP inference attached to the existing recorded simulator transport."""
from collections import deque
import numpy as np
from dp_data import CAMERAS, IMAGE_SIZE, validate_joint_manifest
from dp_runtime import build_policy, verify_checkpoint_identity
from recording_dataset import digest
from recording_time import resolve_sampling


class RecordedPolicy:
    def __init__(self, context, source_dir, manifest):
        import torch
        self.torch=torch;self.manifest=manifest;self.mode="rgb"
        self.dimension=validate_joint_manifest(manifest)
        checkpoint=context["checkpoint"]
        if digest(checkpoint["path"])!=checkpoint["sha256"]:
            raise ValueError("DP checkpoint changed since registration")
        config=context["policy"]["native_config"]
        payload=torch.load(checkpoint["path"],map_location="cpu",weights_only=False)
        verify_checkpoint_identity(payload,config["dataset_manifest_sha256"],config)
        if payload["dimension"]!=self.dimension:raise ValueError("DP checkpoint joint dimension differs")
        self.action_steps=payload["settings"]["action_steps"]
        self.observation_steps=payload["settings"]["observation_steps"]
        sampling=resolve_sampling(manifest,config.get("control_hz"),action_steps=self.action_steps,window_policy="pad",require_validation=True)
        if sampling!=payload["recording_sampling"]:raise ValueError("DP checkpoint sampling differs from evaluation")
        self.control_hz=sampling["control_hz"]
        self.model=build_policy(source_dir,self.dimension,self.action_steps,self.observation_steps)
        self.model.load_state_dict(payload["model"],strict=True);self.model.cuda().eval()
        self.history=deque(maxlen=self.observation_steps)

    def reset(self,seed):
        self.torch.manual_seed(seed);np.random.seed(seed);self.history.clear();self.model.reset()

    def step(self,observation,predict):
        import cv2
        torch=self.torch
        state=np.asarray(observation["state"],dtype=np.float32)
        if state.shape!=(self.dimension,) or not np.isfinite(state).all():
            raise ValueError("Simulator DP joint state is invalid")
        sample={"agent_pos":state.copy()}
        for name in CAMERAS:
            image=np.asarray(observation["images"][name])
            if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[-1]!=3:
                raise ValueError("Simulator DP image is not RGB uint8")
            sample[name]=np.moveaxis(cv2.resize(image,IMAGE_SIZE),-1,0).astype(np.float32)/255
        self.history.append(sample)
        while len(self.history)<self.observation_steps:self.history.appendleft(sample)
        if not predict:return None
        obs={k:torch.as_tensor(np.stack([x[k] for x in self.history]),device="cuda")[None] for k in sample}
        with torch.inference_mode():result=self.model.predict_action(obs)["action"][0].cpu().numpy()
        if result.shape!=(self.action_steps,self.dimension) or not np.isfinite(result).all():
            raise ValueError("DP generated invalid joint action chunks")
        return result
