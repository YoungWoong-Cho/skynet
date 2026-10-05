"""Official Diffusion Policy U-Net/loss/ResNet/EMA with Skynet I/O and step budget.

Architecture and optimizer settings follow the pinned upstream image recipe.
Skynet sets recorded RGB/joints, image dimensions, action horizon, fixed episode
split, training-only normalization, validation checkpoint selection and logging.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import time

REVISION = "5ba07ac6661db573af695b419a7947ecb704690f"
SCHEMA = "skynet.diffusion-policy-checkpoint/v1"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def repository(path):
    root = Path(path).resolve()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if actual != REVISION or dirty:
        raise ValueError("DP requires the clean pinned official source revision")
    if not (root / "diffusion_policy/policy/diffusion_unet_image_policy.py").is_file():
        raise ValueError("Official DP image policy is missing")
    sys.path.insert(0, str(root))
    return root


def build_policy(root, dimension, action_steps=50, observation_steps=2):
    from dp_data import CAMERAS, IMAGE_SIZE, horizon_for
    repository(root)
    from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
    from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
    from diffusion_policy.model.vision.model_getter import get_resnet
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    shape = {"action": {"shape": [dimension]}, "obs": {
        "agent_pos": {"shape": [dimension], "type": "low_dim"},
        **{c: {"shape": [3, IMAGE_SIZE[1], IMAGE_SIZE[0]], "type": "rgb"} for c in CAMERAS}}}
    encoder = MultiImageObsEncoder(shape_meta=shape, rgb_model=get_resnet("resnet18", weights=None),
        crop_shape=None, use_group_norm=True, share_rgb_model=False, imagenet_norm=True)
    scheduler = DDPMScheduler(num_train_timesteps=100, beta_start=.0001, beta_end=.02,
        beta_schedule="squaredcos_cap_v2", variance_type="fixed_small", clip_sample=True, prediction_type="epsilon")
    return DiffusionUnetImagePolicy(shape_meta=shape, noise_scheduler=scheduler, obs_encoder=encoder,
        horizon=horizon_for(action_steps, observation_steps), n_action_steps=action_steps,
        n_obs_steps=observation_steps, num_inference_steps=100, obs_as_global_cond=True,
        diffusion_step_embed_dim=128, down_dims=(512,1024,2048), kernel_size=5, n_groups=8, cond_predict_scale=True)


def verify_checkpoint_identity(payload, manifest_sha, config):
    if payload.get("schema") != SCHEMA or payload.get("repository_revision") != REVISION:
        raise ValueError("Unsupported DP checkpoint source/schema")
    if payload.get("manifest_sha256") != manifest_sha:
        raise ValueError("DP checkpoint does not match the selected dataset")
    for key, default in (("action_steps",50), ("observation_steps",2)):
        if payload["settings"][key] != config.get(key, default):
            raise ValueError("DP checkpoint differs from submitted " + key)
    hz = config.get("control_hz")
    if hz is not None and payload["recording_sampling"]["control_hz"] != hz:
        raise ValueError("DP checkpoint control frequency differs from submission")


def verify_policy_reload(policy, state, observation, *, seed, device, expected_shape):
    """Reload trainable EMA state outside inference mode, then compare predictions.

    Upstream LinearNormalizer recreates parameters in load_state_dict. Loading
    inside inference_mode would make them inference tensors that the next EMA
    update cannot write to, even though both predictions were identical.
    """
    import torch
    device=torch.device(device)
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=="cuda" else []
    with torch.random.fork_rng(devices=devices):
        with torch.inference_mode():
            torch.manual_seed(seed);before=policy.predict_action(observation)["action"]
        policy.load_state_dict(state,strict=True)
        if any(torch.is_inference(p) for p in policy.parameters()):
            raise ValueError("Reload created inference-only training parameters")
        with torch.inference_mode():
            torch.manual_seed(seed);after=policy.predict_action(observation)["action"]
    if tuple(after.shape)!=tuple(expected_shape) or not torch.isfinite(after).all() or not torch.equal(before,after):
        raise ValueError("DP strict checkpoint reload/inference verification failed")
    return dict(shape=list(after.shape),finite=True,reload_max_abs_diff=float((before-after).abs().max()))


def main():
    p = argparse.ArgumentParser()
    for name in ("repository", "dataset", "manifest-sha", "output"):
        p.add_argument("--"+name, required=True)
    for name, default in (("batch-size",16),("max-steps",3000),("num-workers",2),("seed",42),
                           ("action-steps",50),("observation-steps",2),("validation-every",200)):
        p.add_argument("--"+name,type=int,default=default)
    p.add_argument("--control-hz",type=float)
    p.add_argument("--learning-rate",type=float,default=1e-4)
    p.add_argument("--checkpoint-selection",choices=["best","latest"],default="best")
    p.add_argument("--verify-only",action="store_true")
    args=p.parse_args()
    if min(args.batch_size,args.max_steps,args.action_steps,args.observation_steps,args.validation_every)<1 or args.num_workers<0 or args.learning_rate<=0:
        p.error("Invalid DP training settings")
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from dp_data import prepare_data, training_normalizer
    from recording_dataset import close_handles, digest
    from training_parallel import TrainingContext
    root=repository(args.repository)
    recordings,datasets,sampling,dimension=prepare_data(args.dataset,args.manifest_sha,
        control_hz=args.control_hz,action_steps=args.action_steps,observation_steps=args.observation_steps)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    receipt=dict(schema="skynet.dp-loader-validation/v1",manifest_sha256=args.manifest_sha,
        episodes=len(recordings),split=recordings.manifest["split"],dimension=dimension,
        recording_sampling=sampling,observation_steps=args.observation_steps,horizon=datasets["train"].horizon,
        train_windows=len(datasets["train"]),validation_windows=len(datasets["validation"]),
        normalization="training_episodes_only",action_alignment="obs history ends at t; predicted action slice starts at t")
    normalizer=training_normalizer(recordings)
    for data in datasets.values():
        if len(data): data[0]
    close_handles();write_json(output/"dataset-validation.json",receipt)
    if args.verify_only: return
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    context=TrainingContext(1);device=context.device
    policy=build_policy(root,dimension,args.action_steps,args.observation_steps)
    policy.set_normalizer(normalizer)
    policy.to(device)
    ema_policy=copy.deepcopy(policy)
    from diffusion_policy.model.diffusion.ema_model import EMAModel
    from diffusion_policy.model.common.lr_scheduler import get_scheduler
    ema=EMAModel(ema_policy,update_after_step=0,inv_gamma=1.,power=.75,min_value=0.,max_value=.9999)
    optimizer=torch.optim.AdamW(policy.parameters(),lr=args.learning_rate,betas=(.95,.999),eps=1e-8,weight_decay=1e-6)
    lr_scheduler=get_scheduler("cosine",optimizer=optimizer,num_warmup_steps=500,num_training_steps=args.max_steps)
    generator=torch.Generator().manual_seed(args.seed)
    loaders={s:DataLoader(data,batch_size=args.batch_size,shuffle=s=="train",num_workers=args.num_workers,
                         pin_memory=True,generator=generator,**({"prefetch_factor":1} if args.num_workers else {}))
             for s,data in datasets.items() if len(data)}
    if "validation" not in loaders: raise ValueError("DP fit comparison requires a validation split")
    def move(x):
        return {k:move(v) for k,v in x.items()} if isinstance(x,dict) else x.to(device,non_blocking=True)
    def validate():
        ema_policy.eval();total=0.;count=0
        with torch.random.fork_rng(devices=[context.local_rank]),torch.inference_mode():
            torch.manual_seed(args.seed+1)
            for batch in loaders["validation"]:
                batch=move(batch);loss=ema_policy.compute_loss(batch);context.check_loss(loss)
                n=len(batch["action"]);total+=float(loss)*n;count+=n
        return total/count
    directory=output/"checkpoints";directory.mkdir(exist_ok=True)
    def save(name,step,val_loss):
        path=directory/name;temporary=path.with_suffix(".tmp")
        torch.save(dict(schema=SCHEMA,repository_revision=REVISION,model=ema_policy.state_dict(),
            dimension=dimension,settings=vars(args),manifest_sha256=args.manifest_sha,
            recording_sampling=sampling,global_step=step,validation_loss=val_loss),temporary)
        temporary.replace(path)
        write_json(path.with_name(path.name+".skynet-checkpoint.json"),dict(schema="skynet.checkpoint-score/v1",
            global_step=step,selection=dict(score=val_loss,mode="min",metric="validation/loss")))
        return path
    def reload_check(path):
        payload=torch.load(path,map_location=device,weights_only=False)
        verify_checkpoint_identity(payload,args.manifest_sha,vars(args))
        probe=move(next(iter(DataLoader(datasets["train"],batch_size=1,num_workers=0))))["obs"]
        return verify_policy_reload(ema_policy,payload["model"],probe,seed=args.seed+2,
            device=device,expected_shape=(1,args.action_steps,dimension))
    write_json(output/"applied-settings.json",dict(schema="skynet.applied-settings/v1",settings=vars(args),
        repository_revision=REVISION,recording_sampling=sampling,
        original_model="Official image DiffusionUnetImagePolicy, ResNet18 without pretraining, DDPM100, native loss and EMA",
        skynet_changes="Registered RGB/joints, resize320x240 without crop, horizon padded to multiple4, fixed splits, step budget, validation/checkpoint/progress integration"))
    initial=validate();best=initial;best_step=0;save("best.ckpt",0,initial)
    step=0;epoch=0;started=time.monotonic();loss_sum=0.;count=0;gradient_checked=False
    while step<args.max_steps:
        policy.train()
        for raw in loaders["train"]:
            batch=move(raw);optimizer.zero_grad(set_to_none=True)
            loss=policy.compute_loss(batch);context.check_loss(loss);loss.backward()
            if not gradient_checked:
                grads=[p.grad for p in policy.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(torch.count_nonzero(g)>0 for g in grads):
                    raise ValueError("DP training has invalid or empty gradients")
                gradient_checked=True
            optimizer.step();lr_scheduler.step();ema.step(policy);step+=1
            loss_sum+=float(loss.detach())*len(batch["action"]);count+=len(batch["action"])
            if step==8 or step%args.validation_every==0 or step==args.max_steps:
                val=validate();last=save("latest.ckpt",step,val)
                if val<best: best=val;best_step=step;save("best.ckpt",step,val)
                if step==8:
                    write_json(output/"smoke-verification.json",dict(steps=8,nonzero_finite_gradient=True,**reload_check(last)))
            if step%20==0 or step==8 or step==args.max_steps:
                row=dict(global_step=step,train_loss=loss_sum/count,lr=optimizer.param_groups[0]["lr"],elapsed_seconds=time.monotonic()-started)
                if step==8 or step%args.validation_every==0 or step==args.max_steps: row["val_loss"]=val
                with (output/"logs.json.txt").open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
                print(json.dumps(row),flush=True);loss_sum=0.;count=0
            if step>=args.max_steps:break
        epoch+=1
    # Verify the final weights first, then verify the selected best weights.
    verification=reload_check(directory/"latest.ckpt")
    selected=directory/("best.ckpt" if args.checkpoint_selection=="best" else "latest.ckpt")
    if args.checkpoint_selection=="best":
        saved=torch.load(selected,map_location=device,weights_only=False);ema_policy.load_state_dict(saved["model"],strict=True)
        verification=reload_check(selected)
    write_json(output/"training-result.json",dict(global_step=step,selected_checkpoint_step=best_step if args.checkpoint_selection=="best" else step,
        checkpoint=str(selected),checkpoint_sha256=digest(selected),initial_validation_loss=initial,best_validation_loss=best,
        manifest_sha256=args.manifest_sha,epochs=epoch,stop_reason="max_steps",**verification,recorded_task_success="not_evaluated"))
    context.close()


if __name__=="__main__":main()
