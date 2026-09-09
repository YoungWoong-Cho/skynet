"""Run with torchrun on allocated GPUs; compare real DDP gradients and checkpoints."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn
from torch.utils.data import TensorDataset
sys.path.insert(0, str(Path(__file__).parents[1] / 'skynet_app/adapters'))
from training_parallel import TrainingContext, PolicyLoss


class Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(8,32), nn.Tanh(), nn.Linear(32,3))
        self.unused=nn.Parameter(torch.ones(1))
    def compute_loss(self,batch):
        x,y=batch
        return (self.layers(x)-y).square().mean()


def main():
    context=TrainingContext(int(os.environ['WORLD_SIZE']))
    torch.manual_seed(31)
    model=Regression().to(context.device)
    reference=copy.deepcopy(model)
    wrapped=context.wrap(PolicyLoss(model,'compute_loss'))
    x,y=torch.randn(26,8),torch.randn(26,3)
    loader=context.loader(TensorDataset(x,y), 12, shuffle=True, seed=11)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    expected_optimizer=torch.optim.AdamW(reference.parameters(),lr=1e-3)
    max_error=0.0
    for epoch in range(3):
        loader.batch_sampler.set_epoch(epoch)
        order=torch.randperm(26,generator=torch.Generator().manual_seed(11+epoch))
        optimizer.zero_grad();expected_optimizer.zero_grad()
        local_count=global_count=0
        for step,batch in enumerate(loader):
            batch=[item.to(context.device) for item in batch]
            indices=order[step*12:(step+1)*12]
            expected=reference.compute_loss((x[indices].to(context.device),y[indices].to(context.device)))
            actual=wrapped(batch)[0]['loss']
            context.check_loss(actual)
            count=loader.batch_sampler.valid_count(step)
            torch.testing.assert_close(torch.tensor(context.mean(float(actual.detach())*count,count,len(indices))),expected.detach().cpu(),atol=2e-6,rtol=1e-5)
            context.backward(actual,count)
            (expected*len(indices)).backward()
            local_count+=count;global_count+=len(indices)
            if (step+1)%2==0 or step+1==len(loader):
                context.normalize_gradients(model.parameters(),local_count,global_count)
                for p,q in zip(model.parameters(),reference.parameters()):
                    if q.grad is None:
                        assert p.grad is None
                    else:
                        q.grad.div_(global_count)
                        torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=1e-4)
                        max_error=max(max_error,float((p.grad-q.grad).abs().max()))
                optimizer.step();expected_optimizer.step()
                optimizer.zero_grad();expected_optimizer.zero_grad()
                local_count=global_count=0
                for p,q in zip(model.parameters(),reference.parameters()):
                    torch.testing.assert_close(p,q,atol=2e-6,rtol=1e-4)
    if context.primary:
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            torch.save(model.state_dict(),path)
            loaded=Regression()
            loaded.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            torch.testing.assert_close(loaded.layers(x),model.layers(x.to(context.device)).detach().cpu(),atol=2e-6,rtol=1e-4)
        print(json.dumps(dict(event='distributed_correctness_verified',gpus=context.world_size,
            uneven_batches=True,zero_weight_padding=True,gradient_accumulation=True,
            checkpoint_reload=True,max_gradient_error=max_error)),flush=True)
    context.close()

if __name__=='__main__': main()
