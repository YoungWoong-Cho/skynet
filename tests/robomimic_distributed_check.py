"""Run with torchrun: native BC-RNN-GMM updates, global loss and checkpoint reload."""
import json
import os
from pathlib import Path
import sys

import torch
sys.path.insert(0, str(Path(__file__).parents[1] / "skynet_app/adapters"))
from training_parallel import TrainingContext
from robomimic_training import enable_distributed_training


def main():
    from robomimic.config import config_factory
    import robomimic.algo as algorithms
    import robomimic.utils.obs_utils as obs_utils

    context = TrainingContext(int(os.environ["WORLD_SIZE"]))
    torch.manual_seed(42)
    config = config_factory(algo_name="bc")
    with config.values_unlocked():
        config.train.batch_size = 5
        config.observation.modalities.obs.low_dim = ["state"]
        config.algo.actor_layer_dims = (32,)
        config.algo.rnn.enabled = True
        config.algo.rnn.hidden_dim = 32
        config.algo.gmm.enabled = True
    obs_utils.initialize_obs_utils_with_config(config)
    enable_distributed_training(context)
    model = algorithms.algo_factory("bc", config, {"state": (4,)}, 2, context.device)
    before = {k: v.clone() for k, v in model.nets.state_dict().items()}
    for size in (5, 2):
        batch = {"obs": {"state": torch.randn(size, 3, 4, device=context.device)},
                 "goal_obs": None, "actions": torch.randn(size, 3, 2, device=context.device)}
        model.set_train()
        info = model.train_on_batch(batch, epoch=1)
        assert torch.isfinite(info["losses"]["action_loss"])
        model.set_eval()
        validation = model.train_on_batch(batch, epoch=1, validate=True)
        assert torch.isfinite(validation["losses"]["action_loss"])
    assert any(not torch.equal(before[k], v) for k, v in model.nets.state_dict().items())
    restored = algorithms.algo_factory("bc", config, {"state": (4,)}, 2, context.device)
    restored.deserialize(model.serialize())
    for k, v in model.nets.state_dict().items():
        torch.testing.assert_close(v, restored.nets.state_dict()[k])
        total = v.clone()
        context.dist.all_reduce(total)
        torch.testing.assert_close(total / context.world_size, v)
    assert set(model.log_info(info))
    if context.primary:
        print(json.dumps({"event": "robomimic_distributed_verified", "gpus": context.world_size,
                          "native_checkpoint_reload": True, "loss": float(info["losses"]["action_loss"])}))
    context.close()


if __name__ == "__main__":
    main()
