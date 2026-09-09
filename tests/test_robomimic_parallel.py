"""Exercise the native DexMimicGen BC training and checkpoint interfaces on two GPUs."""

from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("robomimic")
sys.path.insert(0, str(Path(__file__).parents[1] / "skynet_app/adapters"))
from robomimic_training import enable_parallel_training


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two allocated CUDA GPUs"
)
def test_native_bc_rnn_gmm_updates_on_both_gpus_and_reloads_checkpoint():
    from robomimic.config import config_factory
    import robomimic.algo as algorithms
    import robomimic.utils.obs_utils as obs_utils

    config = config_factory(algo_name="bc")
    with config.values_unlocked():
        config.train.batch_size = 5
        config.observation.modalities.obs.low_dim = ["state"]
        config.algo.actor_layer_dims = (32,)
        config.algo.rnn.enabled = True
        config.algo.rnn.hidden_dim = 32
        config.algo.gmm.enabled = True
    obs_utils.initialize_obs_utils_with_config(config)
    enable_parallel_training([0, 1])
    model = algorithms.algo_factory(
        "bc", config, {"state": (4,)}, 2, torch.device("cuda:0")
    )
    before = {k: v.clone() for k, v in model.nets.state_dict().items()}
    batch = {
        "obs": {"state": torch.randn(5, 3, 4, device="cuda:0")},
        "goal_obs": None,
        "actions": torch.randn(5, 3, 2, device="cuda:0"),
    }
    model.set_train()
    info = model.train_on_batch(batch, epoch=1)
    assert torch.isfinite(info["losses"]["action_loss"])
    assert model._skynet_parallel.reported_devices == {0, 1}
    assert any(
        not torch.equal(before[k], v) for k, v in model.nets.state_dict().items()
    )
    restored = algorithms.algo_factory(
        "bc", config, {"state": (4,)}, 2, torch.device("cuda:0")
    )
    restored.deserialize(model.serialize())
    for k, v in model.nets.state_dict().items():
        torch.testing.assert_close(v, restored.nets.state_dict()[k])
    assert set(model.log_info(info))
