"""Strict saved-state restoration shared by dataset rendering and evaluation."""
import numpy as np

def restore_state(env, state):
    import torch
    current = env.scene.get_state(is_relative=True)
    def tensors(saved, actual):
        if isinstance(actual, dict):
            if not isinstance(saved, dict) or set(saved) != set(actual):
                raise ValueError("Saved scene entities differ from the renderer")
            return {k: tensors(saved[k], actual[k]) for k in actual}
        value = np.asarray(saved)
        if value.shape != tuple(actual.shape) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError("Invalid saved scene state")
        return torch.as_tensor(value, device=env.device, dtype=actual.dtype)
    env.scene.reset_to(tensors(state, current), is_relative=True)
    env.sim.forward()
    env.sim.render()
    env.scene.update(dt=env.physics_dt)
