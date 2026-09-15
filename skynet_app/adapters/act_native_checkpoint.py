"""Epoch-boundary recovery for the pinned upstream ACT loop.

The original loader, model, loss, optimizer and update order stay upstream.
A self-contained checkpoint carries both inference and recovery state.
"""

import json
import os
from pathlib import Path
import random
import signal
import time

SCHEMA = "skynet.act-native-checkpoint/v1"
ENTRY_SHA256 = "ab0ca2af8b00efe80a340229e86c675455b6537261db09aa73091cd38bd0910a"


def instrument(directory):
    """Apply only audited lifecycle hooks to a private source export."""
    from artifacts import digest

    path = directory / "imitate_episodes.py"
    if digest(path) != ENTRY_SHA256:
        raise ValueError("ACT checkpoint hooks require the audited upstream training source")
    source = path.read_text()
    changes = [
        ("def main(args):\n", "def main(args):\n    args['num_epochs'] = int(os.environ['SKYNET_ACT_EPOCHS'])\n"),
        ('    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)',
         '    config["native_args"] = args\n    config["normalization"] = stats\n    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)'),
        ('    for epoch in tqdm(range(num_epochs)):',
         '    from act_native_checkpoint import TrainingSession\n    session = TrainingSession(config, train_dataloader, val_dataloader)\n    start_epoch, best_ckpt_info = session.restore(policy, optimizer)\n    if best_ckpt_info is not None:\n        min_val_loss = best_ckpt_info[1]\n\n    for epoch in tqdm(range(start_epoch, num_epochs)):'),
        ('compute_dict_mean(train_history[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)])',
         'compute_dict_mean(train_history)'),
        ("        if (epoch + 1) % config['save_freq'] == 0:",
         '        session.completed_epoch(epoch, policy, optimizer, best_ckpt_info, epoch_summary, validation_history[-1])\n        train_history.clear()\n        validation_history.clear()\n\n' + "        if (epoch + 1) % config['save_freq'] == 0:"),
    ]
    for before, after in changes:
        if source.count(before) != 1:
            raise ValueError("ACT checkpoint hook no longer matches upstream")
        source = source.replace(before, after)
    path.write_text(source)
    return {"path": "imitate_episodes.py", "source_sha256": ENTRY_SHA256,
            "runtime_sha256": digest(path), "purpose": "epoch checkpoint, resume and metrics"}


def cpu(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(v) for v in value)
    return value


def atomic_save(payload, path):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(cpu(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def identity(config, train, validation):
    args = {k: v for k, v in config["native_args"].items() if k not in {"ckpt_dir", "num_epochs"}}
    return {
        "source_revision": os.environ["SKYNET_ACT_REVISION"],
        "dataset_manifest_sha256": os.environ["SKYNET_ACT_MANIFEST_SHA"],
        "native_args": args,
        "policy_config": config["policy_config"],
        "state_dim": config["state_dim"],
        "split": {"train": list(map(int, train.dataset.episode_ids)),
                  "validation": list(map(int, validation.dataset.episode_ids))},
    }


def trim_progress(path, next_epoch):
    """Discard metrics after the durable checkpoint before replaying those epochs."""
    if not path.exists():
        return
    rows = []
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue  # A killed writer can leave one incomplete final line.
        if value["epoch"] < next_epoch:
            rows.append(json.dumps(value, allow_nan=False) + "\n")
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(rows))
    temporary.replace(path)


class TrainingSession:
    def __init__(self, config, train, validation):
        self.config = config
        self.identity = identity(config, train, validation)
        self.output = Path(os.environ["SKYNET_ACT_OUTPUT"])
        self.checkpoint = self.output / "checkpoints/last.ckpt"
        self.request = Path(os.environ["SKYNET_ACT_STOP_REQUEST"])
        self.last_saved = time.monotonic()
        self.stop_signal = None
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(sig, self.stop)

    def stop(self, signum, _frame):
        self.stop_signal = signum

    def restore(self, policy, optimizer):
        import numpy as np
        import torch

        path = os.environ.get("SKYNET_ACT_RESUME")
        if not path:
            return 0, None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != SCHEMA or payload.get("identity") != self.identity:
            raise ValueError("Native ACT resume checkpoint belongs to another dataset, source, split or training configuration")
        for key in ("qpos_mean", "qpos_std", "action_mean", "action_std"):
            if not np.array_equal(payload["normalization"][key], self.config["normalization"][key]):
                raise ValueError("Native ACT normalization changed since checkpoint creation")
        epoch = payload["next_epoch"]
        if not isinstance(epoch, int) or not 0 < epoch <= self.config["num_epochs"]:
            raise ValueError("Resume epoch exceeds the selected training length")
        if not self.checkpoint.exists():
            atomic_save(payload, self.checkpoint)
        trim_progress(self.output / "logs.json.txt", epoch)
        policy.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"])
        torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
        self.last_saved = time.monotonic()
        print(json.dumps({"event": "act_native_resumed", "completed_epochs": epoch,
                          "optimizer_steps": sorted({int(v.get('step', 0)) for v in optimizer.state.values()})}), flush=True)
        return epoch, payload["best_ckpt_info"]

    def completed_epoch(self, epoch, policy, optimizer, best, train, validation):
        import numpy as np
        import torch

        # No persistent workers upstream: iterating each next epoch recreates workers
        # from the restored Torch RNG, including sampler order and NumPy seeds.
        metrics = {"epoch": epoch, "train_loss": float(train["loss"]),
                   "val_loss": float(validation["loss"]), "lr": optimizer.param_groups[0]["lr"]}
        for prefix, values in (("train", train), ("validation", validation)):
            metrics.update({f"{prefix}/{key}": float(value) for key, value in values.items()})
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "logs.json.txt").open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            stream.flush()
        stop = self.stop_signal or (signal.SIGTERM if self.request.exists() else None)
        if epoch == 0 or epoch + 1 == self.config["num_epochs"] or stop or time.monotonic() - self.last_saved >= 60:
            payload = {"schema": SCHEMA, "identity": self.identity, "next_epoch": epoch + 1,
                       "model": policy.state_dict(), "optimizer": optimizer.state_dict(),
                       "best_ckpt_info": best, "normalization": self.config["normalization"],
                       "config": self.config,
                       "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                               "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}}
            atomic_save(payload, self.checkpoint)
            self.last_saved = time.monotonic()
            print(json.dumps({"event": "act_native_checkpoint", "completed_epochs": epoch + 1,
                              "path": str(self.checkpoint)}), flush=True)
        if stop:
            raise SystemExit(128 + int(stop))
