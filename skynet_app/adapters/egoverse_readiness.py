"""Cluster checks for native EgoVerse inference and prediction video output."""
import argparse
import importlib
import hashlib
import json
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--pi", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, args.source_dir)
    checks, errors = {}, []
    actual = {"checks": checks, "gpu": {}}
    try:
        import torch
        import numpy as np
        import imageio.v2 as imageio
        for module in ("egomimic.algo.act", "egomimic.algo.hpt",
                       "egomimic.pl_utils.pl_model", "egomimic.rldb.zarr.zarr_dataset_multi"):
            importlib.import_module(module)
        checks["native_imports"] = True
        if args.pi:
            importlib.import_module("egomimic.algo.pi")
            import transformers
            patch = Path(args.source_dir) / "external/openpi/src/openpi/models_pytorch/transformers_replace"
            installed = Path(transformers.__file__).parent
            checks["pi_imports_and_patch"] = all(
                (installed / p.relative_to(patch)).is_file()
                and hashlib.sha256(p.read_bytes()).digest() == hashlib.sha256((installed / p.relative_to(patch)).read_bytes()).digest()
                for p in patch.rglob("*.py"))
        actual["gpu"] = dict(cuda_available=torch.cuda.is_available(), device_count=torch.cuda.device_count())
        layer = torch.nn.Linear(8, 8).cuda()
        loss = layer(torch.ones(2, 8, device="cuda")).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        checks["cuda_forward_backward"] = bool(torch.isfinite(loss) and torch.isfinite(layer.weight.grad).all())
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "prediction.mp4"
            with imageio.get_writer(video, fps=10) as writer:
                writer.append_data(np.full((128, 128, 3), 100, dtype=np.uint8))
            with imageio.get_reader(video) as reader:
                checks["video_encode_decode"] = reader.get_data(0).shape == (128, 128, 3)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    Path(args.result).write_text(json.dumps(dict(schema_version="skynet.evaluator-readiness/v1",
        ready=not errors and all(checks.values()), actual=actual, errors=errors)))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
