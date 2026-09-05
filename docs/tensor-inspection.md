# Inspect a model's tensor flow

Open **Data → Collection**. In a completed DexVerse cycle, choose **Inspect model tensors**. Select a dataset frame (numbered from zero) and choose **Load frame**. Use **Previous step**, **Next step**, or a layer button to follow that frame through the saved model.

The inspector shows:

- The input and output tensor at each step, including shape, float32 type, value ranges and every feature value.
- Normalization, three linear layers, two Tanh activations, action denormalization and action range limiting.
- The learned parameter shapes and normalization/range values used at each applicable step.
- The final prediction beside the demonstrated action for that same frame. This is a training-frame comparison, not task success or held-out evaluation.
- Full-precision JSON export with the dataset and checkpoint checksums.

All tensors have shape `[batch, features]`. The current policy processes one frame at a time, so batch size is 1. Cells show six significant digits; hovering a value or exporting JSON retains its float32 value. Feature indices follow the saved dataset. The dataset did not preserve joint names, so the inspector does not invent them.

## Supported model and provenance

The current trace adapter supports `skynet-state-bc/v1`: 35 observation features → Linear(35, 128) → Tanh → Linear(128, 128) → Tanh → Linear(128, 28). Saved input normalization, output scaling and training-range bounds are applied exactly as in `dexverse_runner.py`.

This is a new CPU inference pass through the saved checkpoint using an observation from the saved dataset. It is **not** a historical evaluation activation recording. CPU and GPU rounding may differ. It does not replay a simulator or issue robot commands. GR00T and pi0.5 checkpoints need their own trace adapters and are explicitly unsupported by this inspector.

## Implementation and limits

`GET /api/collection/processing/jobs/{cycle_id}/tensor-trace?frame=0` returns `skynet.tensor-trace/v1`. `TensorTraceService` verifies the cycle state and compatibility, invokes the standalone trace worker in the cycle's saved runtime and gateway, and validates the returned values and provenance. The worker verifies dataset/checkpoint checksums, uses `torch.load(..., weights_only=True)` on CPU, limits Torch to one thread, and checks the explicit layer walk against a complete forward pass.

The web server has no PyTorch or HDF5 dependency. The trace worker uses the same runtime that trained the policy. The current small-model adapter limits checkpoints to 16 MB and datasets to 256 MB, reporting an explicit error for larger files. A runtime request has a 50-second remote timeout and a 55-second transport timeout. At most two distinct uncached traces load at once; simultaneous requests for the same trace share one calculation.

SQLite caches up to 128 traces, keyed by the checkpoint and dataset checksums and sizes, frame number, and trace-worker source hash. Cached traces refer to those exact artifact versions; they do not recheck the cluster on each click. Changing any artifact or the worker invalidates the key. Layer navigation is entirely local to the browser. Closing the inspector or choosing another cycle invalidates earlier browser requests, so late results cannot replace the current selection.

## Verification

- Normal web environment: `python -m pytest -p no:cacheprovider tests/test_tensor_trace.py -q`.
- Environment with PyTorch and h5py: `python -m pytest -p no:cacheprovider tests/test_tensor_trace_runner.py -q`.
- Browser-state regressions: `node --test tests/tensor_trace_browser.cjs`.
- The live browser was checked against frames 0 and 589 from completed cycle `8bd93c5d-977c-46c9-a26c-2221b78b44ea`, including all nine steps forward and backward, direct layer selection, retry after network failure, full-precision export and the final demonstration comparison.
