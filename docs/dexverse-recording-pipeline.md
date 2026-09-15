# Retired offline tracking experiment

The local tracking importer, recorder installer, offline replay/training/evaluation workflow, and their server APIs have been removed. The retired URL families are `/api/collection/local/*` and `/api/collection/processing/*`; they are no longer registered.

For simulator demonstrations, use **Data → Collect**, then **Recordings → Convert**. See [Live collection](live-dexverse.md) and [Dataset preparation](policy-data-exports.md). This workflow collects simulator trajectories and is not an import replacement for old headset-only JSONL motion files.

Existing files and historical database rows are preserved. The standalone [Vision Pro recorder](../visionpro/README.md) can still export its native files through the system share control. It no longer has a Skynet backend import or offline experiment endpoint.

The old `capture_processing` package and setup command have been removed. Current services retain their shared functionality in `skynet_app/cluster_upload.py`, `skynet_app/isaac_job.py`, and `skynet_app/dexverse_release.py`. Live collection and video rendering use their own runtime configuration.
