# Collection storage

Completed collection sessions are stored on sky2. The collection workstation retains files only while collection, image processing, or a video replay is active, or while a transfer needs attention.

`config/live_storage.json` enables automatic archiving and removal of a verified workstation copy. Startup reconciliation includes existing terminal sessions and resumes interrupted transfers. Session history retains the original execution host and profile; a separate archive receipt records the current data location.

The complete session tree, including recordings, images, videos and diagnostics, is copied to:

`/coc/flash7/ycho420/datasets/raw/dexverse-live/<session-id>/<manifest-sha256>/`

Each file is verified by size and SHA-256. The receipt is saved before cleanup. The destination is verified again, the collection and replay processes must be stopped, and the source must still match the manifest before its exact session directory is removed. Corruption, missing files, an ownership mismatch, or a changed source retains the source and reports an error. Interrupted deletion resumes only for the same verified session. Empty failed sessions get an explicit empty receipt.

New review data and video caches are separate from the immutable archive. Reviews validate bounded bytes in memory and save their data remotely. Videos stream with HTTP byte ranges and `Cache-Control: no-store`; archived replay uses Slurm and never stages a recording back on the workstation. Preparing training data also runs on the cluster and registers a cluster location. Skynet retains operational metadata and frozen code locally, but does not retain collection payloads. Explicit user downloads are still available.

The recordings table shows transfer status and the sky2 location. A failed transfer can be retried without deleting the source. Archive jobs continue after closing the review; app restart reconciles saved progress.

Existing Mac preparation/review caches are first copied to a checksum-verified operator archive under `datasets/raw/operator-cache/`. Registered prepared versions retain their identity and verified cluster locations. Only after these checks are their local payload files removed. Reusable simulator runtimes and hand assets are not collected data and remain available for future collection.
