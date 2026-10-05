# Collection storage

Recordings → View → **Delete** removes the selected native recording using the
shared dependency preview and confirmation dialog. Its identity includes the
source path, so a repeated or stale request cannot delete the next file after
the list changes. Dataset consumers of the session, dependent recordings and
active collection/review/conversion work block removal. Completed source transfer
and workstation cleanup are required before deletion.

The selected PKL, legacy camera sidecars and associated reviews/videos are removed; other
recordings and their storage slots remain unchanged. Deletion verifies the
remaining archive and publishes its updated manifest before committing the new
recording list. The original allocation hash stays in `archive.storage_key`, while
`manifest_sha256` verifies the current inventory. This keeps sibling file paths
stable. An interrupted operation keeps the same durable deletion intent and is
resumed through the common Delete dialog. Deleting the last file leaves an empty
collection session; session-level deletion is a separate action.

Completed collection sessions are stored on the training cluster. New sessions save actions, scene states and preview metadata. The collection workstation retains files while collection or saving is active, or while a transfer needs attention. Any historical image/replay work must also stop before its source is removed.

`config/live_storage.json` enables automatic archiving and removal of a verified workstation copy. Startup reconciliation includes existing terminal sessions and resumes interrupted transfers. Session history retains the original execution host and profile; a separate archive receipt records the current data location.

The complete session tree, including recordings, diagnostics and any historical image/video files, is copied to:

`/coc/flash7/ycho420/datasets/raw/dexverse-live/<session-id>/<manifest-sha256>/`

Each file is verified by size and SHA-256. The receipt is saved before cleanup. The destination is verified again, the collection and replay processes must be stopped, and the source must still match the manifest before its exact session directory is removed. Corruption, missing files, an ownership mismatch, or a changed source retains the source and reports an error. Interrupted deletion resumes only for the same verified session. Empty failed sessions get an explicit empty receipt.

Review data is separate from the immutable archive. Reviews validate bounded bytes in memory and save their data remotely. Recording View uses saved states and frozen assets for 3D playback; it starts no image or video generation. Convert prepares required observations on the cluster, reuses verified matching observations, and registers the resulting dataset at its cluster location. Observation deletion follows the shared source and consumer references described in [observation preparation](observation-preparation.md). Skynet retains operational metadata and frozen code locally, but does not retain collection payloads. Explicit downloads of original recordings and prepared datasets remain available.

The recordings table shows transfer status and the training cluster location. A failed transfer can be retried without deleting the source. Archive jobs continue after closing the review; app restart reconciles saved progress.

Existing Mac preparation/review caches are first copied to a checksum-verified operator archive under `datasets/raw/operator-cache/`. Registered prepared versions retain their identity and verified cluster locations. Only after these checks are their local payload files removed. Reusable simulator runtimes and hand assets are not collected data and remain available for future collection.
