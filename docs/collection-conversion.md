# Retired state-only collection conversion

The old state-only conversion API and its worker were removed after the unified policy-aware preparation workflow replaced the UI. The removed routes are:

- `POST /api/collection/live/sessions/{identifier}/conversions`
- `GET /api/collection/live/conversions/{identifier}`
- `GET /api/collection/live/conversions/{identifier}/logs`
- `GET /api/collection/live/conversions/{identifier}/{name}`

Use **Data → Recordings → Convert** and the current `/api/data/exports` API described in [Dataset preparation](policy-data-exports.md).

This code cleanup does not delete stored recordings, datasets, cluster files, or historical database rows. Registered historical datasets retain their registry entries and verified locations. Recording-deletion checks still account for old conversion records.

The former state-only HDF5 converter, automatic conversion monitor, and its browser QA fixture are no longer shipped. The old validation receipts under `docs/validation/` describe past checks, not current API availability. Current recording review and video rendering continue through their own services.
