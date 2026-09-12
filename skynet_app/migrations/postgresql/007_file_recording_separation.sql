-- Recording ownership belongs to datasets, never environment/model file sets.
-- Remove only the two registry ownership keys; preserve all asset metadata.
UPDATE data_resources
SET metadata_json = (metadata_json::jsonb - 'session_id' - 'recording_session_id')::text
WHERE replace(kind, '-', '_') IN (
    'simulation_assets', 'embodiment_assets', 'calibration', 'checkpoint', 'model', 'file'
) AND metadata_json::jsonb ?| ARRAY['session_id', 'recording_session_id'];

ALTER TABLE data_resources ADD CONSTRAINT file_resources_no_recording_link CHECK (
    replace(kind, '-', '_') NOT IN (
        'simulation_assets', 'embodiment_assets', 'calibration', 'checkpoint', 'model', 'file'
    ) OR NOT (metadata_json::jsonb ?| ARRAY['session_id', 'recording_session_id'])
);
