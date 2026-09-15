-- Stop all app writers before rollback; restore the matching pre-012 source
-- before restarting them. Current edited labels are copied back into metadata.
-- The deployment backup preserves exact pre-migration JSON bytes if needed.
BEGIN;
SET LOCAL lock_timeout = '10s';
SELECT pg_advisory_xact_lock(-8777073831261177637);
LOCK TABLE data_resources IN ACCESS EXCLUSIVE MODE;
DROP TRIGGER resource_identity_guard ON data_resources;
DROP FUNCTION guard_resource_identity();
ALTER TABLE data_resources RENAME CONSTRAINT data_resources_provider_namespace_source_key_key
    TO data_resources_provider_namespace_name_key;
ALTER TABLE data_resources RENAME COLUMN source_key TO name;
ALTER TABLE data_resources DROP CONSTRAINT resource_display_name_single_source;
UPDATE data_resources
SET metadata_json = (metadata_json::jsonb || jsonb_build_object('display_name', display_name))::text;
ALTER TABLE data_resources DROP COLUMN display_name;
DELETE FROM skynet_schema_migrations WHERE version IN (12, 13);
COMMIT;
