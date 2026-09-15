-- Promote the catalog label without changing identity, dates, or frozen receipts.
ALTER TABLE data_resources ADD COLUMN display_name TEXT;
UPDATE data_resources
SET display_name = CASE
        WHEN jsonb_typeof(metadata_json::jsonb->'display_name') = 'string'
             AND btrim(metadata_json::jsonb->>'display_name') <> ''
        THEN metadata_json::jsonb->>'display_name'
        ELSE concat_ws('/', nullif(namespace, ''), nullif(name, ''))
    END,
    metadata_json = CASE WHEN metadata_json::jsonb ? 'display_name'
        THEN (metadata_json::jsonb - 'display_name')::text ELSE metadata_json END;
ALTER TABLE data_resources ALTER COLUMN display_name SET NOT NULL;
ALTER TABLE data_resources ADD CONSTRAINT resource_display_name_nonempty
    CHECK (btrim(display_name) <> '');
ALTER TABLE data_resources ADD CONSTRAINT resource_display_name_single_source
    CHECK (NOT (metadata_json::jsonb ? 'display_name'));

-- Labels may repeat. The existing provider/namespace/name constraint is identity.
-- Version metadata and published manifests are deliberately left untouched.
