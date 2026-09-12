-- Persist the dataset/file distinction. Unknown legacy types deliberately fail
-- NOT NULL instead of silently appearing in the Datasets catalog.
ALTER TABLE data_resources ADD COLUMN category TEXT;
UPDATE data_resources SET kind = replace(kind, '-', '_');
UPDATE data_resources
SET kind = 'dataset',
    metadata_json = (metadata_json::jsonb || '{"test_fixture": true}'::jsonb)::text
WHERE kind = 'qa_fixture';
UPDATE data_resources SET category = CASE
    WHEN kind IN ('dataset', 'demonstrations', 'evaluation_data', 'raw_capture') THEN 'dataset'
    WHEN kind IN ('simulation_assets', 'embodiment_assets', 'calibration', 'checkpoint', 'model', 'file') THEN 'file'
END;
ALTER TABLE data_resources ALTER COLUMN category SET NOT NULL;
ALTER TABLE data_resources ADD CONSTRAINT resource_category_type CHECK (
    (category = 'dataset' AND kind IN ('dataset', 'demonstrations', 'evaluation_data', 'raw_capture')) OR
    (category = 'file' AND kind IN ('simulation_assets', 'embodiment_assets', 'calibration', 'checkpoint', 'model', 'file'))
);

CREATE FUNCTION resource_has_recording_link(metadata jsonb) RETURNS boolean
LANGUAGE sql IMMUTABLE AS $$
    SELECT metadata ?| ARRAY['session_id', 'recording_session_id'] OR EXISTS (
        SELECT 1 FROM jsonb_array_elements(
            CASE WHEN jsonb_typeof(metadata->'sources') = 'array' THEN metadata->'sources' ELSE '[]'::jsonb END
        ) source WHERE source ?| ARRAY['session_id', 'recording_session_id']
    )
$$;
ALTER TABLE data_resources DROP CONSTRAINT file_resources_no_recording_link;
ALTER TABLE data_resources ADD CONSTRAINT file_resources_no_recording_link CHECK (
    category <> 'file' OR NOT resource_has_recording_link(metadata_json::jsonb)
);

-- Published results are immutable. Do not rewrite historical result metadata.
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM data_resource_versions v JOIN data_resources r ON r.id = v.resource_id
               WHERE r.category = 'file' AND resource_has_recording_link(v.metadata_json::jsonb)) THEN
        RAISE EXCEPTION 'File result contains recording provenance; resolve its classification before migrating';
    END IF;
END $$;
CREATE FUNCTION guard_file_result_recordings() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF EXISTS (SELECT 1 FROM data_resources WHERE id = NEW.resource_id AND category = 'file')
       AND resource_has_recording_link(NEW.metadata_json::jsonb) THEN
        RAISE EXCEPTION 'Files cannot be linked to a recording' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER file_result_recording_guard BEFORE INSERT OR UPDATE ON data_resource_versions
    FOR EACH ROW EXECUTE FUNCTION guard_file_result_recordings();

CREATE FUNCTION guard_resource_classification() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.category <> OLD.category OR NEW.kind <> OLD.kind THEN
        RAISE EXCEPTION 'Resource classification is immutable' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER resource_classification_guard BEFORE UPDATE OF category, kind ON data_resources
    FOR EACH ROW EXECUTE FUNCTION guard_resource_classification();

-- Repair stale catalog dates from publications made before this trigger existed.
UPDATE data_resources r SET updated_at = GREATEST(r.updated_at, v.created_at)
FROM (SELECT resource_id, MAX(created_at) AS created_at FROM data_resource_versions GROUP BY resource_id) v
WHERE r.id = v.resource_id;

-- The catalog date reflects publication/removal as well as edits to its name.
CREATE FUNCTION touch_resource_result_parent() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    UPDATE data_resources
    SET updated_at = to_char(clock_timestamp() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.MS') || 'Z'
    WHERE id = COALESCE(NEW.resource_id, OLD.resource_id);
    RETURN NULL;
END $$;
CREATE TRIGGER resource_result_parent_updated AFTER INSERT OR DELETE ON data_resource_versions
    FOR EACH ROW EXECUTE FUNCTION touch_resource_result_parent();
