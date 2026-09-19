-- A published output is the dataset. Catalog labels and archive state are mutable;
-- source manifests and experiment snapshots remain byte-for-byte immutable.
CREATE TABLE data_dataset_presentations (
    version_id TEXT PRIMARY KEY REFERENCES data_resource_versions(id) ON DELETE CASCADE,
    display_name TEXT NOT NULL CHECK (btrim(display_name) <> ''),
    description TEXT NOT NULL DEFAULT '',
    archived_at TEXT,
    updated_at TEXT NOT NULL
);
INSERT INTO data_dataset_presentations(version_id, display_name, description, archived_at, updated_at)
SELECT v.id,
       concat_ws(' · ', coalesce(nullif(btrim(v.metadata_json::jsonb->>'display_name'), ''), r.display_name),
         CASE WHEN nullif(btrim(v.metadata_json::jsonb->'adapter'->>'name'), '') IS NOT NULL
              AND position(v.metadata_json::jsonb->'adapter'->>'name' IN
                  coalesce(nullif(btrim(v.metadata_json::jsonb->>'display_name'), ''), r.display_name)) = 0
              THEN v.metadata_json::jsonb->'adapter'->>'name' END),
       r.description, r.archived_at, greatest(r.updated_at, v.created_at)
FROM data_resource_versions v JOIN data_resources r ON r.id = v.resource_id
WHERE r.category = 'dataset' AND v.format <> 'skynet.episodes/v1';

-- Internal source groups no longer carry dataset-level archive state.
UPDATE data_resources SET archived_at = NULL WHERE category = 'dataset' AND archived_at IS NOT NULL;

CREATE FUNCTION initialize_dataset_presentation() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE resource data_resources%ROWTYPE; label TEXT;
BEGIN
    SELECT * INTO resource FROM data_resources WHERE id = NEW.resource_id;
    IF resource.category <> 'dataset' OR NEW.format = 'skynet.episodes/v1' THEN RETURN NULL; END IF;
    label := nullif(btrim(NEW.metadata_json::jsonb->>'display_name'), '');
    IF label IS NULL THEN
        label := concat_ws(' · ', resource.display_name,
                  nullif(btrim(NEW.metadata_json::jsonb->'adapter'->>'name'), ''));
    END IF;
    INSERT INTO data_dataset_presentations(version_id, display_name, description, archived_at, updated_at)
    VALUES (NEW.id, label, resource.description, NULL, NEW.created_at);
    RETURN NULL;
END;
$$;
CREATE TRIGGER dataset_presentation_created AFTER INSERT ON data_resource_versions
    FOR EACH ROW EXECUTE FUNCTION initialize_dataset_presentation();
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON data_dataset_presentations
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('data,exports,experiments', 'shared');
