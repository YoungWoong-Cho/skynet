-- Variants project their compact display spec and tracking providers; evaluation
-- stages project their frozen target dataset. Offloaded columns hold only an
-- object reference, so their writer (PayloadStore.put) projects the body it holds.
-- Manual rollback (stop app writers first):
--   DROP INDEX document_projections_assignments;
--   DROP TRIGGER skynet_document_projection ON workflow_stages;
--   DELETE FROM document_projections WHERE table_name='workflow_stages';
--   re-create skynet_project_document and skynet_refresh_document_projection from 018
--   (variants rows keep harmless extra keys, or re-run the 018 INSERT for 'variants');
--   DELETE FROM skynet_schema_migrations WHERE version=20;
CREATE OR REPLACE FUNCTION skynet_project_document(body TEXT, kind TEXT) RETURNS JSONB
LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE document JSONB := body::jsonb; paths JSONB; assignments JSONB := '[]'::jsonb;
        target JSONB; result JSONB;
BEGIN
    IF kind='workflow_stages' THEN
        -- The stage context is the durable target reference; the plan carries a copy of it.
        target := CASE WHEN jsonb_typeof(document #> '{context,target_dataset}')='object'
                       THEN document #> '{context,target_dataset}'
                       ELSE document #> '{plan,native_config,canonical_evaluation,target_dataset}' END;
        IF jsonb_typeof(target)='object' THEN
            assignments := jsonb_build_array(jsonb_build_object(
                'role','evaluation_target',
                'version_id', target ->> 'version_id',
                'digest', target ->> 'manifest_sha256',
                'path', target ->> 'path'));
        END IF;
        RETURN jsonb_build_object('assignments', assignments);
    END IF;
    SELECT coalesce(jsonb_agg(value ORDER BY value), '[]'::jsonb) INTO paths FROM (
        SELECT DISTINCT value FROM jsonb_array_elements(jsonb_path_query_array(document,
            'strict $.** ? (@.type() == "string" && @ like_regex "^/")'))
    ) unique_paths;
    IF kind='experiment_revisions' THEN
        SELECT coalesce(jsonb_agg(jsonb_build_object(
            'version_id', value #>> '{version,metadata,registered_version_id}',
            'digest', value #>> '{version,manifest_sha256}',
            'provider', value #>> '{resource,provider}',
            'namespace', value #>> '{resource,namespace}',
            'source_key', value #>> '{resource,name}')), '[]'::jsonb) INTO assignments
        FROM jsonb_array_elements(CASE WHEN jsonb_typeof(document #> '{data,bundle,assignments}')='array'
             THEN document #> '{data,bundle,assignments}' ELSE '[]'::jsonb END);
    END IF;
    result := jsonb_build_object('paths',paths,'assignments',assignments,
        'source', CASE WHEN kind='experiment_revisions' THEN jsonb_build_object(
            'git_revision',document #>> '{source,revision}',
            'repository',document #>> '{source,repository}',
            'project_subdirectory',document #>> '{source,project_subdirectory}') ELSE '{}'::jsonb END);
    IF kind='variants' THEN
        -- 'display' is the run list's spec: the capsule code and per-episode lists removed.
        result := result || jsonb_build_object(
            'tracking_providers', CASE WHEN jsonb_typeof(document #> '{tracking,providers}')='array'
                THEN document #> '{tracking,providers}' ELSE '[]'::jsonb END,
            'display', jsonb_set(
                document #- '{source,adapter_manifest,train,capsule_files}',
                '{data,bundle,assignments}',
                COALESCE((SELECT jsonb_agg(jsonb_set(item, '{version,metadata}',
                    (COALESCE(item #> '{version,metadata}', '{}'::jsonb) - 'episodes' - 'shared_artifacts')
                    || jsonb_build_object('episodes',
                        CASE WHEN jsonb_typeof(item #> '{version,metadata,episodes}') = 'array'
                             THEN to_jsonb(jsonb_array_length(item #> '{version,metadata,episodes}'))
                             ELSE item #> '{version,metadata,episodes}' END)))
                  FROM jsonb_array_elements(COALESCE(document #> '{data,bundle,assignments}', '[]'::jsonb)) item),
                  '[]'::jsonb)));
    END IF;
    RETURN result;
END;
$$;

CREATE OR REPLACE FUNCTION skynet_refresh_document_projection() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE body TEXT;
BEGIN
    IF TG_OP='DELETE' THEN
        DELETE FROM document_projections WHERE table_name=TG_TABLE_NAME AND record_id=OLD.id;
        RETURN NULL;
    END IF;
    IF TG_OP='UPDATE' AND OLD.id<>NEW.id THEN
        DELETE FROM document_projections WHERE table_name=TG_TABLE_NAME AND record_id=OLD.id;
    END IF;
    body := to_jsonb(NEW)->>TG_ARGV[0];
    -- An offloaded column holds only an object reference; PayloadStore.put projects
    -- the body it uploaded, in this same transaction.
    IF body LIKE '{"$skynet_object_v1":%' THEN RETURN NULL; END IF;
    INSERT INTO document_projections VALUES (TG_TABLE_NAME,NEW.id,skynet_project_document(body,TG_TABLE_NAME))
    ON CONFLICT(table_name,record_id) DO UPDATE SET projection_json=excluded.projection_json;
    RETURN NULL;
END;
$$;

CREATE TRIGGER skynet_document_projection
    AFTER INSERT OR UPDATE OF id,resolved_config_json OR DELETE ON workflow_stages
    FOR EACH ROW EXECUTE FUNCTION skynet_refresh_document_projection('resolved_config_json');

-- Inline stage bodies project now; offloaded ones are backfilled by
-- Database.repair_document_projections from the bodies' target subtrees.
INSERT INTO document_projections
SELECT 'workflow_stages', id, skynet_project_document(resolved_config_json,'workflow_stages')
FROM workflow_stages WHERE resolved_config_json NOT LIKE '{"$skynet_object_v1":%'
ON CONFLICT (table_name,record_id) DO UPDATE SET projection_json=excluded.projection_json;

-- Only the variants projection changes shape; the other kinds stay byte-identical.
INSERT INTO document_projections
SELECT 'variants', id, skynet_project_document(resolved_spec_json,'variants') FROM variants
ON CONFLICT (table_name,record_id) DO UPDATE SET projection_json=excluded.projection_json;

CREATE INDEX document_projections_assignments ON document_projections
    USING gin ((projection_json -> 'assignments') jsonb_path_ops);
