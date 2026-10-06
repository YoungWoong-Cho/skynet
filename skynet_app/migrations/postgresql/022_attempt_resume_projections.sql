-- Job attempts project whether their execution pins an initial checkpoint, so the
-- run list reads that flag from a row instead of inspecting each offloaded
-- execution snapshot through the object store on every cold start.
-- Manual rollback (stop app writers first): DROP TRIGGER skynet_document_projection ON
-- job_attempts; DELETE FROM document_projections WHERE table_name='job_attempts';
-- re-create skynet_project_document from 021; DELETE FROM skynet_schema_migrations WHERE version=22.
CREATE OR REPLACE FUNCTION skynet_project_document(body TEXT, kind TEXT) RETURNS JSONB
LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE document JSONB := body::jsonb; paths JSONB; assignments JSONB := '[]'::jsonb;
        target JSONB; result JSONB;
BEGIN
    IF kind='job_attempts' THEN
        -- Whether the execution pins an initial checkpoint (the run list's resume flag).
        -- Truthiness follows the Python reader: null, false, '', 0, [] and {} are false.
        RETURN jsonb_build_object('has_initial_checkpoint', (
            SELECT bool_or(value IS NOT NULL AND value NOT IN ('null'::jsonb, 'false'::jsonb, '""'::jsonb, '0'::jsonb, '[]'::jsonb, '{}'::jsonb))
            FROM unnest(ARRAY[document #> '{plan,native_config,initial_checkpoint}',
                              document #> '{migration_provenance,checkpoint}',
                              document #> '{resolved_spec,native,config,initial_checkpoint}']) AS value) IS TRUE);
    END IF;
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
        -- 'display' is the run list's spec: only the paths the list reads (resources,
        -- the progress contract and its total, the resume pin, and describe()'s view
        -- of each assignment). Key paths equal the full spec's, so the list code
        -- reads it unchanged; run detail keeps the stored specification.
        result := result || jsonb_build_object(
            'tracking_providers', CASE WHEN jsonb_typeof(document #> '{tracking,providers}')='array'
                THEN document #> '{tracking,providers}' ELSE '[]'::jsonb END,
            'display', jsonb_build_object(
                'resources', document -> 'resources',
                'train', jsonb_build_object('max_steps', document #> '{train,max_steps}'),
                'native', jsonb_build_object('config', jsonb_build_object(
                    'epochs', document #> '{native,config,epochs}',
                    'initial_checkpoint', document #> '{native,config,initial_checkpoint}')),
                'source', jsonb_build_object('adapter_manifest', jsonb_build_object('train', jsonb_build_object(
                    'progress', document #> '{source,adapter_manifest,train,progress}'))),
                'data', jsonb_build_object('bundle', jsonb_build_object(
                    'name', document #> '{data,bundle,name}',
                    'assignments', COALESCE((SELECT jsonb_agg(jsonb_build_object(
                        'role', item -> 'role',
                        'resource', jsonb_build_object('id', item #> '{resource,id}', 'name', item #> '{resource,name}'),
                        'version', jsonb_build_object(
                            'format', item #> '{version,format}',
                            'manifest_sha256', item #> '{version,manifest_sha256}',
                            'metadata', jsonb_strip_nulls(jsonb_build_object(
                                'display_name', item #> '{version,metadata,display_name}',
                                'registered_version_id', item #> '{version,metadata,registered_version_id}',
                                'episodes', CASE WHEN jsonb_typeof(item #> '{version,metadata,episodes}')='array'
                                                 THEN to_jsonb(jsonb_array_length(item #> '{version,metadata,episodes}'))
                                                 ELSE item #> '{version,metadata,episodes}' END))
                              -- describe() reads num_episodes by key presence, so presence is kept, not just values.
                              || CASE WHEN COALESCE(item #> '{version,metadata}', '{}'::jsonb) ? 'num_episodes'
                                      THEN jsonb_build_object('num_episodes', item #> '{version,metadata,num_episodes}')
                                      ELSE '{}'::jsonb END)))
                      FROM jsonb_array_elements(CASE WHEN jsonb_typeof(document #> '{data,bundle,assignments}')='array'
                                                     THEN document #> '{data,bundle,assignments}' ELSE '[]'::jsonb END) item),
                      '[]'::jsonb)))));
    END IF;
    RETURN result;
END;
$$;

CREATE TRIGGER skynet_document_projection
    AFTER INSERT OR UPDATE OF id,execution_snapshot_json OR DELETE ON job_attempts
    FOR EACH ROW EXECUTE FUNCTION skynet_refresh_document_projection('execution_snapshot_json');

-- Inline snapshots project now; offloaded ones are backfilled by
-- Database.repair_document_projections from the snapshots' resume subtrees.
INSERT INTO document_projections
SELECT 'job_attempts', id, skynet_project_document(execution_snapshot_json,'job_attempts')
FROM job_attempts WHERE execution_snapshot_json IS NOT NULL AND execution_snapshot_json NOT LIKE '{"$skynet_object_v1":%'
ON CONFLICT (table_name,record_id) DO UPDATE SET projection_json=excluded.projection_json;
