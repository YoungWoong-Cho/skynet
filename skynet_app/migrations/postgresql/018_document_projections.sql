-- Small, transactionally maintained read projections. Original receipts remain
-- unchanged; direct SQL writers get the same insert/update/delete behavior.
CREATE TABLE document_projections (
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    projection_json JSONB NOT NULL,
    PRIMARY KEY (table_name, record_id)
);

CREATE FUNCTION skynet_project_document(body TEXT, kind TEXT) RETURNS JSONB
LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE document JSONB := body::jsonb; paths JSONB; assignments JSONB := '[]'::jsonb;
BEGIN
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
    RETURN jsonb_build_object('paths',paths,'assignments',assignments,
        'source', CASE WHEN kind='experiment_revisions' THEN jsonb_build_object(
            'git_revision',document #>> '{source,revision}',
            'repository',document #>> '{source,repository}',
            'project_subdirectory',document #>> '{source,project_subdirectory}') ELSE '{}'::jsonb END);
END;
$$;

CREATE FUNCTION skynet_refresh_document_projection() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP='DELETE' THEN
        DELETE FROM document_projections WHERE table_name=TG_TABLE_NAME AND record_id=OLD.id;
    ELSE
        IF TG_OP='UPDATE' AND OLD.id<>NEW.id THEN
            DELETE FROM document_projections WHERE table_name=TG_TABLE_NAME AND record_id=OLD.id;
        END IF;
        INSERT INTO document_projections VALUES (TG_TABLE_NAME,NEW.id,
            skynet_project_document(to_jsonb(NEW)->>TG_ARGV[0],TG_TABLE_NAME))
        ON CONFLICT(table_name,record_id) DO UPDATE SET projection_json=excluded.projection_json;
    END IF;
    RETURN NULL;
END;
$$;

DO $$
DECLARE item RECORD;
BEGIN
    FOR item IN SELECT * FROM (VALUES
        ('experiment_revisions','requested_spec_json'),('variants','resolved_spec_json'),
        ('live_xr_sessions','payload_json'),('policy_exports','payload_json'),
        ('data_bundles','manifest_json')) AS columns(table_name,document_column)
    LOOP
        EXECUTE format('CREATE TRIGGER skynet_document_projection AFTER INSERT OR UPDATE OF id,%I OR DELETE ON %I
            FOR EACH ROW EXECUTE FUNCTION skynet_refresh_document_projection(%L)',
            item.document_column,item.table_name,item.document_column);
        EXECUTE format('INSERT INTO document_projections SELECT %L,id,skynet_project_document(%I,%L) FROM %I',
            item.table_name,item.document_column,item.table_name,item.table_name);
    END LOOP;
END;
$$;
