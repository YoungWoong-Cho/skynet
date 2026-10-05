-- Preserve deletion reservations even for direct SQL and older API writers.
-- Network file removal runs outside the repository write transaction.
CREATE OR REPLACE FUNCTION guard_pending_history_fn() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE plan JSONB; fields JSONB := to_jsonb(NEW); field RECORD; identifiers TEXT[]; value TEXT; document JSONB; file JSONB; candidate TEXT; deleting_path TEXT;
BEGIN
IF current_setting('skynet.delete_history', true)='on' THEN RETURN NEW; END IF;
FOR plan IN SELECT plan_json::jsonb FROM maintenance_operations LOOP
    SELECT array_agg(ids.value) INTO identifiers
    FROM jsonb_each(plan->'records') records,
         LATERAL jsonb_array_elements_text(records.value) ids;
    identifiers := array_append(identifiers, plan->>'id');
    FOR field IN SELECT * FROM jsonb_each_text(fields) LOOP
        IF field.key IN ('id','run_id','stage_id','experiment_id','experiment_revision_id',
            'variant_id','checkpoint_id','resume_checkpoint_id','restarted_from_run_id',
            'evaluation_id','entity_id','scope_id','depends_on_stage_id',
            'adapter_key','adapter_version_id','source_adapter_key','evaluation_suite_id',
            'version_id','input_version_id','output_version_id','bundle_id')
           AND field.value=ANY(identifiers) THEN
            RAISE EXCEPTION 'This item is being deleted; finish its pending deletion first' USING ERRCODE='23514';
        END IF;
        -- JSON snapshots can pin a version without a relational foreign key.
        IF right(field.key,5)='_json' THEN
            FOREACH value IN ARRAY identifiers LOOP
                IF position(to_json(value)::text IN field.value)>0 THEN
                    RAISE EXCEPTION 'This item is being deleted; finish its pending deletion first' USING ERRCODE='23514';
                END IF;
            END LOOP;
        END IF;
    END LOOP;
    -- Scalar paths and JSON-pinned paths cannot acquire a reference while
    -- another process is deleting their files. Work without such references
    -- remains writable throughout the remote operation.
    FOR field IN SELECT * FROM jsonb_each_text(fields) LOOP
        IF TG_OP='UPDATE' AND (to_jsonb(OLD)->field.key) IS NOT DISTINCT FROM (fields->field.key) THEN
            CONTINUE;
        END IF;
        document := CASE WHEN right(field.key,5)='_json' AND field.value IS NOT NULL
                         THEN field.value::jsonb ELSE to_jsonb(field.value) END;
        FOR candidate IN SELECT jsonb_array_elements_text(jsonb_path_query_array(document,
            'strict $.** ? (@.type() == "string" && @ like_regex "^/")')) LOOP
            candidate := rtrim(regexp_replace(candidate, '/+', '/', 'g'), '/');
            FOR file IN SELECT * FROM jsonb_array_elements(coalesce(plan->'files','[]'::jsonb)) LOOP
                deleting_path := rtrim(regexp_replace(file->>'path', '/+', '/', 'g'), '/');
                IF candidate=deleting_path OR starts_with(candidate, deleting_path || '/')
                    OR starts_with(deleting_path, candidate || '/') THEN
                    RAISE EXCEPTION 'This file is being deleted; finish its pending deletion first' USING ERRCODE='23514';
                END IF;
            END LOOP;
        END LOOP;
    END LOOP;
    IF TG_TABLE_NAME='metadata_payload_refs' AND EXISTS (
        SELECT 1 FROM metadata_payloads p, jsonb_array_elements(coalesce(plan->'files','[]'::jsonb)) f
        WHERE p.sha256=fields->>'sha256' AND (p.path=f->>'path'
          OR starts_with(p.path, (f->>'path') || '/'))
    ) THEN
        RAISE EXCEPTION 'This metadata body is being deleted; retry later' USING ERRCODE='23514';
    END IF;
    IF TG_TABLE_NAME='adapters' AND plan->>'kind'='adapter' AND EXISTS (
        SELECT 1 FROM adapters a WHERE a.id=ANY(identifiers)
        AND a.adapter_key IN (fields->>'adapter_key', fields->>'source_adapter_key')
    ) THEN
        RAISE EXCEPTION 'This adapter is being deleted; finish its pending deletion first' USING ERRCODE='23514';
    END IF;
    IF TG_TABLE_NAME='evaluation_suites' AND plan->>'kind'='suite' AND EXISTS (
        SELECT 1 FROM evaluation_suites s
        WHERE s.id=ANY(identifiers) AND s.name=fields->>'name'
          AND s.evaluator_adapter=fields->>'evaluator_adapter'
    ) THEN
        RAISE EXCEPTION 'This suite is being deleted; finish its pending deletion first' USING ERRCODE='23514';
    END IF;
END LOOP;
RETURN NEW;
END; $$;


DO $$ DECLARE name TEXT;
BEGIN
FOREACH name IN ARRAY ARRAY['data_locations','data_derivations','data_derivation_inputs',
    'metadata_payloads','metadata_payload_refs','observation_artifacts','observation_producers'] LOOP
 EXECUTE format('CREATE TRIGGER guard_pending_history BEFORE INSERT OR UPDATE ON %I FOR EACH ROW EXECUTE FUNCTION guard_pending_history_fn()', name);
END LOOP;
END $$;
