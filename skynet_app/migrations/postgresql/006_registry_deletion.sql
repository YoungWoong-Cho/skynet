-- Default adapters are editable by the installation owner. Keep immutable versions.
CREATE TABLE registry_exclusions (
    kind TEXT NOT NULL,
    seed_key TEXT NOT NULL,
    PRIMARY KEY(kind, seed_key)
);
UPDATE adapters SET owner_id='legacy' WHERE owner_id IS NULL AND seed_key IS NOT NULL;

CREATE OR REPLACE FUNCTION guard_pending_history_fn() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE plan JSONB; fields JSONB := to_jsonb(NEW); field RECORD; identifiers TEXT[]; value TEXT;
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
            'adapter_key','adapter_version_id','source_adapter_key','evaluation_suite_id')
           AND field.value=ANY(identifiers) THEN
            RAISE EXCEPTION 'This item is being deleted; finish its pending deletion first' USING ERRCODE='23514';
        END IF;
        -- JSON snapshots can pin a version without a relational foreign key.
        IF plan->>'kind' IN ('adapter','suite') AND right(field.key,5)='_json' THEN
            FOREACH value IN ARRAY identifiers LOOP
                IF position(to_json(value)::text IN field.value)>0 THEN
                    RAISE EXCEPTION 'This item is being deleted; finish its pending deletion first' USING ERRCODE='23514';
                END IF;
            END LOOP;
        END IF;
    END LOOP;
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
FOREACH name IN ARRAY ARRAY['adapters','adapter_validations','evaluation_suites'] LOOP
 EXECUTE format('CREATE TRIGGER guard_pending_history BEFORE INSERT OR UPDATE ON %I FOR EACH ROW EXECUTE FUNCTION guard_pending_history_fn()', name);
END LOOP;
END $$;
