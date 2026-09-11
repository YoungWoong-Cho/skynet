-- An older client or a direct SQL writer cannot revive a partially deleted item.
CREATE FUNCTION guard_pending_history_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF current_setting('skynet.delete_history', true)='on' THEN RETURN NEW; END IF;
IF EXISTS (
    SELECT 1 FROM maintenance_operations m,
    LATERAL jsonb_each(m.plan_json::jsonb->'records') records,
    LATERAL jsonb_array_elements_text(records.value) ids,
    LATERAL jsonb_each_text(to_jsonb(NEW)) fields
    WHERE fields.key IN ('id','run_id','stage_id','experiment_id','experiment_revision_id',
        'variant_id','checkpoint_id','resume_checkpoint_id','restarted_from_run_id',
        'evaluation_id','entity_id','scope_id','depends_on_stage_id')
      AND fields.value=ids.value
) THEN
 RAISE EXCEPTION 'This history item is being deleted; finish its pending deletion first' USING ERRCODE='23514';
END IF;
RETURN NEW;
END; $$;
DO $$ DECLARE name TEXT;
BEGIN
FOREACH name IN ARRAY ARRAY['experiments','experiment_revisions','variants','runs','workflow_stages',
 'job_attempts','checkpoints','evaluations','evaluation_episodes','artifacts','manifests',
 'metrics','training_progress_samples','events','tracking_bindings','stage_dependencies'] LOOP
 EXECUTE format('CREATE TRIGGER guard_pending_history BEFORE INSERT OR UPDATE ON %I FOR EACH ROW EXECUTE FUNCTION guard_pending_history_fn()', name);
END LOOP;
END $$;
