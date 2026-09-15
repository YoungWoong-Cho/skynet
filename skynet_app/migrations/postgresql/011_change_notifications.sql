-- Invalidation hints only: no data copies, expiry caches, or persisted event log.
-- NOTIFY is delivered by PostgreSQL only after the writing transaction commits.
CREATE FUNCTION skynet_notify_change() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    before_row jsonb;
    after_row jsonb;
    old_scope text;
    new_scope text;
    topics text[] := string_to_array(TG_ARGV[0], ',');
BEGIN
    IF TG_OP <> 'INSERT' THEN before_row := to_jsonb(OLD); END IF;
    IF TG_OP <> 'DELETE' THEN after_row := to_jsonb(NEW); END IF;
    -- Polling a worker must not turn its last-checked clock into a refresh loop.
    IF TG_OP = 'UPDATE' THEN
        IF before_row ? 'payload_json' THEN
            before_row := jsonb_set(before_row, '{payload_json}',
                (before_row->>'payload_json')::jsonb - 'updated_at' - 'checked_at');
            after_row := jsonb_set(after_row, '{payload_json}',
                (after_row->>'payload_json')::jsonb - 'updated_at' - 'checked_at');
        END IF;
        IF (before_row - 'updated_at' - 'checked_at') IS NOT DISTINCT FROM
           (after_row - 'updated_at' - 'checked_at') THEN
            RETURN NULL;
        END IF;
    END IF;
    IF TG_ARGV[1] = 'shared' THEN
        old_scope := '*';
        new_scope := '*';
    ELSIF TG_ARGV[1] = 'session' THEN
        old_scope := before_row->>'workspace_id';
        new_scope := after_row->>'workspace_id';
    ELSE
        old_scope := before_row->>'owner_id';
        new_scope := after_row->>'owner_id';
        IF TG_TABLE_NAME = 'adapters' THEN
            IF before_row IS NOT NULL AND (old_scope IS NULL OR
               (old_scope = 'legacy' AND before_row->>'seed_key' IS NOT NULL)) THEN
                old_scope := '*';
            END IF;
            IF after_row IS NOT NULL AND (new_scope IS NULL OR
               (new_scope = 'legacy' AND after_row->>'seed_key' IS NOT NULL)) THEN
                new_scope := '*';
            END IF;
        END IF;
    END IF;
    IF old_scope IS NOT NULL THEN
        PERFORM pg_notify('skynet_changes', json_build_object('v', 1, 'scope', old_scope, 'topics', topics)::text);
    END IF;
    IF new_scope IS NOT NULL AND new_scope IS DISTINCT FROM old_scope THEN
        PERFORM pg_notify('skynet_changes', json_build_object('v', 1, 'scope', new_scope, 'topics', topics)::text);
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON data_resources
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('data,exports', 'shared');
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON data_resource_versions
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('data,exports', 'shared');
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON data_locations
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('data,exports', 'shared');
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON data_derivations
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('data', 'shared');
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON data_derivation_inputs
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('data', 'shared');
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON data_imports
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('data', 'shared');
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON data_bundles
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('data', 'shared');
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON data_bundle_assignments
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('data', 'shared');
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON policy_exports
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('exports', 'shared');
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON live_xr_sessions
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('recordings', 'shared');
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON adapters
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('adapters', 'owner');
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON workspace_storage
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('settings', 'owner');
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON tracking_connections
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('settings', 'owner');
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON slack_notifications
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('settings', 'owner');
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON workspace_sessions
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('session', 'session');

-- Dataset usage is shared, while its private experiment details are redacted by
-- the snapshot API. These hints contain no experiment/run identifiers.
CREATE TRIGGER skynet_change AFTER INSERT OR DELETE OR UPDATE OF name ON experiments
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('data,exports', 'shared');
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON experiment_revisions
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('data,exports', 'shared');
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON variants
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('data,exports', 'shared');
CREATE TRIGGER skynet_change AFTER INSERT OR DELETE OR UPDATE OF status ON runs
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('exports', 'shared');
