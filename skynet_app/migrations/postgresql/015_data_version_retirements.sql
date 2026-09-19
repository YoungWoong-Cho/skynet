-- Retire generated copies without deleting immutable experiment/checkpoint history.
CREATE TABLE data_version_retirements (
    version_id TEXT PRIMARY KEY REFERENCES data_resource_versions(id) ON DELETE RESTRICT,
    replacement_version_id TEXT NOT NULL REFERENCES data_resource_versions(id) ON DELETE RESTRICT,
    state TEXT NOT NULL CHECK (state IN ('PENDING','RETIRED','FAILED')),
    plan_json TEXT NOT NULL,
    receipt_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (version_id <> replacement_version_id)
);
CREATE INDEX data_version_retirement_replacement ON data_version_retirements(replacement_version_id);
CREATE FUNCTION guard_data_version_retirement() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF ROW(NEW.version_id, NEW.replacement_version_id, NEW.created_at)
       IS DISTINCT FROM ROW(OLD.version_id, OLD.replacement_version_id, OLD.created_at)
       OR (OLD.state = 'RETIRED' AND NEW IS DISTINCT FROM OLD) THEN
        RAISE EXCEPTION 'Dataset retirement identity and completed receipts are immutable';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER immutable_data_version_retirement BEFORE UPDATE ON data_version_retirements
    FOR EACH ROW EXECUTE FUNCTION guard_data_version_retirement();
CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON data_version_retirements
    FOR EACH ROW EXECUTE FUNCTION skynet_notify_change('data,exports,experiments', 'shared');
