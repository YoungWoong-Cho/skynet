-- Rename only the live registry column; PostgreSQL retains its indexes/references.
ALTER TABLE data_resources RENAME COLUMN name TO source_key;
ALTER TABLE data_resources RENAME CONSTRAINT data_resources_provider_namespace_name_key
    TO data_resources_provider_namespace_source_key_key;

CREATE FUNCTION guard_resource_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF ROW(NEW.id, NEW.provider, NEW.namespace, NEW.source_key)
       IS DISTINCT FROM ROW(OLD.id, OLD.provider, OLD.namespace, OLD.source_key) THEN
        RAISE EXCEPTION 'Resource identity is immutable; edit display_name instead'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER resource_identity_guard
    BEFORE UPDATE OF id, provider, namespace, source_key ON data_resources
    FOR EACH ROW EXECUTE FUNCTION guard_resource_identity();
