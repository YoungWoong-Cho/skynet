-- Experiment notes owned by one workspace: Markdown rows, folders, and attachments
-- whose bodies live in the checksum-addressed object store (metadata_payloads).
-- Attachment rows keep their bodies through metadata_payload_refs rows that the
-- Notes API writes and releases in the same transaction, like other payload owners.
-- Manual rollback (no data was transformed; stop app writers and restore the prior
-- source first):
--   DELETE FROM metadata_payload_refs WHERE table_name='note_attachments';
--   DROP TABLE note_attachments, notes, note_folders;
--   DROP FUNCTION note_owner_guard();
--   DELETE FROM skynet_schema_migrations WHERE version=19;
CREATE TABLE note_folders (
    owner_id TEXT NOT NULL REFERENCES workspaces(id),
    id TEXT NOT NULL CHECK (id ~ '^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$'),
    name TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 240),
    PRIMARY KEY (owner_id, id),
    UNIQUE (owner_id, name)
);
CREATE TABLE notes (
    owner_id TEXT NOT NULL REFERENCES workspaces(id),
    id TEXT NOT NULL CHECK (id ~ '^[a-z0-9][a-z0-9-]{0,127}$'),
    title TEXT NOT NULL CHECK (length(title) BETWEEN 1 AND 240),
    markdown TEXT NOT NULL CHECK (octet_length(markdown) <= 1048576),
    folder_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (owner_id, id),
    FOREIGN KEY (owner_id, folder_id) REFERENCES note_folders(owner_id, id) ON DELETE SET NULL (folder_id)
);
CREATE TABLE note_attachments (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    note_id TEXT NOT NULL,
    name TEXT NOT NULL CHECK (name ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$'),
    media_type TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    sha256 TEXT NOT NULL REFERENCES metadata_payloads(sha256) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    UNIQUE (owner_id, note_id, name),
    FOREIGN KEY (owner_id, note_id) REFERENCES notes(owner_id, id) ON DELETE CASCADE
);

-- Writes outside the session's workspace are refused and ownership is immutable.
CREATE FUNCTION note_owner_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP <> 'INSERT' AND current_workspace_id() IS NOT NULL
       AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
    IF (TG_OP = 'INSERT' AND current_workspace_id() IS NOT NULL
        AND NEW.owner_id IS DISTINCT FROM current_workspace_id())
       OR (TG_OP = 'UPDATE' AND NEW.owner_id IS DISTINCT FROM OLD.owner_id) THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END; $$;

DO $$ DECLARE name TEXT;
BEGIN
FOREACH name IN ARRAY ARRAY['note_folders','notes','note_attachments'] LOOP
    EXECUTE format('CREATE TRIGGER workspace_owner_guard BEFORE INSERT OR UPDATE OR DELETE ON %I
        FOR EACH ROW EXECUTE FUNCTION note_owner_guard()', name);
    EXECUTE format('CREATE TRIGGER skynet_change AFTER INSERT OR UPDATE OR DELETE ON %I
        FOR EACH ROW EXECUTE FUNCTION skynet_notify_change(''notes'', ''owner'')', name);
END LOOP;
END $$;
