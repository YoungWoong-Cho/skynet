-- Reconcile selects workflow stages and job attempts by status first: the workflow
-- repair reads active stages, the Slurm poll and the submission recovery read
-- bound or unconfirmed attempts. Stages were indexed only by run first and
-- attempts only by stage first, so every reconcile pass scanned both tables.
-- Manual rollback: DROP INDEX idx_stages_status_run; DROP INDEX idx_attempts_status_stage;
-- DELETE FROM skynet_schema_migrations WHERE version=25.
CREATE INDEX idx_stages_status_run ON workflow_stages(status, run_id);
CREATE INDEX idx_attempts_status_stage ON job_attempts(status, stage_id);
