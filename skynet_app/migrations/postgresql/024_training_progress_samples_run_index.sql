-- training_progress_samples is read by run (the run list's progress evidence,
-- run detail, tracking sync) but was indexed only by owner and by its
-- (attempt, restart, completed) uniqueness, so every run list load scanned it.
-- Manual rollback: DROP INDEX idx_training_progress_samples_run;
-- DELETE FROM skynet_schema_migrations WHERE version=24.
CREATE INDEX idx_training_progress_samples_run ON training_progress_samples(run_id);
