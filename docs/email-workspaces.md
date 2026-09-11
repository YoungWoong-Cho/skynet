# Email workspaces

Enter your email and select **Open workspace**. The app remembers the choice in this browser for 30 days. **Switch email / sign out** clears the session and returns to email entry. Returning with the same email restores the same records. Whitespace and letter case do not create duplicate workspaces.

There are no passwords, email messages, verification codes or SSO. This is for a trusted team: anyone entering another person's email can use that workspace.

## Personal and shared records

Personal records include projects, experiment specifications and revisions, custom training adapters and their validations, saved repository branches/commits, tracking connections and credentials, Slack notification settings and delivery history, the cluster base path, training runs and attempts, checkpoints, evaluation jobs, episodes, results, logs and tracking links. Browser gateway preferences and tutorial progress also follow the workspace. Logging out does not cancel jobs or discard saved tracking credentials.

Cluster availability, recording sessions, datasets and preparation jobs, hands/poses, collection adapter definitions, built-in training adapters, installed runtimes and evaluation suite definitions are shared. Clone a built-in training adapter to customize it in your workspace. Shared datasets remain protected from deletion while any experiment uses them; other workspaces' experiment names and links are hidden.

The app checks ownership in API requests and database operations. A single background coordinator continues reconciling every workspace's jobs. The browser discards forms and cached records when switching; stale tabs must reload rather than issue requests under the replacement workspace. Cluster jobs still run through the service's configured SSH account; email selection does not create Linux or Slurm accounts.

W&B and MLflow settings, in-memory credentials and OS credential-store namespaces are separate for each workspace. Existing OS-stored and environment credentials belong only to the legacy owner's workspace. A new workspace must connect its own tracking account. Connect validates the supplied credentials and saves them in that server's supported OS credential store. Connected integrations show Disconnect, which removes the saved credentials.

## Personal cluster base path

On first use, **Settings → Cluster storage** requires an explicit base path; no personal path is prefilled or inherited from the server. Enter an absolute path to a new or empty directory and select **Validate and initialize**. Skynet checks syntax, write access, directory emptiness and workspace ownership before creating directories and saving the path. Failed initialization leaves the previous setting unchanged. A completed initialization can be retried safely for the same workspace. The path does not change the Linux or Slurm account.

New training runs record the selected base path when their draft run is created. Job scripts, repository checkouts, logs, caches and checkpoints use that root. Existing runs, retries, submission recovery and evaluations associated with those runs retain their recorded root, even after the preference changes. No files are moved or removed. Shared recording, preparation and import jobs, registered datasets and installed runtime profiles retain their existing shared locations.

The default is the deployment’s configured root. Saving a preference does not change another email workspace’s preference. Simultaneous edits from stale tabs are rejected rather than overwriting a newer value. Local control-plane database and capsule locations remain deployment settings.

## Assign existing records before upgrading

Keep one running Skynet service per database. With the old app stopped, back up its SQLite database using SQLite's backup API, then run:

```bash
python -m skynet_app.workspaces --database /absolute/path/to/data/skynet.db --legacy-owner you@example.com
```

This migrates the database and assigns existing personal records to the specified email. IDs, job receipts, dataset locations and remote jobs are retained. The migration does not copy recordings or move cluster files. Repeating it with the same email is safe; a different owner is rejected.

Alternatively, prepare the owner while the old app is still running:

```bash
python -m skynet_app.workspaces --database /absolute/path/to/data/skynet.db --legacy-owner you@example.com --prepare-owner
```

This writes only `workspace-owner.json` beside the database. It does not open or migrate the database. The upgraded app applies it on its next startup. `SKYNET_LEGACY_OWNER_EMAIL` can supply the same value through deployment configuration instead. Configure this before team members sign in; an existing separate workspace with that email will not be silently merged.

Do not run old code against a migrated database. To roll back, stop the app and restore both the old code and its database backup. Do not run local and deployed schedulers against duplicate copies of the same job history.

## Checks

```bash
python -m pytest tests/test_email_workspaces.py
npm run test:workspaces
```

These check real SQLite isolation, migration, session expiry/logout, credentials, shared adapters, cross-workspace read/cancel rejection, background job selection, and initialization of the real UI scripts after email selection. They do not submit real cluster jobs.
