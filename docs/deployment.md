# Linux team deployment

The current deployment is on `rl2-ws11` (`130.207.121.35`), under the Linux account `youngwoong`. The checkout and virtual environment are `/home/youngwoong/skynet` and `/home/youngwoong/skynet/.venv`. The service listens on TCP 8080. A network administrator must allow that port from the team's trusted network before direct browser access will work.

Email entry separates personal workspaces; it does not verify identity. Keep access limited to the trusted team. An SSH tunnel also works without opening an inbound web port:

```bash
ssh -N -L 127.0.0.1:18080:127.0.0.1:8080 rl2-ws11
```

Open `http://127.0.0.1:18080`. This forwards the remote app; it does not run a local app or scheduler.

## Service installation

Install Python 3.11 or newer and uv, clone this repository to `~/skynet`, then install the locked production environment:

```bash
cd ~/skynet
uv sync --frozen --no-dev
mkdir -p ~/.config/systemd/user
cp deploy/skynet.service ~/.config/systemd/user/
systemd-analyze --user verify ~/.config/systemd/user/skynet.service
systemctl --user daemon-reload
```

Configure the SSH aliases in the cluster profile with non-interactive access and verified host keys. The current server has its own dedicated key for sky1/sky2; it does not contain the Mac's private SSH or GitHub credentials. Its authorized key is restricted to this server's IP, so update that restriction if the server's address changes.

Before starting, assign the existing workspace owner and migrate data as described below. For a new installation, prepare the intended owner instead. Enable user services outside interactive login sessions, then start one server worker:

```bash
loginctl enable-linger "$USER"
systemctl --user enable --now skynet.service
```

The service uses the user's Linux Secret Service for remembered tracking credentials. `gnome-keyring-daemon.service` and the user's D-Bus session must be available. If the login keyring is locked after a host reboot, unlock it through the server's normal login/keyring workflow before using remembered tracking connections. Secrets are not stored in the service unit or repository.

`data/operator.env` is an optional, private environment file for operator settings already approved for this installation. The service does not accept simulator licenses by default. Keep the data directory private to the service user.

## Workspace cutover

Only one Skynet scheduler may manage a copied job history. Stop the previous app, including its development reload supervisor, before taking the final SQLite snapshot. Stopping the app does not stop Slurm training jobs.

1. Preserve a rollback copy of the old database and its matching code version.
2. Use SQLite's backup API for a consistent database copy. Do not copy a live database without its committed WAL contents.
3. Transfer the database and supporting `data/` files over SSH. Existing cluster recordings, prepared datasets, checkpoints and training jobs stay on the cluster.
4. Preserve workspace IDs and ownership. The current legacy owner is `ycho420@gatech.edu`.
5. Relocate mutable submission-host artifact paths to the new checkout, verifying file hashes. Preserve immutable dataset versions, manifests, scientific configurations, Slurm IDs and cluster paths. Reset copied browser sessions to require a fresh sign-in.
6. Transfer remembered tracking credentials directly between secure stores over SSH; retain the source key until the target is verified. Never place credentials in a Git commit, command line, log or plaintext migration file.
7. Verify database integrity, foreign keys, record counts, supporting file checksums and artifact accessibility before starting the new service.
8. Verify email sign-in, run history, recording storage locations, tracking connections and cluster access on the deployed app. Do not submit or cancel real jobs just to test the deployment.

The current server stores the database and supporting metadata under `~/skynet/data/`. The retained Shadow cube recordings remain on sky2. Bonjour connectivity is a separate prerequisite for collecting new data on that workstation.

## Operations

Run these on the server:

```bash
systemctl --user status skynet.service
journalctl --user -u skynet.service -n 100 --no-pager
systemctl --user restart skynet.service
systemctl --user stop skynet.service
curl --fail http://127.0.0.1:8080/api/health
```

Use a single worker and no development reload watcher. Do not restart the old Mac app against its stale database after cutover.

For code updates, transfer a verified commit, install its frozen dependencies, then restart the service. The current checkout was installed from a Git bundle because the server has no personal GitHub token. Its GitHub origin alone does not grant access to the private repository. Keep private runtime data out of bundles and commits.

Check server-local health and direct network access separately: a successful local health check does not prove the host firewall permits team browsers to connect.
