# Linux team deployment

The previous `rl2-ws11` deployment was removed. The following instructions are
for a fresh Linux installation. Configure the same central PostgreSQL endpoint
and cluster object store used by the Mac; the server does not get a DB copy.

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

Configure the SSH aliases in the cluster profile with non-interactive access and verified host keys. Provision a dedicated SSH identity on the new app host; keep private keys out of the repository.

Before starting, configure the central endpoint as described below. For a new installation, prepare the intended workspace owner. Enable user services outside interactive login sessions, then start one server worker:

```bash
loginctl enable-linger "$USER"
systemctl --user enable --now skynet.service
```

The service uses the user's Linux Secret Service for remembered tracking credentials. `gnome-keyring-daemon.service` and the user's D-Bus session must be available. If the login keyring is locked after a host reboot, unlock it through the server's normal login/keyring workflow before using remembered tracking connections. Secrets are not stored in the service unit or repository.

`data/operator.env` is an optional, private environment file for operator settings already approved for this installation. The service does not accept simulator licenses by default. Keep the data directory private to the service user.

## Central workspace connection

1. Install the same code revision as the other app hosts.
2. Configure `config/database.json` and the private DB password file using
   [central database setup](central-database.md). Use the existing object-store root.
3. Verify the same workspace, recording, training and evaluation IDs are visible.
   Cluster data stays at its registered location; there is no DB-file transfer.
4. Configure credentials on the new host through its secure credential store.
   Credential transfer requires explicit authorization; never put keys in Git.
5. Start the service and verify sign-in, history, artifact access and cluster access.
   Do not submit or cancel real jobs just to verify deployment.

PostgreSQL chooses one background coordinator across app hosts. If cutting over,
stop the previous app; this does not stop existing Slurm jobs. Both hosts continue
to reference the same cluster database and files.

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
