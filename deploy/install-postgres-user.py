#!/usr/bin/env python3
"""Install a private, reboot-persistent PostgreSQL user service (no sudo)."""

import argparse
import os
import pwd
import subprocess
from pathlib import Path


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    binary = args.bin.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.stat().st_uid != os.getuid():
        raise SystemExit("Database root must belong to this user")
    root.chmod(0o700)
    user = pwd.getpwuid(os.getuid()).pw_name
    linger = run(
        "loginctl",
        "show-user",
        user,
        "-p",
        "Linger",
        "--value",
        capture_output=True,
        text=True,
    ).stdout.strip()
    if linger != "yes":
        raise SystemExit(
            "Enable lingering for this user before installing a persistent DB service"
        )
    data = root / "data"
    socket = Path("/run/user") / str(os.getuid()) / "skynet-postgres"
    socket.mkdir(mode=0o700, exist_ok=True)
    socket.chmod(0o700)
    if not (data / "PG_VERSION").exists():
        run(
            str(binary / "initdb"),
            "-D",
            str(data),
            "--data-checksums",
            "--auth-local=peer",
            "--auth-host=scram-sha-256",
            "--no-locale",
            "--encoding=UTF8",
        )
    config = root / "server.conf"
    config.write_text(
        "\n".join(
            [
                "listen_addresses = ''",
                "port = 55432",
                "unix_socket_directories = '" + str(socket) + "'",
                "unix_socket_permissions = 0700",
                "max_connections = 40",
                "shared_buffers = '64MB'",
                "work_mem = '4MB'",
                "maintenance_work_mem = '64MB'",
                "fsync = on",
                "synchronous_commit = on",
                "full_page_writes = on",
                "log_min_messages = warning",
                "log_statement = 'none'",
                "timezone = 'UTC'",
                "password_encryption = 'scram-sha-256'",
                "",
            ]
        )
    )
    config.chmod(0o600)
    unit = Path.home() / ".config/systemd/user"
    unit.mkdir(parents=True, exist_ok=True)
    (unit / "skynet-postgres.service").write_text(f"""[Unit]
Description=Skynet central PostgreSQL (private SSH-accessible socket)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={binary}/postgres -D {data} -c config_file={config}
RuntimeDirectory=skynet-postgres
RuntimeDirectoryMode=0700
UMask=0077
NoNewPrivileges=yes
Restart=on-failure
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=60

[Install]
WantedBy=default.target
""")
    backup_source = Path(__file__).with_name("backup-postgres.py")
    if backup_source.is_file():
        backup = root / "backup-postgres.py"
        backup.write_bytes(backup_source.read_bytes())
        backup.chmod(0o700)
        (unit / "skynet-postgres-backup.service").write_text(f"""[Unit]
Description=Back up the Skynet central database
Requires=skynet-postgres.service
After=skynet-postgres.service

[Service]
Type=oneshot
UMask=0077
ExecStart=/usr/bin/python3 {backup} --bin {binary} --socket {socket} --destination {root}/backups
""")
        (unit / "skynet-postgres-backup.timer").write_text("""[Unit]
Description=Skynet database backup every six hours

[Timer]
OnCalendar=*-*-* 00,06,12,18:15:00
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
""")
    run("systemctl", "--user", "daemon-reload")
    run("systemctl", "--user", "enable", "--now", "skynet-postgres.service")
    import time

    for _ in range(30):
        if (
            subprocess.run(
                [str(binary / "pg_isready"), "-h", str(socket), "-p", "55432"],
                stdout=subprocess.DEVNULL,
            ).returncode
            == 0
        ):
            break
        time.sleep(1)
    else:
        raise SystemExit("PostgreSQL did not become ready")
    found = run(
        str(binary / "psql"),
        "-h",
        str(socket),
        "-p",
        "55432",
        "-d",
        "postgres",
        "-Atc",
        "SELECT 1 FROM pg_database WHERE datname='skynet'",
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not found:
        run(str(binary / "createdb"), "-h", str(socket), "-p", "55432", "skynet")
    if backup_source.is_file():
        run("systemctl", "--user", "enable", "--now", "skynet-postgres-backup.timer")
    print("Database is ready; SSH socket:", socket / ".s.PGSQL.55432")


if __name__ == "__main__":
    main()
