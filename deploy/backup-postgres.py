#!/usr/bin/env python3
"""Create an atomic, private PostgreSQL backup through the local Unix socket."""

import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--database", default="skynet")
    parser.add_argument("--keep", type=int, default=28)
    args = parser.parse_args()
    if args.keep < 2:
        parser.error("Keep at least two verified backups")
    os.umask(0o077)
    args.destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    name = "skynet-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    temporary = args.destination / (name + ".partial")
    final = args.destination / (name + ".dump")
    try:
        subprocess.run(
            [
                str(args.bin / "pg_dump"),
                "-h",
                args.socket,
                "-p",
                "55432",
                "-d",
                args.database,
                "--format=custom",
                "--file",
                str(temporary),
            ],
            check=True,
        )
        subprocess.run(
            [str(args.bin / "pg_restore"), "--list", str(temporary)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(final)
        for old in sorted(args.destination.glob("skynet-*.dump"), reverse=True)[
            args.keep :
        ]:
            old.unlink()
        print(final)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
