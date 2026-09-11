"""Strict quota parsing and an authoritative fallback for omitted idle accounts."""

import re


def account_gpu_quota(output, account, columns, *, fallback_columns=None):
    fallback = {gpu: index for index, gpu in enumerate(fallback_columns or columns, start=1)}
    header = None
    for line in output.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells[0].lower() == "account":
            header = {cell.strip().lower(): index for index, cell in enumerate(cells)}
        elif cells[0] == account:
            values = {}
            for gpu in columns:
                position = header.get(gpu, -1) if header is not None else fallback.get(gpu, -1)
                if position < 0 or position >= len(cells):
                    raise ValueError(f"gpu_usage is missing the quota column for {gpu}")
                match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", cells[position])
                if not match:
                    raise ValueError(f"gpu_usage returned an invalid quota value for {gpu}: {cells[position]}")
                values[gpu] = tuple(map(int, match.groups()))
            return values
    return None


def idle_partition_quota(account, partition):
    """Shipped to the gateway; never infer zero allocation from a missing row."""
    import re
    import subprocess

    def read(argv):
        return subprocess.check_output(argv, text=True, timeout=15).strip()

    metadata = read(["scontrol", "show", "partition", partition, "--oneliner"])
    fields = dict(item.split("=", 1) for item in metadata.split() if "=" in item)
    if fields.get("PartitionName") != partition:
        raise ValueError("Slurm did not return the requested normal partition")
    qos = fields.get("QoS") or fields.get("QOS")
    if not qos or not re.fullmatch(r"[A-Za-z0-9_.-]+", qos):
        raise ValueError("The normal partition has no explicit QOS; its idle quota cannot be verified")
    raw = read(["sacctmgr", "--noheader", "--parsable2", "show", "qos", "where",
                "name=" + qos, "format=Name,MaxTRESPU,GrpTRES"])
    rows = [line.split("|") for line in raw.splitlines() if line.split("|", 1)[0] == qos]
    if len(rows) != 1 or len(rows[0]) < 3:
        raise ValueError("Slurm did not return the normal partition's QOS limits")
    limits = {}
    for cell in rows[0][1:3]:
        for item in filter(None, cell.split(",")):
            resource, separator, value = item.partition("=")
            if resource == "gres/gpu" or resource.startswith("gres/gpu:"):
                if not separator or not value.isdigit():
                    raise ValueError("Slurm returned an invalid GPU limit")
                key = resource.removeprefix("gres/")
                limits[key] = min(limits.get(key, int(value)), int(value))
    # gpu_usage excludes overcap/scavenger from the normal account's usage.
    # Require the entire normal partition to be idle, including transitional
    # allocations, so another account cannot consume its partition QOS budget.
    allocations = read(["squeue", "--noheader", "--partition=" + partition,
                        "--states=RUNNING,COMPLETING,CONFIGURING,SUSPENDED,RESIZING", "--format=%i"])
    if allocations:
        raise ValueError("The account is absent from gpu_usage but the normal partition has active allocations; refresh its quota")
    if not limits:
        raise ValueError("Slurm returned no explicit GPU limits for the idle normal account")
    return {"account": account, "partition": partition, "qos": qos, "limits": limits,
            "active_allocations": 0}
