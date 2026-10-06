"""Cluster facts live in config/clusters, never in code, pages or shipped configuration.

Comment lines are not scanned: prose may name the current cluster. Each exception
below names a file that still carries a literal on purpose, with the reason.
"""

import re
import subprocess
from pathlib import Path, PurePosixPath

from skynet_app.cluster_config import CLUSTER

ROOT = Path(__file__).resolve().parents[1]
GENERATED = {"static/hands-viewer.js", "static/episode-scene.js"}  # Built from frontend/ by esbuild.
SUFFIXES = (".py", ".js", ".html", ".css", ".json", ".sh", ".sbatch")
SCANNED = ("skynet_app/", "ops/", "frontend/", "static/", "config/", "deploy/")
SKIPPED = ("config/clusters/", "static/vendor/")
# Literals that stay until the change named here lands; each is checked by another test.
EXCEPTIONS = {
    "config/live_video.json": "frozen into DexVerse evaluation identity; replaced by a queue_policy/runtime_profile loader after the HAT study",
    "config/collection_adapters/dexverse-cloudxr.json": "the DexVerse seed's GPU requirement and Isaac Lab checkout; validated against the configured aliases and profiles",
    "skynet_app/adapters/__init__.py": "adapter manifest defaults are part of each adapter's identity; validated against the configured aliases",
    "skynet_app/adapters/act_manifest.py": "adapter identity; validated against the configured aliases",
    "skynet_app/adapters/dp_manifest.py": "adapter identity; validated against the configured aliases",
    "skynet_app/adapters/egoverse_manifest.py": "adapter identity; validated against the configured aliases",
    "skynet_app/adapters/hat_manifest.py": "adapter identity; validated against the configured aliases",
    "skynet_app/adapters/evaluation_workers.py": "shipped inside evaluation capsules; changing it changes every adapter's evaluator identity",
    "ops/datasets/observation_render.py": "a docstring shipped inside conversion capsules; changing it changes conversion identity",
}
COMMENT = re.compile(r"^\s*(#|//|/\*|\*|<!--|--)")


def token(value: str, *, flags: int = 0) -> re.Pattern[str]:
    return re.compile(r"(?<![\w-])" + re.escape(value) + r"(?![\w-])", flags)


def cluster_fact_patterns() -> dict[str, re.Pattern[str]]:
    patterns = {f"gateway {host}": token(host) for host in CLUSTER.gateways}
    for name, queue in CLUSTER.queues.items():
        patterns[f"partition {queue.partition}"] = token(queue.partition)
        patterns[f"account {queue.account}"] = token(queue.account)
        patterns[f"queue key {name!r}"] = re.compile(r"([\"'])" + re.escape(name) + r"\1")
    for alias, target in CLUSTER.gpu_aliases.items():
        if target:  # The alias without a target is the generic "any".
            patterns[f"GPU {target}"] = re.compile(
                r"(?<![\w-])" + re.escape(target).replace("_", "[ _-]?") + r"(?![\w-])", re.IGNORECASE)
    if CLUSTER.isaac_evaluation_placement:
        for node in CLUSTER.isaac_evaluation_placement.nodes:
            patterns[f"node {node}"] = token(node)
    for partition in CLUSTER.dashboard.overflow_partitions:
        patterns[f"overflow partition {partition}"] = token(partition)
    patterns["overflow label"] = re.compile(re.escape(CLUSTER.dashboard.overflow_account_label))
    for name, value in CLUSTER.paths.model_dump().items():
        patterns[f"path {name}"] = re.compile(re.escape(value) + r"(?![\w-])")
    patterns["user root"] = re.compile(re.escape(str(PurePosixPath(CLUSTER.paths.work_root).parent)) + "/")
    patterns["home root"] = re.compile(re.escape(CLUSTER.paths.home_root) + r"(?![\w-])")
    for name, value in CLUSTER.commands.model_dump().items():
        if isinstance(value, str) and value.startswith("/"):
            patterns[f"command {name}"] = re.compile(re.escape(value))
    return patterns


def scanned_files() -> list[str]:
    listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", *SCANNED], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.split()
    return [path for path in listed if path.endswith(SUFFIXES) and not path.startswith(SKIPPED) and path not in GENERATED]


def test_configured_cluster_facts_do_not_appear_in_code_pages_or_shipped_configuration():
    patterns = cluster_fact_patterns()
    found, unused_exceptions = [], set(EXCEPTIONS)
    for path in scanned_files():
        text = (ROOT / path).read_text(errors="ignore")
        hits = [f"{path}:{number} ({name})" for number, line in enumerate(text.splitlines(), 1)
                if not COMMENT.match(line) for name, pattern in patterns.items() if pattern.search(line)]
        if not hits:
            continue
        if path in EXCEPTIONS:
            unused_exceptions.discard(path)
            continue
        found.extend(hits)
    assert not found, "Cluster facts belong in config/clusters:\n" + "\n".join(found)
    assert not unused_exceptions, "Exceptions no longer needed: " + ", ".join(sorted(unused_exceptions))


def test_forms_take_cluster_numbers_from_rendered_placeholders():
    # Resource inputs carry no literal defaults or bounds; index() renders them from the configuration.
    html = (ROOT / "static/index.html").read_text()
    for element in ("experiment-gpu-count", "resource-nodes", "resource-cpus", "resource-memory", "resource-time",
                    "checkpoint-max-attempts", "evaluation-max-attempts", "collection-gpu-count", "collection-cpu-count",
                    "collection-memory-gb", "collection-time-limit", "data-import-cpus", "data-import-memory", "data-import-time"):
        tag = re.search(rf'<(?:input|select)[^>]*\bid="{element}"[^>]*>', html.replace("\n", " "))
        assert tag, element
        literal = re.findall(r'\b(?:value|max)="(\d[\d:]*)"', tag.group(0))
        assert literal in ([], ["1"]), f"{element} carries a literal {literal}"
    assert not re.search(r"CPUS_PER_GPU\s*=\s*\d", (ROOT / "static/app.js").read_text())
