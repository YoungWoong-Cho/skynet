"""Login host names live in the cluster configuration, never in code or pages."""

import re
from pathlib import Path

from skynet_app.cluster_config import CLUSTER

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = {"episode-scene.js", "hands-viewer.js"}  # Built from frontend/ by esbuild.


def test_configured_gateway_names_do_not_appear_in_code_or_pages():
    names = re.compile(r"\b(" + "|".join(map(re.escape, CLUSTER.gateways)) + r")\b")
    sources = [
        *(ROOT / "skynet_app").rglob("*.py"),
        *(ROOT / "ops").rglob("*.py"),
        *(ROOT / "frontend").glob("*.js"),
        *(path for path in (ROOT / "static").glob("*.js") if path.name not in BUNDLES),
        ROOT / "static/index.html",
    ]
    found = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in sources
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1)
        if names.search(line)
    ]
    assert not found, "Gateway host names belong in config/clusters: " + ", ".join(found)
