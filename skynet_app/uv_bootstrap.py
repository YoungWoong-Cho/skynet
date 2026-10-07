"""Shell prelude shared by every cluster job script: workspace exports and uv discovery.

uv itself is astral-sh's installer and runner. Skynet-specific is the order in
which a job looks for it (an explicit executable, the pinned bootstrap under
``$UV_CACHE_DIR``, a user install under ``$WORK_ROOT`` or ``$HOME``, the PATH,
then the PyPI package through the given interpreter) and the venv+pip bootstrap
that installs the configured ``defaults.uv_version`` when nothing is found.
"""

from __future__ import annotations

import shlex

from .cluster_config import ClusterPaths
from .workspace_storage import personal_directories


UV_MISSING = (
    "TODO/preflight: uv not found; configure runtime.uv_executable, an existing env, or a container"
)
# The shell function the locator defines: runs uv in whichever form was found.
UV_COMMAND = "skynet_uv"


def workspace_exports(paths: ClusterPaths) -> dict[str, str]:
    """The workspace variables a job exports before touching uv or the caches."""
    return {
        "HOME": paths.home_root,
        "WORK_ROOT": paths.work_root,
        "XDG_CACHE_HOME": f"{paths.work_root}/.cache",
        "UV_CACHE_DIR": paths.uv_cache,
        "HF_HOME": paths.huggingface_cache,
        "TORCH_HOME": paths.torch_cache,
    }


def workspace_directories_line(paths: ClusterPaths) -> str:
    """One ``mkdir -p`` for the cache root and every directory of the workspace root."""
    directories = " ".join(
        shlex.quote(f"{paths.work_root}/{relative}")
        for relative in personal_directories(paths.work_root)
    )
    return f'mkdir -p "$XDG_CACHE_HOME" {directories}'


def workspace_prelude_lines(paths: ClusterPaths) -> list[str]:
    """Export the workspace variables, then create the layout they point at."""
    exports = [
        f"export {key}={shlex.quote(value)}" for key, value in workspace_exports(paths).items()
    ]
    return [*exports, workspace_directories_line(paths)]


def uv_module_command(python: str = "python3") -> str:
    """How uv runs when only its PyPI package is importable from ``python``."""
    return f"{shlex.quote(python)} -m uv"


def uv_locator_lines(
    *,
    uv_version: str,
    bootstrap: bool,
    python: str = "python3",
    explicit: str | None = None,
) -> list[str]:
    """Set ``UV_BIN`` and define ``skynet_uv``; exit 69 when no uv is found or installed.

    Needs ``UV_CACHE_DIR``, ``WORK_ROOT`` and ``HOME`` exported first
    (see :func:`workspace_prelude_lines`).
    """
    module = uv_module_command(python)
    interpreter = shlex.quote(python)
    lines = [
        f'UV_BOOTSTRAP_ROOT="$UV_CACHE_DIR"/bootstrap-{shlex.quote(uv_version)}',
        "UV_BIN=",
    ]
    if explicit:
        lines.append(
            f"if [[ -x {shlex.quote(explicit)} ]]; then UV_BIN={shlex.quote(explicit)}; fi"
        )
    lines.extend(
        [
            'if [[ -z "$UV_BIN" && -x "$UV_BOOTSTRAP_ROOT/bin/uv" ]]; then UV_BIN="$UV_BOOTSTRAP_ROOT/bin/uv"; fi',
            'if [[ -z "$UV_BIN" && -x "$WORK_ROOT/.local/bin/uv" ]]; then UV_BIN="$WORK_ROOT/.local/bin/uv"; fi',
            'if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then UV_BIN="$HOME/.local/bin/uv"; fi',
            'if [[ -z "$UV_BIN" ]] && command -v uv >/dev/null 2>&1; then UV_BIN="$(command -v uv)"; fi',
            f'if [[ -z "$UV_BIN" ]] && {interpreter} -c "import uv" >/dev/null 2>&1; then UV_BIN={shlex.quote(module)}; fi',
        ]
    )
    if bootstrap:
        lines.extend(
            [
                'if [[ -z "$UV_BIN" ]]; then',
                f'  {interpreter} -m venv "$UV_BOOTSTRAP_ROOT" || {{ echo "TODO/preflight: python venv unavailable for uv bootstrap" >&2; exit 69; }}',
                f'  "$UV_BOOTSTRAP_ROOT/bin/python" -m pip install --disable-pip-version-check --no-input "uv=={uv_version}" || {{ echo "TODO/preflight: pinned uv bootstrap failed" >&2; exit 69; }}',
                '  UV_BIN="$UV_BOOTSTRAP_ROOT/bin/uv"',
                "fi",
            ]
        )
    lines.extend(
        [
            f'if [[ -z "$UV_BIN" ]]; then echo {shlex.quote(UV_MISSING)} >&2; exit 69; fi',
            f'{UV_COMMAND}() {{ if [[ "$UV_BIN" == {shlex.quote(module)} ]]; then {module} "$@"; else "$UV_BIN" "$@"; fi; }}',
        ]
    )
    return lines


__all__ = [
    "UV_COMMAND",
    "UV_MISSING",
    "uv_locator_lines",
    "uv_module_command",
    "workspace_directories_line",
    "workspace_exports",
    "workspace_prelude_lines",
]
