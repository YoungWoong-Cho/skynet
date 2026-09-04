from __future__ import annotations

import ast
import copy
import base64
import hashlib
import json
import os
import re
import shlex
import threading
import time
import tomllib
from collections import OrderedDict
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from typing import Any, Mapping
from urllib.parse import urlsplit

from .cluster_runtime import ClusterClient, ClusterError
from .cluster_config import CLUSTER


FULL_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SCP_REPOSITORY_RE = re.compile(
    r"^(?:[A-Za-z0-9._-]+@)?"
    r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?:"
    r"[A-Za-z0-9_./~+-]+$"
)
BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,254}$")
REVISION_RE = re.compile(r"^[A-Za-z0-9._/+~-]{1,128}$")
INSPECTION_FILES = (
    ".skynet.json",
    ".skynet.toml",
    ".skynet.yml",
    ".skynet.yaml",
    "uv.lock",
    "pyproject.toml",
    "conda-lock.yml",
    "conda-lock.yaml",
    "environment.yml",
    "environment.yaml",
    "requirements.txt",
    "requirements-dev.txt",
    "poetry.lock",
    "Pipfile.lock",
    "Dockerfile",
    "Containerfile",
    "Apptainer.def",
    "Singularity.def",
    "README.md",
    "README.rst",
)
TEXT_INSPECTION_FILES = {
    ".skynet.json",
    ".skynet.toml",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
}
CHOICE_MAX_FILES = 16
CHOICE_MAX_FILE_BYTES = 262_144
CHOICE_MAX_TOTAL_BYTES = 2_097_152
CHOICE_MAX_AST_DEPTH = 8
YAML_CATALOG_MAX_FILES = 128


def _bounded_environment_integer(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


class _StaticRegistryEvaluator:
    """Resolve a deliberately small Python registry grammar without executing it."""

    def __init__(self, contents: Mapping[str, bytes], rule: Mapping[str, Any]) -> None:
        self.rule = rule
        self.trees: dict[str, ast.Module] = {}
        self.warnings: list[str] = []
        self.complete = True
        self._module_paths: dict[str, list[str]] = {}
        for path, content in contents.items():
            try:
                source = content.decode("utf-8")
                self.trees[path] = ast.parse(source, filename=path)
            except (UnicodeDecodeError, SyntaxError) as error:
                self._fail(f"{path} could not be parsed as UTF-8 Python: {error}")
                continue
            stem = path[:-3] if path.endswith(".py") else path
            parts = stem.split("/")
            if parts[-1:] == ["__init__"]:
                parts = parts[:-1]
            for index in range(len(parts)):
                module = ".".join(parts[index:])
                self._module_paths.setdefault(module, []).append(path)

    def _fail(self, message: str) -> None:
        self.complete = False
        if message not in self.warnings:
            self.warnings.append(message)

    @staticmethod
    def _call_name(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _imports(self, path: str) -> dict[str, tuple[str, str | None]]:
        aliases: dict[str, tuple[str, str | None]] = {}
        tree = self.trees[path]
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".", 1)[0]
                    aliases[local] = (alias.name, None)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for alias in node.names:
                    if alias.name != "*":
                        aliases[alias.asname or alias.name] = (node.module, alias.name)
        return aliases

    def _module_path(self, module: str) -> str | None:
        matches = sorted(set(self._module_paths.get(module, [])))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            self._fail(
                f"imported registry module {module} was not declared in choice_source supporting_files"
            )
        else:
            self._fail(f"imported registry module {module} is ambiguous across declared files")
        return None

    def _factory_target(self, path: str, call: ast.Call) -> tuple[str, str] | None:
        imports = self._imports(path)
        if isinstance(call.func, ast.Name):
            imported = imports.get(call.func.id)
            if imported and imported[1] is not None:
                target_path = self._module_path(imported[0])
                return (target_path, imported[1]) if target_path else None
            return path, call.func.id
        if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
            imported = imports.get(call.func.value.id)
            if imported and imported[1] is None:
                target_path = self._module_path(imported[0])
                return (target_path, call.func.attr) if target_path else None
        self._fail(f"{path} contains an unsupported registry factory reference")
        return None

    @staticmethod
    def _literal_value(node: ast.expr) -> tuple[bool, Any]:
        if isinstance(node, ast.Constant) and isinstance(
            node.value, (str, int, float, bool, type(None))
        ):
            return True, node.value
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
            and not isinstance(node.operand.value, bool)
        ):
            value = node.operand.value
            return True, value if isinstance(node.op, ast.UAdd) else -value
        return False, None

    def _constructor_default_nodes(self, constructor: str) -> dict[str, ast.expr]:
        entrypoint = str(self.rule["entrypoint"])
        tree = self.trees.get(entrypoint)
        if tree is None:
            return {}
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == constructor
        ]
        if len(classes) != 1:
            return {}
        defaults: dict[str, ast.expr] = {}
        for node in classes[0].body:
            name: str | None = None
            value: ast.expr | None = None
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                name, value = node.targets[0].id, node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name, value = node.target.id, node.value
            if name is None or value is None:
                continue
            if isinstance(value, ast.Call) and self._call_name(value.func) == "field":
                default = next(
                    (
                        keyword.value
                        for keyword in value.keywords
                        if keyword.arg == "default"
                    ),
                    None,
                )
                if default is None:
                    continue
                value = default
            defaults[name] = value
        return defaults

    @staticmethod
    def _nested_expression(node: ast.expr, parts: list[str]) -> ast.expr | None:
        current = node
        for part in parts:
            if not isinstance(current, ast.Call):
                return None
            matches = [
                keyword.value for keyword in current.keywords if keyword.arg == part
            ]
            if len(matches) != 1:
                return None
            current = matches[0]
        return current

    def _constructor_metadata(
        self, path: str, node: ast.Call, constructor: str
    ) -> dict[str, Any]:
        defaults = self._constructor_default_nodes(constructor)
        keywords = {
            keyword.arg: keyword.value
            for keyword in node.keywords
            if keyword.arg is not None
        }
        values: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        for binding in self.rule.get("metadata_fields") or []:
            if not isinstance(binding, Mapping):
                continue
            source_path = str(binding.get("source_path") or "")
            canonical_path = str(binding.get("canonical_path") or "")
            parts = source_path.split(".")
            root = keywords.get(parts[0])
            origin = "constructor_argument"
            if root is None:
                root = defaults.get(parts[0])
                origin = "constructor_class_default"
            if root is None:
                continue
            expression = self._nested_expression(root, parts[1:])
            if expression is None:
                continue
            exact, value = self._literal_value(expression)
            if not exact:
                continue
            value_map = binding.get("value_map")
            if isinstance(value_map, Mapping) and value_map:
                if str(value) not in value_map:
                    continue
                value = value_map[str(value)]
            values[canonical_path] = value
            evidence[canonical_path] = {
                "file": (
                    path
                    if origin == "constructor_argument"
                    else str(self.rule["entrypoint"])
                ),
                "source_path": source_path,
                "origin": origin,
                "line": getattr(expression, "lineno", None),
                "column": getattr(expression, "col_offset", None),
                "expression_sha256": hashlib.sha256(
                    ast.dump(expression, include_attributes=False).encode("utf-8")
                ).hexdigest(),
            }
        return {"values": values, "evidence": evidence}

    def _constructor_value(
        self, path: str, node: ast.expr
    ) -> list[dict[str, Any]]:
        constructor = str(self.rule["constructor"])
        keyword_name = str(self.rule.get("value_keyword") or "name")
        if not isinstance(node, ast.Call) or self._call_name(node.func) != constructor:
            self._fail(f"{path} registry contains an expression other than {constructor}(...)")
            return []
        keywords = [keyword for keyword in node.keywords if keyword.arg == keyword_name]
        if len(keywords) != 1 or not isinstance(keywords[0].value, ast.Constant):
            self._fail(f"{path} has a {constructor} with a non-literal {keyword_name}")
            return []
        value = keywords[0].value.value
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 255
            or any(character in value for character in ("\x00", "\n", "\r"))
        ):
            self._fail(f"{path} has an invalid literal {constructor}.{keyword_name}")
            return []
        return [
            {
                "choice": value,
                **self._constructor_metadata(path, node, constructor),
            }
        ]

    def _factory_values(
        self,
        path: str,
        call: ast.Call,
        depth: int,
        active: set[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        if call.args or call.keywords:
            self._fail(f"{path} registry factory calls must have no arguments")
            return []
        target = self._factory_target(path, call)
        if target is None:
            return []
        target_path, function_name = target
        marker = (target_path, function_name)
        if marker in active:
            self._fail(f"registry factory cycle detected at {target_path}:{function_name}")
            return []
        if depth >= CHOICE_MAX_AST_DEPTH:
            self._fail("repository choice registry exceeded the static recursion limit")
            return []
        tree = self.trees.get(target_path)
        if tree is None:
            self._fail(f"registry factory source is unavailable: {target_path}")
            return []
        functions = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
        ]
        if len(functions) != 1:
            self._fail(f"registry factory {target_path}:{function_name} was not uniquely defined")
            return []
        returns = [node for node in functions[0].body if isinstance(node, ast.Return)]
        if len(returns) != 1 or returns[0].value is None:
            self._fail(f"registry factory {target_path}:{function_name} has no single literal return")
            return []
        return self._sequence_values(
            target_path, returns[0].value, depth + 1, {*active, marker}
        )

    def _sequence_values(
        self,
        path: str,
        node: ast.expr,
        depth: int,
        active: set[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return [
                *self._sequence_values(path, node.left, depth, active),
                *self._sequence_values(path, node.right, depth, active),
            ]
        if not isinstance(node, (ast.List, ast.Tuple)):
            self._fail(f"{path} registry must be a literal list or tuple")
            return []
        values: list[dict[str, Any]] = []
        for element in node.elts:
            if isinstance(element, ast.Starred):
                if not isinstance(element.value, ast.Call):
                    self._fail(f"{path} registry has an unsupported starred expression")
                    continue
                values.extend(self._factory_values(path, element.value, depth, active))
            else:
                values.extend(self._constructor_value(path, element))
        return values

    def resolve(self) -> dict[str, Any]:
        entrypoint = str(self.rule["entrypoint"])
        registry = str(self.rule["registry"])
        tree = self.trees.get(entrypoint)
        if tree is None:
            self._fail(f"choice_source entrypoint is unavailable: {entrypoint}")
            return {"choices": [], "complete": False, "warnings": self.warnings}
        assignments: list[ast.expr] = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == registry for target in node.targets
            ):
                assignments.append(node.value)
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == registry
                and node.value is not None
            ):
                assignments.append(node.value)
        if len(assignments) != 1:
            self._fail(f"{entrypoint} does not define exactly one {registry} registry")
            return {"choices": [], "complete": False, "warnings": self.warnings}
        records = self._sequence_values(entrypoint, assignments[0], 0, set())
        choices = [str(record["choice"]) for record in records]
        unique = list(dict.fromkeys(choices))
        if len(unique) != len(choices):
            self._fail(f"{entrypoint}:{registry} contains duplicate choice names")
        if not unique:
            self._fail(f"{entrypoint}:{registry} did not yield any literal choices")
        metadata = {
            str(record["choice"]): {
                "values": dict(record.get("values") or {}),
                "evidence": dict(record.get("evidence") or {}),
            }
            for record in records
        }
        return {
            "choices": unique,
            "metadata": metadata,
            "complete": self.complete,
            "warnings": self.warnings,
        }


def _resolve_python_static_registry(
    contents: Mapping[str, bytes], rule: Mapping[str, Any]
) -> dict[str, Any]:
    return _StaticRegistryEvaluator(contents, rule).resolve()


def _resolve_python_enum(
    contents: Mapping[str, bytes], rule: Mapping[str, Any]
) -> dict[str, Any]:
    entrypoint = str(rule.get("entrypoint") or "")
    class_name = str(rule.get("registry") or "")
    value_mode = str(rule.get("value_keyword") or "name")
    content = contents.get(entrypoint)
    if content is None:
        return {
            "choices": [],
            "complete": False,
            "warnings": [f"choice_source entrypoint is unavailable: {entrypoint}"],
        }
    try:
        tree = ast.parse(content.decode("utf-8"), filename=entrypoint)
    except (UnicodeDecodeError, SyntaxError) as error:
        return {
            "choices": [],
            "complete": False,
            "warnings": [f"{entrypoint} could not be parsed as UTF-8 Python: {error}"],
        }
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1:
        return {
            "choices": [],
            "complete": False,
            "warnings": [f"{entrypoint} does not define exactly one {class_name} class"],
        }
    choices: list[str] = []
    for node in classes[0].body:
        target: ast.Name | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target, node.value
        if (
            target is None
            or target.id.startswith("_")
            or not isinstance(value, ast.Constant)
            or not isinstance(value.value, str)
        ):
            continue
        choices.append(target.id if value_mode == "name" else value.value)
    unique = list(dict.fromkeys(choices))
    warnings: list[str] = []
    if len(unique) != len(choices):
        warnings.append(f"{entrypoint}:{class_name} contains duplicate {value_mode} choices")
    if not unique:
        warnings.append(f"{entrypoint}:{class_name} did not yield any literal string enum choices")
    return {"choices": unique, "complete": not warnings, "warnings": warnings}


_YAML_MAPPING_RE = re.compile(
    r"^(?P<key>[A-Za-z_][A-Za-z0-9_-]*)\s*:(?P<value>.*)$"
)
_YAML_INTEGER_RE = re.compile(r"[-+]?(?:0|[1-9][0-9_]*)")
_YAML_NUMBER_RE = re.compile(
    r"[-+]?(?:(?:0|[1-9][0-9_]*)(?:\.[0-9_]*)?|\.[0-9_]+)"
    r"(?:[eE][-+]?[0-9_]+)?"
)


def _strip_yaml_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for index, character in enumerate(value):
        if quote == '"':
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif quote == "'":
            if character == quote:
                if index + 1 < len(value) and value[index + 1] == quote:
                    continue
                quote = None
        elif character in {"'", '"'}:
            quote = character
        elif character == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.rstrip()


def _static_yaml_scalar(expression: str) -> tuple[bool, Any]:
    value = _strip_yaml_comment(expression).strip()
    if not value or value[0] in "!&*[{>|":
        return False, None
    if value.startswith('"') and value.endswith('"'):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return False, None
        return (True, decoded) if isinstance(decoded, str) else (False, None)
    if value.startswith("'") and value.endswith("'"):
        return True, value[1:-1].replace("''", "'")
    normalized = value.casefold()
    if normalized in {"null", "~"}:
        return True, None
    if normalized in {"true", "false"}:
        return True, normalized == "true"
    compact = value.replace("_", "")
    if _YAML_INTEGER_RE.fullmatch(value):
        try:
            return True, int(compact, 10)
        except ValueError:
            return False, None
    if _YAML_NUMBER_RE.fullmatch(value):
        try:
            return True, float(compact)
        except ValueError:
            return False, None
    if any(character in value for character in ("\x00", "\n", "\r")):
        return False, None
    return True, value


def _yaml_scalar_paths(
    content: bytes, paths: set[str], source_file: str
) -> tuple[dict[str, tuple[Any, int, int, str]], list[str]]:
    try:
        source = content.decode("utf-8")
    except UnicodeDecodeError as error:
        return {}, [f"{source_file} could not be parsed as UTF-8 YAML: {error}"]
    found: dict[str, tuple[Any, int, int, str]] = {}
    warnings: list[str] = []
    stack: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith(("#", "---", "...")):
            continue
        leading = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        if "\t" in leading:
            continue
        indentation = len(leading)
        line = raw_line[indentation:]
        if line.startswith("-"):
            continue
        match = _YAML_MAPPING_RE.fullmatch(line)
        if match is None:
            continue
        while stack and indentation <= stack[-1][0]:
            stack.pop()
        key = match.group("key")
        path = ".".join([*(item[1] for item in stack), key])
        raw_value = match.group("value")
        expression = _strip_yaml_comment(raw_value).strip()
        if not expression:
            stack.append((indentation, key))
            continue
        if not any(_yaml_path_matches(pattern, path) for pattern in paths):
            continue
        if path in found:
            warnings.append(f"{source_file} contains duplicate YAML path {path}")
            continue
        exact, value = _static_yaml_scalar(expression)
        if not exact:
            warnings.append(
                f"{source_file}:{line_number} YAML path {path} is not a literal scalar"
            )
            continue
        value_column = raw_line.index(raw_value) + len(raw_value) - len(raw_value.lstrip())
        found[path] = (value, line_number, value_column, expression)
    return found, warnings


def _yaml_path_matches(pattern: str, path: str) -> bool:
    expected = pattern.split(".")
    actual = path.split(".")
    return len(expected) == len(actual) and all(
        wanted == "*" or wanted == observed
        for wanted, observed in zip(expected, actual, strict=True)
    )


def _resolve_yaml_static_mapping(
    contents: Mapping[str, bytes], rule: Mapping[str, Any]
) -> dict[str, Any]:
    choice = str(rule.get("choice") or "")
    fields = [
        dict(field)
        for field in rule.get("metadata_fields") or []
        if isinstance(field, Mapping)
    ]
    by_file: dict[str, set[str]] = {}
    for field in fields:
        source_file = str(field.get("source_file") or "")
        source_path = str(field.get("source_path") or "")
        by_file.setdefault(source_file, set()).add(source_path)

    parsed: dict[str, dict[str, tuple[Any, int, int, str]]] = {}
    warnings: list[str] = []
    for source_file, source_paths in by_file.items():
        content = contents.get(source_file)
        if content is None:
            warnings.append(f"YAML metadata source is unavailable: {source_file}")
            continue
        values, file_warnings = _yaml_scalar_paths(
            content, source_paths, source_file
        )
        parsed[source_file] = values
        warnings.extend(file_warnings)

    values: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for field in fields:
        source_file = str(field.get("source_file") or "")
        source_path = str(field.get("source_path") or "")
        canonical_path = str(field.get("canonical_path") or "")
        record = parsed.get(source_file, {}).get(source_path)
        if record is None:
            warnings.append(
                f"{source_file} does not expose literal YAML path {source_path}"
            )
            continue
        value, line, column, expression = record
        value_map = field.get("value_map")
        if isinstance(value_map, Mapping) and value_map:
            if str(value) not in value_map:
                warnings.append(
                    f"{source_file}:{line} YAML path {source_path} has no declared value mapping"
                )
                continue
            value = value_map[str(value)]
        values[canonical_path] = value
        evidence[canonical_path] = {
            "file": source_file,
            "source_path": source_path,
            "origin": "yaml_literal_scalar",
            "line": line,
            "column": column,
            "expression_sha256": hashlib.sha256(
                expression.encode("utf-8")
            ).hexdigest(),
        }

    metadata = {choice: {"values": values, "evidence": evidence}} if choice else {}
    if not choice:
        warnings.append("YAML static mapping does not declare a config choice")
    return {
        "choices": [choice] if choice else [],
        "metadata": metadata,
        "complete": not warnings and len(values) == len(fields),
        "warnings": list(dict.fromkeys(warnings)),
    }


def _yaml_defaults_entries(
    content: bytes, source_file: str
) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        source = content.decode("utf-8")
    except UnicodeDecodeError as error:
        return [], [f"{source_file} could not be parsed as UTF-8 YAML: {error}"]
    entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    defaults_indent: int | None = None
    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        leading = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        if "\t" in leading:
            continue
        indentation = len(leading)
        line = raw_line[indentation:]
        if defaults_indent is None:
            if indentation == 0 and re.fullmatch(r"defaults\s*:\s*(?:#.*)?", line):
                defaults_indent = indentation
            continue
        if indentation <= defaults_indent:
            break
        if not line.startswith("-"):
            continue
        expression = _strip_yaml_comment(line[1:]).strip()
        if expression == "_self_":
            entries.append({"kind": "self"})
            continue
        override = False
        optional = False
        while True:
            if expression.startswith("override "):
                override = True
                expression = expression.removeprefix("override ").strip()
                continue
            if expression.startswith("optional "):
                optional = True
                expression = expression.removeprefix("optional ").strip()
                continue
            break
        if ":" in expression:
            raw_group, raw_name = expression.split(":", 1)
            group = raw_group.strip()
            exact, name = _static_yaml_scalar(raw_name)
            if (
                not group
                or not exact
                or (name is not None and not isinstance(name, str))
            ):
                warnings.append(
                    f"{source_file}:{line_number} has a non-literal Hydra defaults entry"
                )
                continue
            entries.append(
                {
                    "kind": "group",
                    "group": group,
                    "name": name,
                    "override": override,
                    "optional": optional,
                }
            )
            continue
        exact, name = _static_yaml_scalar(expression)
        if not exact or not isinstance(name, str):
            warnings.append(
                f"{source_file}:{line_number} has a non-literal Hydra defaults entry"
            )
            continue
        entries.append(
            {
                "kind": "relative",
                "name": name,
                "override": override,
                "optional": optional,
            }
        )
    return entries, warnings


def _yaml_config_file(
    contents: Mapping[str, bytes], relative: PurePosixPath
) -> str | None:
    candidates = [str(relative)] if relative.suffix in {".yaml", ".yml"} else [
        f"{relative}.yaml",
        f"{relative}.yml",
    ]
    return next((candidate for candidate in candidates if candidate in contents), None)


def _yaml_catalog_layers(
    contents: Mapping[str, bytes], directory: str, entrypoint: str
) -> tuple[list[tuple[str, str]], list[str]]:
    root = PurePosixPath(directory)
    layers: list[tuple[str, str]] = []
    warnings: list[str] = []
    active: set[tuple[str, str]] = set()

    def visit(source_file: str, package: str) -> None:
        marker = (source_file, package)
        if marker in active:
            warnings.append(f"Hydra defaults cycle detected at {source_file}")
            return
        content = contents.get(source_file)
        if content is None:
            warnings.append(f"Hydra defaults source is unavailable: {source_file}")
            return
        active.add(marker)
        entries, entry_warnings = _yaml_defaults_entries(content, source_file)
        warnings.extend(entry_warnings)
        emitted_self = False
        current = PurePosixPath(source_file)
        try:
            current_group = current.parent.relative_to(root)
        except ValueError:
            warnings.append(f"Hydra defaults source escapes catalog directory: {source_file}")
            active.remove(marker)
            return
        for entry in entries:
            if entry["kind"] == "self":
                layers.append((source_file, package))
                emitted_self = True
                continue
            if entry["kind"] == "relative":
                relative = current.parent / str(entry["name"]).lstrip("/")
                child_package = package
            else:
                name = entry.get("name")
                if name is None:
                    continue
                raw_group = str(entry["group"])
                group_name, separator, explicit_package = raw_group.partition("@")
                absolute = group_name.startswith("/")
                group_name = group_name.lstrip("/")
                group_path = PurePosixPath(group_name)
                if not absolute and current_group != PurePosixPath("."):
                    group_path = current_group / group_path
                relative = root / group_path / str(name).lstrip("/")
                default_package = ".".join(group_path.parts)
                if not separator or explicit_package == "_group_":
                    child_package = default_package
                elif explicit_package == "_global_":
                    child_package = ""
                elif explicit_package == "_here_":
                    child_package = package
                else:
                    child_package = explicit_package.replace("/", ".")
            target = _yaml_config_file(contents, relative)
            if target is None:
                if not entry.get("optional"):
                    warnings.append(
                        f"{source_file} selects unavailable Hydra default {relative}"
                    )
                continue
            if entry.get("override") and child_package:
                layers[:] = [
                    layer
                    for layer in layers
                    if not (
                        layer[1] == child_package
                        or layer[1].startswith(child_package + ".")
                    )
                ]
            visit(target, child_package)
        if not emitted_self:
            layers.append((source_file, package))
        active.remove(marker)

    visit(entrypoint, "")
    return layers, list(dict.fromkeys(warnings))


def _resolve_yaml_static_catalog(
    contents: Mapping[str, bytes], rule: Mapping[str, Any]
) -> dict[str, Any]:
    directory = str(rule.get("directory") or "")
    filename_pattern = str(rule.get("filename_pattern") or "")
    root = PurePosixPath(directory)
    fields = [
        dict(field)
        for field in rule.get("metadata_fields") or []
        if isinstance(field, Mapping)
    ]
    catalog: dict[str, str] = {}
    warnings: list[str] = []
    for source_file in sorted(contents):
        path = PurePosixPath(source_file)
        if path.parent != root or not fnmatchcase(path.name, filename_pattern):
            continue
        choice = path.stem
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", choice):
            warnings.append(f"YAML catalog contains an unsafe config name: {path.name}")
            continue
        if choice in catalog:
            warnings.append(f"YAML catalog contains duplicate config name: {choice}")
            catalog.pop(choice, None)
            continue
        catalog[choice] = source_file
    if not catalog:
        warnings.append(
            f"YAML catalog has no top-level files matching {directory}/{filename_pattern}"
        )

    metadata: dict[str, Any] = {}
    for choice, entrypoint in sorted(catalog.items()):
        layers, composition_warnings = _yaml_catalog_layers(
            contents, directory, entrypoint
        )
        values: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        unresolved: dict[str, Any] = {}
        choice_warnings = list(composition_warnings)
        for field in fields:
            selector = str(field.get("source_path") or "")
            canonical_path = str(field.get("canonical_path") or "")
            matches: dict[str, dict[str, Any]] = {}
            selector_warnings: list[str] = []
            for source_file, package in layers:
                prefix = f"{package}." if package else ""
                if prefix and not selector.startswith(prefix):
                    continue
                local_selector = selector[len(prefix):] if prefix else selector
                found, file_warnings = _yaml_scalar_paths(
                    contents[source_file], {local_selector}, source_file
                )
                selector_warnings.extend(file_warnings)
                for local_path, record in found.items():
                    value, line, column, expression = record
                    composed_path = prefix + local_path
                    matches[composed_path] = {
                        "file": source_file,
                        "source_path": composed_path,
                        "origin": "hydra_composed_yaml_literal_scalar",
                        "line": line,
                        "column": column,
                        "expression_sha256": hashlib.sha256(
                            expression.encode("utf-8")
                        ).hexdigest(),
                        "value": value,
                    }
            ordered_matches = [matches[path] for path in sorted(matches)]
            distinct_values = {
                json.dumps(item["value"], sort_keys=True, separators=(",", ":"))
                for item in ordered_matches
            }
            if not ordered_matches:
                status = (
                    "not_statically_resolvable"
                    if selector_warnings
                    else "not_declared"
                )
                warning = (
                    f"{choice}: no repository default is declared for {selector} "
                    "in the selected Hydra defaults chain"
                )
                unresolved[canonical_path] = {
                    "status": status,
                    "source_path": selector,
                    "warning": warning,
                }
                choice_warnings.extend(selector_warnings)
                choice_warnings.append(warning)
                continue
            if len(distinct_values) != 1:
                warning = (
                    f"{choice}: repository defaults for {selector} are ambiguous "
                    "across wildcard matches"
                )
                unresolved[canonical_path] = {
                    "status": "ambiguous",
                    "source_path": selector,
                    "warning": warning,
                    "matches": ordered_matches,
                }
                choice_warnings.extend(selector_warnings)
                choice_warnings.append(warning)
                continue
            value = ordered_matches[0]["value"]
            value_map = field.get("value_map")
            if isinstance(value_map, Mapping) and value_map:
                if str(value) not in value_map:
                    warning = (
                        f"{choice}: repository default for {selector} has no declared "
                        "value mapping"
                    )
                    unresolved[canonical_path] = {
                        "status": "unmapped",
                        "source_path": selector,
                        "warning": warning,
                    }
                    choice_warnings.append(warning)
                    continue
                value = value_map[str(value)]
            values[canonical_path] = value
            public_matches = [
                {key: item[key] for key in item if key != "value"}
                for item in ordered_matches
            ]
            first = public_matches[0]
            evidence[canonical_path] = {
                **first,
                "source_path": selector,
                "matches": public_matches,
                "collapsed_identical_matches": len(public_matches) > 1,
            }
            choice_warnings.extend(selector_warnings)
        metadata[choice] = {
            "values": values,
            "evidence": evidence,
            "unresolved": unresolved,
            "complete": not unresolved and not choice_warnings,
            "warnings": list(dict.fromkeys(choice_warnings)),
        }
    return {
        "choices": sorted(catalog),
        "metadata": metadata,
        "complete": not warnings,
        "warnings": list(dict.fromkeys(warnings)),
    }


class SourceDiscovery:
    """Discover remote Git refs through the configured SSH gateway pair."""

    def __init__(
        self,
        cluster: ClusterClient,
        *,
        cache_seconds: int | None = None,
        cache_entries: int | None = None,
    ) -> None:
        self.cluster = cluster
        self.cache_seconds = cache_seconds or _bounded_environment_integer(
            "SKYNET_SOURCE_CACHE_SECONDS", 60, 1, 600
        )
        self.cache_entries = cache_entries or _bounded_environment_integer(
            "SKYNET_SOURCE_CACHE_ENTRIES", 128, 1, 512
        )
        self._cache: OrderedDict[tuple[Any, ...], tuple[float, dict[str, Any]]] = OrderedDict()
        self._cache_lock = threading.Lock()

    @staticmethod
    def repository_url(value: str) -> str:
        repository = value.strip()
        if not repository or len(repository) > 2048:
            raise ValueError("repository URL must contain between 1 and 2048 characters")
        if any(character.isspace() or ord(character) < 32 for character in repository):
            raise ValueError("repository URL cannot contain whitespace or control characters")

        parsed = urlsplit(repository)
        if parsed.scheme:
            if parsed.scheme.lower() not in {"https", "ssh", "git"}:
                raise ValueError("repository URL must use https, ssh, or git")
            if not parsed.hostname or not parsed.path or parsed.path == "/":
                raise ValueError("repository URL must include a host and repository path")
            if parsed.password or (parsed.scheme.lower() == "https" and parsed.username):
                raise ValueError("repository URL must not contain embedded credentials")
            if parsed.query or parsed.fragment:
                raise ValueError("repository URL must not contain a query string or fragment")
            return repository

        if not SCP_REPOSITORY_RE.fullmatch(repository):
            raise ValueError(
                "repository URL must be an https/ssh/git URL or an SSH form such as git@host:owner/repo"
            )
        return repository

    @staticmethod
    def branch_name(value: str) -> str:
        branch = value.strip()
        invalid = (
            not BRANCH_RE.fullmatch(branch)
            or ".." in branch
            or "//" in branch
            or "@{" in branch
            or branch.endswith(("/", ".", ".lock"))
            or "/." in branch
        )
        if invalid:
            raise ValueError("branch contains unsupported Git ref characters")
        return branch

    def _get_cached(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            expires_at, payload = cached
            if expires_at <= now:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return copy.deepcopy(payload)

    def _set_cached(self, key: tuple[Any, ...], payload: dict[str, Any]) -> dict[str, Any]:
        with self._cache_lock:
            self._cache[key] = (time.monotonic() + self.cache_seconds, copy.deepcopy(payload))
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_entries:
                self._cache.popitem(last=False)
        return payload

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    def branches(self, repo_url: str, gateway: str = "auto") -> dict[str, Any]:
        repository = self.repository_url(repo_url)
        key = ("branches", gateway, repository)
        if cached := self._get_cached(key):
            return cached

        repository_q = shlex.quote(repository)
        command = (
            "set -eu; export GIT_TERMINAL_PROMPT=0; "
            "git -c credential.interactive=never ls-remote --symref "
            f"{repository_q} HEAD 'refs/heads/*'"
        )
        host, output = self.cluster.run_with_fallback(command, gateway, timeout=25)
        default_branch: str | None = None
        tips: dict[str, str] = {}
        for line in output.splitlines():
            if line.startswith("ref: "):
                fields = line.split("\t", 1)
                if len(fields) == 2 and fields[1] == "HEAD":
                    target = fields[0].removeprefix("ref: ")
                    if target.startswith("refs/heads/"):
                        candidate = target.removeprefix("refs/heads/")
                        try:
                            default_branch = self.branch_name(candidate)
                        except ValueError as error:
                            raise ClusterError(f"repository returned an invalid default branch: {candidate}") from error
                continue
            fields = line.split("\t", 1)
            if len(fields) != 2 or not FULL_COMMIT_RE.fullmatch(fields[0]):
                continue
            ref = fields[1]
            if not ref.startswith("refs/heads/"):
                continue
            try:
                name = self.branch_name(ref.removeprefix("refs/heads/"))
            except ValueError as error:
                raise ClusterError(f"repository returned an invalid branch ref: {ref}") from error
            tips[name] = fields[0].lower()

        if not tips:
            raise ClusterError("the repository did not expose any branch refs")
        if default_branch is not None and default_branch not in tips:
            raise ClusterError(f"repository default branch {default_branch} was not present in its branch refs")
        branches = [
            {"name": name, "sha": tips[name]}
            for name in sorted(tips, key=lambda item: (item != default_branch, item.lower()))
        ]
        return self._set_cached(
            key,
            {
                "repository": repository,
                "gateway": host,
                "default_branch": default_branch,
                "branches": branches,
            },
        )

    def commits(
        self,
        repo_url: str,
        branch: str,
        limit: int = 50,
        gateway: str = "auto",
    ) -> dict[str, Any]:
        repository = self.repository_url(repo_url)
        branch = self.branch_name(branch)
        if not 1 <= limit <= 100:
            raise ValueError("commit limit must be between 1 and 100")
        key = ("commits", gateway, repository, branch, limit)
        if cached := self._get_cached(key):
            return cached

        repository_q = shlex.quote(repository)
        ref_q = shlex.quote(f"refs/heads/{branch}")
        command = f'''set -eu
export GIT_TERMINAL_PROMPT=0
tmp=$(mktemp -d "${{TMPDIR:-/tmp}}/skynet-git-XXXXXX")
trap 'rm -rf "$tmp"' EXIT
git -C "$tmp" init -q
git -C "$tmp" -c credential.interactive=never -c protocol.version=2 fetch -q \
  --no-tags --depth={limit} --filter=blob:none {repository_q} {ref_q}
git -C "$tmp" log -n {limit} \
  --format='%H%x1f%h%x1f%s%x1f%an%x1f%cI%x1e' FETCH_HEAD
'''
        host, output = self.cluster.run_with_fallback(command, gateway, timeout=45)
        commits: list[dict[str, str]] = []
        for raw_record in output.split("\x1e"):
            record = raw_record.strip("\r\n")
            if not record:
                continue
            fields = record.split("\x1f")
            if len(fields) != 5 or not FULL_COMMIT_RE.fullmatch(fields[0]):
                continue
            commits.append(
                {
                    "sha": fields[0].lower(),
                    "short_sha": fields[1],
                    "subject": fields[2],
                    "author": fields[3],
                    "timestamp": fields[4],
                }
            )
        if not commits:
            raise ClusterError(f"branch {branch} did not return any commits")
        return self._set_cached(
            key,
            {
                "repository": repository,
                "gateway": host,
                "branch": branch,
                "commits": commits,
            },
        )

    def resolve_revision(self, repo_url: str, revision: str, gateway: str = "auto") -> str:
        repository = self.repository_url(repo_url)
        requested = revision.strip()
        if not REVISION_RE.fullmatch(requested):
            raise ValueError("revision contains unsupported Git ref characters")
        if FULL_COMMIT_RE.fullmatch(requested):
            return requested.lower()
        key = ("revision", gateway, repository, requested)
        if cached := self._get_cached(key):
            return str(cached["commit"])
        repository_q = shlex.quote(repository)
        revision_q = shlex.quote(requested)
        command = (
            "set -eu; export GIT_TERMINAL_PROMPT=0; "
            "git -c credential.interactive=never ls-remote --exit-code "
            f"{repository_q} {revision_q} {shlex.quote(requested + '^{}')}"
        )
        host, output = self.cluster.run_with_fallback(command, gateway, timeout=25)
        candidates: list[tuple[str, str]] = []
        for line in output.splitlines():
            fields = line.split(None, 1)
            if len(fields) == 2 and FULL_COMMIT_RE.fullmatch(fields[0]):
                candidates.append((fields[1], fields[0].lower()))
        if not candidates:
            raise ClusterError(f"revision {requested} did not resolve to a commit")
        peeled = [sha for ref, sha in candidates if ref.endswith("^{}")]
        if len(peeled) == 1:
            commit = peeled[0]
        elif len(peeled) > 1 or len(candidates) != 1:
            raise ClusterError(f"revision {requested} resolved ambiguously; provide a full commit SHA")
        else:
            commit = candidates[0][1]
        self._set_cached(
            key,
            {"repository": repository, "requested_revision": requested, "commit": commit, "gateway": host},
        )
        return commit

    @staticmethod
    def project_subdirectory(value: str) -> str:
        raw = value.strip()
        if raw.startswith("/"):
            raise ValueError("project_subdirectory must be relative to the repository")
        normalized = raw.strip("/") or "."
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts or any(
            character in normalized for character in ("\x00", "\n", "\r")
        ):
            raise ValueError("project_subdirectory must stay inside the repository")
        return normalized

    def _read_choice_files(
        self,
        repo_url: str,
        revision: str,
        paths: list[str],
        gateway: str,
        project_subdirectory: str,
    ) -> dict[str, Any]:
        repository = self.repository_url(repo_url)
        commit = revision.lower()
        if not FULL_COMMIT_RE.fullmatch(commit):
            commit = self.resolve_revision(repository, revision, gateway)
        subdirectory = self.project_subdirectory(project_subdirectory)
        if not 1 <= len(paths) <= CHOICE_MAX_FILES:
            raise ValueError(f"repository choice discovery supports 1-{CHOICE_MAX_FILES} files")
        normalized_paths: list[str] = []
        for path in paths:
            normalized = self.project_subdirectory(path)
            if normalized == "." or PurePosixPath(normalized).suffix not in {
                ".py",
                ".yaml",
                ".yml",
            }:
                raise ValueError(
                    "repository choice files must be relative Python or YAML files"
                )
            normalized_paths.append(normalized)
        if len(normalized_paths) != len(set(normalized_paths)):
            raise ValueError("repository choice files must be unique")
        key = (
            "choice-files", gateway, repository, commit, subdirectory, tuple(normalized_paths)
        )
        if cached := self._get_cached(key):
            return cached

        prefix = "" if subdirectory == "." else subdirectory.rstrip("/") + "/"
        requested = [prefix + path for path in normalized_paths]
        path_words = " ".join(shlex.quote(path) for path in requested)
        command = f'''set -eu
export GIT_TERMINAL_PROMPT=0
tmp=$(mktemp -d "${{TMPDIR:-/tmp}}/skynet-choices-XXXXXX")
trap 'rm -rf "$tmp"' EXIT
git -C "$tmp" init -q
git -C "$tmp" -c credential.interactive=never -c protocol.version=2 fetch -q \
  --no-tags --depth=1 --filter=blob:none {shlex.quote(repository)} {shlex.quote(commit)}
actual=$(git -C "$tmp" rev-parse FETCH_HEAD^{{commit}})
test "$actual" = {shlex.quote(commit)}
printf 'COMMIT\t%s\n' "$actual"
total=0
for path in {path_words}; do
  object="$actual:$path"
  object_type=$(git -C "$tmp" cat-file -t "$object" 2>/dev/null || true)
  if test "$object_type" != blob; then
    printf 'MISSING\t%s\n' "$path"
    continue
  fi
  size=$(git -C "$tmp" cat-file -s "$object")
  next_total=$((total + size))
  if test "$size" -gt {CHOICE_MAX_FILE_BYTES} || test "$next_total" -gt {CHOICE_MAX_TOTAL_BYTES}; then
    printf 'TOO_LARGE\t%s\t%s\n' "$path" "$size"
    continue
  fi
  total=$next_total
  sha=$(git -C "$tmp" show "$object" | sha256sum | awk '{{print $1}}')
  printf 'FILE\t%s\t%s\t%s\n' "$path" "$size" "$sha"
  printf 'CONTENT\t%s\t' "$path"
  git -C "$tmp" show "$object" | base64 -w0
  printf '\n'
done
'''
        host, output = self.cluster.run_with_fallback(command, gateway, timeout=60)
        actual_commit: str | None = None
        contents: dict[str, bytes] = {}
        files: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []
        for line in output.splitlines():
            fields = line.split("\t")
            if fields[0] == "COMMIT" and len(fields) == 2:
                actual_commit = fields[1].lower()
            elif fields[0] == "FILE" and len(fields) == 4:
                relative = fields[1][len(prefix):] if prefix and fields[1].startswith(prefix) else fields[1]
                files[relative] = {
                    "path": relative,
                    "size_bytes": int(fields[2]),
                    "sha256": fields[3].lower(),
                }
            elif fields[0] == "CONTENT" and len(fields) == 3:
                relative = fields[1][len(prefix):] if prefix and fields[1].startswith(prefix) else fields[1]
                try:
                    contents[relative] = base64.b64decode(fields[2], validate=True)
                except ValueError:
                    warnings.append(f"repository choice file has invalid transport encoding: {relative}")
            elif fields[0] == "MISSING" and len(fields) == 2:
                relative = fields[1][len(prefix):] if prefix and fields[1].startswith(prefix) else fields[1]
                warnings.append(f"repository choice file is missing: {relative}")
            elif fields[0] == "TOO_LARGE" and len(fields) == 3:
                relative = fields[1][len(prefix):] if prefix and fields[1].startswith(prefix) else fields[1]
                warnings.append(f"repository choice file exceeds inspection bounds: {relative}")
        if actual_commit != commit:
            raise ClusterError("repository choice inspection returned a different commit")
        for path in normalized_paths:
            if path not in contents and not any(path in warning for warning in warnings):
                warnings.append(f"repository choice file content was not returned: {path}")
        return self._set_cached(
            key,
            {
                "repository": repository,
                "commit": commit,
                "project_subdirectory": subdirectory,
                "gateway": host,
                "files": files,
                "contents": contents,
                "warnings": warnings,
            },
        )

    def _read_yaml_catalog(
        self,
        repo_url: str,
        revision: str,
        directory: str,
        gateway: str,
        project_subdirectory: str,
    ) -> dict[str, Any]:
        repository = self.repository_url(repo_url)
        commit = revision.lower()
        if not FULL_COMMIT_RE.fullmatch(commit):
            commit = self.resolve_revision(repository, revision, gateway)
        subdirectory = self.project_subdirectory(project_subdirectory)
        catalog_directory = self.project_subdirectory(directory)
        if catalog_directory == ".":
            raise ValueError("repository YAML catalog directory cannot be the repository root")
        key = (
            "yaml-catalog-files",
            gateway,
            repository,
            commit,
            subdirectory,
            catalog_directory,
        )
        if cached := self._get_cached(key):
            return cached

        prefix = "" if subdirectory == "." else subdirectory.rstrip("/") + "/"
        requested_directory = prefix + catalog_directory
        command = f'''set -eu
export GIT_TERMINAL_PROMPT=0
tmp=$(mktemp -d "${{TMPDIR:-/tmp}}/skynet-yaml-catalog-XXXXXX")
trap 'rm -rf "$tmp"' EXIT
git -C "$tmp" init -q
git -C "$tmp" -c credential.interactive=never -c protocol.version=2 fetch -q \
  --no-tags --depth=1 --filter=blob:none {shlex.quote(repository)} {shlex.quote(commit)}
actual=$(git -C "$tmp" rev-parse FETCH_HEAD^{{commit}})
test "$actual" = {shlex.quote(commit)}
printf 'COMMIT\t%s\n' "$actual"
git -C "$tmp" ls-tree -r --name-only "$actual" -- {shlex.quote(requested_directory)} > "$tmp/catalog-paths"
count=0
total=0
while IFS= read -r path; do
  case "$path" in
    *.yaml|*.yml) ;;
    *) continue ;;
  esac
  count=$((count + 1))
  if test "$count" -gt {YAML_CATALOG_MAX_FILES}; then
    printf 'TOO_MANY\t%s\n' "$count"
    break
  fi
  object="$actual:$path"
  size=$(git -C "$tmp" cat-file -s "$object")
  next_total=$((total + size))
  if test "$size" -gt {CHOICE_MAX_FILE_BYTES} || test "$next_total" -gt {CHOICE_MAX_TOTAL_BYTES}; then
    printf 'TOO_LARGE\t%s\t%s\n' "$path" "$size"
    continue
  fi
  total=$next_total
  sha=$(git -C "$tmp" show "$object" | sha256sum | awk '{{print $1}}')
  printf 'FILE\t%s\t%s\t%s\n' "$path" "$size" "$sha"
  printf 'CONTENT\t%s\t' "$path"
  git -C "$tmp" show "$object" | base64 -w0
  printf '\n'
done < "$tmp/catalog-paths"
'''
        host, output = self.cluster.run_with_fallback(command, gateway, timeout=60)
        actual_commit: str | None = None
        contents: dict[str, bytes] = {}
        files: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []
        catalog_root = PurePosixPath(catalog_directory)
        for line in output.splitlines():
            fields = line.split("\t")
            if fields[0] == "COMMIT" and len(fields) == 2:
                actual_commit = fields[1].lower()
                continue
            if fields[0] == "TOO_MANY" and len(fields) == 2:
                warnings.append(
                    f"repository YAML catalog exceeds {YAML_CATALOG_MAX_FILES} files"
                )
                continue
            if fields[0] not in {"FILE", "CONTENT", "TOO_LARGE"} or len(fields) < 2:
                continue
            relative = (
                fields[1][len(prefix):]
                if prefix and fields[1].startswith(prefix)
                else fields[1]
            )
            path = PurePosixPath(relative)
            if (
                path.suffix not in {".yaml", ".yml"}
                or catalog_root not in path.parents
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                warnings.append(f"repository YAML catalog returned unsafe path: {relative}")
                continue
            if fields[0] == "FILE" and len(fields) == 4:
                files[relative] = {
                    "path": relative,
                    "size_bytes": int(fields[2]),
                    "sha256": fields[3].lower(),
                }
            elif fields[0] == "CONTENT" and len(fields) == 3:
                try:
                    contents[relative] = base64.b64decode(fields[2], validate=True)
                except ValueError:
                    warnings.append(
                        f"repository YAML catalog file has invalid transport encoding: {relative}"
                    )
            elif fields[0] == "TOO_LARGE" and len(fields) == 3:
                warnings.append(
                    f"repository YAML catalog file exceeds inspection bounds: {relative}"
                )
        if actual_commit != commit:
            raise ClusterError("repository YAML catalog inspection returned a different commit")
        if not contents and not warnings:
            warnings.append(f"repository YAML catalog is empty: {catalog_directory}")
        return self._set_cached(
            key,
            {
                "repository": repository,
                "commit": commit,
                "project_subdirectory": subdirectory,
                "gateway": host,
                "files": files,
                "contents": contents,
                "warnings": warnings,
            },
        )

    def input_options(
        self,
        repo_url: str,
        revision: str,
        input_fields: list[Mapping[str, Any]],
        gateway: str = "auto",
        project_subdirectory: str = ".",
    ) -> dict[str, Any]:
        repository = self.repository_url(repo_url)
        commit = revision.lower()
        if not FULL_COMMIT_RE.fullmatch(commit):
            commit = self.resolve_revision(repository, revision, gateway)
        options: dict[str, Any] = {}
        for field in input_fields:
            raw_rule = field.get("choice_source")
            if not isinstance(raw_rule, Mapping):
                continue
            path = str(field.get("path") or "")
            rule = dict(raw_rule)
            kind = str(rule.get("kind") or "")
            entrypoint = str(rule.get("entrypoint") or "")
            supporting = [str(item) for item in rule.get("supporting_files") or []]
            files = [entrypoint, *supporting]
            rule_sha256 = hashlib.sha256(
                json.dumps(rule, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            ).hexdigest()
            if kind not in {
                "python_static_registry",
                "python_enum",
                "yaml_static_mapping",
                "yaml_static_catalog",
            }:
                resolved = {
                    "choices": [],
                    "complete": False,
                    "warnings": [f"unsupported repository choice source: {kind or 'missing kind'}"],
                }
                bundle = {"files": {}, "warnings": [], "gateway": gateway}
            elif kind == "yaml_static_catalog":
                bundle = self._read_yaml_catalog(
                    repository,
                    commit,
                    str(rule.get("directory") or ""),
                    gateway,
                    project_subdirectory,
                )
                resolved = _resolve_yaml_static_catalog(bundle["contents"], rule)
                files = sorted(bundle.get("files", {}))
            else:
                bundle = self._read_choice_files(
                    repository, commit, files, gateway, project_subdirectory
                )
                if kind == "python_static_registry":
                    resolved = _resolve_python_static_registry(
                        bundle["contents"], rule
                    )
                elif kind == "python_enum":
                    resolved = _resolve_python_enum(bundle["contents"], rule)
                else:
                    resolved = _resolve_yaml_static_mapping(
                        bundle["contents"], rule
                    )
            warnings = [*bundle.get("warnings", []), *resolved.get("warnings", [])]
            evidence = [bundle["files"][name] for name in files if name in bundle.get("files", {})]
            options[path] = {
                "choices": list(resolved.get("choices") or []),
                "metadata": dict(resolved.get("metadata") or {}),
                "complete": bool(resolved.get("complete")) and not warnings,
                "source": {
                    "kind": kind,
                    "commit": commit,
                    "entrypoint": entrypoint,
                    "registry": rule.get("registry"),
                    "choice": rule.get("choice"),
                    "directory": rule.get("directory"),
                    "filename_pattern": rule.get("filename_pattern"),
                    "files": evidence,
                    "rule_sha256": rule_sha256,
                },
                "warnings": warnings,
            }
        return options

    def inspect(
        self,
        repo_url: str,
        revision: str,
        gateway: str = "auto",
        project_subdirectory: str = ".",
    ) -> dict[str, Any]:
        repository = self.repository_url(repo_url)
        commit = revision.lower()
        if not FULL_COMMIT_RE.fullmatch(commit):
            commit = self.resolve_revision(repository, revision, gateway)
        subdirectory = self.project_subdirectory(project_subdirectory)
        key = ("inspection", gateway, repository, commit, subdirectory)
        if cached := self._get_cached(key):
            return cached

        prefix = "" if subdirectory == "." else subdirectory.rstrip("/") + "/"
        paths = [prefix + name for name in INSPECTION_FILES]
        content_paths = {prefix + name for name in TEXT_INSPECTION_FILES}
        path_words = " ".join(shlex.quote(path) for path in paths)
        content_case = "|".join(shlex.quote(path) for path in sorted(content_paths))
        command = f'''set -eu
export GIT_TERMINAL_PROMPT=0
tmp=$(mktemp -d "${{TMPDIR:-/tmp}}/skynet-inspect-XXXXXX")
trap 'rm -rf "$tmp"' EXIT
git -C "$tmp" init -q
git -C "$tmp" -c credential.interactive=never -c protocol.version=2 fetch -q \
  --no-tags --depth=1 --filter=blob:none {shlex.quote(repository)} {shlex.quote(commit)}
actual=$(git -C "$tmp" rev-parse FETCH_HEAD^{{commit}})
test "$actual" = {shlex.quote(commit)}
printf 'COMMIT\\t%s\\n' "$actual"
for path in {path_words}; do
  object="$actual:$path"
  object_type=$(git -C "$tmp" cat-file -t "$object" 2>/dev/null || true)
  if test "$object_type" = blob; then
    size=$(git -C "$tmp" cat-file -s "$object")
    sha=$(git -C "$tmp" show "$object" | sha256sum | awk '{{print $1}}')
    printf 'FILE\\t%s\\t%s\\t%s\\n' "$path" "$size" "$sha"
    case "$path" in
      {content_case})
        if test "$size" -le 262144; then
          printf 'CONTENT\\t%s\\t' "$path"
          git -C "$tmp" show "$object" | base64 -w0
          printf '\\n'
        fi
        ;;
    esac
  fi
done
'''
        host, output = self.cluster.run_with_fallback(command, gateway, timeout=60)
        files: dict[str, dict[str, Any]] = {}
        contents: dict[str, bytes] = {}
        actual_commit: str | None = None
        for line in output.splitlines():
            fields = line.split("\t")
            if fields[0] == "COMMIT" and len(fields) == 2:
                actual_commit = fields[1].lower()
            elif fields[0] == "FILE" and len(fields) == 4:
                relative = fields[1][len(prefix):] if prefix and fields[1].startswith(prefix) else fields[1]
                files[relative] = {
                    "path": relative,
                    "size_bytes": int(fields[2]),
                    "sha256": fields[3].lower(),
                }
            elif fields[0] == "CONTENT" and len(fields) == 3:
                relative = fields[1][len(prefix):] if prefix and fields[1].startswith(prefix) else fields[1]
                try:
                    contents[relative] = base64.b64decode(fields[2], validate=True)
                except ValueError as error:
                    raise ClusterError(f"repository inspection returned invalid content for {relative}") from error
        if actual_commit != commit:
            raise ClusterError("repository inspection returned a different commit")

        candidates: list[dict[str, Any]] = []
        warnings: list[str] = []
        repository_manifest: dict[str, Any] | None = None
        repository_manifest_path: str | None = None
        if ".skynet.json" in contents:
            try:
                repository_manifest = json.loads(contents[".skynet.json"].decode("utf-8"))
                repository_manifest_path = ".skynet.json"
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                warnings.append(f".skynet.json is invalid: {error}")
        elif ".skynet.toml" in contents:
            try:
                repository_manifest = tomllib.loads(contents[".skynet.toml"].decode("utf-8"))
                repository_manifest_path = ".skynet.toml"
            except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
                warnings.append(f".skynet.toml is invalid: {error}")
        if ".skynet.yml" in files or ".skynet.yaml" in files:
            warnings.append(
                "YAML repository manifests are advisory only; use .skynet.toml or .skynet.json for automatic resolution"
            )

        def add_candidate(
            backend: str,
            source: str,
            strength: str,
            runnable: bool,
            evidence: list[str],
            configuration: dict[str, Any],
            missing: list[str] | None = None,
            priority: int = 100,
        ) -> None:
            candidates.append(
                {
                    "id": f"{source}:{backend}:{len(candidates) + 1}",
                    "type": backend,
                    "backend": backend,
                    "source": source,
                    "strength": strength,
                    "confidence": strength,
                    "runnable": runnable,
                    "evidence": evidence,
                    "configuration": configuration,
                    "missing_requirements": missing or [],
                    "priority": priority,
                }
            )

        manifest_runtime = repository_manifest.get("runtime") if isinstance(repository_manifest, dict) else None
        if isinstance(manifest_runtime, dict):
            backend = {"container": "apptainer"}.get(
                str(manifest_runtime.get("backend", "")), str(manifest_runtime.get("backend", ""))
            )
            configuration = {
                key: value
                for key, value in manifest_runtime.items()
                if key in {
                    "lock_file", "environment_path", "container_image", "container_digest",
                    "uv_executable", "bootstrap_uv", "uv_version", "profile",
                }
            }
            missing: list[str] = []
            if backend == "uv":
                lock = str(configuration.get("lock_file") or "uv.lock")
                configuration["lock_file"] = lock
                if lock not in files:
                    missing.append(lock)
                elif files[lock].get("sha256"):
                    configuration["lock_sha256"] = files[lock]["sha256"]
            elif backend == "conda":
                lock = configuration.get("lock_file")
                if lock and str(lock) not in files:
                    missing.append(str(lock))
                elif lock:
                    configuration["lock_sha256"] = files[str(lock)]["sha256"]
                if not lock and not configuration.get("environment_path"):
                    missing.append("lock_file or environment_path")
            elif backend == "apptainer" and not configuration.get("container_image"):
                missing.append("container_image")
            elif backend == "existing" and not configuration.get("environment_path"):
                missing.append("environment_path")
            elif backend not in {"uv", "conda", "apptainer", "existing"}:
                missing.append("supported runtime.backend")
            add_candidate(
                backend or "unknown",
                "repository-manifest",
                "strong" if not missing else "invalid",
                not missing,
                [repository_manifest_path or ".skynet"],
                configuration,
                missing,
                priority=0,
            )

        pyproject_uv = False
        if "pyproject.toml" in contents:
            try:
                pyproject = tomllib.loads(contents["pyproject.toml"].decode("utf-8"))
                pyproject_uv = isinstance(pyproject.get("tool", {}).get("uv"), dict)
            except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
                warnings.append(f"pyproject.toml could not be parsed: {error}")
        if "uv.lock" in files:
            missing = [] if "pyproject.toml" in files else ["pyproject.toml"]
            add_candidate(
                "uv", "repository-detection", "strong" if not missing else "weak", not missing,
                [name for name in ("uv.lock", "pyproject.toml") if name in files],
                {"lock_file": "uv.lock", "lock_sha256": files["uv.lock"]["sha256"]}, missing,
            )
        elif "pyproject.toml" in files:
            add_candidate(
                "uv", "repository-detection", "weak", False, ["pyproject.toml"], {},
                ["uv.lock"],
            )
        for lock_name in ("conda-lock.yml", "conda-lock.yaml"):
            if lock_name in files:
                digest = files[lock_name]["sha256"]
                add_candidate(
                    "conda", "repository-detection", "strong", True, [lock_name],
                    {
                        "lock_file": lock_name,
                        "lock_sha256": digest,
                        "environment_path": f"{CLUSTER.paths.work_root}/.cache/conda/envs/{digest[:20]}",
                    },
                )
                break
        if not any(item["backend"] == "conda" and item["strength"] == "strong" for item in candidates):
            environment_files = [name for name in ("environment.yml", "environment.yaml") if name in files]
            if environment_files:
                add_candidate(
                    "conda", "repository-detection", "weak", False, environment_files, {},
                    ["conda-lock.yml or explicit environment_path"],
                )
        container_files = [
            name for name in ("Apptainer.def", "Singularity.def", "Containerfile", "Dockerfile") if name in files
        ]
        if container_files:
            add_candidate(
                "apptainer", "repository-detection", "weak", False, container_files, {},
                ["built immutable container_image and container_digest"],
            )
        requirement_files = [name for name in ("requirements.txt", "requirements-dev.txt") if name in files]
        if requirement_files:
            hashed = all(
                b"--hash=sha256:" in contents.get(name, b"") for name in requirement_files
            )
            add_candidate(
                "uv", "requirements-evidence", "weak", False, requirement_files, {},
                ["uv.lock or an explicit versioned runtime manifest"],
            )
            if hashed:
                warnings.append("requirements files contain hashes but need an explicit install command/runtime manifest")
        if pyproject_uv:
            warnings.append("pyproject.toml declares [tool.uv]")
        if "README.md" in files or "README.rst" in files:
            warnings.append("README setup instructions are advisory and were not converted into executable commands")

        strong = [candidate for candidate in candidates if candidate["strength"] == "strong" and candidate["runnable"]]
        detected_backends = sorted({candidate["backend"] for candidate in strong if candidate["source"] != "repository-manifest"})
        conflicts = detected_backends if len(detected_backends) > 1 else []
        missing_requirements = sorted(
            {item for candidate in candidates for item in candidate["missing_requirements"]}
        )
        evidence = [files[name] for name in sorted(files)]
        payload: dict[str, Any] = {
            "schema_version": 1,
            "repository": repository,
            "requested_revision": revision,
            "commit": commit,
            "project_subdirectory": subdirectory,
            "gateway": host,
            "evidence": evidence,
            "repository_manifest": repository_manifest,
            "repository_manifest_path": repository_manifest_path,
            "candidates": candidates,
            "conflicts": conflicts,
            "missing_requirements": missing_requirements,
            "warnings": warnings,
        }
        payload["inspection_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        return self._set_cached(key, payload)


def resolve_runtime(
    inspection: dict[str, Any],
    request: dict[str, Any],
    adapter_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requested = str(request.get("type") or request.get("backend") or "auto")
    requested = {"container": "apptainer", "adapter-default": "auto"}.get(requested, requested)
    if requested not in {"auto", "uv", "conda", "apptainer", "existing"}:
        raise ValueError(f"unsupported runtime backend: {requested}")
    candidates = list(inspection.get("candidates") or [])
    adapter_runtime = dict(adapter_runtime or {})
    recommendation = adapter_runtime.get("recommended_backend")
    allowed = set(adapter_runtime.get("allowed_backends") or {"uv", "conda", "apptainer", "existing"})

    selected: dict[str, Any] | None = None
    mode = "manual"
    if requested == "auto":
        manifest_candidates = [
            item for item in candidates
            if item.get("source") == "repository-manifest" and item.get("strength") == "strong" and item.get("runnable")
        ]
        strong = manifest_candidates or [
            item for item in candidates
            if item.get("strength") == "strong" and item.get("runnable")
        ]
        strong = [item for item in strong if item.get("backend") in allowed]
        backends = sorted({str(item.get("backend")) for item in strong})
        if len(strong) == 1 or len(backends) == 1 and strong:
            selected = sorted(strong, key=lambda item: int(item.get("priority", 100)))[0]
            mode = str(selected.get("source"))
        elif strong:
            evidence = ", ".join(
                f"{item['backend']} ({', '.join(item.get('evidence') or [])})" for item in strong
            )
            raise ValueError(
                f"runtime auto-detection is ambiguous: {evidence}; choose a runtime explicitly or add .skynet.toml"
            )
        else:
            candidate_summary = ", ".join(
                f"{item.get('backend')} ({item.get('strength')})" for item in candidates
            ) or "no dependency/runtime files"
            recommendation_text = f" Adapter recommends {recommendation}." if recommendation else ""
            raise ValueError(
                "runtime auto-detection found no single strong runnable candidate: "
                f"{candidate_summary}.{recommendation_text} Select a runtime explicitly and provide its lock/environment/image."
            )
    else:
        if requested not in allowed:
            raise ValueError(
                f"adapter does not allow {requested}; choose one of {sorted(allowed)}"
            )
        matching = [item for item in candidates if item.get("backend") == requested and item.get("runnable")]
        if matching:
            selected = sorted(
                matching,
                key=lambda item: (item.get("strength") != "strong", int(item.get("priority", 100))),
            )[0]

    configuration = dict(selected.get("configuration") or {}) if selected else {}
    profile = request.get("profile")
    for key in (
        "lock_file", "lock_sha256", "environment_path", "container_image", "container_digest",
        "uv_executable", "bootstrap_uv", "uv_version",
    ):
        if request.get(key) is not None:
            configuration[key] = request[key]
    if requested == "conda" and profile and not configuration.get("environment_path"):
        configuration["environment_path"] = str(profile)
    if requested == "existing" and profile:
        configuration["environment_path"] = str(profile)
    if requested == "apptainer" and profile and not configuration.get("container_image"):
        configuration["container_image"] = str(profile)
    backend = str(selected.get("backend")) if selected else requested
    if backend == "uv" and not configuration.get("lock_file"):
        raise ValueError("uv runtime requires uv.lock at the selected commit or an explicit lock_file")
    if backend == "conda" and not configuration.get("environment_path"):
        raise ValueError("conda runtime requires a conda-lock file or explicit environment_path")
    if backend == "apptainer" and not configuration.get("container_image"):
        raise ValueError("Apptainer runtime requires an immutable image path/reference")
    if backend == "existing" and not configuration.get("environment_path"):
        raise ValueError("existing runtime is never selected implicitly; provide environment_path")

    resolution = {
        "schema_version": 1,
        "mode": mode,
        "requested_backend": requested,
        "selected_backend": backend,
        "selected_candidate": selected,
        "adapter_recommendation": recommendation,
        "inspection": inspection,
    }
    result = {
        "backend": backend,
        "profile": str(profile or "default"),
        **configuration,
        "resolution": resolution,
        "resolution_sha256": hashlib.sha256(
            json.dumps(resolution, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest(),
    }
    if backend != "uv":
        result.setdefault("bootstrap_uv", False)
    return result


__all__ = ["SourceDiscovery", "resolve_runtime"]
