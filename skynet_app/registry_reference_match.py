"""Exact registry reference checks, also embedded in the remote metadata reader."""


def registry_reference_match(value, kind, identifiers, aliases):
    identifiers, aliases = set(identifiers), set(aliases)
    identifier_lengths = {len(identifier) for identifier in identifiers}

    def strings(item):
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            for child in item.values():
                yield from strings(child)
        elif isinstance(item, list):
            for child in item:
                yield from strings(child)

    # Each subtree was previously rescanned once for every ancestor dictionary.
    # Execution receipts contain deeply nested dataset manifests; walk them once.
    pending = [(value, False)]
    while pending:
        item, inside_object = pending.pop()
        if isinstance(item, str):
            if inside_object and len(item) in identifier_lengths and item in identifiers:
                return True
        elif isinstance(item, list):
            pending.extend((child, inside_object) for child in item)
        elif isinstance(item, dict):
            if kind == "adapter":
                source = item.get("source")
                if isinstance(source, dict) and not any(source.get(key) for key in ("adapter_id", "adapter_version_id")):
                    if isinstance(source.get("adapter"), str) and source["adapter"] in aliases:
                        return True
            else:
                evaluation = item.get("evaluation")
                if isinstance(evaluation, dict) and aliases.intersection(strings(evaluation.get("suites", []))):
                    return True
            pending.extend((child, True) for child in item.values())
    return False
