"""Exact registry reference checks, also embedded in the remote metadata reader."""


def suite_reference(adapter, suite, version):
    """Identity of a canonical evaluation entry, which names its suite version instead of its ID."""
    return "\t".join(("suite", adapter, suite, version))


def registry_reference_match(value, identifiers):
    identifiers = set(identifiers)
    identifier_lengths = {len(identifier) for identifier in identifiers}
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
            named = (item.get("adapter"), item.get("suite"), item.get("suite_version"))
            if all(isinstance(part, str) for part in named) and suite_reference(*named) in identifiers:
                return True
            pending.extend((child, True) for child in item.values())
    return False
