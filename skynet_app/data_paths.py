"""Path contracts shared by Registry requests and persisted bundle manifests."""


def validate_mount_path(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    value = value.strip()
    if (
        value.startswith("/")
        or "\\" in value
        or ":" in value
        or any(ord(character) < 32 for character in value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("Mount path must be a relative directory without empty, '.' or '..' components")
    return value
