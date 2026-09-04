from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


_RECORD_VERSION = 1
_DEFAULT_SERVICE_NAME = "io.skynet-control.tracking"
_UNSAFE_BACKEND_MARKERS = (
    "chainer",
    "failkeyring",
    "keyrings.alt.file",
    "nullkeyring",
    "plaintext",
)


class CredentialStoreError(RuntimeError):
    """A credential-store operation failed without exposing credential material."""


class CredentialStoreUnavailable(CredentialStoreError):
    """The operating-system credential store cannot be used safely."""


@dataclass(frozen=True)
class StoredCredential:
    provider: str
    endpoint: str
    credentials: Mapping[str, str] = field(repr=False)


class CredentialStore(Protocol):
    def load(self, provider: str) -> StoredCredential | None: ...

    def save(
        self, provider: str, endpoint: str, credentials: Mapping[str, str]
    ) -> None: ...

    def delete(self, provider: str) -> None: ...


class KeyringCredentialStore:
    """Store endpoint-bound credentials in the platform's secure credential manager.

    The ``keyring`` package selects macOS Keychain, Windows Credential Locker, or a
    Linux Secret Service/KWallet backend. File, plaintext, fail, null, and chained
    fallback backends are rejected instead of silently persisting secrets to disk.
    """

    def __init__(
        self,
        *,
        service_name: str = _DEFAULT_SERVICE_NAME,
        keyring_backend: Any | None = None,
    ) -> None:
        self.service_name = service_name
        self._keyring_backend = keyring_backend

    @staticmethod
    def _account(provider: str) -> str:
        normalized = provider.strip().lower()
        if not normalized or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", normalized):
            raise ValueError("credential provider must be a non-empty identifier")
        return f"tracking:{normalized}"

    def _backend(self) -> Any:
        backend = self._keyring_backend
        if backend is None:
            try:
                import keyring
            except ImportError as error:  # pragma: no cover - dependency installation failure.
                raise CredentialStoreUnavailable(
                    "OS credential store support is unavailable; install the keyring dependency "
                    "or connect with remember=false"
                ) from error
            try:
                backend = keyring.get_keyring()
            except Exception as error:
                raise CredentialStoreUnavailable(
                    "OS credential store initialization failed; connect with remember=false "
                    "or configure credentials through the environment"
                ) from error
            self._keyring_backend = backend
        backend_name = (
            f"{type(backend).__module__}.{type(backend).__qualname__}"
        ).lower()
        try:
            priority = float(getattr(backend, "priority", 0))
        except Exception as error:
            raise CredentialStoreUnavailable(
                "OS credential store is unavailable; connect with remember=false "
                "or configure credentials through the environment"
            ) from error
        if priority <= 0 or any(marker in backend_name for marker in _UNSAFE_BACKEND_MARKERS):
            raise CredentialStoreUnavailable(
                "No supported secure OS credential store is available; plaintext and file "
                "credential backends are not allowed. Connect with remember=false or configure "
                "credentials through the environment"
            )
        return backend

    @staticmethod
    def _operation_error(operation: str, error: Exception) -> CredentialStoreUnavailable:
        return CredentialStoreUnavailable(
            f"OS credential store {operation} failed ({type(error).__name__}); "
            "connect with remember=false or configure credentials through the environment"
        )

    def load(self, provider: str) -> StoredCredential | None:
        account = self._account(provider)
        try:
            raw = self._backend().get_password(self.service_name, account)
        except CredentialStoreError:
            raise
        except Exception as error:
            raise self._operation_error("read", error) from error
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
            if (
                not isinstance(payload, dict)
                or payload.get("version") != _RECORD_VERSION
                or payload.get("provider") != provider
                or not isinstance(payload.get("endpoint"), str)
                or not payload["endpoint"]
                or not isinstance(payload.get("credentials"), dict)
                or not all(
                    isinstance(key, str) and isinstance(value, str) and value
                    for key, value in payload["credentials"].items()
                )
            ):
                raise ValueError("invalid record")
            credentials = dict(payload["credentials"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise CredentialStoreError(
                f"Stored credential record for {provider} is invalid; disconnect and reconnect"
            ) from error
        return StoredCredential(
            provider=provider,
            endpoint=str(payload["endpoint"]),
            credentials=credentials,
        )

    def save(
        self, provider: str, endpoint: str, credentials: Mapping[str, str]
    ) -> None:
        account = self._account(provider)
        values = {
            str(key): str(value)
            for key, value in credentials.items()
            if value is not None and str(value)
        }
        if not endpoint or not values:
            raise ValueError("endpoint and at least one credential value are required")
        payload = json.dumps(
            {
                "version": _RECORD_VERSION,
                "provider": provider,
                "endpoint": endpoint,
                "credentials": values,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            self._backend().set_password(self.service_name, account, payload)
        except CredentialStoreError:
            raise
        except Exception as error:
            raise self._operation_error("write", error) from error

    def delete(self, provider: str) -> None:
        account = self._account(provider)
        try:
            backend = self._backend()
            if backend.get_password(self.service_name, account) is not None:
                backend.delete_password(self.service_name, account)
        except CredentialStoreError:
            raise
        except Exception as error:
            raise self._operation_error("delete", error) from error


__all__ = [
    "CredentialStore",
    "CredentialStoreError",
    "CredentialStoreUnavailable",
    "KeyringCredentialStore",
    "StoredCredential",
]
