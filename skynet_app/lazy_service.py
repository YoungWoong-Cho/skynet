"""Module-level services that are built on first use, never at import.

Building an API module's service opens the configured database, which applies
pending migrations and seeds registries. A module that defines
``service = LazyService(factory)`` keeps its public name for routes, other
modules and tests, while importing it opens no connection.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")
_UNBUILT = object()


class LazyService(Generic[T]):
    """Stand in for the instance ``factory`` returns, built once on first use.

    Attribute reads, writes and deletes reach that instance, so
    ``monkeypatch.setattr(module.service, name, value)`` patches the real
    service. The proxy has no public attributes of its own, which could hide
    the service's (LiveArchiveService has its own ``resolve``). Operations that
    skip attribute lookup (isinstance, identity, operators, ``with``) see the
    proxy, so pass ``resolve(proxy)`` wherever the object itself matters.
    Create it as ``LazyService(factory)``: a subscripted call such as
    ``LazyService[T](factory)`` sets ``__orig_class__``, which would build it.
    """

    __slots__ = ("__factory", "__lock", "__instance")

    def __init__(self, factory: Callable[[], T]) -> None:
        object.__setattr__(self, "_LazyService__factory", factory)
        object.__setattr__(self, "_LazyService__lock", threading.Lock())
        object.__setattr__(self, "_LazyService__instance", _UNBUILT)

    def __resolve(self) -> T:
        instance = self.__instance
        if instance is _UNBUILT:
            with self.__lock:
                instance = self.__instance
                if instance is _UNBUILT:
                    instance = self.__factory()
                    object.__setattr__(self, "_LazyService__instance", instance)
        return instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__resolve(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.__resolve(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self.__resolve(), name)


def resolve(service: LazyService[T] | T) -> T:
    """Return the instance behind ``service``, building it on first use.

    Any other object is already an instance and is returned unchanged, so a
    caller also accepts a module service that was replaced by its instance.
    """
    if type(service) is LazyService:
        return service._LazyService__resolve()
    return service
