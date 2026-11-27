"""Factories for PEP 562 lazy submodule attribute loading."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

LazyAttrMap = Mapping[str, tuple[str, str]]


def lazy_submodule_getattr(
    module_globals: dict[str, Any],
    lazy_attrs: LazyAttrMap,
    *,
    cache: bool = True,
) -> Callable[[str], Any]:
    """Return a ``__getattr__`` that imports ``(module, attr)`` entries on demand."""

    def __getattr__(name: str) -> Any:
        if name not in lazy_attrs:
            module_name = module_globals.get("__name__", "<module>")
            msg = f"module {module_name!r} has no attribute {name!r}"
            raise AttributeError(msg)
        module_name, attr = lazy_attrs[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        if cache:
            module_globals[name] = value
        return value

    return __getattr__


def lazy_submodule_dir(all_names: Sequence[str]) -> Callable[[], list[str]]:
    """Return a ``__dir__`` that lists ``all_names`` sorted."""

    def __dir__() -> list[str]:
        return sorted(all_names)

    return __dir__
