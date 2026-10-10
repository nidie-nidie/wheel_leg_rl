from __future__ import annotations

from collections.abc import Callable
from typing import Any


_finalizer: Callable[[int], None] | None = None
_finalized = False
_guard: Any | None = None


def register_bootstrap_guard(guard: Any) -> None:
    global _guard
    if _guard is not None:
        raise RuntimeError("Bootstrap guard is already registered")
    _guard = guard


def current_bootstrap_guard() -> Any:
    if _guard is None:
        raise RuntimeError("Worker is not running under the guarded bootstrap")
    return _guard


def register_bootstrap_finalizer(finalizer: Callable[[int], None]) -> None:
    global _finalizer, _finalized
    if _finalizer is not None:
        raise RuntimeError("Bootstrap finalizer is already registered")
    _finalizer = finalizer
    _finalized = False


def finalize_bootstrap(return_code: int) -> None:
    global _finalized
    if _finalizer is None:
        raise RuntimeError("Worker is not running under the guarded bootstrap")
    if _finalized:
        raise RuntimeError("Bootstrap evidence was already finalized")
    _finalizer(int(return_code))
    _finalized = True


def bootstrap_finalized() -> bool:
    return _finalized


def clear_bootstrap_finalizer() -> None:
    global _finalizer, _finalized, _guard
    _finalizer = None
    _finalized = False
    _guard = None
