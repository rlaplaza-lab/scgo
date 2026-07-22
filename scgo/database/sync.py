"""Database retry helpers for transient SQLite and filesystem errors."""

from __future__ import annotations

import functools
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from scgo.exceptions import SCGORuntimeError
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

HPC_DATABASE_EXCEPTIONS = (sqlite3.OperationalError, OSError)
F = Callable[..., Any]


@dataclass(frozen=True)
class RetryConfig:
    """Retry counts and exponential backoff for transient SQLite / I/O errors."""

    max_retries: int = 3
    initial_delay: float = 0.1
    max_delay: float = 5.0
    backoff_factor: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_retries", max(1, self.max_retries))
        object.__setattr__(self, "initial_delay", max(0.0, self.initial_delay))
        object.__setattr__(self, "max_delay", max(self.initial_delay, self.max_delay))
        object.__setattr__(self, "backoff_factor", max(1.0, self.backoff_factor))

    def get_delay(self, attempt: int) -> float:
        delay = self.initial_delay * (self.backoff_factor**attempt)
        return min(delay, self.max_delay)


PRESET_AGGRESSIVE = RetryConfig(max_retries=5, initial_delay=0.1, backoff_factor=2.0)
PRESET_CONSERVATIVE = RetryConfig(max_retries=3, initial_delay=0.5, backoff_factor=1.5)
PRESET_DEFAULT = RetryConfig(max_retries=3, initial_delay=0.2, backoff_factor=2.0)


def database_retry(
    operation: Callable[[], Any],
    config: RetryConfig | None = None,
    operation_name: str = "database operation",
    log_level: str = "debug",
    *,
    exception_types: tuple[type[BaseException], ...] | None = None,
    max_retries: int | None = None,
    initial_delay: float | None = None,
    backoff_factor: float | None = None,
) -> Any:
    """Run ``operation`` with exponential backoff on transient errors.

    When ``exception_types`` is ``None`` (default), only
    :exc:`~sqlite3.OperationalError` messages classified by
    :func:`is_retryable_error` and any :exc:`OSError` are retried.

    When ``exception_types`` is set (e.g. :data:`HPC_DATABASE_EXCEPTIONS`), those
    exception types are retried without the SQLite message filter — matching the
    historical :func:`retry_with_backoff` behavior used by BH/simple.

    Optional ``max_retries`` / ``initial_delay`` / ``backoff_factor`` override
    fields on ``config`` (or :data:`PRESET_DEFAULT`) for call-site compatibility.
    """
    base = config or PRESET_DEFAULT
    effective_config = RetryConfig(
        max_retries=max_retries if max_retries is not None else base.max_retries,
        initial_delay=(
            initial_delay if initial_delay is not None else base.initial_delay
        ),
        max_delay=base.max_delay,
        backoff_factor=(
            backoff_factor if backoff_factor is not None else base.backoff_factor
        ),
    )
    n_retries = effective_config.max_retries

    log_methods = {
        "debug": logger.debug,
        "info": logger.info,
        "warning": logger.warning,
        "error": logger.error,
    }
    log_func = log_methods.get(log_level.lower(), logger.debug)

    for attempt in range(n_retries):
        try:
            return operation()
        except Exception as e:
            if exception_types is None:
                if isinstance(e, sqlite3.OperationalError):
                    if not is_retryable_error(e):
                        raise
                elif not isinstance(e, OSError):
                    raise
            elif not isinstance(e, exception_types):
                raise

            if attempt < n_retries - 1:
                delay = effective_config.get_delay(attempt)
                log_func(
                    f"{operation_name}: failed (attempt {attempt + 1}/{n_retries}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"{operation_name}: failed after {n_retries} attempts: {e}"
                )
                raise


def is_retryable_error(error: Exception) -> bool:
    """True for sqlite OperationalError messages that often clear after a short wait."""
    if not isinstance(error, sqlite3.OperationalError):
        return False

    msg = str(error).lower()
    if "locked" in msg or "readonly" in msg:
        return True

    return any(
        kw in msg
        for kw in (
            "disk i/o error",
            "resource temporarily unavailable",
            "input/output error",
        )
    )


def retry_on_lock(
    config: RetryConfig | None = None,
    operation_name: str = "database operation",
    log_retries: bool = True,
) -> Callable[[F], F]:
    """Decorator to retry callable on sqlite locked/readonly OperationalError."""
    effective_config = config or PRESET_CONSERVATIVE

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(effective_config.max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if not is_retryable_error(e):
                        raise

                    if attempt < effective_config.max_retries - 1:
                        delay = effective_config.get_delay(attempt)
                        if log_retries:
                            logger.warning(
                                f"{operation_name}: database locked, retrying in {delay:.2f}s "
                                f"(attempt {attempt + 1}/{effective_config.max_retries})"
                            )
                        time.sleep(delay)
                    else:
                        if log_retries:
                            logger.error(
                                f"{operation_name}: database locked after {effective_config.max_retries} attempts"
                            )
                        raise
            raise SCGORuntimeError(f"{operation_name} failed unexpectedly")

        return wrapper  # type: ignore[return-value]

    return decorator


def retry_transaction(
    db_connection,
    operation: Callable[[Any], Any],
    config: RetryConfig | None = None,
    operation_name: str = "transaction",
    isolation_level: str = "DEFERRED",
) -> Any:
    """Retry a full database transaction on transient lock errors.

    Runs ``operation(conn)`` inside a fresh
    :func:`~scgo.database.transactions.database_transaction` on each attempt.
    Unlike a yielded context manager, this correctly retries when the body or
    commit raises a retryable :exc:`~sqlite3.OperationalError`.

    Args:
        db_connection: ASE ``DataConnection`` (or compatible) for the DB.
        operation: Callable that receives the SQLite connection and returns a
            result (may be ``None``).
        config: Retry/backoff settings (defaults to ``PRESET_AGGRESSIVE``).
        operation_name: Label used in retry log messages.
        isolation_level: SQLite isolation level for each attempt.

    Returns:
        The return value of ``operation`` on success.
    """
    from scgo.database.transactions import database_transaction

    effective_config = config or PRESET_AGGRESSIVE
    for attempt in range(effective_config.max_retries):
        try:
            with database_transaction(
                db_connection,
                isolation_level=isolation_level,
            ) as conn:
                return operation(conn)
        except sqlite3.OperationalError as e:
            if not is_retryable_error(e):
                raise
            if attempt < effective_config.max_retries - 1:
                delay = effective_config.get_delay(attempt)
                logger.warning(
                    f"{operation_name}: database locked, retrying in {delay:.2f}s "
                    f"(attempt {attempt + 1}/{effective_config.max_retries})"
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"{operation_name}: database locked after "
                    f"{effective_config.max_retries} attempts"
                )
                raise
    raise SCGORuntimeError(f"{operation_name} failed unexpectedly")


def retry_with_backoff(
    operation: Callable[[], Any],
    max_retries: int = 3,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
    exception_types: tuple[type[BaseException], ...] = (OSError,),
    operation_name: str = "operation",
    log_level: str = "debug",
) -> Any:
    """Thin wrapper around :func:`database_retry` for backward compatibility."""
    return database_retry(
        operation,
        operation_name=operation_name,
        log_level=log_level,
        exception_types=exception_types,
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
    )
