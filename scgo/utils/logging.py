"""Root logging setup, verbosity levels, and TRACE support for SCGO."""

from __future__ import annotations

import logging
import os
import sys

from scgo.exceptions import SCGOValidationError
from scgo.utils.runtime_warnings import apply_scgo_runtime_warning_filters

# Define custom TRACE logging level (below DEBUG)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def _trace(self, message, *args, **kwargs):
    """Log at TRACE; defers formatting when TRACE is disabled."""
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


# Install trace method immediately on import (modern logging support)
logging.Logger.trace = _trace  # type: ignore[attr-defined]


# Verbosity level mapping: user-friendly integers to Python logging levels
VERBOSITY_LEVELS: dict[int, int] = {
    0: logging.WARNING,  # quiet - only warnings and errors
    1: logging.INFO,  # normal - key updates + warnings (default)
    2: logging.DEBUG,  # verbose - detailed information
    3: TRACE,  # ultra-verbose - trace-level diagnostics
}


def get_logger(name: str) -> logging.Logger:
    """Return a logger for ``name`` (typically ``__name__``)."""
    return logging.getLogger(name)


def configure_logging(
    verbosity: int = 1,
    format_string: str | None = None,
    hpc_mode: bool | None = None,
) -> None:
    """Configure the root logger for the SCGO package.

    SCGO targets HPC batch jobs: by default third-party loggers are suppressed
    aggressively. Set environment variable ``SCGO_LOCAL_DEV=1`` to use milder
    suppression when ``hpc_mode`` is omitted, or pass ``hpc_mode=False``
    explicitly.

    Args:
        verbosity: Verbosity level (0=quiet, 1=normal, 2=debug, 3=trace).
        format_string: Custom format string. If None, uses default format.
        hpc_mode: If True, suppresses more third-party logs (default when None,
            unless ``SCGO_LOCAL_DEV=1``). If False, only WARNING+ for most libs.
    """
    if hpc_mode is None:
        hpc_mode = os.environ.get("SCGO_LOCAL_DEV") != "1"
    apply_scgo_runtime_warning_filters()
    if verbosity not in VERBOSITY_LEVELS:
        raise SCGOValidationError(
            f"Invalid verbosity level: {verbosity}. Must be 0, 1, 2, or 3."
        )

    level = VERBOSITY_LEVELS[verbosity]
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if format_string is None:
        format_string = "%(message)s"

    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    _suppress_third_party_loggers(level, hpc_mode=hpc_mode)


def _suppress_third_party_loggers(level: int, hpc_mode: bool = False) -> None:
    """Suppress or control third-party library loggers to prevent noise.

    Args:
        level: The logging level to set for third-party loggers.
        hpc_mode: If True, suppresses more aggressively for HPC environments.
    """
    third_party_loggers = [
        "ase",
        "ase.calculators",
        "ase.optimize",
        "torch",
        "torch_sim",
        "mace",
        "tqdm",
        "urllib3",
        "requests",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "h5py",
        "netCDF4",
    ]

    for logger_name in third_party_loggers:
        logger = logging.getLogger(logger_name)
        suppression_level = logging.ERROR if hpc_mode else max(level, logging.WARNING)
        logger.setLevel(suppression_level)
        logger.propagate = False


def should_show_progress(verbosity: int) -> bool:
    """True when verbosity >= 1 (progress bars enabled for normal+)."""
    return verbosity >= 1


# ---------------------------------------------------------------------------
# Verbosity-gated logging helpers
# ---------------------------------------------------------------------------
# These helpers provide consistent verbosity-gated logging with lazy evaluation.
# Use these instead of scattering `if verbosity >= X:` checks in code.
#
# Style guidelines for SCGO logging:
# - Prefer %-style formatting: logger.info("Processing %s", item)
# - Avoid f-strings: logger.info(f"Processing {item}") - eager evaluation wasteful
# - Use these helpers for verbosity-gated messages
# - Use logger.exception() for unexpected errors with automatic traceback
# - Use exc_info=(verbosity >= 2) for handled errors with conditional traceback


def log_debug_v(
    logger: logging.Logger,
    message: str,
    *args: object,
    verbosity: int = 1,
    min_verbosity: int = 2,
) -> None:
    """Log debug message if verbosity >= min_verbosity (default 2).

    Uses lazy %-style formatting. Message is only formatted if it will be logged.

    Args:
        logger: The logger instance.
        message: Format string for the message.
        *args: Arguments for the format string.
        verbosity: Current verbosity level (0-3).
        min_verbosity: Minimum verbosity to log (default 2 = DEBUG).
    """
    if verbosity >= min_verbosity:
        logger.debug(message, *args)


def log_info_v(
    logger: logging.Logger,
    message: str,
    *args: object,
    verbosity: int = 1,
    min_verbosity: int = 1,
) -> None:
    """Log info message if verbosity >= min_verbosity (default 1).

    Uses lazy %-style formatting. Message is only formatted if it will be logged.

    Args:
        logger: The logger instance.
        message: Format string for the message.
        *args: Arguments for the format string.
        verbosity: Current verbosity level (0-3).
        min_verbosity: Minimum verbosity to log (default 1 = INFO).
    """
    if verbosity >= min_verbosity:
        logger.info(message, *args)


def log_warning_v(
    logger: logging.Logger,
    message: str,
    *args: object,
    verbosity: int = 1,
    min_verbosity: int = 1,
) -> None:
    """Log warning message if verbosity >= min_verbosity (default 1).

    Warnings are typically always shown, but this allows conditional suppression.

    Args:
        logger: The logger instance.
        message: Format string for the message.
        *args: Arguments for the format string.
        verbosity: Current verbosity level (0-3).
        min_verbosity: Minimum verbosity to log (default 1).
    """
    if verbosity >= min_verbosity:
        logger.warning(message, *args)


def log_error_v(
    logger: logging.Logger,
    message: str,
    *args: object,
    verbosity: int = 0,
    min_verbosity: int = 0,
) -> None:
    """Log error message if verbosity >= min_verbosity (default 0).

    Errors are typically always shown, but this allows conditional suppression.

    Args:
        logger: The logger instance.
        message: Format string for the message.
        *args: Arguments for the format string.
        verbosity: Current verbosity level (0-3).
        min_verbosity: Minimum verbosity to log (default 0 = always).
    """
    if verbosity >= min_verbosity:
        logger.error(message, *args)


def log_exception_v(
    logger: logging.Logger,
    message: str,
    *args: object,
    verbosity: int = 1,
    min_verbosity: int = 1,
    min_verbosity_for_traceback: int = 2,
) -> None:
    """Log exception with traceback if verbosity >= min_verbosity_for_traceback.

    For unexpected errors, use logger.exception() directly instead.
    This helper is for handled exceptions where you want conditional traceback.

    Args:
        logger: The logger instance.
        message: Format string for the message.
        *args: Arguments for the format string.
        verbosity: Current verbosity level (0-3).
        min_verbosity: Minimum verbosity to log error (default 1).
        min_verbosity_for_traceback: Minimum verbosity for traceback (default 2).
    """
    if verbosity >= min_verbosity:
        if verbosity >= min_verbosity_for_traceback:
            logger.exception(message, *args)
        else:
            logger.error(message, *args)
