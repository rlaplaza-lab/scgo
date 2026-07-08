"""SCGO exception hierarchy."""

from __future__ import annotations

import logging


class SCGOError(Exception):
    """Base exception for all SCGO errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class SCGOValidationError(SCGOError):
    """Input validation errors (logged at ERROR when logging is configured)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        # Log validation errors at ERROR level when logging is configured
        # Only log if we have handlers (configure_logging has been called)
        # This avoids logging during module import before configure_logging is called
        root_logger = logging.getLogger()
        if root_logger.handlers:
            logger = logging.getLogger("scgo.validation")
            logger.error("Validation error: %s", message)


class SCGOConfigurationError(SCGOError):
    """Configuration or system-setup errors."""


class SCGORuntimeError(SCGOError):
    """Runtime errors during optimization."""


class SCGODatabaseError(SCGOError):
    """Database operation errors."""


class SCGONotImplementedError(SCGOError):
    """Unimplemented or unavailable functionality."""


class SCGOFileError(SCGOError):
    """File I/O errors."""
