"""SCGO exception hierarchy."""

from __future__ import annotations


class SCGOError(Exception):
    """Base exception for all SCGO errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class SCGOValidationError(SCGOError):
    """Input validation errors.

    Callers that want user-facing ERROR logs should log at the raise site
    (or API/runner boundary); construction alone does not log.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


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
