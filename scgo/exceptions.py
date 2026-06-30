"""SCGO exception hierarchy."""

from __future__ import annotations


class SCGOError(Exception):
    """Base exception for all SCGO errors."""


class SCGOConfigurationError(SCGOError):
    """Configuration or system-setup errors."""


class SCGOValidationError(SCGOError):
    """Input validation errors."""


class SCGORuntimeError(SCGOError):
    """Runtime errors during optimization."""


class SCGODatabaseError(SCGOError):
    """Database operation errors."""


class SCGONotImplementedError(SCGOError):
    """Unimplemented or unavailable functionality."""


class SCGOFileError(SCGOError):
    """File I/O errors."""
