"""Custom exceptions for database operations in SCGO."""

from __future__ import annotations

from scgo.exceptions import SCGODatabaseError


class DatabaseSetupError(SCGODatabaseError):
    """Raised when database setup or initialization fails."""
