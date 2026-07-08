"""Custom exceptions for database operations in SCGO."""

from __future__ import annotations


class DatabaseSetupError(Exception):
    """Raised when database setup or initialization fails."""


class DatabaseMigrationError(Exception):
    """Raised when database schema migration fails."""
