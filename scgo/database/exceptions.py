"""Custom exceptions for database operations in SCGO."""


class DatabaseSetupError(Exception):
    """Raised when database setup or initialization fails."""


class DatabaseMigrationError(Exception):
    """Raised when database schema migration fails."""
