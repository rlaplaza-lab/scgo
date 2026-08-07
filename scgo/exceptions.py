"""SCGO exception hierarchy."""

from __future__ import annotations


class SCGOError(Exception):
    """Base exception for all SCGO errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class SCGOValidationError(SCGOError, ValueError):
    """Input validation errors.

    Subclasses :class:`ValueError` so pre-existing ``except ValueError`` handlers
    catch it without edits.

    Callers that want user-facing ERROR logs should log at the raise site
    (or API/runner boundary); construction alone does not log.
    """


class SCGOConfigurationError(SCGOError, ValueError):
    """Configuration or system-setup errors.

    Subclasses :class:`ValueError` so callers checking for user-input/setup
    failures and ``pytest.raises(ValueError)`` tests keep working.
    """


class SCGODependencyError(SCGOConfigurationError, ImportError):
    """Missing optional dependency errors.

    Subclasses :class:`ImportError` so callers checking for import failures and
    ``pytest.raises(ImportError)`` tests keep working.
    """


class SCGORuntimeError(SCGOError, RuntimeError):
    """Runtime errors during optimization.

    Subclasses :class:`RuntimeError` so pre-existing ``except RuntimeError``
    handlers catch it without edits.
    """


class SCGODatabaseError(SCGOError, OSError):
    """Database operation errors.

    Subclasses :class:`OSError` so pre-existing ``except OSError`` handlers catch
    it without edits.
    """


class SCGONotImplementedError(SCGOError, NotImplementedError):
    """Unimplemented or unavailable functionality."""


class SCGOFileError(SCGOError, OSError):
    """File I/O errors.

    Subclasses :class:`OSError` so pre-existing ``except OSError`` handlers catch
    it without edits.
    """
