"""SCGO exception hierarchy.

This module provides a structured exception hierarchy for SCGO to improve
error handling consistency and make it easier to catch specific categories
of errors.
"""

from __future__ import annotations


class SCGOError(Exception):
    """Base exception for all SCGO errors.

    All SCGO-specific exceptions should inherit from this class to allow
    for catching all SCGO-related errors with a single except clause.
    """


class SCGOConfigurationError(SCGOError):
    """Errors related to SCGO configuration.

    Raised when there are issues with parameter configuration, system setup,
    or invalid configuration combinations.
    """


class SCGOValidationError(SCGOError):
    """Validation errors for SCGO inputs.

    Raised when input validation fails for compositions, system types,
    parameters, or other user-provided data.
    """


class SCGORuntimeError(SCGOError):
    """Runtime errors during optimization execution.

    Raised when errors occur during the execution of optimization algorithms,
    such as convergence failures, numerical issues, or unexpected states.
    """


class SCGODatabaseError(SCGOError):
    """Database-related errors.

    Raised when there are issues with database operations, including
    connection failures, query errors, or data integrity problems.
    """


class SCGONotImplementedError(SCGOError):
    """Features or functionality not yet implemented.

    Raised when attempting to use features that are planned but not yet
    implemented, or when hitting known limitations.
    """


class SCGOFileError(SCGOError):
    """File I/O related errors.

    Raised when there are issues reading or writing files, including
    missing files, permission issues, or corrupt data.
    """
