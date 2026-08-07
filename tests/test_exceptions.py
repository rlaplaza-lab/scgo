"""Tests asserting the SCGO exception class hierarchy."""

from __future__ import annotations

import pytest

from scgo.database import DatabaseMigrationError, DatabaseSetupError
from scgo.exceptions import (
    SCGOConfigurationError,
    SCGODatabaseError,
    SCGODependencyError,
    SCGOError,
    SCGOFileError,
    SCGONotImplementedError,
    SCGORuntimeError,
    SCGOValidationError,
)


def test_scgo_error_is_root_of_typed_family():
    assert issubclass(SCGOError, Exception)
    for cls in (
        SCGOValidationError,
        SCGOConfigurationError,
        SCGODependencyError,
        SCGORuntimeError,
        SCGODatabaseError,
        SCGONotImplementedError,
        SCGOFileError,
    ):
        assert issubclass(cls, SCGOError)


def test_validation_error_is_value_error():
    assert issubclass(SCGOValidationError, ValueError)


def test_runtime_error_is_runtime_error():
    assert issubclass(SCGORuntimeError, RuntimeError)


def test_dependency_error_is_configuration_and_import_error():
    assert issubclass(SCGODependencyError, SCGOConfigurationError)
    assert issubclass(SCGODependencyError, ImportError)


def test_file_error_is_os_error():
    assert issubclass(SCGOFileError, OSError)


def test_not_implemented_error_is_not_implemented_error():
    assert issubclass(SCGONotImplementedError, NotImplementedError)


def test_database_errors_reparented_under_scgo_database_error():
    assert issubclass(DatabaseSetupError, SCGODatabaseError)
    assert issubclass(DatabaseSetupError, SCGOError)
    assert issubclass(DatabaseMigrationError, SCGODatabaseError)
    assert issubclass(DatabaseMigrationError, SCGOError)


def test_broad_handlers_catch_typed_errors():
    with pytest.raises(ValueError):
        raise SCGOValidationError("boom")
    with pytest.raises(RuntimeError):
        raise SCGORuntimeError("boom")
    with pytest.raises(ImportError):
        raise SCGODependencyError("boom")
