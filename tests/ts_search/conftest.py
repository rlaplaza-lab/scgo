"""Shared fixtures for TS search tests."""

from __future__ import annotations

import tempfile

import pytest


@pytest.fixture
def temp_output_dir():
    """Temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
