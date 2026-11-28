"""Atomic JSON / XYZ writes via a same-directory temporary file."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from typing import Any

from ase import Atoms
from ase.io import write as ase_write

from scgo.exceptions import SCGOFileError


def atomic_write_json(
    path: str,
    payload: dict[str, Any],
    *,
    cls: type[json.JSONEncoder] | None = None,
    tmp_prefix: str = ".tmp_",
) -> None:
    """Write JSON via a same-directory temp file then ``os.replace``.

    Raises :class:`~scgo.exceptions.SCGOFileError` on I/O failure.
    """
    directory = os.path.dirname(path) or "."
    tmp_path: str | None = None
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=tmp_prefix,
            suffix=".json",
            dir=directory,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, cls=cls)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    except OSError as exc:
        raise SCGOFileError(f"Failed to write JSON file {path}: {exc}") from exc
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


def atomic_write_xyz(path: str, atoms: Atoms, *, tmp_prefix: str = ".tmp_") -> None:
    """Write XYZ via a same-directory temp file then ``os.replace``."""
    directory = os.path.dirname(path) or "."
    tmp_path: str | None = None
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=tmp_prefix,
            suffix=".xyz",
            dir=directory,
        )
        os.close(fd)
        ase_write(tmp_path, atoms)
        os.replace(tmp_path, path)
        tmp_path = None
    except OSError as exc:
        raise SCGOFileError(f"Failed to write XYZ file {path}: {exc}") from exc
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
