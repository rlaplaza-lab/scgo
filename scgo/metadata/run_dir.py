"""Run-directory record (``metadata.json``) and run-id helpers."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scgo.metadata.provenance import output_json_provenance
from scgo.utils.logging import get_logger, log_info_v


class RunDirJSONEncoder(json.JSONEncoder):
    """JSON encoder: ``type`` objects become their ``__name__`` (for params snapshots)."""

    def default(self, obj: Any) -> Any:
        """Serialize types by name for JSON run-tracking metadata."""
        if isinstance(obj, type):
            return obj.__name__
        return super().default(obj)


@dataclass
class RunDirRecord:
    """Per-run params/composition snapshot written to ``metadata.json``."""

    run_id: str
    timestamp: str
    composition: list[str] | None = None
    formula: str | None = None
    params: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunDirRecord:
        """Create record from dictionary (ignores provenance header keys)."""
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            composition=data.get("composition"),
            formula=data.get("formula"),
            params=data.get("params"),
        )


def _formula_from_composition(composition: list[str]) -> str:
    counts = Counter(composition)
    return "".join(
        f"{elem}{count if count > 1 else ''}" for elem, count in sorted(counts.items())
    )


def generate_run_id() -> str:
    """Generate timestamp-based run ID with microsecond granularity.

    Returns:
        Run ID in format: run_YYYYMMDD_HHMMSS_ffffff
    """
    now = datetime.now(UTC)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    microseconds = now.microsecond
    return f"run_{timestamp}_{microseconds:06d}"


def ensure_run_id(run_id: str | None, verbosity: int = 0, logger=None) -> str:
    """Ensure a run_id exists, generating one if needed and logging if appropriate."""
    if run_id is None:
        run_id = generate_run_id()
        log_info_v(
            logger if logger is not None else get_logger(__name__),
            "Generated run ID: %s",
            run_id,
            verbosity=verbosity,
        )
    return run_id


def save_run_dir_record(
    run_dir: str,
    run_id: str,
    record: dict[str, Any] | None = None,
) -> None:
    """Save run directory record to ``metadata.json``."""
    os.makedirs(run_dir, exist_ok=True)

    composition = record.get("composition") if record else None
    formula = record.get("formula") if record else None
    if composition and not formula:
        formula = _formula_from_composition(composition)

    record_obj = RunDirRecord(
        run_id=run_id,
        timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        composition=composition,
        formula=formula,
        params=record.get("params") if record else None,
    )

    payload = {**output_json_provenance(), **record_obj.to_dict()}

    metadata_file = os.path.join(run_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(payload, f, indent=2, cls=RunDirJSONEncoder)


def load_run_dir_record(run_dir: str) -> RunDirRecord | None:
    """Load run directory record from ``metadata.json``."""
    metadata_file = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        return None

    try:
        with open(metadata_file) as f:
            data = json.load(f)
        return RunDirRecord.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger = get_logger(__name__)
        logger.warning(f"Failed to load run dir record from {metadata_file}: {e}")
        return None


def get_run_directories(base_output_dir: str) -> list[str]:
    """Get list of all run directories in base output directory."""
    if not os.path.exists(base_output_dir):
        return []

    run_dirs = [
        os.path.join(base_output_dir, item)
        for item in os.listdir(base_output_dir)
        if (
            item.startswith("run_")
            and os.path.isdir(os.path.join(base_output_dir, item))
            and get_run_id_from_dir(item) is not None
        )
    ]

    return sorted(run_dirs)


def resolve_run_id_from_db_path(
    db_path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> str:
    """Resolve GO run ID from a database path (``run_*`` segment when present)."""
    db_path_str = os.path.abspath(str(db_path))
    parts: Sequence[str]
    if base_dir is not None:
        base_s = os.path.abspath(str(base_dir))
        try:
            rel = os.path.relpath(db_path_str, base_s)
            parts = rel.split(os.sep)
        except ValueError:
            parts = Path(db_path_str).parts
    else:
        parts = Path(db_path_str).parts

    for part in parts:
        resolved = get_run_id_from_dir(part)
        if resolved is not None:
            return resolved
        if part.startswith("run_"):
            return part

    parent_name = Path(db_path_str).parent.name
    resolved = get_run_id_from_dir(parent_name)
    if resolved is not None:
        return resolved
    if parent_name.startswith("run_"):
        return parent_name

    basename = os.path.basename(db_path_str)
    logger = get_logger(__name__)
    logger.warning(
        "Could not resolve run_id from path %s; using database basename %r as fallback",
        db_path,
        basename,
    )
    return basename


def get_run_id_from_dir(run_dir: str) -> str | None:
    """Extract run ID from directory name."""
    dir_name = os.path.basename(run_dir)
    if dir_name.startswith("run_") and len(dir_name) == 26:
        parts = dir_name.split("_")
        if (
            len(parts) == 4
            and len(parts[1]) == 8
            and len(parts[2]) == 6
            and len(parts[3]) == 6
        ):
            return dir_name
    return None
