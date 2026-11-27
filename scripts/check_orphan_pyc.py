"""Fail when ``tests/`` has ``.pyc`` files whose ``.py`` source is gone."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Exit 1 when any orphan bytecode remains under ``tests/``."""
    orphans: list[Path] = []
    for pyc in (_ROOT / "tests").rglob("*.pyc"):
        if pyc.parent.name != "__pycache__":
            orphans.append(pyc)
            continue
        module = pyc.name.split(".cpython-", 1)[0]
        if not (pyc.parent.parent / f"{module}.py").is_file():
            orphans.append(pyc)
    if not orphans:
        return 0
    print(f"orphan bytecode under tests/ ({len(orphans)} files):", file=sys.stderr)
    for path in sorted(orphans):
        print(path.relative_to(_ROOT), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
