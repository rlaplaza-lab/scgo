#!/usr/bin/env python3
"""Install SCGO + MLIP + dev deps into a Modal image (torch already installed)."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("mace", "upet"):
        print(
            f"Usage: {sys.argv[0]} mace|upet",
            file=sys.stderr,
        )
        return 2
    mlip_extra = sys.argv[1]
    data = tomllib.loads(Path("/build/pyproject.toml").read_text(encoding="utf-8"))
    deps = list(data["project"]["dependencies"])
    deps.extend(data["project"]["optional-dependencies"][mlip_extra])
    deps.extend(data["project"]["optional-dependencies"]["dev"])
    skip_prefixes = ("ruff", "pre-commit")
    install_deps = [
        dep
        for dep in deps
        if not dep.startswith(skip_prefixes) and not dep.startswith("torch>=")
    ]
    subprocess.run(
        ["pip", "install", "--no-cache-dir", *install_deps],
        check=True,
    )
    if mlip_extra == "upet":
        # metatomic-torchsim declares vesin<0.6 but needs skin= from 0.6.0
        subprocess.run(
            [
                "pip",
                "install",
                "--no-cache-dir",
                "vesin==0.6.0",
                "--force-reinstall",
                "--no-deps",
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
