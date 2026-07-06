#!/usr/bin/env python3
"""Kaggle kernel entry point for SCGO GPU CI (rendered by kaggle-gpu.yml)."""

from __future__ import annotations

import os
import subprocess
import sys

REPO_URL = "https://github.com/rlaplaza-lab/scgo.git"
GIT_REF = "__GIT_REF__"
PYTEST_MARKER = "__PYTEST_MARKER__"
CONDA_ENV = "scgo-gpu"


def run(cmd: list[str], *, cwd: str | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> int:
    workdir = "/kaggle/working/scgo"
    if os.path.isdir(workdir):
        run(["rm", "-rf", workdir])
    run(["git", "clone", "--depth", "1", "--branch", GIT_REF, REPO_URL, workdir])
    os.chdir(workdir)

    # scgo requires Python >= 3.12; Kaggle base images may be older.
    run(["conda", "create", "-y", "-n", CONDA_ENV, "python=3.12"])
    pip = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        "python",
        "-m",
        "pip",
    ]
    run([*pip, "install", "--upgrade", "pip"])
    run([*pip, "install", "-e", ".[mace,dev]"])

    py = ["conda", "run", "--no-capture-output", "-n", CONDA_ENV, "python"]
    run([*py, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA required'"])

    env = os.environ.copy()
    env["SCGO_BATCH_TEST_SAMPLES"] = "15"

    pytest_cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        "pytest",
        "tests/",
        "-m",
        PYTEST_MARKER,
        "-v",
        "--tb=short",
    ]
    print("+", " ".join(pytest_cmd), flush=True)
    completed = subprocess.run(pytest_cmd, env=env)
    return int(completed.returncode)


if __name__ == "__main__":
    sys.exit(main())
