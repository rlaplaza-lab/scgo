#!/usr/bin/env python3
"""Kaggle kernel entry point for SCGO GPU CI (rendered by kaggle-gpu.yml)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import traceback
import urllib.request

REPO_URL = "https://github.com/rlaplaza-lab/scgo.git"
GIT_REF = "__GIT_REF__"
PYTEST_MARKER = "__PYTEST_MARKER__"
CONDA_ENV = "scgo-gpu"
WORKDIR = "/kaggle/working/scgo"


def log(message: str) -> None:
    print(message, flush=True)


def run(
    cmd: list[str], *, cwd: str | None = None, env: dict[str, str] | None = None
) -> None:
    log("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def _python_ok(version: str) -> bool:
    major, minor, *_ = (int(part) for part in version.split("."))
    return (major, minor) >= (3, 12)


def _system_python() -> list[str] | None:
    for candidate in (sys.executable, "python3", "python"):
        if candidate == "python" and not shutil.which("python"):
            continue
        try:
            completed = subprocess.run(
                [
                    candidate,
                    "-c",
                    "import sys; print('.'.join(map(str, sys.version_info[:3])))",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        version = completed.stdout.strip()
        log(f"Found {candidate} version {version}")
        if _python_ok(version):
            return [candidate]
    return None


def _conda_exe() -> str:
    for candidate in (
        os.environ.get("CONDA_EXE", ""),
        "/opt/conda/bin/conda",
        shutil.which("conda") or "",
    ):
        if candidate and (candidate == "conda" or os.path.isfile(candidate)):
            return candidate
    return "conda"


def _conda_python() -> list[str]:
    conda = _conda_exe()
    conda_env = os.environ.copy()
    conda_env["CONDA_PLUGINS_AUTO_ACCEPT_TOS"] = "yes"
    for tos_cmd in (
        [
            conda,
            "tos",
            "accept",
            "--override-channels",
            "--channel",
            "https://repo.anaconda.com/pkgs/main",
        ],
        [
            conda,
            "tos",
            "accept",
            "--override-channels",
            "--channel",
            "https://repo.anaconda.com/pkgs/r",
        ],
    ):
        subprocess.run(tos_cmd, env=conda_env, check=False)
    run([conda, "create", "-y", "-n", CONDA_ENV, "python=3.12"], env=conda_env)
    return [conda, "run", "--no-capture-output", "-n", CONDA_ENV, "python"]


def _resolve_python() -> list[str]:
    system = _system_python()
    if system is not None:
        log("Using system Python (>= 3.12)")
        return system
    log("System Python unavailable or < 3.12; creating conda env")
    return _conda_python()


def _ensure_git() -> None:
    if shutil.which("git"):
        return
    if shutil.which("apt-get"):
        run(["apt-get", "update"])
        run(["apt-get", "install", "-y", "git"])


def _fetch_repo() -> None:
    if os.path.isdir(WORKDIR):
        shutil.rmtree(WORKDIR)
    os.makedirs(WORKDIR, exist_ok=True)
    _ensure_git()
    if shutil.which("git"):
        try:
            run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    GIT_REF,
                    REPO_URL,
                    WORKDIR,
                ]
            )
            return
        except subprocess.CalledProcessError as exc:
            log(f"git clone failed ({exc}); falling back to source tarball")
    archive_url = (
        f"https://github.com/rlaplaza-lab/scgo/archive/refs/heads/{GIT_REF}.tar.gz"
    )
    archive_path = "/kaggle/working/scgo-src.tar.gz"
    log(f"Downloading {archive_url}")
    urllib.request.urlretrieve(archive_url, archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path="/kaggle/working")
    extracted = f"/kaggle/working/scgo-{GIT_REF}"
    if not os.path.isdir(extracted):
        raise FileNotFoundError(f"Expected extracted source at {extracted}")
    shutil.move(extracted, WORKDIR)


def main() -> int:
    try:
        _fetch_repo()
        os.chdir(WORKDIR)

        py = _resolve_python()
        pip = [*py, "-m", "pip"]
        run([*pip, "install", "--upgrade", "pip"])
        run(
            [
                *pip,
                "install",
                "--no-cache-dir",
                "torch>=2.12.0,<2.13",
                "--index-url",
                "https://download.pytorch.org/whl/cu124",
            ]
        )
        run([*pip, "install", "--no-cache-dir", "-e", ".[mace,dev]"])

        run(
            [
                *py,
                "-c",
                "import torch; assert torch.cuda.is_available(), 'CUDA required'",
            ]
        )

        env = os.environ.copy()
        env["SCGO_BATCH_TEST_SAMPLES"] = "15"

        pytest_cmd = [
            *py,
            "-m",
            "pytest",
            "tests/",
            "-m",
            PYTEST_MARKER,
            "-v",
            "--tb=short",
        ]
        log("+ " + " ".join(pytest_cmd))
        completed = subprocess.run(pytest_cmd, env=env)
        return int(completed.returncode)
    except Exception:
        log("SCGO Kaggle runner failed:")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
