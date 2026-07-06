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
from pathlib import Path

REPO_URL = "https://github.com/rlaplaza-lab/scgo.git"
GIT_REF = "__GIT_REF__"
PYTEST_MARKER = "__PYTEST_MARKER__"
CONDA_ENV = "scgo-gpu"
# Use /tmp so pytest/pip artifacts are not saved as Kaggle kernel output.
WORKDIR = Path("/tmp/scgo")
DATASET_INPUT = Path("/kaggle/input/scgocisrc")
SOURCE_ARCHIVE = "scgo-src.tar.gz"
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"
PYPI_INDEX = "https://pypi.org/simple"


def log(message: str) -> None:
    print(message, flush=True)


def run(
    cmd: list[str], *, cwd: str | Path | None = None, env: dict[str, str] | None = None
) -> None:
    log("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def _python_ok(version: str) -> bool:
    major, minor, *_ = (int(part) for part in version.split("."))
    return (major, minor) >= (3, 12)


def _log_kaggle_inputs() -> None:
    inputs_root = Path("/kaggle/input")
    if not inputs_root.is_dir():
        log("No /kaggle/input directory mounted")
        return
    log("Kaggle input mounts:")
    for path in sorted(inputs_root.rglob("*")):
        if path.is_file():
            log(f"  {path} ({path.stat().st_size} bytes)")


def _system_python() -> list[str] | None:
    for candidate in ("python3.12", "python3", sys.executable, "python"):
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


def _safe_extractall(tar: tarfile.TarFile, path: Path) -> None:
    if hasattr(tarfile, "data_filter"):
        tar.extractall(path=path, filter="data")
    else:
        tar.extractall(path=path)


def _extract_dataset_archive(archive: Path) -> None:
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    log(f"Extracting bundled source from {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        _safe_extractall(tar, WORKDIR)


def _find_dataset_archive() -> Path | None:
    if not DATASET_INPUT.is_dir():
        return None
    direct = DATASET_INPUT / SOURCE_ARCHIVE
    if direct.is_file():
        return direct
    matches = sorted(DATASET_INPUT.rglob(SOURCE_ARCHIVE))
    return matches[0] if matches else None


def _dataset_tree_ready() -> bool:
    return DATASET_INPUT.is_dir() and (DATASET_INPUT / "pyproject.toml").is_file()


def _copy_dataset_tree() -> None:
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    log(f"Copying bundled source tree from {DATASET_INPUT}")
    shutil.copytree(
        DATASET_INPUT,
        WORKDIR,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        dirs_exist_ok=True,
    )


def _fetch_repo_from_dataset() -> bool:
    archive = _find_dataset_archive()
    if archive is not None:
        _extract_dataset_archive(archive)
        return True
    if _dataset_tree_ready():
        _copy_dataset_tree()
        return True
    return False


def _ensure_git() -> None:
    if shutil.which("git"):
        return
    if shutil.which("apt-get"):
        run(["apt-get", "update"])
        run(["apt-get", "install", "-y", "git"])


def _fetch_repo_from_network() -> None:
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir(parents=True, exist_ok=True)
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
                    str(WORKDIR),
                ]
            )
            return
        except subprocess.CalledProcessError as exc:
            log(f"git clone failed ({exc}); falling back to source tarball")
    archive_url = (
        f"https://github.com/rlaplaza-lab/scgo/archive/refs/heads/{GIT_REF}.tar.gz"
    )
    archive_path = Path("/tmp/scgo-src-download.tar.gz")
    log(f"Downloading {archive_url}")
    urllib.request.urlretrieve(archive_url, archive_path)
    extracted = Path(f"/tmp/scgo-{GIT_REF}")
    if extracted.exists():
        shutil.rmtree(extracted)
    with tarfile.open(archive_path, "r:gz") as tar:
        _safe_extractall(tar, Path("/tmp"))
    if not extracted.is_dir():
        raise FileNotFoundError(f"Expected extracted source at {extracted}")
    shutil.move(str(extracted), str(WORKDIR))


def _fetch_repo() -> None:
    if _fetch_repo_from_dataset():
        log("Using CI source bundle from Kaggle dataset input")
        return
    log("Dataset bundle not found; fetching source over the network")
    _fetch_repo_from_network()


def _install_torch_stack(py: list[str], pip: list[str]) -> None:
    """Install CUDA torch on Kaggle where cu124 wheels may be 2.4–2.6 only."""
    attempts = (
        [
            *pip,
            "install",
            "--no-cache-dir",
            "torch>=2.12.0,<2.13",
            "torchvision",
            "--index-url",
            PYTORCH_CUDA_INDEX,
            "--extra-index-url",
            PYPI_INDEX,
        ],
        [
            *pip,
            "install",
            "--no-cache-dir",
            "torch",
            "torchvision",
            "--index-url",
            PYTORCH_CUDA_INDEX,
        ],
    )
    for cmd in attempts:
        log("+ " + " ".join(cmd))
        completed = subprocess.run(cmd)
        if completed.returncode == 0:
            return
        log(f"Torch install failed (exit {completed.returncode}); trying fallback")
    raise subprocess.CalledProcessError(1, attempts[-1])


def _install_scgo_mace(py: list[str], pip: list[str]) -> None:
    """Install SCGO + MACE without pyproject's torch>=2.12 pin (unavailable on Kaggle)."""
    run([*pip, "install", "--no-cache-dir", "-e", ".", "--no-deps"])
    run(
        [
            *pip,
            "install",
            "--no-cache-dir",
            "ase>=3.22.0",
            "ase-ga>=0.1.0",
            "numpy>=2.2",
            "scipy>=1.14,<3",
            "tqdm>=4.60.0",
            "e3nn==0.4.4",
            "mace-torch==0.3.16",
            "nvalchemi-toolkit-ops==0.3.1",
            "nvidia-nccl-cu12>=2.28",
            "torch-sim-atomistic[mace]==0.6.0",
            "pytest>=7.0.0",
            "pytest-xdist>=3.0.0",
            "pytest-cov>=4.0.0",
            "psutil>=7.0.0",
        ]
    )


def _assert_cuda_usable(py: list[str]) -> None:
    run(
        [
            *py,
            "-c",
            (
                "import torch\n"
                "if not torch.cuda.is_available():\n"
                "    raise SystemExit('CUDA required')\n"
                "name = torch.cuda.get_device_name()\n"
                "cap = torch.cuda.get_device_capability()\n"
                "print(f'GPU: {name}, capability sm_{cap[0]}{cap[1]}')\n"
                "if cap[0] < 7:\n"
                "    raise SystemExit(\n"
                "        f'GPU {name} (sm_{cap[0]}{cap[1]}) is incompatible with the '\n"
                "        'installed PyTorch CUDA build; use machine_shape NvidiaTeslaT4'\n"
                "    )\n"
                "torch.ones(1, device='cuda')\n"
                "print('CUDA smoke test passed')\n"
            ),
        ]
    )


def main() -> int:
    try:
        _log_kaggle_inputs()
        _fetch_repo()
        os.chdir(WORKDIR)

        py = _resolve_python()
        pip = [*py, "-m", "pip"]
        run([*pip, "install", "--upgrade", "pip"])
        _install_torch_stack(py, pip)
        _install_scgo_mace(py, pip)
        _assert_cuda_usable(py)

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
