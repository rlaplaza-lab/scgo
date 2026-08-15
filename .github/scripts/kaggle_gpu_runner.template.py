#!/usr/bin/env python3
"""Kaggle kernel entry point for SCGO GPU CI (rendered by kaggle-gpu.yml)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import traceback
from pathlib import Path

GIT_REF = "__GIT_REF__"
PYTEST_MARKER = "__PYTEST_MARKER__"
MLIP_EXTRA = "__MLIP_EXTRA__"
CONDA_ENV = "scgo-gpu"
# Use /tmp so pytest/pip artifacts are not saved as Kaggle kernel output.
WORKDIR = Path("/tmp/scgo")
# Kaggle mounts datasets either at /kaggle/input/<slug> or the newer
# /kaggle/input/datasets/<owner>/<slug>; _resolve_dataset_dir() probes for
# whichever layout this kernel actually got.
DATASET_OWNER = "rlaplaza"
DATASET_SLUG = "scgocisrc"
DATASET_INPUT = Path("/kaggle/input") / DATASET_SLUG
SOURCE_ARCHIVE = "scgo-src.tar.gz"
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"
PYPI_INDEX = "https://pypi.org/simple"

# SCGO log phrases that mark a *real* GPU-memory degradation: a fused NEB force
# batch that ran out of memory, the half-budget retry failing again, or a band
# dropped without producing a saddle. Generic torch OOM text is deliberately not
# matched: torch-sim's autobatcher probes memory by triggering OOM on purpose,
# and the warm probe logs a non-fatal "Memory probing failed" on the way.
# Unit tests that simulate these paths tag their message with
# ``SYNTHETIC_FAILURE_TOKEN`` so they can never trip this guard.
#
# A "Parallel NEB band unusable" line is only counted as memory degradation when
# it also carries a genuine degradation/never-ran substring (e.g. "out of
# memory", "batched force evaluation", "neb not processed"). Non-finite forces
# or bad-input ("model weights are corrupt") band failures are physics/numeric,
# not GPU memory pressure, and a green run that contains them is not a
# regression. This keeps the guard robust to unit-test simulations even if they
# forget to tag their message with ``SYNTHETIC_FAILURE_TOKEN``.
SYNTHETIC_FAILURE_TOKEN = "scgo-simulated-failure"


def log(message: str) -> None:
    print(message, flush=True)


def run(
    cmd: list[str], *, cwd: str | Path | None = None, env: dict[str, str] | None = None
) -> None:
    log("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def _log_kaggle_inputs() -> None:
    inputs_root = Path("/kaggle/input")
    if not inputs_root.is_dir():
        log("No /kaggle/input directory mounted")
        return
    log("Kaggle input mounts:")
    for path in sorted(inputs_root.rglob("*")):
        if path.is_file():
            log(f"  {path} ({path.stat().st_size} bytes)")


def _conda_exe() -> str | None:
    """Return a usable conda executable, or ``None`` when conda is unavailable.

    Kaggle's default GPU image ships a plain CPython at
    ``/usr/local/lib/python3.12`` with no conda; the newer GPU runtimes in
    particular omit conda entirely. Callers must fall back to the system
    interpreter when this returns ``None``.
    """
    for candidate in (
        os.environ.get("CONDA_EXE", ""),
        "/opt/conda/bin/conda",
        shutil.which("conda") or "",
    ):
        if candidate and (candidate == "conda" or os.path.isfile(candidate)):
            return candidate
    return None


def _conda_python() -> list[str]:
    conda = _conda_exe()
    if conda is None:
        raise RuntimeError("conda required for conda Python path but not found")
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
    """Prefer the conda env interpreter; fall back to the system interpreter when conda is absent."""
    if _conda_exe() is not None:
        return _conda_python()
    log(f"conda not found; using system interpreter {sys.executable}")
    return [sys.executable]


def _safe_extractall(tar: tarfile.TarFile, path: Path) -> None:
    # Python 3.12+ (project minimum) provides tarfile.data_filter.
    tar.extractall(path=path, filter="data")


def _extract_dataset_archive(archive: Path) -> None:
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    log(f"Extracting bundled source from {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        _safe_extractall(tar, WORKDIR)


def _resolve_dataset_dir() -> Path | None:
    """Locate the mounted CI source dataset regardless of Kaggle's mount layout.

    Kaggle exposes dataset inputs at either ``/kaggle/input/<slug>`` or the
    newer ``/kaggle/input/datasets/<owner>/<slug>``; probe both, then fall back
    to a recursive search so a path-layout change can't silently break source
    discovery (the dataset is published and polled to 'complete' by the
    kaggle-gpu.yml workflow before the kernel launches).
    """
    candidates = [
        DATASET_INPUT,
        Path("/kaggle/input/datasets") / DATASET_OWNER / DATASET_SLUG,
    ]
    for cand in candidates:
        if cand.is_dir() and (cand / "pyproject.toml").is_file():
            return cand
    root = Path("/kaggle/input")
    if root.is_dir():
        for match in sorted(root.rglob(DATASET_SLUG)):
            if match.is_dir() and (match / "pyproject.toml").is_file():
                return match
    return None


def _find_dataset_archive() -> Path | None:
    dataset_dir = _resolve_dataset_dir()
    if dataset_dir is None:
        return None
    direct = dataset_dir / SOURCE_ARCHIVE
    if direct.is_file():
        return direct
    matches = sorted(dataset_dir.rglob(SOURCE_ARCHIVE))
    return matches[0] if matches else None


def _dataset_tree_ready() -> bool:
    dataset_dir = _resolve_dataset_dir()
    return dataset_dir is not None and (dataset_dir / "pyproject.toml").is_file()


def _copy_dataset_tree() -> None:
    dataset_dir = _resolve_dataset_dir()
    assert dataset_dir is not None
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    log(f"Copying bundled source tree from {dataset_dir}")
    shutil.copytree(
        dataset_dir,
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


def _fetch_repo() -> None:
    if not _fetch_repo_from_dataset():
        raise FileNotFoundError(
            "Kaggle dataset bundle 'rlaplaza/scgocisrc' not found in the kernel "
            "input; the kaggle-gpu.yml workflow publishes and polls it to "
            "'complete' before launching the kernel."
        )
    log("Using CI source bundle from Kaggle dataset input")


def _numpy_requirement() -> str:
    """Match pyproject.toml so Kaggle uses the same NumPy pin as CI tests."""
    data = tomllib.loads((WORKDIR / "pyproject.toml").read_text(encoding="utf-8"))
    for dep in data["project"]["dependencies"]:
        if dep.startswith("numpy"):
            return dep
    raise RuntimeError("numpy requirement missing from pyproject.toml")


def _install_numpy(py: list[str], pip: list[str]) -> None:
    """Install project NumPy before torch/torchvision (Kaggle base image ships 2.0.x)."""
    spec = _numpy_requirement()
    run([*pip, "install", "--no-cache-dir", spec])


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
            # Last-resort Kaggle workaround: unpinned torch when cu124 index lacks 2.12.x.
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


def _install_scgo_mlip(py: list[str], pip: list[str], *, mlip_extra: str) -> None:
    """Install SCGO + one MLIP extra (``mace`` or ``upet``); torch is pre-installed."""
    if mlip_extra not in ("mace", "upet"):
        raise SystemExit(
            f"Unsupported MLIP_EXTRA={mlip_extra!r}; expected 'mace' or 'upet' "
            "(UMA is not run on Kaggle — HuggingFace auth for fairchem weights)."
        )
    data = tomllib.loads((WORKDIR / "pyproject.toml").read_text(encoding="utf-8"))
    deps = list(data["project"]["dependencies"])
    deps.extend(data["project"]["optional-dependencies"][mlip_extra])
    deps.extend(data["project"]["optional-dependencies"]["dev"])
    # Torch comes from Kaggle's CUDA index; lint tooling is not needed on the kernel.
    skip_prefixes = ("ruff", "pre-commit")
    install_deps = [
        dep
        for dep in deps
        if not dep.startswith(skip_prefixes) and not dep.startswith("torch>=")
    ]

    run([*pip, "install", "--no-cache-dir", "-e", f".[{mlip_extra},dev]", "--no-deps"])
    run([*pip, "install", "--no-cache-dir", *install_deps])
    if mlip_extra == "upet":
        # metatomic-torchsim declares vesin<0.6 but needs skin= from 0.6.0
        run(
            [
                *pip,
                "install",
                "--no-cache-dir",
                "vesin==0.6.0",
                "--force-reinstall",
                "--no-deps",
            ]
        )


def _assert_numpy_version(py: list[str]) -> None:
    spec = _numpy_requirement()
    run(
        [
            *py,
            "-c",
            (
                "import re\n"
                "import numpy as np\n"
                f"spec = {spec!r}\n"
                "match = re.fullmatch(r'numpy>=(\\d+)\\.(\\d+)(?:,<(\\d+)\\.(\\d+))?', spec)\n"
                "if match is None:\n"
                "    raise SystemExit(f'Unsupported numpy spec: {spec!r}')\n"
                "lo_major, lo_minor, hi_major, hi_minor = match.groups()\n"
                "lo = (int(lo_major), int(lo_minor))\n"
                "hi = (int(hi_major), int(hi_minor)) if hi_major else None\n"
                "parts = [int(part) for part in np.__version__.split('.')[:2]]\n"
                "version = (parts[0], parts[1])\n"
                "if version < lo or (hi is not None and version >= hi):\n"
                "    raise SystemExit(\n"
                "        f'NumPy {np.__version__} does not satisfy {spec!r}'\n"
                "    )\n"
                "print(f'NumPy {np.__version__} satisfies {spec!r}')\n"
            ),
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


def _is_unexpected_oom_line(line: str) -> bool:
    """True when ``line`` reports a genuine (non-simulated) GPU degradation.

    The genuine-OOM substrings below mirror the canonical rule in
    ``scgo.metadata.provenance.is_cuda_oom_error`` (``"out of memory"``).
    """
    lowered = line.lower()
    if SYNTHETIC_FAILURE_TOKEN in lowered:
        return False
    # Genuine torch OOM text. ``torch.cuda.OutOfMemoryError`` and the cuBLAS/cuDNN
    # ``RuntimeError`` both print "out of memory". Do not match generic
    # "Memory Estimation" probe chatter (no OOM substring appears there).
    if "out of memory" in lowered or "outofmemory" in lowered:
        return True
    # torch-sim's InFlight/BinningAutoBatcher raises a ``ValueError`` whose message
    # contains "max_metric" when a later batch's metric exceeds the sticky cached
    # scaler. That is a memory-degradation failure that can masquerade behind a
    # "band unusable" / example-failure line, so treat it as unexpected.
    if "max_metric" in lowered:
        return True
    # Synthetic degradation markers (kept for backwards-compat log scanning).
    if "hit cuda oom" in lowered or "retry still oom" in lowered:
        return True
    # "Parallel NEB band unusable" is emitted for *any* band failure: non-finite
    # forces, bad-input errors, etc. Only count it as memory degradation when the
    # band line also names a genuine degradation / never-ran cause (the same
    # substrings the TS degradation guard matches).
    if "parallel neb band unusable" in lowered:
        return any(
            m in lowered
            for m in (
                "out of memory",
                "outofmemory",
                "batched force evaluation",
                "neb not processed",
            )
        )
    return False


def _run_pytest_streaming(cmd: list[str], env: dict[str, str]) -> tuple[int, list[str]]:
    """Run pytest, tee its output, and collect unexpected GPU-degradation lines."""
    oom_lines: list[str] = []
    with subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if len(oom_lines) < 20 and _is_unexpected_oom_line(line):
                oom_lines.append(line.rstrip())
        returncode = proc.wait()
    return returncode, oom_lines


def main() -> int:
    try:
        _log_kaggle_inputs()
        _fetch_repo()
        os.chdir(WORKDIR)

        py = _resolve_python()
        pip = [*py, "-m", "pip"]
        run([*pip, "install", "--upgrade", "pip"])
        _install_numpy(py, pip)
        _install_torch_stack(py, pip)
        log(f"Installing scgo[{MLIP_EXTRA},dev] for GPU suite")
        _install_scgo_mlip(py, pip, mlip_extra=MLIP_EXTRA)
        _assert_numpy_version(py)
        _assert_cuda_usable(py)

        env = os.environ.copy()
        env["SCGO_BATCH_TEST_SAMPLES"] = "15"
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Reduce allocator fragmentation on the 16 GB T4: the torch OOM
        # traceback itself recommends this, and the fused NEB force batches are
        # exactly the large short-lived allocations it helps with.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # e3nn (pulled by scgo[mace]) unpickles constants.pt via torch.load;
        # PyTorch >=2.6 defaults weights_only=True and rejects it. Force the
        # legacy default off so MACE imports regardless of the torch version the
        # cu124 index resolves. Harmless on the UPET suite.
        env.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")

        pytest_cmd = [
            *py,
            "-m",
            "pytest",
            "tests/",
            "-m",
            PYTEST_MARKER,
            "-v",
            "--tb=short",
            "--timeout=1800",
            "--capture=tee-sys",
            "--log-cli-level=INFO",
            "--log-cli-format=%(asctime)s %(levelname)s %(name)s: %(message)s",
            "-rA",
            "--durations=25",
        ]
        log("+ " + " ".join(pytest_cmd))
        returncode, oom_lines = _run_pytest_streaming(pytest_cmd, env)
        if oom_lines:
            log("")
            log(
                "SCGO GPU CI: NEB bands were dropped due to GPU memory pressure. "
                "Green tests are not enough here: this means the transition-state "
                "stage silently degraded. Failing the job."
            )
            for line in oom_lines:
                log(f"  OOM> {line}")
            return returncode or 1
        return int(returncode)
    except Exception:
        log("SCGO Kaggle runner failed:")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
