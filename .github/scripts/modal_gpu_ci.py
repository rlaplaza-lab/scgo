#!/usr/bin/env python3
"""Modal H100 entry point for SCGO GPU CI (invoked by modal-gpu.yml).

Builds a cached image with CUDA torch + MLIP deps from ``pyproject.toml``,
mounts the checked-out repo at runtime, and runs the same pytest marker /
unexpected-OOM guard as the Kaggle GPU runner.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

import modal

REPO_REMOTE = "/root/scgo"
SYNTHETIC_FAILURE_TOKEN = "scgo-simulated-failure"
IGNORE = [
    ".git",
    ".git/**",
    "dist",
    "dist/**",
    "Pt4_searches",
    "Pt4_searches/**",
    "**/__pycache__",
    "**/*.pyc",
    ".pytest_cache",
    ".pytest_cache/**",
    ".kaggle",
    ".kaggle/**",
    "**/kaggle.json",
    ".env",
]


def _parse_early_mlip_extra() -> str:
    """Read ``--mlip-extra`` before Modal builds the image for this process."""
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--mlip-extra" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--mlip-extra="):
            return arg.split("=", 1)[1]
    return os.environ.get("SCGO_MLIP_EXTRA", "mace")


MLIP_EXTRA = _parse_early_mlip_extra()
if MLIP_EXTRA not in ("mace", "upet"):
    raise SystemExit(
        f"Unsupported MLIP_EXTRA={MLIP_EXTRA!r}; expected 'mace' or 'upet' "
        "(UMA is not run on Modal — HuggingFace auth for fairchem weights)."
    )


def _build_image(mlip_extra: str) -> modal.Image:
    """Layer CUDA torch, then MLIP deps from pyproject; mount the repo at runtime.

    Install scripts are copied as files (not inlined) because Modal's
    ``run_commands`` Dockerfile generator cannot parse multiline shell with
    ``for``/``break`` or heredocs.
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        .add_local_file(
            ".github/scripts/modal_install_torch.sh",
            "/build/modal_install_torch.sh",
            copy=True,
        )
        .run_commands("bash /build/modal_install_torch.sh")
        .add_local_file("pyproject.toml", "/build/pyproject.toml", copy=True)
        .add_local_file(
            ".github/scripts/modal_install_deps.py",
            "/build/modal_install_deps.py",
            copy=True,
        )
        .run_commands(f"python /build/modal_install_deps.py {mlip_extra}")
        .add_local_dir(
            ".",
            remote_path=REPO_REMOTE,
            copy=False,
            ignore=IGNORE,
        )
    )


app = modal.App(f"scgo-gpu-ci-{MLIP_EXTRA}")
image = _build_image(MLIP_EXTRA)


def _log(message: str) -> None:
    print(message, flush=True)


def _is_unexpected_oom_line(line: str) -> bool:
    """True when ``line`` reports a genuine (non-simulated) GPU degradation.

    Mirrors ``.github/scripts/kaggle_gpu_runner.template.py``.
    """
    lowered = line.lower()
    if SYNTHETIC_FAILURE_TOKEN in lowered:
        return False
    if "out of memory" in lowered or "outofmemory" in lowered:
        return True
    if re.search(r"\bmax_metric\b", lowered):
        return True
    if "hit cuda oom" in lowered or "retry still oom" in lowered:
        return True
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


def _assert_cuda_usable() -> None:
    # torch is only on the Modal image, not on the GitHub Actions runner that
    # imports this module to launch ``modal run``.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    name = torch.cuda.get_device_name()
    cap = torch.cuda.get_device_capability()
    _log(f"GPU: {name}, capability sm_{cap[0]}{cap[1]}")
    if cap[0] < 7:
        raise RuntimeError(
            f"GPU {name} (sm_{cap[0]}{cap[1]}) is incompatible with the "
            "installed PyTorch CUDA build"
        )
    torch.ones(1, device="cuda")
    _log("CUDA smoke test passed")


def _run_suite(mlip_extra: str, marker: str) -> int:
    workdir = Path(REPO_REMOTE)
    if not (workdir / "pyproject.toml").is_file():
        raise FileNotFoundError(
            f"Repo mount missing at {workdir}; expected pyproject.toml from "
            "add_local_dir in the Modal image."
        )
    os.chdir(workdir)

    subprocess.run(
        [
            "pip",
            "install",
            "--no-cache-dir",
            "-e",
            f".[{mlip_extra},dev]",
            "--no-deps",
        ],
        check=True,
    )
    _assert_cuda_usable()

    env = os.environ.copy()
    env["SCGO_BATCH_TEST_SAMPLES"] = "15"
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")

    pytest_timeout = "1800" if "gpu_smoke" in marker else "3600"
    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-m",
        marker,
        "-v",
        "--tb=short",
        f"--timeout={pytest_timeout}",
        "--capture=tee-sys",
        "--log-cli-level=INFO",
        "--log-cli-format=%(asctime)s %(levelname)s %(name)s: %(message)s",
        "-rA",
        "--durations=25",
    ]
    _log("+ " + " ".join(pytest_cmd))
    returncode, oom_lines = _run_pytest_streaming(pytest_cmd, env)
    if oom_lines:
        _log("")
        _log(
            "SCGO GPU CI: NEB bands were dropped due to GPU memory pressure. "
            "Green tests are not enough here: this means the transition-state "
            "stage silently degraded. Failing the job."
        )
        for line in oom_lines:
            _log(f"  OOM> {line}")
        return returncode or 1
    return int(returncode)


@app.function(image=image, gpu="H100", timeout=4 * 60 * 60)
def run_gpu_tests(mlip_extra: str, marker: str) -> int:
    """Install the editable package and run the GPU pytest suite on an H100."""
    try:
        return _run_suite(mlip_extra, marker)
    except Exception:
        _log("SCGO Modal runner failed:")
        _log(traceback.format_exc())
        return 1


@app.local_entrypoint()
def main(mlip_extra: str = MLIP_EXTRA, marker: str = "") -> None:
    """Launch the remote H100 suite; exit with the remote pytest return code."""
    if mlip_extra != MLIP_EXTRA:
        raise SystemExit(
            f"--mlip-extra={mlip_extra!r} does not match image built for "
            f"{MLIP_EXTRA!r}; pass the same value Modal saw at import time."
        )
    if not marker.strip():
        raise SystemExit("--marker is required (pytest -m expression)")
    code = run_gpu_tests.remote(mlip_extra, marker)
    raise SystemExit(code)
