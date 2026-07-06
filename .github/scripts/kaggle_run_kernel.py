#!/usr/bin/env python3
"""Push a Kaggle GPU kernel, wait for completion, and report parsed logs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

_REDACT_PATTERNS = (
    re.compile(r"KGAT_[A-Za-z0-9_-]+"),
    re.compile(r"(?i)(api[_-]?token|access[_-]?token|kaggle[_-]?key)\s*[:=]\s*\S+"),
)


def redact_text(text: str) -> str:
    redacted = text
    for pattern in _REDACT_PATTERNS:
        redacted = pattern.sub("***", redacted)
    return redacted


def parse_kaggle_log(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    entries: list[dict] = []
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            entries = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            entries = [payload]
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            while idx < len(text) and text[idx] in ", \n\r\t":
                idx += 1
            if idx >= len(text):
                break
            try:
                obj, end = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                return text
            if isinstance(obj, dict):
                entries.append(obj)
            idx = end

    if not entries:
        return text

    chunks: list[str] = []
    for entry in entries:
        data = entry.get("data")
        if isinstance(data, str):
            chunks.append(data)
    return "".join(chunks) if chunks else text


def run_cmd(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.stdout:
        print(
            redact_text(completed.stdout),
            end="" if completed.stdout.endswith("\n") else "\n",
        )
    if completed.stderr:
        print(
            redact_text(completed.stderr),
            end="" if completed.stderr.endswith("\n") else "\n",
            file=sys.stderr,
        )
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode, cmd, output=completed.stdout, stderr=completed.stderr
        )
    return completed


def write_kernel_metadata(
    *,
    staging_dir: Path,
    slug: str,
    title: str,
    code_file: Path,
    dataset_sources: list[str],
    timeout_seconds: int,
) -> None:
    run_cmd(["kaggle", "kernels", "init", "-p", str(staging_dir)])
    metadata_path = staging_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    owner = str(metadata.get("id", slug)).split("/")[0]
    metadata["id"] = f"{owner}/{slug}"
    metadata["title"] = title
    metadata["code_file"] = str(code_file.resolve())
    metadata["language"] = "python"
    metadata["kernel_type"] = "script"
    metadata["is_private"] = True
    metadata["enable_gpu"] = True
    metadata["enable_tpu"] = False
    metadata["enable_internet"] = True
    metadata["dataset_sources"] = dataset_sources
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    push_cmd = ["kaggle", "kernels", "push", "-p", str(staging_dir)]
    if timeout_seconds > 0:
        push_cmd.extend(["--timeout", str(timeout_seconds)])
    run_cmd(push_cmd)


def download_kernel_log(slug: str, output_dir: Path) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(["kaggle", "kernels", "output", slug, "-p", str(output_dir)], check=False)
    log_files = sorted(output_dir.glob("*.log"))
    if not log_files:
        log_files = sorted(output_dir.rglob("*.log"))
    return log_files[0] if log_files else None


def wait_for_kernel(slug: str, *, fetch_seconds: int) -> str:
    while True:
        completed = run_cmd(["kaggle", "kernels", "status", slug], check=False)
        status = (completed.stdout or completed.stderr or "").strip()
        print(status)
        lowered = status.lower()
        if "error" in lowered:
            return "error"
        if "cancel" in lowered:
            return "cancel"
        if "complete" in lowered:
            return "complete"
        time.sleep(fetch_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default="scgogpuci")
    parser.add_argument("--title", default="ScgoGpuCI")
    parser.add_argument("--code-file", default=".github/scripts/kaggle_gpu_runner.py")
    parser.add_argument("--dataset", action="append", default=["rlaplaza/scgocisrc"])
    parser.add_argument("--fetch-seconds", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--log-dir", default="/tmp/kaggle-kernel-log")
    args = parser.parse_args()

    code_file = Path(args.code_file)
    if not code_file.is_file():
        raise SystemExit(f"Kernel code file not found: {code_file}")

    staging = Path("/tmp/kaggle-kernel-push")
    staging.mkdir(parents=True, exist_ok=True)
    staged_code = staging / code_file.name
    staged_code.write_text(code_file.read_text(encoding="utf-8"), encoding="utf-8")

    write_kernel_metadata(
        staging_dir=staging,
        slug=args.slug,
        title=args.title,
        code_file=staged_code,
        dataset_sources=args.dataset,
        timeout_seconds=args.timeout_seconds,
    )

    final_status = wait_for_kernel(args.slug, fetch_seconds=args.fetch_seconds)
    log_path = download_kernel_log(args.slug, Path(args.log_dir))
    if log_path is not None:
        print(f"::group::Kaggle kernel log ({log_path.name})")
        print(
            redact_text(
                parse_kaggle_log(log_path.read_text(encoding="utf-8", errors="replace"))
            )
        )
        print("::endgroup::")
    else:
        print("No Kaggle kernel .log file found", file=sys.stderr)

    if final_status == "complete":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
