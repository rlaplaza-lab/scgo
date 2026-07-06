#!/usr/bin/env python3
"""Download and print a Kaggle kernel log (NDJSON or plain text)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
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


def main() -> int:
    if len(sys.argv) != 3:
        print(
            f"usage: {sys.argv[0]} <kernel-slug-or-owner/slug> <output-dir>",
            file=sys.stderr,
        )
        return 2

    kernel = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        ["kaggle", "kernels", "output", kernel, "-p", str(output_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
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

    log_files = sorted(output_dir.glob("*.log"))
    if not log_files:
        log_files = sorted(output_dir.rglob("*.log"))
    if not log_files:
        print(f"No .log files under {output_dir}", file=sys.stderr)
        return 1

    for log_path in log_files:
        print(f"::group::Kaggle kernel log ({log_path.name})")
        print(
            redact_text(
                parse_kaggle_log(log_path.read_text(encoding="utf-8", errors="replace"))
            )
        )
        print("::endgroup::")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
