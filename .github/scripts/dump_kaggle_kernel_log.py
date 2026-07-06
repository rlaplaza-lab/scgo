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


def _render_log(log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return text

    lines = text.splitlines()
    chunks: list[str] = []
    parsed_ndjson = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            chunks.append(line)
            continue
        parsed_ndjson = True
        data = payload.get("data")
        if isinstance(data, str):
            chunks.append(data)

    if parsed_ndjson:
        return "".join(chunks)
    return text


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

    log_files = sorted(output_dir.rglob("*.log"))
    if not log_files:
        print(f"No .log files under {output_dir}", file=sys.stderr)
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                print(f"===== {path} =====")
                print(redact_text(path.read_text(encoding="utf-8", errors="replace")))
        return 1

    for log_path in log_files:
        print(f"===== {log_path.name} =====")
        print(redact_text(_render_log(log_path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
