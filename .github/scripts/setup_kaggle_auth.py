#!/usr/bin/env python3
"""Write ~/.kaggle credentials from normalized workflow environment variables."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    api_token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()

    if api_token:
        token_path = kaggle_dir / "access_token"
        token_path.write_text(api_token, encoding="utf-8")
        token_path.chmod(0o600)
        return

    if username and key:
        cred_path = kaggle_dir / "kaggle.json"
        cred_path.write_text(
            json.dumps({"username": username, "key": key}),
            encoding="utf-8",
        )
        cred_path.chmod(0o600)
        return

    raise SystemExit(
        "No Kaggle credentials: set KAGGLE_API_TOKEN or KAGGLE_USERNAME+KAGGLE_KEY"
    )


if __name__ == "__main__":
    main()
