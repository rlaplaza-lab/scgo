"""Calculator family names."""

from __future__ import annotations


def normalize_calculator_name(name: object) -> str:
    return str(name).strip().upper()
