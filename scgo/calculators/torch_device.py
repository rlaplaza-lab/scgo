"""Shared Torch device resolution for MACE / UMA / UPET / TorchSim helpers."""

from __future__ import annotations

import warnings

import torch

from scgo.exceptions import SCGOValidationError

_warned_unsupported: set[str] = set()


def _normalize_device_str(device: str | torch.device) -> str:
    """Normalize a device specifier to a lowercase string (e.g. ``cuda:0``)."""
    if isinstance(device, torch.device):
        if device.index is not None and device.type == "cuda":
            return f"cuda:{device.index}"
        return device.type
    text = str(device).strip().lower()
    if text.startswith("cuda"):
        return text
    if text in ("cpu", "mps"):
        return text
    return text


def _is_supported(device_str: str, *, allow_mps: bool) -> bool:
    if device_str == "cpu":
        return True
    if device_str == "mps":
        return allow_mps
    return device_str == "cuda" or device_str.startswith("cuda:")


def resolve_torch_device(
    device: str | torch.device | None,
    *,
    allow_mps: bool,
    backend_name: str,
) -> str:
    """Resolve a Torch device string.

    Auto-select prefers CUDA, then MPS when ``allow_mps`` and available, else CPU.
    Explicit unsupported devices warn once and raise :exc:`SCGOValidationError`
    instead of silently coercing to CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        if allow_mps and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    selected = _normalize_device_str(device)
    if _is_supported(selected, allow_mps=allow_mps):
        return selected

    key = f"{backend_name}:{selected}:mps={allow_mps}"
    if key not in _warned_unsupported:
        _warned_unsupported.add(key)
        warnings.warn(
            f"{backend_name}: device {selected!r} is not supported "
            f"(allow_mps={allow_mps}).",
            UserWarning,
            stacklevel=2,
        )
    raise SCGOValidationError(
        f"{backend_name}: unsupported device {selected!r} "
        f"(allow_mps={allow_mps}); use cuda, cpu"
        + (", or mps" if allow_mps else "")
        + "."
    )
