"""Shared Torch device resolution for MACE / UMA / UPET / TorchSim helpers."""

from __future__ import annotations

try:
    import torch
except ImportError:
    torch = None  # PyTorch unavailable; module still importable.

from scgo.exceptions import SCGOConfigurationError, SCGOValidationError


def _normalize_device_str(device: str | object) -> str:
    return str(device).strip().lower()


def _is_supported(device_str: str, *, allow_mps: bool) -> bool:
    if device_str == "cpu":
        return True
    if device_str == "mps":
        return allow_mps
    return device_str == "cuda" or device_str.startswith("cuda:")


def resolve_torch_device(
    device: str | object | None,
    *,
    allow_mps: bool,
    backend_name: str,
) -> str:
    """Resolve a Torch device string (CUDA, optional MPS, else CPU).

    Explicit unsupported devices raise :exc:`SCGOValidationError`.
    """
    if torch is None:
        raise SCGOConfigurationError(
            f"{backend_name}: PyTorch is not installed. "
            "Install an MLIP extra (e.g. pip install 'scgo[mace]', "
            "'scgo[uma]', or 'scgo[upet]')."
        )

    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        if allow_mps and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    selected = _normalize_device_str(device)
    if _is_supported(selected, allow_mps=allow_mps):
        return selected

    raise SCGOValidationError(
        f"{backend_name}: unsupported device {selected!r} "
        f"(allow_mps={allow_mps}); use cuda, cpu"
        + (", or mps" if allow_mps else "")
        + "."
    )
