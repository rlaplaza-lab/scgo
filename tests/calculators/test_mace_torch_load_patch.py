"""Scoped torch.load patch for MACE checkpoint loading."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.requires_mace
def test_torch_load_weights_only_false_restores_original() -> None:
    from scgo.calculators.mace_helpers import torch_load_weights_only_false

    original = torch.load
    seen: list[bool | None] = []

    def _probe(*args, **kwargs):
        seen.append(kwargs.get("weights_only"))
        return {"ok": True}

    torch.load = _probe  # type: ignore[method-assign]
    try:
        with torch_load_weights_only_false():
            assert torch.load is not _probe
            torch.load("ignored")  # type: ignore[call-arg]
            assert seen == [False]
        assert torch.load is _probe
    finally:
        torch.load = original  # type: ignore[method-assign]
    assert torch.load is original
