"""A collection of mutations that can be used.

This module is kept as a thin backwards-compatible re-export. The actual
implementations live in the :mod:`scgo.ase_ga_patches.mutations` package,
split by mutation family.
"""

from __future__ import annotations

from scgo.ase_ga_patches.mutations import (
    AnisotropicRattleMutation,
    BreathingMutation,
    CustomPermutationMutation,
    FlatteningMutation,
    InPlaneSlideMutation,
    MirrorMutation,
    OverlapReliefMutation,
    PermutationMutation,
    RattleMutation,
    RotationalMutation,
    ShellSwapMutation,
    _ensure_rng,
)

__all__ = [
    "RattleMutation",
    "AnisotropicRattleMutation",
    "OverlapReliefMutation",
    "PermutationMutation",
    "CustomPermutationMutation",
    "ShellSwapMutation",
    "MirrorMutation",
    "RotationalMutation",
    "FlatteningMutation",
    "BreathingMutation",
    "InPlaneSlideMutation",
    "_ensure_rng",
]
