"""Mutation operators for the ASE GA patches.

This package splits what used to be a single, large ``standardmutations``
module into one module per mutation family, plus shared helpers
(``_common.py``, ``_finalize.py``). All public mutation classes are
re-exported here so that ``scgo.ase_ga_patches.standardmutations`` (which now
just re-exports from this package) keeps working unchanged.
"""

from __future__ import annotations

from scgo.ase_ga_patches.mutations._common import _ensure_rng
from scgo.ase_ga_patches.mutations.breathing import BreathingMutation
from scgo.ase_ga_patches.mutations.flattening import FlatteningMutation
from scgo.ase_ga_patches.mutations.in_plane_slide import InPlaneSlideMutation
from scgo.ase_ga_patches.mutations.mirror import MirrorMutation
from scgo.ase_ga_patches.mutations.overlap_relief import OverlapReliefMutation
from scgo.ase_ga_patches.mutations.permutation import (
    CustomPermutationMutation,
    PermutationMutation,
)
from scgo.ase_ga_patches.mutations.rattle import (
    AnisotropicRattleMutation,
    RattleMutation,
)
from scgo.ase_ga_patches.mutations.rotational import RotationalMutation
from scgo.ase_ga_patches.mutations.shell_swap import ShellSwapMutation

__all__ = [
    "AnisotropicRattleMutation",
    "BreathingMutation",
    "CustomPermutationMutation",
    "FlatteningMutation",
    "InPlaneSlideMutation",
    "MirrorMutation",
    "OverlapReliefMutation",
    "PermutationMutation",
    "RattleMutation",
    "RotationalMutation",
    "ShellSwapMutation",
    "_ensure_rng",
]
