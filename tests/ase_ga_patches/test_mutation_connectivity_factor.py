"""Mutation connectivity gate must honor the run-resolved factor."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.algorithms.ga_common import create_mutation_operators
from scgo.ase_ga_patches.mutations._common import _mobile_is_connected
from scgo.initialization.atomic_radii import build_blmin_from_zs


def test_mutation_operators_use_run_connectivity_factor() -> None:
    blmin = build_blmin_from_zs([78, 79], ratio=0.7)
    ops, _ = create_mutation_operators(
        composition=["Pt", "Au", "Pt"],
        n_to_optimize=3,
        blmin=blmin,
        rng=np.random.default_rng(0),
        connectivity_factor=1.1,
    )
    assert ops and all(op.connectivity_factor == 1.1 for op in ops)

    # Pt–Pt at 3.5 Å: connected at default 1.4 (~3.81 Å), not at 1.1 (~2.99 Å).
    dimer = Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    assert _mobile_is_connected(dimer, use_mic=False, connectivity_factor=1.4)
    assert not _mobile_is_connected(dimer, use_mic=False, connectivity_factor=1.1)
