"""TorchSim force reads used by NEB Hessian checks."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from scgo.exceptions import SCGORuntimeError


def forces_from_torchsim_batch(
    relaxer: Any,
    displaced: list[Atoms],
) -> list[np.ndarray]:
    """Forces from a zero-step TorchSim batch, for a Hessian finite difference."""
    batch = relaxer.relax_batch(displaced, steps=0)
    forces_out: list[np.ndarray] = []
    for _energy, relaxed_atoms in batch:
        forces = relaxed_atoms.arrays.get("forces")
        if forces is None and relaxed_atoms.calc is not None:
            forces = relaxed_atoms.get_forces()
        if forces is None:
            raise SCGORuntimeError(
                "TorchSim relax_batch did not return forces for Hessian"
            )
        forces_out.append(np.asarray(forces, dtype=float))
    return forces_out
