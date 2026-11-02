"""Default rigid-body templates for adsorbate fragments (hierarchical surface init)."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.initialization.atomic_radii import get_covalent_radius
from scgo.initialization.initialization_config import (
    BOND_LENGTH_SCALE,
    CONNECTIVITY_FACTOR,
    NONBONDED_GAP_SCALE,
)

# O–H length from the covalent sum. Callers may still pass ``oh_bond_length``.
DEFAULT_OH_BOND_LENGTH = (
    get_covalent_radius("O") + get_covalent_radius("H")
) * BOND_LENGTH_SCALE
# Separation that keeps two parallel OH units in different components.
DEFAULT_PAIRED_OH_SEPARATION = (
    (get_covalent_radius("O") + get_covalent_radius("O"))
    * CONNECTIVITY_FACTOR
    * NONBONDED_GAP_SCALE
)


def build_default_fragment_template(
    symbols: list[str], *, oh_bond_length: float = DEFAULT_OH_BOND_LENGTH
) -> Atoms | None:
    """Return a gas-phase template for simple ``symbols`` lists, or ``None``.

    Supported patterns (exact symbol order) for a **single** template:
        - ``["O", "H"]``: one OH
        - ``["O", "H", "O", "H"]``: two OH separated by ``DEFAULT_PAIRED_OH_SEPARATION``

    For multiple identical fragments, pass ``adsorbates=[frag, frag, ...]`` at the
    runner API; each fragment is placed on its own adsorption site.
    """
    s = [str(x) for x in symbols]
    if s == ["O", "H"]:
        pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, oh_bond_length]], dtype=float)
        return Atoms(symbols=s, positions=pos)
    if s == ["O", "H", "O", "H"]:
        sep = DEFAULT_PAIRED_OH_SEPARATION
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, oh_bond_length],
                [sep, 0.0, 0.0],
                [sep, 0.0, oh_bond_length],
            ],
            dtype=float,
        )
        return Atoms(symbols=s, positions=pos)
    return None
