"""Default rigid-body templates for adsorbate fragments (hierarchical surface init)."""

from __future__ import annotations

import numpy as np
from ase import Atoms

# Default O–H length (Å) for built-in OH and paired-OH patterns
DEFAULT_OH_BOND_LENGTH = 0.96


def build_default_fragment_template(
    symbols: list[str], *, oh_bond_length: float = DEFAULT_OH_BOND_LENGTH
) -> Atoms | None:
    """Return a gas-phase template for simple ``adsorbate_symbols`` lists, or ``None``.

    Supported patterns (exact symbol order) for a **single** template:
        - ``["O", "H"]``: one OH

    For multiple identical fragments, pass ``adsorbates=[frag, frag, ...]`` at the
    runner API; each fragment is placed on its own adsorption site.
    """
    s = [str(x) for x in symbols]
    if s == ["O", "H"]:
        pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, oh_bond_length]], dtype=float)
        return Atoms(symbols=s, positions=pos)
    if s == ["O", "H", "O", "H"]:
        sep = 2.2
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
