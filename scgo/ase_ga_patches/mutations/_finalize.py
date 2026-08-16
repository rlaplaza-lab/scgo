"""Shared finalization helper for mutation ``get_new_individual`` methods."""

# fmt: off

from __future__ import annotations

import numpy as np
from ase import Atoms

from scgo.ase_ga_patches.mutations._common import (
    _IDENTITY_ATOL,
    _preserves_mobile_connectivity,
    _resolve_op_connectivity_factor,
)
from scgo.exceptions import SCGOValidationError
from scgo.utils.helpers import get_composition_counts

__all__ = ["_finalize_mutant"]


def _finalize_mutant(creator, parent, mutant, description):
    """Finalize a mutated ``Atoms`` object into a new individual.

    Wraps the common ``initialize_individual``/``finalize_individual`` pattern
    duplicated across every mutation's ``get_new_individual`` method. ``mutant``
    is the result of the operator's ``mutate`` call; if it is ``None`` (mutation
    failed), it is returned unchanged alongside ``description``.

    A post-mutation atom-count + stoichiometry guard ensures no mutation silently
    drops, duplicates, or transmutes atoms. Identity geometry and a disconnected
    mobile region are declined (``None``) so the GA does not store no-ops or
    fragments that the production storage gate would reject.
    """
    if mutant is None:
        return mutant, description

    if not isinstance(mutant, Atoms):
        raise SCGOValidationError(
            f"{getattr(creator, 'descriptor', type(creator).__name__)} produced a "
            f"non-Atoms mutant ({type(mutant).__name__})."
        )
    if len(mutant) != len(parent):
        raise SCGOValidationError(
            f"{getattr(creator, 'descriptor', type(creator).__name__)} changed atom "
            f"count: {len(mutant)} vs {len(parent)}."
        )
    mutant_counts = get_composition_counts(mutant.get_chemical_symbols())
    parent_counts = get_composition_counts(parent.get_chemical_symbols())
    if mutant_counts != parent_counts:
        raise SCGOValidationError(
            f"{getattr(creator, 'descriptor', type(creator).__name__)} changed "
            f"stoichiometry: {dict(mutant_counts)} vs {dict(parent_counts)}."
        )
    if np.allclose(mutant.get_positions(), parent.get_positions(), atol=_IDENTITY_ATOL):
        return None, description

    n_top = getattr(creator, "n_top", None)
    n_mobile = len(mutant) if n_top is None else int(n_top)
    if n_mobile > 0:
        mobile = mutant[len(mutant) - n_mobile :]
        policy = getattr(creator, "_policy", None)
        use_mic = bool(getattr(policy, "uses_surface", False))
        if not _preserves_mobile_connectivity(
            parent[len(parent) - n_mobile :],
            mobile,
            use_mic=use_mic,
            connectivity_factor=_resolve_op_connectivity_factor(creator),
        ):
            return None, description

    indi = creator.initialize_individual(parent, mutant)
    indi.info["data"]["parents"] = [parent.info.get("confid")]

    return creator.finalize_individual(indi), description

# fmt: on
