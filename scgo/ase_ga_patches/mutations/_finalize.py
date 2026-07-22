# fmt: off

from __future__ import annotations

"""Shared finalization helper for mutation ``get_new_individual`` methods."""

__all__ = ["_finalize_mutant"]


def _finalize_mutant(creator, parent, mutant, description):
    """Finalize a mutated ``Atoms`` object into a new individual.

    Wraps the common ``initialize_individual``/``finalize_individual`` pattern
    duplicated across every mutation's ``get_new_individual`` method. ``mutant``
    is the result of the operator's ``mutate`` call; if it is ``None`` (mutation
    failed), it is returned unchanged alongside ``description``.
    """
    if mutant is None:
        return mutant, description

    indi = creator.initialize_individual(parent, mutant)
    indi.info["data"]["parents"] = [parent.info.get("confid")]

    return creator.finalize_individual(indi), description

# fmt: on
