"""Energy metrics for adsorption (optional objectives)."""

from __future__ import annotations


def adsorption_energy(
    e_adsorbate_slab: float,
    e_slab: float,
    e_isolated_cluster: float,
) -> float:
    """Classical adsorption energy: E(ads+slab) - E(slab) - E(cluster).

    Negative values usually indicate binding. This is the raw electronic
    total-energy difference only: it neglects zero-point energy, thermal
    (vibrational/entropic) corrections, and cell / finite-size effects, and it
    assumes the slab and isolated cluster were evaluated in the same cell.
    Basis-set superposition error (BSSE) does NOT apply: the MLIP potentials used
    here are not atom-centered basis-set methods, so no counterpoise correction
    is warranted.

    Args:
        e_adsorbate_slab: Total energy of the relaxed adsorbate+slab system.
        e_slab: Total energy of the bare relaxed slab (same cell/supercell).
        e_isolated_cluster: Total energy of the isolated cluster.

    Returns:
        Adsorption energy in the same units as inputs (typically eV).
    """
    return e_adsorbate_slab - e_slab - e_isolated_cluster
