"""Generic slab-first surface runner helpers for SCGO example scripts.

Provides reusable utilities that work with *any* ASE slab ``Atoms`` object,
so example runner scripts only need to build or load their slab and call
these helpers — no surface-specific module required.
"""

from __future__ import annotations

from ase import Atoms

from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.pbc import normalize_slab_pbc


def make_surface_config(
    slab: Atoms,
    *,
    adsorption_height_min: float = 2.0,
    adsorption_height_max: float = 3.5,
    fix_all_slab_atoms: bool = True,
    comparator_use_mic: bool = True,
    max_placement_attempts: int = 500,
) -> SurfaceSystemConfig:
    """Build a ``SurfaceSystemConfig`` from an arbitrary ASE slab.

    Parameters
    ----------
    slab:
        ASE ``Atoms`` for the substrate. In-plane periodicity is preserved;
        periodicity along the vacuum axis (default ``z``) is turned off when
        present so the slab is suitable for MLIP/ASE relaxations.
    adsorption_height_min:
        Minimum adsorbate height above the slab surface (Angstrom).
    adsorption_height_max:
        Maximum adsorbate height above the slab surface (Angstrom).
    fix_all_slab_atoms:
        Whether to freeze every slab atom during local relaxation.
    comparator_use_mic:
        Use minimum image convention for duplicate detection (recommended
        when the slab has in-plane periodicity).
    max_placement_attempts:
        Maximum random placement retries per candidate structure.

    See Also
    --------
    attach_slab_constraints_from_surface_config :
        Lower-level helper applied automatically when ``surface_config`` is passed
        to ``run_transition_state_search``.
    """
    slab = slab.copy()
    normalize_slab_pbc(slab)
    return SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=adsorption_height_min,
        adsorption_height_max=adsorption_height_max,
        fix_all_slab_atoms=fix_all_slab_atoms,
        comparator_use_mic=comparator_use_mic,
        max_placement_attempts=max_placement_attempts,
    )
