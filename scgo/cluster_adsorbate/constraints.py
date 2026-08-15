"""ASE constraints for adsorbates on a cluster."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ase import Atoms
from ase.constraints import FixBondLengths

from scgo.cluster_adsorbate.helpers import parse_positive_fragment_lengths
from scgo.exceptions import SCGOValidationError

if TYPE_CHECKING:
    from scgo.surface.config import SurfaceSystemConfig
    from scgo.system_types import AdsorbateDefinition, AdsorbateFragmentInput


def attach_fix_bond_lengths(
    atoms: Atoms,
    bond_pairs: Sequence[tuple[int, int]],
) -> None:
    """Append one :class:`~ase.constraints.FixBondLengths` for all pairs (global indices).

    A single multi-pair constraint is used (instead of one
    :class:`~ase.constraints.FixBondLength` per pair) because the per-pair
    constraints are applied sequentially and each one undoes the previous
    correction, so a rigid multi-atom fragment is not actually held rigid.

    Raises:
        SCGOValidationError: If a pair is out of range, self-paired, or duplicated.
    """
    n = len(atoms)
    seen: set[tuple[int, int]] = set()
    new_constraints: list = list(atoms.constraints) if atoms.constraints else []
    pairs: list[tuple[int, int]] = []
    for a, b in bond_pairs:
        if not (0 <= a < n and 0 <= b < n):
            raise SCGOValidationError(
                f"Invalid bond pair ({a}, {b}) for structure with {n} atoms"
            )
        if a == b:
            raise SCGOValidationError(
                f"bond pair ({a}, {b}) must be two distinct atoms"
            )
        key = (min(a, b), max(a, b))
        if key in seen:
            raise SCGOValidationError(f"duplicate bond pair {key}")
        seen.add(key)
        pairs.append((int(a), int(b)))
    if pairs:
        new_constraints.append(FixBondLengths(pairs))
    if new_constraints:
        atoms.set_constraint(new_constraints)


def attach_adsorbate_internal_geometry_constraints(
    atoms: Atoms,
    *,
    n_slab: int,
    adsorbate_definition: AdsorbateDefinition | None,
) -> None:
    """Freeze pairwise distances inside each adsorbate fragment.

    This enforces rigid internal geometry for adsorbates while still allowing
    collective translation/rotation of each adsorbate fragment.
    """
    if adsorbate_definition is None:
        return
    core_symbols = adsorbate_definition.core_symbols
    fragment_lengths = parse_positive_fragment_lengths(
        adsorbate_definition.adsorbate_fragment_lengths
    )

    ads_start = int(n_slab) + len(core_symbols)
    bond_pairs: list[tuple[int, int]] = []
    cursor = ads_start
    for frag_len in fragment_lengths:
        for i in range(cursor, cursor + frag_len):
            for j in range(i + 1, cursor + frag_len):
                bond_pairs.append((i, j))
        cursor += frag_len

    if bond_pairs:
        attach_fix_bond_lengths(atoms, bond_pairs)


def prepare_atoms_for_local_relax(
    atoms: Atoms,
    *,
    surface_mode: bool,
    surface_config: SurfaceSystemConfig | None,
    n_slab: int,
    freeze_adsorbate_internal_geometry: bool,
    adsorbate_definition: AdsorbateDefinition | None,
    adsorbate_fragment_templates: AdsorbateFragmentInput | None,
) -> Atoms:
    """Copy ``atoms`` and attach slab + adsorbate constraints for local relaxation.

    Mirrors the freeze/constrain sequence used by both basin hopping and the
    genetic algorithm: optional slab ``FixAtoms`` plus optional rigid
    adsorbate-internal-geometry constraints. Returns the copied, constrained
    structure (the input is not modified).
    """
    from scgo.cluster_adsorbate.rigid import enforce_frozen_adsorbate_geometry
    from scgo.surface.constraints import attach_slab_constraints

    eff_n_slab = n_slab if surface_mode else 0
    c = atoms.copy()
    if surface_mode and surface_config is not None and n_slab > 0:
        attach_slab_constraints(
            c,
            n_slab,
            fix_all_slab_atoms=surface_config.fix_all_slab_atoms,
            n_fix_bottom_slab_layers=surface_config.n_fix_bottom_slab_layers,
            n_relax_top_slab_layers=surface_config.n_relax_top_slab_layers,
            surface_normal_axis=surface_config.surface_normal_axis,
        )
    if freeze_adsorbate_internal_geometry:
        enforce_frozen_adsorbate_geometry(
            c,
            n_slab=eff_n_slab,
            adsorbate_definition=adsorbate_definition,
            fragment_templates=adsorbate_fragment_templates,
        )
        attach_adsorbate_internal_geometry_constraints(
            c,
            n_slab=eff_n_slab,
            adsorbate_definition=adsorbate_definition,
        )
    return c
