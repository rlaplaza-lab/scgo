"""Shared NEB endpoint copy / constraint / validation prep."""

from __future__ import annotations

from typing import Any

from ase import Atoms

from scgo.surface.constraints import attach_slab_constraints_from_surface_config
from scgo.system_types import validate_structure_for_system_type
from scgo.utils.helpers import copy_atoms
from scgo.utils.ts_runner_kwargs import NebRunConfig


def prepare_neb_endpoints(
    atoms_i: Atoms,
    atoms_j: Atoms,
    neb_cfg: NebRunConfig,
) -> tuple[Atoms, Atoms]:
    """Copy minima endpoints, attach slab FixAtoms, and validate both sides.

    Raises:
        ValueError: Propagated from :func:`validate_structure_for_system_type`
            when an endpoint fails system-type checks.
    """
    react = copy_atoms(atoms_i)
    prod = copy_atoms(atoms_j)
    if neb_cfg.surface_config is not None:
        attach_slab_constraints_from_surface_config(react, neb_cfg.surface_config)
        attach_slab_constraints_from_surface_config(prod, neb_cfg.surface_config)
    validate_kwargs: dict[str, Any] = {
        "system_type": neb_cfg.system_type,
        "surface_config": neb_cfg.surface_config,
        "n_slab": neb_cfg.n_slab,
        "adsorbate_definition": neb_cfg.adsorbate_definition,
        "connectivity_factor": neb_cfg.connectivity_factor,
        "cluster_adsorbate_config": neb_cfg.cluster_adsorbate_config,
        "allow_cluster_fragmentation": neb_cfg.allow_cluster_fragmentation,
        "allow_adsorbate_surface_detachment": neb_cfg.allow_adsorbate_surface_detachment,
        "enforce_adsorbate_subgraph_integrity": neb_cfg.enforce_adsorbate_subgraph_integrity,
    }
    for ep in (react, prod):
        validate_structure_for_system_type(ep, **validate_kwargs)
    return react, prod
