"""Rigid-body restoration for adsorbate fragments with frozen internal geometry."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase import Atoms

from scgo.cluster_adsorbate.constraints import (
    attach_adsorbate_internal_geometry_constraints,
)
from scgo.cluster_adsorbate.helpers import parse_positive_fragment_lengths
from scgo.exceptions import SCGOValidationError
from scgo.system_types import (
    AdsorbateDefinition,
    AdsorbateFragmentInput,
    resolve_adsorbate_fragments,
)


def _kabsch_place_template(template: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Place ``template`` onto ``target`` preserving target centroid and orientation."""
    if len(template) != len(target):
        raise SCGOValidationError("template and target must have the same length")
    if len(template) == 1:
        return target.copy()

    template_com = template.mean(axis=0)
    target_com = target.mean(axis=0)
    template_centered = template - template_com
    target_centered = target - target_com
    covariance = template_centered.T @ target_centered
    u, _singular, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    return (template_centered @ rotation.T) + target_com


def restore_rigid_adsorbate_fragments(
    atoms: Atoms,
    *,
    n_slab: int,
    adsorbate_definition: AdsorbateDefinition,
    fragment_templates: Sequence[Atoms],
) -> None:
    """Reset adsorbate internal coordinates to templates in-place.

    Each fragment keeps its current center and best-fit rigid orientation.
    """
    core_symbols = adsorbate_definition.get("core_symbols", [])
    if not isinstance(core_symbols, list):
        return
    lengths = parse_positive_fragment_lengths(
        adsorbate_definition.get("adsorbate_fragment_lengths", [])
    )
    if len(lengths) != len(fragment_templates):
        return

    positions = atoms.get_positions()
    ads_start = int(n_slab) + len(core_symbols)
    offset = 0
    for frag_len, template in zip(lengths, fragment_templates, strict=True):
        start = ads_start + offset
        end = start + frag_len
        template_pos = np.asarray(template.get_positions(), dtype=float)
        current = np.asarray(positions[start:end], dtype=float)
        positions[start:end] = _kabsch_place_template(template_pos, current)
        offset += frag_len
    atoms.set_positions(positions)


def enforce_frozen_adsorbate_geometry(
    atoms: Atoms,
    *,
    n_slab: int,
    adsorbate_definition: AdsorbateDefinition | None,
    fragment_templates: AdsorbateFragmentInput | None,
    reattach_constraints: bool = False,
) -> None:
    """Snap adsorbate fragments back to templates and optionally reattach constraints."""
    if adsorbate_definition is None or fragment_templates is None:
        return

    fragments = resolve_adsorbate_fragments(
        fragment_templates,
        adsorbate_definition,
        context="enforce_frozen_adsorbate_geometry",
    )
    restore_rigid_adsorbate_fragments(
        atoms,
        n_slab=n_slab,
        adsorbate_definition=adsorbate_definition,
        fragment_templates=fragments,
    )
    if reattach_constraints:
        atoms.set_constraint([])
        attach_adsorbate_internal_geometry_constraints(
            atoms,
            n_slab=n_slab,
            adsorbate_definition=adsorbate_definition,
        )
