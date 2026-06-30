"""Re-place adsorbate fragments on fresh core-hull sites during GA evolution."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase_ga.offspring_creator import OffspringCreator
from ase_ga.utilities import atoms_too_close_two_sets
from numpy.random import Generator

from scgo.ase_ga_patches.standardmutations import _ensure_rng
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.cluster_adsorbate.helpers import (
    parse_positive_fragment_lengths,
    resolve_fragment_anchor_and_bond_axis,
)
from scgo.cluster_adsorbate.placement import place_fragment_on_cluster
from scgo.system_types import (
    AdsorbateDefinition,
    AdsorbateFragmentInput,
    SystemType,
    get_system_policy,
    resolve_adsorbate_fragments,
)


class FragmentRepositionMutation(OffspringCreator):
    """Rigidly re-place one adsorbate fragment on a new core surface site.

    Core atoms (tag 0) define the convex-hull site pool; other adsorbate
    fragments remain fixed as clash obstacles. Preserves fragment internal
    geometry by using the current fragment pose (or an input template) as
    the rigid body.
    """

    def __init__(
        self,
        blmin: dict,
        n_top: int,
        system_type: SystemType,
        adsorbate_definition: AdsorbateDefinition,
        fragment_templates: AdsorbateFragmentInput | None = None,
        cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
        *,
        rng: Generator | None = None,
        verbose: bool = False,
    ) -> None:
        rng = _ensure_rng(rng)
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.system_type = system_type
        self._policy = get_system_policy(system_type)
        self.adsorbate_definition = adsorbate_definition
        self.fragment_templates = fragment_templates
        self.cluster_adsorbate_config = cluster_adsorbate_config
        self.descriptor = "FragmentRepositionMutation"
        self.min_inputs = 1

    def get_new_individual(self, parents):
        parent = parents[0]
        mutant = self.mutate(parent)
        if mutant is None:
            return mutant, "mutation: fragment_reposition"
        mutant = self.initialize_individual(parent, mutant)
        mutant.info["data"]["parents"] = [parent.info.get("confid")]
        return self.finalize_individual(mutant), "mutation: fragment_reposition"

    def _fragment_template_for_tag(
        self, mobile: Atoms, tag: int, frag_index: int
    ) -> Atoms:
        if self.fragment_templates is not None:
            fragments = resolve_adsorbate_fragments(
                self.fragment_templates,
                self.adsorbate_definition,
                context="FragmentRepositionMutation",
            )
            if 0 <= frag_index < len(fragments):
                return fragments[frag_index].copy()
        mask = mobile.get_tags() == tag
        return mobile[mask].copy()

    def mutate(self, atoms: Atoms) -> Atoms | None:
        n_top = int(self.n_top)
        slab = atoms[: len(atoms) - n_top]
        mobile = atoms[-n_top:].copy()
        tags = mobile.get_tags()
        ads_tags = sorted({int(t) for t in tags if int(t) > 0})
        if not ads_tags:
            return None

        lengths = parse_positive_fragment_lengths(
            self.adsorbate_definition.get("adsorbate_fragment_lengths", [])
        )
        target_tag = int(self.rng.choice(ads_tags))
        frag_index = target_tag - 1
        if lengths and not (0 <= frag_index < len(lengths)):
            return None

        core_mask = tags == 0
        ads_mask = tags == target_tag
        if not np.any(core_mask) or not np.any(ads_mask):
            return None

        metal_core = mobile[core_mask]
        clash_mobile = mobile[~ads_mask]

        ca = self.cluster_adsorbate_config or ClusterAdsorbateConfig()
        anchor, bond_axis = resolve_fragment_anchor_and_bond_axis(
            self.adsorbate_definition
        )

        fragment_tmpl = self._fragment_template_for_tag(mobile, target_tag, frag_index)
        placed = place_fragment_on_cluster(
            metal_core,
            fragment_tmpl,
            self.rng,
            ca,
            anchor_index=anchor,
            bond_axis=bond_axis,
            site_core=metal_core,
            clash_atoms=clash_mobile,
        )
        if placed is None:
            return None

        new_mobile = mobile.copy()
        new_mobile.positions[ads_mask] = placed.get_positions()
        if len(slab) > 0 and atoms_too_close_two_sets(new_mobile, slab, self.blmin):
            return None
        if not self._policy.uses_surface:
            new_mobile.center()
        return slab + new_mobile
