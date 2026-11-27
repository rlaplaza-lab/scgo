"""Shared setup for adsorbate-fragment GA mutations."""

from __future__ import annotations

from ase import Atoms
from numpy.random import Generator

from scgo.ase_ga_patches.mutations._common import MutationOperator
from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.system_types import (
    AdsorbateDefinition,
    AdsorbateFragmentInput,
    SystemType,
    resolve_adsorbate_fragments,
)


class _FragmentMutation(MutationOperator):
    """Creator setup and fragment-template lookup shared by fragment operators.

    ``allow_disconnected_mobile`` is a class flag. Detach sets it so the
    shared finalizer keeps an adsorbate-only child.
    """

    allow_disconnected_mobile = False

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
        surface_normal_axis: int = 2,
    ) -> None:
        self._init_operator(verbose, rng, system_type, n_top=n_top, blmin=blmin)
        self.adsorbate_definition = adsorbate_definition
        self.fragment_templates = fragment_templates
        self.cluster_adsorbate_config = cluster_adsorbate_config
        self.surface_normal_axis = surface_normal_axis

    def _fragment_template_for_tag(
        self, mobile: Atoms, tag: int, frag_index: int
    ) -> Atoms:
        if self.fragment_templates is not None:
            fragments = resolve_adsorbate_fragments(
                self.fragment_templates,
                self.adsorbate_definition,
                context=type(self).__name__,
            )
            if 0 <= frag_index < len(fragments):
                return fragments[frag_index].copy()
        mask = mobile.get_tags() == tag
        return mobile[mask].copy()
