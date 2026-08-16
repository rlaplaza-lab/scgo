"""Structural validation gates for system types."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms

from scgo.exceptions import SCGOValidationError
from scgo.initialization.geometry_helpers import (
    _find_connected_components,
    validate_cluster_structure,
)
from scgo.initialization.initialization_config import (
    CONNECTIVITY_FACTOR,
    MIN_DISTANCE_FACTOR_DEFAULT,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types.composition import AdsorbateDefinition
from scgo.system_types.connectivity_factor import (
    ConnectivityFactorInput,
    NormalizedConnectivityFactor,
    format_connectivity_factor,
    normalize_connectivity_factor,
)
from scgo.system_types.policy import (
    SystemType,
    get_system_policy,
    resolve_connectivity_factor,
    resolve_structure_mic,
)

if TYPE_CHECKING:
    from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig


def _not_connected_message(n_components: int) -> str:
    return (
        "Structure is not connected "
        f"(found {n_components} connected components; enable "
        "allow_cluster_fragmentation and/or allow_adsorbate_surface_detachment "
        "to permit splits)"
    )


def _n_slab_for_mobile(
    *,
    system_type: SystemType,
    n_slab: int | None,
    surface_config: SurfaceSystemConfig | None,
) -> int:
    """Resolve slab prefix length for mobile-region checks."""
    if n_slab is not None:
        return int(n_slab)
    if surface_config is not None and get_system_policy(system_type).uses_surface:
        return len(surface_config.slab)
    return 0


def validate_connectivity_policy(
    atoms: Atoms,
    *,
    uses_surface: bool,
    n_slab: int,
    n_core_mobile: int,
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
    use_mic: bool,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    surface_normal_axis: int = 2,
    slab_stacking_cutoff_a: float | None = None,
) -> tuple[bool, str]:
    """Single system-dependent connectivity gate (connected components + slab contact).

    Surface systems operate on the mobile slice ``atoms[n_slab:]`` and require every
    connected subgroup (classified as ``core`` / ``mixed`` / ``ads_only`` per
    ``n_core_mobile``) to touch the slab, honoring the fragmentation / detachment
    toggles. When ``slab_stacking_cutoff_a`` is set (layered-crystal templates),
    nearest-neighbor stacking against the frozen top layer also counts as contact.
    Non-surface systems operate on the whole structure and require exactly
    one connected component (core and adsorbate must be bonded). The
    ``allow_cluster_fragmentation`` / ``allow_adsorbate_surface_detachment`` toggles
    are surface-only and ignored on the non-surface path: a non-surface structure is
    always required to form a single connected component.

    Connected components are computed exactly once.
    """
    if uses_surface:
        return _validate_surface_connectivity_policy(
            atoms,
            n_slab=n_slab,
            n_core_mobile=n_core_mobile,
            connectivity_factor=connectivity_factor,
            use_mic=use_mic,
            allow_cluster_fragmentation=allow_cluster_fragmentation,
            allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
            surface_normal_axis=surface_normal_axis,
            slab_stacking_cutoff_a=slab_stacking_cutoff_a,
        )
    return _validate_gas_connectivity_policy(atoms, connectivity_factor, use_mic)


def _validate_gas_connectivity_policy(
    atoms: Atoms,
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
    use_mic: bool,
) -> tuple[bool, str]:
    """Non-surface path: the whole structure must be one connected component."""
    components, _ = _find_connected_components(atoms, connectivity_factor, use_mic)
    if len(components) != 1:
        return False, _not_connected_message(len(components))
    return True, ""


def _validate_surface_connectivity_policy(
    atoms: Atoms,
    *,
    n_slab: int,
    n_core_mobile: int,
    connectivity_factor: ConnectivityFactorInput | NormalizedConnectivityFactor,
    use_mic: bool,
    allow_cluster_fragmentation: bool,
    allow_adsorbate_surface_detachment: bool,
    surface_normal_axis: int,
    slab_stacking_cutoff_a: float | None = None,
) -> tuple[bool, str]:
    """Surface path: connected subgroups, each touching the slab.

    The slab-contact helpers live in ``scgo.surface.validation``; keep this
    import inside the function. Hoisting it while that module top-imports
    ``validate_connectivity_policy`` from this package would recreate a cycle.
    """
    from scgo.surface.validation import (
        _adsorbate_subgroup_touches_slab,
        _check_mobile_touches_slab,
        _classify_mobile_component,
    )

    mobile = atoms[n_slab:]
    n_ads = len(mobile)
    if n_ads < 2:
        return _check_mobile_touches_slab(
            atoms,
            n_slab,
            connectivity_factor=connectivity_factor,
            use_mic=use_mic,
            surface_normal_axis=surface_normal_axis,
            stacking_cutoff_a=slab_stacking_cutoff_a,
        )

    components, _ = _find_connected_components(mobile, connectivity_factor, use_mic)
    subgroups = list(components.values())
    allow_split = allow_cluster_fragmentation or allow_adsorbate_surface_detachment

    if not allow_split:
        if len(subgroups) != 1:
            return False, _not_connected_message(len(subgroups))
        return _check_mobile_touches_slab(
            atoms,
            n_slab,
            connectivity_factor=connectivity_factor,
            use_mic=use_mic,
            surface_normal_axis=surface_normal_axis,
            stacking_cutoff_a=slab_stacking_cutoff_a,
        )

    core_like: list[list[int]] = []
    ads_only: list[list[int]] = []
    for subgroup in subgroups:
        kind = _classify_mobile_component(subgroup, n_core_mobile)
        if kind == "ads_only":
            ads_only.append(subgroup)
        else:
            core_like.append(subgroup)

    if allow_cluster_fragmentation and not allow_adsorbate_surface_detachment:
        if ads_only:
            return (
                False,
                "Detached adsorbate-only mobile subgroups are not allowed "
                "(set allow_adsorbate_surface_detachment=True to permit adsorbates "
                "on the slab without cluster contact)",
            )
    elif (
        allow_adsorbate_surface_detachment
        and not allow_cluster_fragmentation
        and len(core_like) != 1
    ):
        return (
            False,
            "Exactly one core-connected mobile component is required when "
            f"allow_cluster_fragmentation=False (found {len(core_like)} "
            "core/mixed components)",
        )

    for subgroup in subgroups:
        if not _adsorbate_subgroup_touches_slab(
            atoms,
            n_slab,
            subgroup,
            connectivity_factor=connectivity_factor,
            use_mic=use_mic,
            surface_normal_axis=surface_normal_axis,
            stacking_cutoff_a=slab_stacking_cutoff_a,
        ):
            return (
                False,
                "Every mobile subgroup must touch the slab "
                f"(subgroup size {len(subgroup)} has no slab contact within "
                f"connectivity_factor={format_connectivity_factor(normalize_connectivity_factor(connectivity_factor))})",
            )
    return True, ""


def validate_mobile_symbols_match_adsorbate_definition(
    atoms: Atoms,
    n_slab: int,
    adsorbate_definition: AdsorbateDefinition,
) -> None:
    """Ensure ``atoms`` mobile slice matches ``core_symbols + adsorbate_symbols`` in order."""
    core_list = [str(s) for s in adsorbate_definition.core_symbols]
    ads_list = [str(s) for s in adsorbate_definition.adsorbate_symbols]
    expected = core_list + ads_list

    n = len(atoms)
    if n_slab < 0 or n_slab > n:
        raise SCGOValidationError(
            f"Invalid n_slab={n_slab} for len(atoms)={n} in mobile symbol validation."
        )

    mobile = atoms.get_chemical_symbols()[n_slab:]
    if len(mobile) != len(expected):
        raise SCGOValidationError(
            f"Mobile region length mismatch: len(mobile)={len(mobile)} vs expected={len(expected)}"
        )

    if mobile != expected:
        head_limit = 12
        expected_head = str(expected[:head_limit]) + (
            "..." if len(expected) > head_limit else ""
        )
        mobile_head = str(mobile[:head_limit]) + (
            "..." if len(mobile) > head_limit else ""
        )
        raise SCGOValidationError(
            f"Mobile symbols mismatch. Expected: {expected_head}; got: {mobile_head}."
        )


def validate_structure_for_system_type(
    atoms: Atoms,
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    n_slab: int | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor
    | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
    binding_penetration_tolerance_a: float = 0.1,
    n_slab_deposit: int | None = None,
) -> None:
    """Apply system-type-specific structural validation.

    When ``adsorbate_definition`` is set for a ``*_adsorbate`` system type, the
    mobile region must match ``core_symbols + adsorbate_symbols`` in order (after
    the slab prefix for surface systems).

    Args:
        atoms: The Atoms object to validate
        system_type: The system type
        surface_config: Surface configuration (for surface systems)
        n_slab: Number of slab atoms (for surface systems)
        adsorbate_definition: Adsorbate definition (for adsorbate systems)
        connectivity_factor: Explicit connectivity factor (float or per-element/pair
            dict), used verbatim when not None. Otherwise resolved from
            ``cluster_adsorbate_config`` then ``surface_config`` then the module
            default (precedence matches :func:`resolve_connectivity_factor`).
        cluster_adsorbate_config: Per-config connectivity factor source. When
            ``connectivity_factor`` is None, its ``structure_connectivity_factor``
            is honored (overriding ``surface_config``).
        allow_cluster_fragmentation: For surface systems, allow multiple disconnected
            core-bearing mobile subgroups (each must touch the slab).
        allow_adsorbate_surface_detachment: For surface systems, allow adsorbate-only
            mobile subgroups on the slab without cluster contact (requires exactly
            one core/mixed subgroup when fragmentation is disabled).
        enforce_adsorbate_subgraph_integrity: When True, require each adsorbate
            fragment to remain internally connected.
        n_slab_deposit: Frozen-prefix length for slab-as-search-target deposit
            and connectivity checks. Tag partition and mobile-symbol matching
            still use ``n_slab`` (the full slab). When the search-mobile region
            includes original top-layer atoms, slab contact follows the
            template: covalent if the template layers covalently bind, otherwise
            van der Waals stacking derived from the template interlayer.
    """
    # Lazy on purpose: surface.validation top-imports validate_connectivity_policy
    # from this package; hoisting both sides would recreate a two-module cycle.
    from scgo.cluster_adsorbate.validation import (
        validate_adsorbate_fragment_integrity,
    )
    from scgo.surface.validation import (
        layer_stacking_cutoff_from_template,
        validate_supported_cluster_deposit,
        validate_surface_config_slab_prefix,
    )

    policy = get_system_policy(system_type)
    cf = resolve_connectivity_factor(
        connectivity_factor,
        cluster_adsorbate_config=cluster_adsorbate_config,
        surface_config=surface_config,
    )
    if policy.uses_surface:
        if surface_config is None:
            raise SCGOValidationError(
                "surface_config is required for surface system validation."
            )
        if policy.requires_slab_prefix_validation:
            validate_surface_config_slab_prefix(atoms, surface_config)
        if policy.needs_supported_deposit_validation:
            # For slab-as-search-target the mobile search region is
            # [n_fixed : total] (mobile top layers + adsorbate), so the deposit
            # boundary must be the fixed prefix, NOT the total slab. The total
            # slab is still the correct boundary for the slab-prefix structural
            # and tag-partition checks (handled elsewhere).
            n_slab_full = int(
                n_slab if n_slab is not None else len(surface_config.slab)
            )
            if policy.slab_is_search_target and n_slab_deposit is not None:
                n_slab_eff = int(n_slab_deposit)
                n_mobile_slab_search = max(0, n_slab_full - n_slab_eff)
            else:
                n_slab_eff = n_slab_full
                n_mobile_slab_search = 0
            use_mic = resolve_structure_mic(system_type, surface_config)
            stacking_cutoff = None
            if n_mobile_slab_search > 0:
                stacking_cutoff = layer_stacking_cutoff_from_template(
                    surface_config.slab,
                    n_slab_eff,
                    surface_normal_axis=surface_config.surface_normal_axis,
                    connectivity_factor=cf,
                    use_mic=use_mic,
                )
            if adsorbate_definition is not None:
                n_core_for_deposit = n_mobile_slab_search + int(
                    adsorbate_definition.n_core
                )
            else:
                n_core_for_deposit = None
            ok, msg = validate_supported_cluster_deposit(
                atoms,
                n_slab_eff,
                surface_normal_axis=surface_config.surface_normal_axis,
                use_mic=use_mic,
                connectivity_factor=cf,
                n_core_mobile=n_core_for_deposit,
                adsorbate_fragment_lengths=(
                    list(adsorbate_definition.adsorbate_fragment_lengths)
                    if adsorbate_definition is not None
                    else None
                ),
                allow_cluster_fragmentation=allow_cluster_fragmentation,
                allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
                binding_penetration_tolerance_a=binding_penetration_tolerance_a,
                slab_stacking_cutoff_a=stacking_cutoff,
            )
            if not ok:
                raise SCGOValidationError(msg)
    else:
        # Non-surface (gas_cluster_adsorbate or bare gas_cluster). The clash check
        # is a sibling that preserves today's rejection of clashing minima; the
        # unified connectivity policy then requires a single connected component
        # across the whole region (core + adsorbate must be bonded).
        ok, msg = validate_cluster_structure(
            atoms,
            MIN_DISTANCE_FACTOR_DEFAULT,
            cf,
            check_clashes=True,
            check_connectivity=False,
            use_mic=False,
        )
        if not ok:
            raise SCGOValidationError(msg)
        ok, msg = validate_connectivity_policy(
            atoms,
            uses_surface=False,
            n_slab=0,
            n_core_mobile=len(atoms),
            connectivity_factor=cf,
            use_mic=False,
            allow_cluster_fragmentation=False,
            allow_adsorbate_surface_detachment=False,
        )
        if not ok:
            raise SCGOValidationError(msg)

    if policy.has_adsorbate and adsorbate_definition is not None:
        if policy.uses_surface:
            if surface_config is None:
                raise SCGOValidationError(
                    "surface_config is required for surface adsorbate mobile symbol validation."
                )
            n_mobile_slab = _n_slab_for_mobile(
                system_type=system_type,
                n_slab=n_slab,
                surface_config=surface_config,
            )
        else:
            n_mobile_slab = 0
        validate_mobile_symbols_match_adsorbate_definition(
            atoms, n_mobile_slab, adsorbate_definition
        )
        # Surface branch already applies fragment integrity in
        # validate_supported_cluster_deposit(...). Avoid duplicate errors.
        if enforce_adsorbate_subgraph_integrity and not policy.uses_surface:
            ok, msg = validate_adsorbate_fragment_integrity(
                atoms,
                n_slab=n_mobile_slab,
                n_core_mobile=adsorbate_definition.n_core,
                adsorbate_fragment_lengths=list(
                    adsorbate_definition.effective_fragment_lengths
                ),
                connectivity_factor=cf,
                use_mic=False,
            )
            if not ok:
                raise SCGOValidationError(msg)


def _validate_adsorbate_tag_partition(
    atoms: Atoms,
    n_slab: int,
    adsorbate_definition: AdsorbateDefinition,
) -> None:
    """Assert ``atoms.get_tags()`` matches the core/adsorbate fragment partition.

    The mobile region (``atoms[n_slab:]``) must be partitioned exactly as
    ``adsorbate_definition`` implies: the first ``len(core_symbols)`` atoms are
    one fragment (sharing a single tag), and each subsequent adsorbate fragment
    (of length ``adsorbate_fragment_lengths[k]``, or the whole adsorbate when
    lengths are absent) is one distinct fragment sharing a single, unique tag.
    Only the grouping/index alignment is checked, not the absolute tag values,
    so the check is robust to whatever slab/adsorbate tag convention the
    upstream builder used.
    """
    n_core = adsorbate_definition.n_core
    frag_lengths = list(adsorbate_definition.effective_fragment_lengths)
    n_expected = n_slab + n_core + sum(frag_lengths)
    if len(atoms) != n_expected:
        raise SCGOValidationError(
            f"Adsorbate tag partition mismatch: structure has {len(atoms)} atoms "
            f"but core+adsorbate definition implies {n_expected} "
            f"(n_slab={n_slab}, n_core={n_core}, fragments={frag_lengths})."
        )
    mobile_tags = np.asarray(atoms.get_tags(), dtype=int)[n_slab:]
    offset = 0
    seen: set[int] = set()
    if n_core > 0:
        core_block = mobile_tags[:n_core]
        if int(np.unique(core_block).size) != 1:
            raise SCGOValidationError(
                "Adsorbate tag partition mismatch: core atoms must share a single tag."
            )
        seen.add(int(core_block[0]))
        offset = n_core
    for frag_idx, frag_len in enumerate(frag_lengths):
        frag = mobile_tags[offset : offset + int(frag_len)]
        if int(np.unique(frag).size) != 1:
            raise SCGOValidationError(
                f"Adsorbate tag partition mismatch: adsorbate fragment {frag_idx} "
                "atoms must share a single tag."
            )
        ftag = int(frag[0])
        if ftag in seen:
            raise SCGOValidationError(
                "Adsorbate tag partition mismatch: adsorbate fragment tag collides "
                f"with another fragment or the core (tag {ftag})."
            )
        seen.add(ftag)
        offset += int(frag_len)


def validate_minimum_structure(
    atoms: Atoms,
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    n_slab: int | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
    connectivity_factor: ConnectivityFactorInput
    | NormalizedConnectivityFactor
    | None = None,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
    binding_penetration_tolerance_a: float = 0.1,
    n_slab_deposit: int | None = None,
) -> None:
    """Shared structural gate for every global-optimization minimum.

    Thin wrapper around :func:`validate_structure_for_system_type` (same rule set
    for GO/TS callers) that also checks adsorbate tag partition when an
    adsorbate definition is present. Connectivity-factor precedence matches
    :func:`resolve_connectivity_factor`.

    Raises:
        SCGOValidationError: If the structure violates its system-type rules.
    """
    validate_structure_for_system_type(
        atoms,
        system_type=system_type,
        surface_config=surface_config,
        n_slab=n_slab,
        adsorbate_definition=adsorbate_definition,
        connectivity_factor=connectivity_factor,
        cluster_adsorbate_config=cluster_adsorbate_config,
        allow_cluster_fragmentation=allow_cluster_fragmentation,
        allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
        enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
        n_slab_deposit=n_slab_deposit,
    )

    if (
        get_system_policy(system_type).has_adsorbate
        and adsorbate_definition is not None
    ):
        n_slab_eff = _n_slab_for_mobile(
            system_type=system_type,
            n_slab=n_slab,
            surface_config=surface_config,
        )
        _validate_adsorbate_tag_partition(atoms, n_slab_eff, adsorbate_definition)


def _validate_input_adsorbate_fragments_connected(
    adsorbates: list[Atoms], *, context: str
) -> None:
    """Ensure each provided input adsorbate fragment is internally connected."""
    prefix = f"{context}: " if context else ""
    for idx, frag in enumerate(adsorbates):
        if len(frag) <= 1:
            continue
        components, _ = _find_connected_components(
            frag,
            connectivity_factor=CONNECTIVITY_FACTOR,
            use_mic=False,
        )
        if len(components) > 1:
            raise SCGOValidationError(
                f"{prefix}adsorbates[{idx}] is disconnected under "
                f"connectivity_factor={CONNECTIVITY_FACTOR}. Provide a connected "
                "initial adsorbate geometry."
            )
