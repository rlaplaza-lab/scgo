"""Canonical system-type definitions and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from ase import Atoms

from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig
from scgo.cluster_adsorbate.feasibility import validate_adsorbate_placement_feasibility
from scgo.cluster_adsorbate.helpers import (
    parse_positive_fragment_lengths,
    resolve_fragment_anchor_and_bond_axis,  # noqa: F401
)
from scgo.cluster_adsorbate.validation import (
    validate_adsorbate_fragment_integrity,
    validate_combined_cluster_structure,
)
from scgo.initialization.geometry_helpers import _find_connected_components
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.validation import (
    validate_supported_cluster_deposit,
    validate_surface_config_slab_prefix,
)

SystemType = Literal[
    "gas_cluster",
    "surface_cluster",
    "gas_cluster_adsorbate",
    "surface_cluster_adsorbate",
]


class AdsorbateDefinition(TypedDict, total=False):
    """Role and layout for ``*_adsorbate`` system types (gas or surface mobile region).

    Both ``core_symbols`` and ``adsorbate_symbols`` must be set (use ``[]`` for
    the side that is empty). They must form an **ordered** partition of the run
    ``composition`` such that
    ``composition == core_symbols + adsorbate_symbols`` (list equality, same
    length and order for the mobile atoms). Element symbols may appear in both
    lists (e.g. oxide cores with O-containing adsorbates). The slab, if any, is *not* part of
    ``composition``.

    **Empty core** (``core_symbols=[]``): all mobile atoms are in
    ``adsorbate_symbols``.

    Build a core cluster, place rigid fragment(s) with
    :func:`scgo.cluster_adsorbate.place_fragment_on_cluster` (one site per fragment),
    then (for surface) deposit. Pass ``adsorbates`` as ``list[Atoms]`` with one
    entry per fragment at the runner API; ``adsorbate_fragment_lengths`` must match.
    """

    core_symbols: list[str]
    adsorbate_symbols: list[str]
    adsorbate_fragment_lengths: list[int]
    fragment_anchor_index: NotRequired[int]
    fragment_bond_axis: NotRequired[list[int]]


AdsorbatesInput = Atoms | list[Atoms]
AdsorbateFragmentInput = Atoms | list[Atoms]


@dataclass(frozen=True)
class SystemPolicy:
    """Behavior flags for a concrete system type."""

    system_type: SystemType
    uses_surface: bool
    has_adsorbate: bool
    requires_slab_prefix_validation: bool
    needs_supported_deposit_validation: bool
    neb_force_mic: bool
    neb_disable_alignment: bool
    neb_surface_cell_remap: bool
    neb_surface_lattice_rotation: bool
    constrain_adsorbate_moves: bool
    adsorbate_move_scale: float
    allow_composition_permutations: bool


SYSTEM_TYPE_POLICIES: dict[SystemType, SystemPolicy] = {
    "gas_cluster": SystemPolicy(
        system_type="gas_cluster",
        uses_surface=False,
        has_adsorbate=False,
        requires_slab_prefix_validation=False,
        needs_supported_deposit_validation=False,
        neb_force_mic=False,
        neb_disable_alignment=False,
        neb_surface_cell_remap=False,
        neb_surface_lattice_rotation=False,
        constrain_adsorbate_moves=False,
        adsorbate_move_scale=1.0,
        allow_composition_permutations=True,
    ),
    "surface_cluster": SystemPolicy(
        system_type="surface_cluster",
        uses_surface=True,
        has_adsorbate=False,
        requires_slab_prefix_validation=True,
        needs_supported_deposit_validation=True,
        neb_force_mic=True,
        neb_disable_alignment=False,
        neb_surface_cell_remap=True,
        neb_surface_lattice_rotation=True,
        constrain_adsorbate_moves=False,
        adsorbate_move_scale=1.0,
        allow_composition_permutations=True,
    ),
    "gas_cluster_adsorbate": SystemPolicy(
        system_type="gas_cluster_adsorbate",
        uses_surface=False,
        has_adsorbate=True,
        requires_slab_prefix_validation=False,
        needs_supported_deposit_validation=False,
        neb_force_mic=False,
        neb_disable_alignment=False,
        neb_surface_cell_remap=False,
        neb_surface_lattice_rotation=False,
        constrain_adsorbate_moves=True,
        adsorbate_move_scale=0.6,
        allow_composition_permutations=False,
    ),
    "surface_cluster_adsorbate": SystemPolicy(
        system_type="surface_cluster_adsorbate",
        uses_surface=True,
        has_adsorbate=True,
        requires_slab_prefix_validation=True,
        needs_supported_deposit_validation=True,
        neb_force_mic=True,
        neb_disable_alignment=False,
        neb_surface_cell_remap=True,
        neb_surface_lattice_rotation=True,
        constrain_adsorbate_moves=True,
        adsorbate_move_scale=0.6,
        allow_composition_permutations=False,
    ),
}


def get_system_policy(system_type: SystemType) -> SystemPolicy:
    """Return centralized behavior policy for one explicit system type."""
    return SYSTEM_TYPE_POLICIES[system_type]


def resolve_connectivity_factor(
    connectivity_factor: float | None,
    *,
    cluster_adsorbate_config: ClusterAdsorbateConfig | None = None,
    surface_config: SurfaceSystemConfig | None = None,
) -> float:
    """Resolve structure connectivity factor from explicit value or configs."""
    if connectivity_factor is not None:
        return float(connectivity_factor)
    if cluster_adsorbate_config is not None:
        return float(cluster_adsorbate_config.structure_connectivity_factor)
    if surface_config is not None:
        return float(surface_config.structure_connectivity_factor)
    return float(CONNECTIVITY_FACTOR)


def validate_system_type_settings(
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
) -> None:
    """Validate system-type companion settings."""
    surface_type = get_system_policy(system_type).uses_surface
    if surface_type and surface_config is None:
        raise ValueError(
            f"system_type={system_type!r} requires surface_config to be provided."
        )
    if not surface_type and surface_config is not None:
        raise ValueError(
            f"system_type={system_type!r} does not allow surface_config. "
            "Use surface_cluster or surface_cluster_adsorbate."
        )


def uses_surface(system_type: SystemType) -> bool:
    return get_system_policy(system_type).uses_surface


def validate_mobile_symbols_match_adsorbate_definition(
    atoms: Atoms,
    n_slab: int,
    adsorbate_definition: AdsorbateDefinition,
) -> None:
    """Ensure ``atoms`` mobile slice matches ``core_symbols + adsorbate_symbols`` in order."""
    cr = adsorbate_definition.get("core_symbols", [])
    ad = adsorbate_definition.get("adsorbate_symbols", [])
    if not isinstance(cr, list) or not isinstance(ad, list):
        raise ValueError(
            "adsorbate_definition['core_symbols'] and ['adsorbate_symbols'] must be lists."
        )

    core_list = [str(s) for s in cr]
    ads_list = [str(s) for s in ad]
    expected = core_list + ads_list

    n = len(atoms)
    if n_slab < 0 or n_slab > n:
        raise ValueError(
            f"Invalid n_slab={n_slab} for len(atoms)={n} in mobile symbol validation."
        )

    mobile = atoms.get_chemical_symbols()[n_slab:]
    if len(mobile) != len(expected):
        raise ValueError(
            f"Mobile region length mismatch: len(mobile)={len(mobile)} vs expected={len(expected)}"
        )

    if mobile != expected:

        def _head(syms: list[str], k: int = 12) -> str:
            h = syms[:k]
            return str(h) + ("..." if len(syms) > k else "")

        raise ValueError(
            f"Mobile symbols mismatch. Expected: {_head(expected)}; got: {_head(mobile)}."
        )


def _n_core_mobile_from_adsorbate_definition(
    adsorbate_definition: AdsorbateDefinition | None,
) -> int | None:
    """Return mobile core atom count from ``adsorbate_definition``, or ``None`` if absent."""
    if adsorbate_definition is None:
        return None
    cr = adsorbate_definition.get("core_symbols", [])
    if not isinstance(cr, list):
        return None
    return len(cr)


def _adsorbate_fragment_lengths_from_definition(
    adsorbate_definition: AdsorbateDefinition | None,
) -> list[int] | None:
    """Return adsorbate fragment lengths from definition, validating shape."""
    if adsorbate_definition is None:
        return None
    raw = adsorbate_definition.get("adsorbate_fragment_lengths")
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(isinstance(x, int) for x in raw):
        raise ValueError(
            "adsorbate_definition['adsorbate_fragment_lengths'] must be a list[int]."
        )
    if any(int(x) <= 0 for x in raw):
        raise ValueError(
            "adsorbate_definition['adsorbate_fragment_lengths'] must contain positive integers."
        )
    return [int(x) for x in raw]


def validate_structure_for_system_type(
    atoms: Atoms,
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    n_slab: int | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
    connectivity_factor: float | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
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
        connectivity_factor: Connectivity factor to use for cluster connectivity
            validation. If None, defaults to CONNECTIVITY_FACTOR from config.
        allow_cluster_fragmentation: For surface systems, allow multiple disconnected
            core-bearing mobile subgroups (each must touch the slab).
        allow_adsorbate_surface_detachment: For surface systems, allow adsorbate-only
            mobile subgroups on the slab without cluster contact (requires exactly
            one core/mixed subgroup when fragmentation is disabled).
        enforce_adsorbate_subgraph_integrity: When True, require each adsorbate
            fragment to remain internally connected.
    """
    policy = get_system_policy(system_type)
    cf = connectivity_factor if connectivity_factor is not None else CONNECTIVITY_FACTOR
    if policy.uses_surface:
        if surface_config is None:
            raise ValueError(
                "surface_config is required for surface system validation."
            )
        if policy.requires_slab_prefix_validation:
            validate_surface_config_slab_prefix(atoms, surface_config)
        if policy.needs_supported_deposit_validation:
            n_slab_eff = int(n_slab if n_slab is not None else len(surface_config.slab))
            ok, msg = validate_supported_cluster_deposit(
                atoms,
                n_slab_eff,
                surface_normal_axis=surface_config.surface_normal_axis,
                use_mic=bool(surface_config.comparator_use_mic),
                connectivity_factor=cf,
                n_core_mobile=_n_core_mobile_from_adsorbate_definition(
                    adsorbate_definition
                ),
                adsorbate_fragment_lengths=_adsorbate_fragment_lengths_from_definition(
                    adsorbate_definition
                ),
                allow_cluster_fragmentation=allow_cluster_fragmentation,
                allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
            )
            if not ok:
                raise ValueError(msg)
    elif policy.has_adsorbate:
        ok, msg = validate_combined_cluster_structure(atoms, connectivity_factor=cf)
        if not ok:
            raise ValueError(msg)

    if policy.has_adsorbate and adsorbate_definition is not None:
        if policy.uses_surface:
            if surface_config is None:
                raise ValueError(
                    "surface_config is required for surface adsorbate mobile symbol validation."
                )
            n_mobile_slab = int(
                n_slab if n_slab is not None else len(surface_config.slab)
            )
        else:
            n_mobile_slab = 0
        validate_mobile_symbols_match_adsorbate_definition(
            atoms, n_mobile_slab, adsorbate_definition
        )
        if enforce_adsorbate_subgraph_integrity:
            core_len = _n_core_mobile_from_adsorbate_definition(adsorbate_definition)
            if core_len is None:
                core_len = 0
            fragment_lengths = _adsorbate_fragment_lengths_from_definition(
                adsorbate_definition
            )
            if fragment_lengths is None:
                ads_list = adsorbate_definition.get("adsorbate_symbols", [])
                fragment_lengths = (
                    [len(ads_list)]
                    if isinstance(ads_list, list) and len(ads_list) > 0
                    else []
                )
            # Surface branch already applies this check in
            # validate_supported_cluster_deposit(...). Avoid duplicate errors.
            if not policy.uses_surface:
                use_mic = False
                ok, msg = validate_adsorbate_fragment_integrity(
                    atoms,
                    n_slab=n_mobile_slab,
                    n_core_mobile=core_len,
                    adsorbate_fragment_lengths=fragment_lengths,
                    connectivity_factor=cf,
                    use_mic=use_mic,
                )
                if not ok:
                    raise ValueError(msg)


def validate_composition_against_adsorbate(
    composition: list[str],
    adsorbate_definition: AdsorbateDefinition,
    *,
    context: str = "",
) -> tuple[list[str], list[str]]:
    """Check ordered partition and return ``(core_list, ads_list)`` as ``list[str]``.

    Both ``core_symbols`` and ``adsorbate_symbols`` must be present (use ``[]`` if
    empty). The run ``composition`` must equal ``core_symbols + adsorbate_symbols``
    in order. Element symbols may repeat across core and adsorbate (e.g. lattice
    O in an oxide core plus O in an OH adsorbate); partitioning is by atom index,
    not by chemical element.

    Raises:
        ValueError: If keys are missing, both sides are empty for non-empty
            composition, or the ordered partition does not match ``composition``.
    """
    prefix = f"{context}: " if context else ""

    cr = adsorbate_definition.get("core_symbols", [])
    ad = adsorbate_definition.get("adsorbate_symbols", [])
    if not isinstance(cr, list) or not isinstance(ad, list):
        raise ValueError(
            f"{prefix}adsorbate_definition['core_symbols'] and ['adsorbate_symbols'] must be lists."
        )

    core_list = [str(s) for s in cr]
    ads_list = [str(s) for s in ad]

    if not composition and not core_list and not ads_list:
        return core_list, ads_list
    if len(core_list) == 0 and len(ads_list) == 0:
        raise ValueError(
            f"{prefix}core_symbols and adsorbate_symbols cannot both be empty unless composition is also empty."
        )

    expected = core_list + ads_list
    if list(composition) != expected:
        raise ValueError(
            f"{prefix}composition must equal core_symbols + adsorbate_symbols. Got {composition}, expected {expected}."
        )

    return core_list, ads_list


def validate_adsorbate_definition(
    *,
    system_type: SystemType,
    composition: list[str],
    adsorbate_definition: AdsorbateDefinition | None,
    context: str,
) -> None:
    """Validate explicit adsorbate role definition for high-level runners."""
    policy = get_system_policy(system_type)
    if not policy.has_adsorbate:
        if adsorbate_definition is not None:
            raise ValueError(
                f"{context} received adsorbate_definition for non-adsorbate "
                f"system_type={system_type!r}."
            )
        return

    if adsorbate_definition is None:
        raise ValueError(
            f"{context} requires adsorbate_definition when system_type={system_type!r}."
        )

    core_list, _ads_list = validate_composition_against_adsorbate(
        composition, adsorbate_definition, context=context
    )

    fba = adsorbate_definition.get("fragment_bond_axis")
    if fba is not None and (
        not isinstance(fba, list)
        or len(fba) != 2
        or not all(isinstance(x, int) for x in fba)
    ):
        raise ValueError(
            f"adsorbate_definition['fragment_bond_axis'] must be a list of two int indices or omitted, got {fba!r}"
        )

    ai = adsorbate_definition.get("fragment_anchor_index")
    if ai is not None and not isinstance(ai, int):
        raise ValueError(
            f"adsorbate_definition['fragment_anchor_index'] must be int or omitted, got {ai!r}"
        )

    frag_lengths = adsorbate_definition.get("adsorbate_fragment_lengths")
    if frag_lengths is not None:
        if not isinstance(frag_lengths, list) or not all(
            isinstance(x, int) for x in frag_lengths
        ):
            raise ValueError(
                "adsorbate_definition['adsorbate_fragment_lengths'] must be a list of integers."
            )
        if any(int(x) <= 0 for x in frag_lengths):
            raise ValueError(
                "adsorbate_definition['adsorbate_fragment_lengths'] values must be positive."
            )
        expected_ads_len = len(composition) - len(core_list)
        if sum(int(x) for x in frag_lengths) != expected_ads_len:
            raise ValueError(
                "adsorbate_definition['adsorbate_fragment_lengths'] must sum to the adsorbate "
                f"length ({expected_ads_len}), got {sum(int(x) for x in frag_lengths)}."
            )


def resolve_adsorbate_fragments(
    templates: AdsorbateFragmentInput | None,
    adsorbate_definition: AdsorbateDefinition,
    *,
    context: str = "",
) -> list[Atoms]:
    """Normalize fragment templates and validate them against the adsorbate definition."""
    prefix = f"{context}: " if context else ""
    if templates is None:
        raise ValueError(f"{prefix}adsorbate fragment template(s) are required.")

    fragments = (
        [templates.copy()]
        if isinstance(templates, Atoms)
        else [frag.copy() for frag in templates]
    )
    if not fragments:
        raise ValueError(f"{prefix}adsorbate fragment template(s) must not be empty.")

    lengths = adsorbate_definition.get("adsorbate_fragment_lengths")
    if lengths is None:
        parsed_lengths: list[int] = []
    else:
        parsed_lengths = parse_positive_fragment_lengths(lengths)
    lengths = parsed_lengths
    raw_ads = adsorbate_definition.get("adsorbate_symbols", [])
    ads_symbols = [str(s) for s in raw_ads] if isinstance(raw_ads, list) else []
    if not lengths and ads_symbols:
        lengths = [len(ads_symbols)]

    if len(fragments) != len(lengths):
        if (
            len(fragments) == 1
            and len(fragments[0]) == sum(lengths)
            and len(lengths) > 1
        ):
            raise ValueError(
                f"{prefix}found one combined adsorbate template for "
                f"{len(lengths)} fragments. Pass adsorbates as list[Atoms] "
                "with one entry per fragment."
            )
        raise ValueError(
            f"{prefix}fragment template count ({len(fragments)}) must match "
            f"adsorbate_fragment_lengths ({len(lengths)})."
        )

    offset = 0
    for idx, (frag, frag_len) in enumerate(zip(fragments, lengths, strict=True)):
        if len(frag) != frag_len:
            raise ValueError(
                f"{prefix}adsorbate fragment {idx} has len={len(frag)}, "
                f"expected {frag_len}."
            )
        expected_symbols = ads_symbols[offset : offset + frag_len]
        if expected_symbols and list(frag.get_chemical_symbols()) != expected_symbols:
            raise ValueError(
                f"{prefix}adsorbate fragment {idx} symbols "
                f"{frag.get_chemical_symbols()!r} do not match expected "
                f"{expected_symbols!r}."
            )
        offset += frag_len
    return fragments


def normalize_adsorbates_input(
    adsorbates: AdsorbatesInput | None, *, context: str
) -> list[Atoms]:
    prefix = f"{context}: " if context else ""
    if adsorbates is None:
        raise ValueError(f"{prefix}adsorbates is required for adsorbate system types.")

    items = adsorbates if isinstance(adsorbates, list) else [adsorbates]
    out: list[Atoms] = []

    for idx, item in enumerate(items):
        if not isinstance(item, Atoms):
            raise TypeError(
                f"{prefix}adsorbates[{idx}] must be ase.Atoms, got {type(item).__name__}."
            )
        if len(item) == 0:
            raise ValueError(f"{prefix}adsorbates[{idx}] must not be empty.")
        out.append(item.copy())

    if not out:
        raise ValueError(f"{prefix}adsorbates must contain at least one fragment.")
    return out


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
            raise ValueError(
                f"{prefix}adsorbates[{idx}] is disconnected under "
                f"connectivity_factor={CONNECTIVITY_FACTOR}. Provide a connected "
                "initial adsorbate geometry."
            )


def flatten_adsorbate_symbols(adsorbates: list[Atoms]) -> list[str]:
    symbols: list[str] = []
    for frag in adsorbates:
        symbols.extend([str(s) for s in frag.get_chemical_symbols()])
    return symbols


def combine_adsorbates_to_template(adsorbates: list[Atoms]) -> Atoms:
    if not adsorbates:
        raise ValueError("adsorbates must contain at least one fragment")
    combined = adsorbates[0].copy()
    for frag in adsorbates[1:]:
        combined += frag.copy()
    return combined


def build_adsorbate_definition_from_inputs(
    *,
    system_type: SystemType,
    composition: list[str],
    adsorbates: AdsorbatesInput | None,
    context: str,
) -> tuple[AdsorbateDefinition | None, list[Atoms] | None, list[str]]:
    policy = get_system_policy(system_type)
    if not policy.has_adsorbate:
        if adsorbates is not None:
            raise ValueError(
                f"{context} does not accept adsorbates for system_type={system_type!r}."
            )
        return None, None, list(composition)
    core_list = [str(s) for s in composition]
    fragments = normalize_adsorbates_input(adsorbates, context=context)
    _validate_input_adsorbate_fragments_connected(fragments, context=context)
    ads_list = flatten_adsorbate_symbols(fragments)
    full_mobile_composition = list(core_list) + list(ads_list)
    ads_def: AdsorbateDefinition = {
        "core_symbols": core_list,
        "adsorbate_symbols": ads_list,
        "adsorbate_fragment_lengths": [len(frag) for frag in fragments],
    }
    validate_adsorbate_definition(
        system_type=system_type,
        composition=full_mobile_composition,
        adsorbate_definition=ads_def,
        context=context,
    )
    validate_adsorbate_placement_feasibility(
        core_list,
        ads_def["adsorbate_fragment_lengths"],
        fragments,
        context=context,
    )
    return ads_def, fragments, full_mobile_composition
