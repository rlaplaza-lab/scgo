"""Canonical system-type definitions and validation helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

from ase import Atoms
from ase.calculators.calculator import Calculator

from scgo.exceptions import SCGOValidationError
from scgo.initialization.geometry_helpers import (
    _find_connected_components,
)
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.validation import (
    validate_supported_cluster_deposit,
    validate_surface_config_slab_prefix,
)
from scgo.utils.helpers import get_composition_counts

if TYPE_CHECKING:
    from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig

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


# Type aliases for calculator types
CalculatorType = type[Calculator]
CalculatorInstance = Calculator | None


class CalculatorKwargs(TypedDict, total=False):
    """Calculator kwargs in ``params['calculator_kwargs']``."""

    model_name: str
    device: str
    dtype: str


class OptimizerSlotParams(TypedDict, total=False):
    """Parameters for one ``optimizer_params`` slot (``simple``, ``bh``, or ``ga``)."""

    optimizer: str
    fmax: float
    niter: int | str
    niter_local_relaxation: int | str
    system_type: SystemType
    surface_config: NotRequired[SurfaceSystemConfig]
    temperature: NotRequired[float]
    dr: NotRequired[float]
    move_fraction: NotRequired[float]
    move_strategy: NotRequired[str]
    deduplicate: NotRequired[bool]
    population_size: NotRequired[int | str]
    mutation_probability: NotRequired[float]
    offspring_fraction: NotRequired[float]
    vacuum: NotRequired[float]
    use_adaptive_mutations: NotRequired[bool]
    stagnation_trigger: NotRequired[int]
    stagnation_full_trigger: NotRequired[int]
    recovery_window: NotRequired[int]
    aggressive_burst_multiplier: NotRequired[float]
    max_mutation_probability: NotRequired[float]
    early_stopping_niter: NotRequired[int]
    n_jobs_population_init: NotRequired[int]
    n_jobs_offspring: NotRequired[int]
    batch_size: NotRequired[int | None]
    relaxer: NotRequired[Any]
    energy_tolerance: NotRequired[float]
    comparator_tol: NotRequired[float]
    comparator_pair_cor_max: NotRequired[float]
    comparator_n_top: NotRequired[int | None]
    fitness_strategy: NotRequired[str | None]
    diversity_reference_db: NotRequired[str | None]
    diversity_max_references: NotRequired[int]
    diversity_update_interval: NotRequired[int]


class GLOptimizerParams(TypedDict, total=False):
    """Top-level GO ``params`` / ``go_params`` dict."""

    calculator: str
    calculator_kwargs: CalculatorKwargs
    surface_config: NotRequired[SurfaceSystemConfig]
    validate_with_hessian: bool
    fmax_threshold: float
    check_hessian: bool
    imag_freq_threshold: float
    optimizer_params: dict[str, OptimizerSlotParams]
    fitness_strategy: str
    diversity_reference_db: NotRequired[str | None]
    diversity_max_references: NotRequired[int]
    diversity_update_interval: NotRequired[int]
    adsorbate_definition: NotRequired[AdsorbateDefinition]
    adsorbate_fragment_template: NotRequired[Atoms | list[Atoms]]
    cluster_adsorbate_config: NotRequired[Any]
    connectivity_factor: float
    allow_cluster_fragmentation: bool
    allow_adsorbate_surface_detachment: bool
    enforce_adsorbate_subgraph_integrity: bool
    freeze_adsorbate_internal_geometry: bool
    seed: NotRequired[int | None]
    tag_final_minima: bool


class TSParams(TypedDict, total=False):
    """Top-level TS ``ts_params`` dict."""

    neb_align_endpoints: bool
    neb_interpolation_mic: bool
    neb_surface_cell_remap: bool
    neb_surface_lattice_rotation: bool
    neb_surface_max_lattice_shift: int
    neb_n_images: int
    neb_spring_constant: float
    neb_fmax: float
    neb_steps: int | str
    neb_climb: bool
    neb_perturb_sigma: float
    neb_interpolation_method: str
    neb_tangent_method: str
    torchsim_fmax: float
    torchsim_max_steps: int | str
    calculator: NotRequired[str]
    calculator_kwargs: NotRequired[CalculatorKwargs]
    surface_config: NotRequired[SurfaceSystemConfig]
    system_type: NotRequired[SystemType]


class SystemConfig(TypedDict, total=False):
    """System type plus optional surface and adsorbate settings."""

    system_type: SystemType
    surface_config: NotRequired[SurfaceSystemConfig]
    adsorbate_definition: NotRequired[AdsorbateDefinition]


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
        raise SCGOValidationError(
            f"system_type={system_type!r} requires surface_config to be provided."
        )
    if not surface_type and surface_config is not None:
        raise SCGOValidationError(
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
        raise SCGOValidationError(
            "adsorbate_definition['core_symbols'] and ['adsorbate_symbols'] must be lists."
        )

    core_list = [str(s) for s in cr]
    ads_list = [str(s) for s in ad]
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

        def _head(syms: list[str], k: int = 12) -> str:
            h = syms[:k]
            return str(h) + ("..." if len(syms) > k else "")

        raise SCGOValidationError(
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
        raise SCGOValidationError(
            "adsorbate_definition['adsorbate_fragment_lengths'] must be a list[int]."
        )
    if any(int(x) <= 0 for x in raw):
        raise SCGOValidationError(
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
    from scgo.cluster_adsorbate.validation import (
        validate_adsorbate_fragment_integrity,
        validate_combined_cluster_structure,
    )

    policy = get_system_policy(system_type)
    cf = connectivity_factor if connectivity_factor is not None else CONNECTIVITY_FACTOR
    if policy.uses_surface:
        if surface_config is None:
            raise SCGOValidationError(
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
                raise SCGOValidationError(msg)
    elif policy.has_adsorbate:
        ok, msg = validate_combined_cluster_structure(atoms, connectivity_factor=cf)
        if not ok:
            raise SCGOValidationError(msg)

    if policy.has_adsorbate and adsorbate_definition is not None:
        if policy.uses_surface:
            if surface_config is None:
                raise SCGOValidationError(
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
                    raise SCGOValidationError(msg)


def _core_and_ads_lists(
    adsorbate_definition: AdsorbateDefinition,
    *,
    context: str = "",
) -> tuple[list[str], list[str]]:
    prefix = f"{context}: " if context else ""
    cr = adsorbate_definition.get("core_symbols", [])
    ad = adsorbate_definition.get("adsorbate_symbols", [])
    if not isinstance(cr, list) or not isinstance(ad, list):
        raise SCGOValidationError(
            f"{prefix}adsorbate_definition['core_symbols'] and ['adsorbate_symbols'] must be lists."
        )
    return [str(s) for s in cr], [str(s) for s in ad]


def _strip_adsorbate_symbols(
    composition: list[str],
    adsorbate_symbols: list[str],
) -> list[str] | None:
    """Return ``composition`` minus one of each adsorbate symbol, or ``None``."""
    remaining = list(composition)
    for symbol in adsorbate_symbols:
        try:
            remaining.remove(symbol)
        except ValueError:
            return None
    return remaining


def resolve_mobile_composition(
    composition: list[str],
    adsorbate_definition: AdsorbateDefinition,
    *,
    context: str = "",
) -> list[str]:
    """Return ``core_symbols + adsorbate_symbols`` for a matching composition.

    Updates ``adsorbate_definition['core_symbols']`` in place when a full mobile
    formula is reconciled by stripping known ``adsorbate_symbols``.
    """
    prefix = f"{context}: " if context else ""
    core_list, ads_list = _core_and_ads_lists(adsorbate_definition, context=context)
    expected = core_list + ads_list
    comp = [str(s) for s in composition]

    if comp == expected:
        return expected

    comp_counts = get_composition_counts(comp)
    exp_counts = get_composition_counts(expected)
    if comp_counts == exp_counts or (
        ads_list and comp_counts == get_composition_counts(core_list)
    ):
        return expected

    if ads_list:
        derived_core = _strip_adsorbate_symbols(comp, ads_list)
        if derived_core is not None:
            adsorbate_definition["core_symbols"] = derived_core
            return derived_core + ads_list

    raise SCGOValidationError(
        f"{prefix}composition must match core_symbols + adsorbate_symbols: "
        f"got counts {dict(comp_counts)}, expected {dict(exp_counts)}."
    )


def extract_adsorbate_definition_from_params(
    params: dict[str, Any] | None,
) -> AdsorbateDefinition | None:
    if not params:
        return None
    top = params.get("adsorbate_definition")
    if isinstance(top, dict):
        return top  # type: ignore[return-value]
    for slot in (params.get("optimizer_params") or {}).values():
        if isinstance(slot, dict):
            ex = slot.get("adsorbate_definition")
            if isinstance(ex, dict):
                return ex  # type: ignore[return-value]
    return None


def resolve_adsorbate_run_composition(
    *,
    system_type: SystemType,
    composition: list[str],
    adsorbates: AdsorbatesInput | None,
    preset_adsorbate_definition: AdsorbateDefinition | None,
    context: str,
) -> tuple[AdsorbateDefinition | None, list[Atoms] | None, list[str]]:
    """Build or reconcile mobile composition for adsorbate runs (gas or surface).

    Uses explicit ``adsorbates`` when provided; otherwise reconciles ``composition``
    against a preset ``adsorbate_definition`` from params.
    """
    policy = get_system_policy(system_type)
    comp = [str(s) for s in composition]

    if not policy.has_adsorbate:
        if adsorbates is not None:
            raise SCGOValidationError(
                f"{context} does not accept adsorbates for system_type={system_type!r}."
            )
        if preset_adsorbate_definition is not None:
            raise SCGOValidationError(
                f"{context} does not accept adsorbate_definition for "
                f"system_type={system_type!r}."
            )
        return None, None, comp

    if adsorbates is not None:
        return build_adsorbate_definition_from_inputs(
            system_type=system_type,
            composition=comp,
            adsorbates=adsorbates,
            context=context,
        )

    if preset_adsorbate_definition is not None:
        ads_def = copy.deepcopy(preset_adsorbate_definition)
        full_comp = resolve_mobile_composition(comp, ads_def, context=context)
        validate_adsorbate_definition(
            system_type=system_type,
            composition=full_comp,
            adsorbate_definition=ads_def,
            context=context,
        )
        return ads_def, None, full_comp

    return build_adsorbate_definition_from_inputs(
        system_type=system_type,
        composition=comp,
        adsorbates=None,
        context=context,
    )


def validate_composition_against_adsorbate(
    composition: list[str],
    adsorbate_definition: AdsorbateDefinition,
    *,
    context: str = "",
) -> tuple[list[str], list[str]]:
    """Check composition against the core/adsorbate partition; return both lists.

    ``composition`` may match ``core_symbols + adsorbate_symbols`` exactly, share
    the same element counts in a different order, list only ``core_symbols``, or
    be a full mobile formula from which ``adsorbate_symbols`` are stripped.
    """
    prefix = f"{context}: " if context else ""
    core_list, ads_list = _core_and_ads_lists(adsorbate_definition, context=context)

    if not composition and not core_list and not ads_list:
        return core_list, ads_list
    if len(core_list) == 0 and len(ads_list) == 0:
        raise SCGOValidationError(
            f"{prefix}core_symbols and adsorbate_symbols cannot both be empty unless composition is also empty."
        )

    resolve_mobile_composition(list(composition), adsorbate_definition, context=context)
    return _core_and_ads_lists(adsorbate_definition, context=context)


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
            raise SCGOValidationError(
                f"{context} received adsorbate_definition for non-adsorbate "
                f"system_type={system_type!r}."
            )
        return

    if adsorbate_definition is None:
        raise SCGOValidationError(
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
        raise SCGOValidationError(
            f"adsorbate_definition['fragment_bond_axis'] must be a list of two int indices or omitted, got {fba!r}"
        )

    ai = adsorbate_definition.get("fragment_anchor_index")
    if ai is not None and not isinstance(ai, int):
        raise SCGOValidationError(
            f"adsorbate_definition['fragment_anchor_index'] must be int or omitted, got {ai!r}"
        )

    frag_lengths = adsorbate_definition.get("adsorbate_fragment_lengths")
    if frag_lengths is not None:
        if not isinstance(frag_lengths, list) or not all(
            isinstance(x, int) for x in frag_lengths
        ):
            raise SCGOValidationError(
                "adsorbate_definition['adsorbate_fragment_lengths'] must be a list of integers."
            )
        if any(int(x) <= 0 for x in frag_lengths):
            raise SCGOValidationError(
                "adsorbate_definition['adsorbate_fragment_lengths'] values must be positive."
            )
        expected_ads_len = len(composition) - len(core_list)
        if sum(int(x) for x in frag_lengths) != expected_ads_len:
            raise SCGOValidationError(
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
    from scgo.cluster_adsorbate.helpers import parse_positive_fragment_lengths

    prefix = f"{context}: " if context else ""
    if templates is None:
        raise SCGOValidationError(
            f"{prefix}adsorbate fragment template(s) are required."
        )

    fragments = (
        [templates.copy()]
        if isinstance(templates, Atoms)
        else [frag.copy() for frag in templates]
    )
    if not fragments:
        raise SCGOValidationError(
            f"{prefix}adsorbate fragment template(s) must not be empty."
        )

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
            raise SCGOValidationError(
                f"{prefix}found one combined adsorbate template for "
                f"{len(lengths)} fragments. Pass adsorbates as list[Atoms] "
                "with one entry per fragment."
            )
        raise SCGOValidationError(
            f"{prefix}fragment template count ({len(fragments)}) must match "
            f"adsorbate_fragment_lengths ({len(lengths)})."
        )

    offset = 0
    for idx, (frag, frag_len) in enumerate(zip(fragments, lengths, strict=True)):
        if len(frag) != frag_len:
            raise SCGOValidationError(
                f"{prefix}adsorbate fragment {idx} has len={len(frag)}, "
                f"expected {frag_len}."
            )
        expected_symbols = ads_symbols[offset : offset + frag_len]
        if expected_symbols and list(frag.get_chemical_symbols()) != expected_symbols:
            raise SCGOValidationError(
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
        raise SCGOValidationError(
            f"{prefix}adsorbates is required for adsorbate system types."
        )

    items = adsorbates if isinstance(adsorbates, list) else [adsorbates]
    out: list[Atoms] = []

    for idx, item in enumerate(items):
        if not isinstance(item, Atoms):
            raise SCGOValidationError(
                f"{prefix}adsorbates[{idx}] must be ase.Atoms, got {type(item).__name__}."
            )
        if len(item) == 0:
            raise SCGOValidationError(f"{prefix}adsorbates[{idx}] must not be empty.")
        out.append(item.copy())

    if not out:
        raise SCGOValidationError(
            f"{prefix}adsorbates must contain at least one fragment."
        )
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
            raise SCGOValidationError(
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
        raise SCGOValidationError("adsorbates must contain at least one fragment")
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
    from scgo.cluster_adsorbate.feasibility import (
        validate_adsorbate_placement_feasibility,
    )

    policy = get_system_policy(system_type)
    if not policy.has_adsorbate:
        if adsorbates is not None:
            raise SCGOValidationError(
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
