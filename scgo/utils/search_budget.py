"""Automatic global-search budgets for genetic algorithms and basin hopping.

The number of search slots follows a lattice-animal arrangement count, tempered
by a growth factor fitted to gas-phase runs and Lennard-Jones waiting times.
Genetic-algorithm generations follow how many atoms are moving. Extra slots
widen the population. Basin hopping uses the same growth from a three-atom
anchor and never goes below the atom-count hop count. See the search-effort page.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, exp, lgamma, log

from scgo.exceptions import SCGOValidationError
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.partition import resolve_slab_search_partition
from scgo.system_types.composition import (
    AdsorbateDefinition,
    resolve_search_mobile_composition,
)
from scgo.system_types.policy import SystemType, get_system_policy
from scgo.utils.helpers import (
    POPULATION_MAX,
    POPULATION_MIN,
    POPULATION_OFFSET,
    POPULATION_SLOPE,
    _scale_by_atom_count,
)
from scgo.utils.logging import get_logger

logger = get_logger(__name__)


def _moving_atom_count(n_atoms: int) -> int:
    """Atom-count floor for this many moving atoms."""
    return _scale_by_atom_count(
        n_atoms,
        base=POPULATION_OFFSET,
        scaling=POPULATION_SLOPE,
        min_val=POPULATION_MIN,
        max_val=POPULATION_MAX,
    )


# Arrangement count, then slot count relative to a unary vacuum anchor:
#   log Ω = N log μ - log|G| + log(N! / product(n_i!)) + log(site placements)
#   log E = log(slots_ref) + GROWTH_FACTOR * (log Ω - log Ω_ref)
# μ is MU_VACUUM. |G| is SYMMETRY_VACUUM or SYMMETRY_SUPPORTED.
# See the search-effort page.
GROWTH_FACTOR = 0.03
GA_REFERENCE_ATOMS = 10
BH_REFERENCE_ATOMS = 3
MU_VACUUM = 8.35
SYMMETRY_VACUUM = 24
SYMMETRY_SUPPORTED = 4
MAX_POPULATION = POPULATION_MAX
MAX_GENERATIONS = POPULATION_MAX
_MAX_SLOTS = MAX_POPULATION * MAX_GENERATIONS
# Square of the atom-count budget at the genetic-algorithm anchor.
GA_REFERENCE_SLOTS = _moving_atom_count(GA_REFERENCE_ATOMS) ** 2

__all__ = [
    "GROWTH_FACTOR",
    "POPULATION_MAX",
    "POPULATION_MIN",
    "POPULATION_OFFSET",
    "POPULATION_SLOPE",
    "ComplexityEstimate",
    "bh_auto_budget",
    "estimate_search_complexity",
    "ga_auto_budget",
]


@dataclass(frozen=True)
class ComplexityEstimate:
    """Search-space summary used to set an automatic budget."""

    n_core: int
    n_mobile: int
    dimension: int | None
    log_core: float
    log_sites: float
    evaluations: float


def _symmetry(dimension: int) -> int:
    if dimension == 2:
        return SYMMETRY_SUPPORTED
    if dimension == 3:
        return SYMMETRY_VACUUM
    raise SCGOValidationError("dimension must be 2 or 3")


def _counts(symbols: Sequence[str]) -> list[int]:
    if not symbols:
        return []
    return list(Counter(str(symbol) for symbol in symbols).values())


def _log_multinomial(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return lgamma(total + 1) - sum(lgamma(count + 1) for count in counts)


def _log_site_assignments(n_sites: int, fragment_counts: Sequence[int]) -> float:
    n_fragments = sum(fragment_counts)
    if n_fragments <= 0:
        return 0.0
    sites = max(int(n_sites), n_fragments)
    return (
        lgamma(sites + 1)
        - lgamma(sites - n_fragments + 1)
        - sum(lgamma(count + 1) for count in fragment_counts)
    )


def _fragment_counts(definition: AdsorbateDefinition | None) -> list[int]:
    if definition is None:
        return []
    symbols = [str(symbol) for symbol in definition.adsorbate_symbols]
    if not symbols:
        return []
    lengths = list(definition.adsorbate_fragment_lengths)
    if not lengths or sum(lengths) != len(symbols):
        groups = [tuple(sorted(symbols))]
    else:
        groups = []
        start = 0
        for length in lengths:
            chunk = symbols[start : start + length]
            groups.append(tuple(sorted(chunk)))
            start += length
    return list(Counter(groups).values())


def _core_symbols(
    composition: Sequence[str],
    definition: AdsorbateDefinition | None,
) -> list[str]:
    if definition is not None and (
        definition.core_symbols or definition.adsorbate_symbols
    ):
        return [str(symbol) for symbol in definition.core_symbols]
    return [str(symbol) for symbol in composition]


def _dimension(system_type: SystemType) -> int | None:
    policy = get_system_policy(system_type)
    if policy.slab_is_search_target:
        return None
    if policy.uses_surface:
        return 2
    return 3


def _mobile_symbols(
    composition: Sequence[str],
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None,
    adsorbate_definition: AdsorbateDefinition | None,
) -> list[str]:
    return resolve_search_mobile_composition(
        system_type=system_type,
        composition=[str(symbol) for symbol in composition],
        surface_config=surface_config,
        adsorbate_definition=adsorbate_definition,
    )


def _mobile_slab_count(surface_config: SurfaceSystemConfig | None) -> int:
    if surface_config is None:
        return 0
    return len(resolve_slab_search_partition(surface_config).mobile_slab_symbols)


def _site_count(
    *,
    dimension: int | None,
    n_core: int,
    n_fragments: int,
    n_mobile_slab: int,
) -> int:
    if n_fragments <= 0:
        return 0
    if dimension in (2, 3):
        base = round(n_core ** (2.0 / 3.0)) if n_core > 0 else 0
    else:
        base = n_mobile_slab
    return max(int(base), n_fragments)


def _capped_exp(log_value: float, ceiling: float) -> float:
    if log_value >= log(ceiling):
        return float(ceiling)
    return exp(log_value)


def estimate_search_complexity(
    composition: Sequence[str],
    *,
    system_type: SystemType = "gas_cluster",
    surface_config: SurfaceSystemConfig | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
    reference_atoms: int | None = None,
    reference_slots: float | None = None,
) -> ComplexityEstimate:
    """Estimate how many relaxed candidates a global search should try.

    Adsorbate atoms stay out of the core count and add a placement term.
    A genetic algorithm is anchored on a unary vacuum cluster of
    ``GA_REFERENCE_ATOMS``. ``reference_atoms`` and ``reference_slots``
    replace that anchor; basin hopping passes its three-atom hop count here.
    The atom-count budget is a floor on the split, not the slot count itself.
    """
    mobile = _mobile_symbols(
        composition,
        system_type=system_type,
        surface_config=surface_config,
        adsorbate_definition=adsorbate_definition,
    )
    n_mobile = max(len(mobile), 1)
    floor = _moving_atom_count(n_mobile)
    dimension = _dimension(system_type)
    core = _core_symbols(composition, adsorbate_definition)
    core_counts = _counts(core)
    n_core = sum(core_counts)
    fragments = _fragment_counts(adsorbate_definition)
    n_mobile_slab = _mobile_slab_count(surface_config) if dimension is None else 0
    n_sites = _site_count(
        dimension=dimension,
        n_core=n_core,
        n_fragments=sum(fragments),
        n_mobile_slab=n_mobile_slab,
    )
    log_sites = _log_site_assignments(n_sites, fragments)

    if dimension is None or n_core <= 0:
        # An exact integer product avoids ceil() seeing floor**2 as slightly
        # larger than itself after a round trip through exp(log(.)).
        if log_sites == 0.0 or GROWTH_FACTOR == 0.0:
            evaluations = float(min(floor * floor, _MAX_SLOTS))
        else:
            evaluations = _capped_exp(
                log(floor * floor) + GROWTH_FACTOR * log_sites,
                _MAX_SLOTS,
            )
        return ComplexityEstimate(
            n_core=n_core,
            n_mobile=n_mobile,
            dimension=dimension,
            log_core=0.0,
            log_sites=log_sites,
            evaluations=evaluations,
        )

    mixing = _log_multinomial(core_counts)
    # The anchor is a unary vacuum cluster, including when this system is supported.
    log_core = n_core * log(MU_VACUUM) - log(_symmetry(dimension)) + mixing
    if reference_slots is None:
        n_ref = GA_REFERENCE_ATOMS
        baseline = float(GA_REFERENCE_SLOTS)
    else:
        n_ref = n_core if reference_atoms is None else reference_atoms
        baseline = float(reference_slots)
    log_reference = n_ref * log(MU_VACUUM) - log(SYMMETRY_VACUUM)
    excess = log_core + log_sites - log_reference
    # An exact anchor product avoids ceil() seeing slots_ref as slightly
    # larger than itself after a round trip through exp(log(.)).
    if GROWTH_FACTOR == 0.0 or excess == 0.0:
        evaluations = float(min(baseline, _MAX_SLOTS))
    else:
        evaluations = _capped_exp(
            log(baseline) + GROWTH_FACTOR * excess,
            _MAX_SLOTS,
        )
    return ComplexityEstimate(
        n_core=n_core,
        n_mobile=n_mobile,
        dimension=dimension,
        log_core=log_core,
        log_sites=log_sites,
        evaluations=evaluations,
    )


def _split_slots(evaluations: float, floor: int) -> tuple[int, int, bool]:
    """Split a slot budget into population and generations."""
    floor = max(POPULATION_MIN, min(MAX_GENERATIONS, floor))
    budget = min(float(evaluations), float(_MAX_SLOTS))
    if budget <= float(floor * floor) + 1e-6:
        return floor, floor, False
    population = min(MAX_POPULATION, max(floor, ceil(budget / floor - 1e-9)))
    generations = floor
    raised = False
    if population * generations + 1e-6 < budget:
        generations = min(MAX_GENERATIONS, max(floor, ceil(budget / population)))
        raised = generations > floor
    return population, generations, raised


def ga_auto_budget(
    composition: Sequence[str],
    *,
    system_type: SystemType = "gas_cluster",
    surface_config: SurfaceSystemConfig | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
) -> tuple[int, int]:
    """Return ``(population_size, generations)`` for an automatic GA run."""
    estimate = estimate_search_complexity(
        composition,
        system_type=system_type,
        surface_config=surface_config,
        adsorbate_definition=adsorbate_definition,
    )
    floor = _moving_atom_count(estimate.n_mobile)
    population, generations, raised = _split_slots(estimate.evaluations, floor)
    if raised:
        logger.info(
            "GA population reached %d; generations raised from %d to %d",
            MAX_POPULATION,
            floor,
            generations,
        )
    return population, generations


def bh_auto_budget(
    composition: Sequence[str],
    *,
    system_type: SystemType = "gas_cluster",
    surface_config: SurfaceSystemConfig | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
) -> int:
    """Return the automatic basin-hopping hop count.

    A unary three-atom search uses the atom-count hop count. Larger clusters
    grow from that anchor with the same animal-count factor and never fall
    below the atom-count hop count for their own size. A system with no core
    atoms, including a slab, grows from its own atom-count hop count: ``M``
    with no placement term, and ``ceil(M * exp(α log Ω_sites))`` when
    molecules supply one, capped at 1000.
    """
    dimension = _dimension(system_type)
    anchor_slots = float(_moving_atom_count(BH_REFERENCE_ATOMS))
    use_bh_anchor = dimension is not None
    estimate = estimate_search_complexity(
        composition,
        system_type=system_type,
        surface_config=surface_config,
        adsorbate_definition=adsorbate_definition,
        reference_atoms=BH_REFERENCE_ATOMS if use_bh_anchor else None,
        reference_slots=anchor_slots if use_bh_anchor else None,
    )
    floor = _moving_atom_count(estimate.n_mobile)
    if dimension is None or estimate.n_core <= 0:
        hops = ceil(floor * exp(GROWTH_FACTOR * estimate.log_sites))
    else:
        hops = max(floor, ceil(estimate.evaluations))
    return min(MAX_GENERATIONS, hops)
