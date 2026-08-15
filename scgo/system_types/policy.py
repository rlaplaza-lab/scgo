"""System-type policies and derived defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from scgo.exceptions import SCGOValidationError
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
from scgo.surface.config import SurfaceSystemConfig

if TYPE_CHECKING:
    from scgo.cluster_adsorbate.config import ClusterAdsorbateConfig

SystemType = Literal[
    "gas_cluster",
    "surface_cluster",
    "gas_cluster_adsorbate",
    "surface_cluster_adsorbate",
    "surface",
    "surface_adsorbate",
]


@dataclass(frozen=True)
class SystemPolicy:
    """Behavior flags for a concrete system type."""

    system_type: SystemType
    uses_surface: bool
    has_adsorbate: bool
    slab_is_search_target: bool
    requires_slab_prefix_validation: bool
    needs_supported_deposit_validation: bool

    @property
    def neb_force_mic(self) -> bool:
        return self.uses_surface

    @property
    def neb_disable_alignment(self) -> bool:
        return False

    @property
    def neb_surface_cell_remap(self) -> bool:
        return self.uses_surface

    @property
    def neb_surface_lattice_rotation(self) -> bool:
        # Continuous in-plane Kabsch breaks adsorbate–slab registry (multi-eV
        # endpoint energy jumps); skip free rotation when an adsorbate is present.
        return self.uses_surface and not self.has_adsorbate

    @property
    def constrain_adsorbate_moves(self) -> bool:
        return self.has_adsorbate

    @property
    def adsorbate_move_scale(self) -> float:
        return 0.6 if self.has_adsorbate else 1.0

    @property
    def allow_composition_permutations(self) -> bool:
        return not self.has_adsorbate


SYSTEM_TYPE_POLICIES: dict[SystemType, SystemPolicy] = {
    "gas_cluster": SystemPolicy(
        system_type="gas_cluster",
        uses_surface=False,
        has_adsorbate=False,
        slab_is_search_target=False,
        requires_slab_prefix_validation=False,
        needs_supported_deposit_validation=False,
    ),
    "surface_cluster": SystemPolicy(
        system_type="surface_cluster",
        uses_surface=True,
        has_adsorbate=False,
        slab_is_search_target=False,
        requires_slab_prefix_validation=True,
        needs_supported_deposit_validation=True,
    ),
    "gas_cluster_adsorbate": SystemPolicy(
        system_type="gas_cluster_adsorbate",
        uses_surface=False,
        has_adsorbate=True,
        slab_is_search_target=False,
        requires_slab_prefix_validation=False,
        needs_supported_deposit_validation=False,
    ),
    "surface_cluster_adsorbate": SystemPolicy(
        system_type="surface_cluster_adsorbate",
        uses_surface=True,
        has_adsorbate=True,
        slab_is_search_target=False,
        requires_slab_prefix_validation=True,
        needs_supported_deposit_validation=True,
    ),
    "surface": SystemPolicy(
        system_type="surface",
        uses_surface=True,
        has_adsorbate=False,
        slab_is_search_target=True,
        requires_slab_prefix_validation=True,
        needs_supported_deposit_validation=False,
    ),
    "surface_adsorbate": SystemPolicy(
        system_type="surface_adsorbate",
        uses_surface=True,
        has_adsorbate=True,
        slab_is_search_target=True,
        requires_slab_prefix_validation=True,
        needs_supported_deposit_validation=True,
    ),
}


def get_system_policy(system_type: SystemType) -> SystemPolicy:
    """Return centralized behavior policy for one explicit system type."""
    try:
        return SYSTEM_TYPE_POLICIES[system_type]
    except KeyError as e:
        raise SCGOValidationError(
            f"Unknown system_type: {system_type!r}. Expected one of "
            f"{sorted(SYSTEM_TYPE_POLICIES)!r}."
        ) from e
    except TypeError as e:  # unhashable inputs (e.g. dict/list)
        raise SCGOValidationError(
            f"Invalid system_type: {system_type!r}. Expected one of "
            f"{sorted(SYSTEM_TYPE_POLICIES)!r}."
        ) from e


def select_scgo_minima_algorithm(
    n_atoms: int, system_type: SystemType
) -> Literal["simple", "bh", "ga"]:
    """Select global optimizer for composition size and system type.

    Uses the mobile-atom count (core + adsorbate symbols for adsorbate modes).
    Plain ``gas_cluster`` alone may use ``simple`` for 1-2 atoms; adsorbate and
    surface modes never select ``simple``.

    Placed here (rather than in :mod:`scgo.runner_go`) so that
    :mod:`scgo.runner_params` can use it without importing :mod:`scgo.runner_go`,
    which avoids a circular import between those two modules.
    """
    policy = get_system_policy(system_type)
    simple_allowed = not policy.uses_surface and not policy.has_adsorbate
    if n_atoms <= 2 and simple_allowed:
        return "simple"
    if n_atoms == 3:
        if policy.has_adsorbate:
            return "ga"
        return "bh"
    return "ga"


def resolve_structure_mic(
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
) -> bool:
    """Resolve MIC for GO/GA/BH structure comparators and population dedupe.

    Gas types always return ``False``. Surface types require ``surface_config``
    and return ``surface_config.comparator_use_mic``.

    TS pre-pair minima dedupe and NEB MIC use :func:`resolve_neb_mic` instead.
    """
    if not get_system_policy(system_type).uses_surface:
        return False
    if surface_config is None:
        raise SCGOValidationError(
            f"system_type={system_type!r} requires surface_config to resolve "
            "structure MIC."
        )
    return bool(surface_config.comparator_use_mic)


def resolve_neb_mic(system_type: SystemType) -> bool:
    """Resolve MIC for NEB interpolation, pair scoring geometry, and TS dedupe."""
    return bool(get_system_policy(system_type).neb_force_mic)


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
            "Use a surface_* system type."
        )
    if get_system_policy(system_type).slab_is_search_target:
        if surface_config is None:
            raise SCGOValidationError(
                f"system_type={system_type!r} requires surface_config."
            )
        from scgo.surface.partition import validate_slab_search_config

        validate_slab_search_config(surface_config)


def uses_surface(system_type: SystemType) -> bool:
    return get_system_policy(system_type).uses_surface


def slab_is_search_target(system_type: SystemType) -> bool:
    """Return True when GA/BH search top slab layers (not only deposited mobile)."""
    return get_system_policy(system_type).slab_is_search_target
