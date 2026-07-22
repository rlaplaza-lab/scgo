"""Configuration for cluster-on-surface (adsorbate + slab) workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ase import Atoms

from scgo.exceptions import (
    SCGOValidationError,
)
from scgo.initialization.initialization_config import CONNECTIVITY_FACTOR
from scgo.surface.pbc import normalize_slab_pbc
from scgo.utils.config_aliases import _UNSET, resolve_aliased_float
from scgo.utils.logging import get_logger
from scgo.utils.validation import validate_positive

logger = get_logger(__name__)


@dataclass(frozen=True, init=False)
class SurfaceSystemConfig:
    """Describe a fixed slab plus a movable adsorbate cluster for GA.

    Atom ordering in combined systems must be ``slab`` atoms first, then the
    ``len(composition)`` adsorbate atoms (matching ASE GA patches: ``n_top``
    trailing atoms are optimized). Pass the same instance to TS search
    (``get_ts_search_params(..., surface_config=...)`` or
    ``run_ts_search(..., ts_params=..., system_type=..., surface_config=...)``) so NEB uses the
    identical slab ``FixAtoms`` policy as local relaxation.
    At runtime, :func:`scgo.surface.validation.validate_surface_config_slab_prefix`
    checks that combined systems still begin with ``slab``'s symbols in order.

    Height may be set via ``adsorption_height_*`` or alias ``height_*``.

    **Slab motion during local relaxation** (three common modes; ``L`` is the
    number of distinct slab coordinate layers along ``surface_normal_axis``):

    ================================  ============================================
    Intent                            Settings
    ================================  ============================================
    Full slab frozen                  ``fix_all_slab_atoms=True`` (default)
    Frozen except top N slab layers   ``fix_all_slab_atoms=False`` and either
                                      ``n_relax_top_slab_layers=N``, or
                                      ``n_fix_bottom_slab_layers=L - N``
    Nothing on the slab frozen        ``fix_all_slab_atoms=False``,
                                      ``n_fix_bottom_slab_layers=None``,
                                      ``n_relax_top_slab_layers=None``
    ================================  ============================================

    For a typical slab with vacuum along ``z``, the adsorbate sits on the
    high-``z`` side; fixing the bottom ``L - N`` distinct layers is the
    same as leaving only the top ``N`` layers free to relax.

    Do not set ``n_relax_top_slab_layers`` together with
    ``n_fix_bottom_slab_layers``, or together with ``fix_all_slab_atoms=True``.
    """

    slab: Atoms
    adsorption_height_min: float
    adsorption_height_max: float
    surface_normal_axis: int
    fix_all_slab_atoms: bool
    n_fix_bottom_slab_layers: int | None
    n_relax_top_slab_layers: int | None
    comparator_use_mic: bool
    cluster_init_vacuum: float
    init_mode: str
    max_placement_attempts: int
    structure_connectivity_factor: float

    def __init__(
        self,
        slab: Atoms,
        *,
        adsorption_height_min: Any = _UNSET,
        adsorption_height_max: Any = _UNSET,
        height_min: Any = _UNSET,
        height_max: Any = _UNSET,
        surface_normal_axis: int = 2,
        fix_all_slab_atoms: bool = True,
        n_fix_bottom_slab_layers: int | None = None,
        n_relax_top_slab_layers: int | None = None,
        comparator_use_mic: bool = False,
        cluster_init_vacuum: float = 8.0,
        init_mode: str = "smart",
        max_placement_attempts: int = 200,
        structure_connectivity_factor: float = CONNECTIVITY_FACTOR,
    ) -> None:
        object.__setattr__(self, "slab", slab)
        object.__setattr__(
            self,
            "adsorption_height_min",
            resolve_aliased_float(
                "adsorption_height_min",
                adsorption_height_min,
                "height_min",
                height_min,
                1.2,
            ),
        )
        object.__setattr__(
            self,
            "adsorption_height_max",
            resolve_aliased_float(
                "adsorption_height_max",
                adsorption_height_max,
                "height_max",
                height_max,
                3.0,
            ),
        )
        object.__setattr__(self, "surface_normal_axis", surface_normal_axis)
        object.__setattr__(self, "fix_all_slab_atoms", fix_all_slab_atoms)
        object.__setattr__(self, "n_fix_bottom_slab_layers", n_fix_bottom_slab_layers)
        object.__setattr__(self, "n_relax_top_slab_layers", n_relax_top_slab_layers)
        object.__setattr__(self, "comparator_use_mic", comparator_use_mic)
        object.__setattr__(self, "cluster_init_vacuum", cluster_init_vacuum)
        object.__setattr__(self, "init_mode", init_mode)
        object.__setattr__(self, "max_placement_attempts", max_placement_attempts)
        object.__setattr__(
            self, "structure_connectivity_factor", structure_connectivity_factor
        )
        self.__post_init__()

    @property
    def height_min(self) -> float:
        """Alias for :attr:`adsorption_height_min`."""
        return self.adsorption_height_min

    @property
    def height_max(self) -> float:
        """Alias for :attr:`adsorption_height_max`."""
        return self.adsorption_height_max

    def __post_init__(self) -> None:
        # Copy slab so post-init pbc adjustments do not mutate a shared Atoms.
        object.__setattr__(self, "slab", self.slab.copy())
        slab = self.slab
        if self.surface_normal_axis not in (0, 1, 2):
            raise SCGOValidationError("surface_normal_axis must be 0, 1, or 2")
        validate_positive(
            "adsorption_height_min", self.adsorption_height_min, strict=True
        )
        validate_positive(
            "adsorption_height_max", self.adsorption_height_max, strict=True
        )
        validate_positive(
            "structure_connectivity_factor",
            self.structure_connectivity_factor,
            strict=True,
        )
        if self.adsorption_height_min > self.adsorption_height_max:
            raise SCGOValidationError(
                "adsorption_height_min must be <= adsorption_height_max, "
                f"got {self.adsorption_height_min} and {self.adsorption_height_max}"
            )
        if len(slab) == 0:
            raise SCGOValidationError("slab must contain at least one atom")

        if not any(slab.pbc):
            raise SCGOValidationError("Slab must have at least one periodic dimension.")

        normalize_slab_pbc(slab, surface_normal_axis=self.surface_normal_axis)

        vacuum_length = slab.cell.lengths()[self.surface_normal_axis]
        if vacuum_length < 10.0:
            logger.warning(
                f"Slab vacuum size ({vacuum_length:.2f} A) on axis {self.surface_normal_axis} "
                "might be too small to prevent periodic interaction.",
            )

        if (
            self.n_fix_bottom_slab_layers is not None
            and self.n_fix_bottom_slab_layers < 1
        ):
            raise SCGOValidationError("n_fix_bottom_slab_layers must be >= 1 when set")
        if (
            self.n_relax_top_slab_layers is not None
            and self.n_relax_top_slab_layers < 1
        ):
            raise SCGOValidationError("n_relax_top_slab_layers must be >= 1 when set")
        if self.fix_all_slab_atoms and self.n_relax_top_slab_layers is not None:
            raise SCGOValidationError(
                "n_relax_top_slab_layers is incompatible with fix_all_slab_atoms=True"
            )
        if (
            self.n_fix_bottom_slab_layers is not None
            and self.n_relax_top_slab_layers is not None
        ):
            raise SCGOValidationError(
                "set at most one of n_fix_bottom_slab_layers and "
                "n_relax_top_slab_layers"
            )


def make_surface_config(
    slab: Atoms,
    *,
    adsorption_height_min: float = 2.0,
    adsorption_height_max: float = 3.5,
    fix_all_slab_atoms: bool = True,
    comparator_use_mic: bool = True,
    max_placement_attempts: int = 500,
) -> SurfaceSystemConfig:
    """Build a :class:`SurfaceSystemConfig` from an arbitrary ASE slab."""
    return SurfaceSystemConfig(
        slab=slab.copy(),
        adsorption_height_min=adsorption_height_min,
        adsorption_height_max=adsorption_height_max,
        fix_all_slab_atoms=fix_all_slab_atoms,
        comparator_use_mic=comparator_use_mic,
        max_placement_attempts=max_placement_attempts,
    )


def describe_surface_config(cfg: SurfaceSystemConfig) -> str:
    """Summarize key surface/deposition fields for logging and provenance."""
    return (
        f"SurfaceSystemConfig(n_slab={len(cfg.slab)}, "
        f"adsorption_height=({cfg.adsorption_height_min}, {cfg.adsorption_height_max}), "
        f"surface_normal_axis={cfg.surface_normal_axis}, "
        f"fix_all_slab_atoms={cfg.fix_all_slab_atoms}, "
        f"n_fix_bottom_slab_layers={cfg.n_fix_bottom_slab_layers}, "
        f"n_relax_top_slab_layers={cfg.n_relax_top_slab_layers}, "
        f"comparator_use_mic={cfg.comparator_use_mic}, "
        f"cluster_init_vacuum={cfg.cluster_init_vacuum}, init_mode={cfg.init_mode!r}, "
        f"max_placement_attempts={cfg.max_placement_attempts})"
    )
