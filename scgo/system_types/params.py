"""TypedDict schemas for GO / optimizer params."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from ase import Atoms

from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types.composition import AdsorbateDefinition


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
    n_jobs_population_init: NotRequired[int | None]
    n_jobs_offspring: NotRequired[int | None]
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
    n_jobs: NotRequired[int]
    validation_n_jobs: NotRequired[int | None]
    seed: NotRequired[int | None]
    tag_final_minima: bool
