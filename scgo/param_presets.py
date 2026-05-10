"""Parameter presets for SCGO campaigns."""

from __future__ import annotations

from typing import Any

from scgo.constants import (
    BOLTZMANN_K_EV_PER_K,
    DEFAULT_COMPARATOR_TOL,
    DEFAULT_ENERGY_TOLERANCE,
    DEFAULT_NEB_TANGENT_METHOD,
    DEFAULT_PAIR_COR_MAX,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import SYSTEM_TYPE_POLICIES, SystemType, get_system_policy

# Available MACE model names for use in calculator_kwargs["model_name"]
AVAILABLE_MACE_MODELS = [
    "mace_matpes_0",  # r2scan variant (default in MACE class)
    "mace_mp_small",  # Small MACE-MP
    "mace_mpa_medium",  # Medium MACE-MPA
    "mace_off_small",  # Small MACE-OFF
]

# Common fairchem pretrained names (see fairchem.core.calculate.pretrained_mlip)
AVAILABLE_UMA_MODELS = [
    "uma-s-1p2",
    "uma-s-1p1",
    "uma-m-1p1",
]

__all__ = [
    "AVAILABLE_MACE_MODELS",
    "AVAILABLE_UMA_MODELS",
    "TS_DEFAULTS_BY_SYSTEM_TYPE",
    "get_default_params",
    "get_minimal_ga_params",
    "get_testing_params",
    "get_torchsim_ga_params",
    "get_diversity_params",
    "get_high_energy_params",
    "get_ts_defaults",
    "get_ts_search_params",
    "get_default_uma_params",
    "get_ts_search_params_uma",
    "get_uma_ga_benchmark_params",
]


# Per-system-type NEB defaults consumed by `get_ts_search_params` and
# `coerce_ts_params_to_runner_kwargs`. Keep `neb_align_endpoints` and
# `neb_interpolation_mic` coherent with `SystemPolicy.neb_disable_alignment` /
# `neb_force_mic` (an import-time assertion below guards against drift). Other
# knobs (n_images, fmax, steps, ...) are independent and benchmarked per type.
TS_DEFAULTS_BY_SYSTEM_TYPE: dict[SystemType, dict[str, Any]] = {
    "gas_cluster": {
        "neb_align_endpoints": True,
        "neb_interpolation_mic": False,
        "neb_n_images": 5,
        "neb_spring_constant": 0.1,
        "neb_fmax": 0.05,
        "neb_steps": "auto",
        "neb_climb": False,
        "neb_perturb_sigma": 0.0,
        "neb_interpolation_method": "idpp",
        "neb_tangent_method": DEFAULT_NEB_TANGENT_METHOD,
        "torchsim_fmax": 0.05,
        "torchsim_max_steps": "auto",
    },
    "gas_cluster_adsorbate": {
        "neb_align_endpoints": True,
        "neb_interpolation_mic": False,
        "neb_n_images": 5,
        "neb_spring_constant": 0.1,
        "neb_fmax": 0.05,
        "neb_steps": "auto",
        "neb_climb": False,
        "neb_perturb_sigma": 0.0,
        "neb_interpolation_method": "idpp",
        "neb_tangent_method": DEFAULT_NEB_TANGENT_METHOD,
        "torchsim_fmax": 0.05,
        "torchsim_max_steps": "auto",
    },
    "surface_cluster": {
        "neb_align_endpoints": False,
        "neb_interpolation_mic": True,
        "neb_n_images": 5,
        "neb_spring_constant": 0.1,
        "neb_fmax": 0.1,
        "neb_steps": 500,
        "neb_climb": False,
        "neb_perturb_sigma": 0.0,
        "neb_interpolation_method": "idpp",
        "neb_tangent_method": DEFAULT_NEB_TANGENT_METHOD,
        "torchsim_fmax": 0.1,
        "torchsim_max_steps": 500,
    },
    "surface_cluster_adsorbate": {
        "neb_align_endpoints": False,
        "neb_interpolation_mic": True,
        "neb_n_images": 5,
        "neb_spring_constant": 0.1,
        "neb_fmax": 0.1,
        "neb_steps": 500,
        "neb_climb": False,
        "neb_perturb_sigma": 0.0,
        "neb_interpolation_method": "idpp",
        "neb_tangent_method": DEFAULT_NEB_TANGENT_METHOD,
        "torchsim_fmax": 0.1,
        "torchsim_max_steps": 500,
    },
}


def _assert_ts_defaults_match_system_policies() -> None:
    """Guard against drift between TS defaults and ``SystemPolicy`` flags."""
    missing = set(SYSTEM_TYPE_POLICIES) - set(TS_DEFAULTS_BY_SYSTEM_TYPE)
    extra = set(TS_DEFAULTS_BY_SYSTEM_TYPE) - set(SYSTEM_TYPE_POLICIES)
    if missing or extra:
        raise RuntimeError(
            "TS_DEFAULTS_BY_SYSTEM_TYPE keys must match SYSTEM_TYPE_POLICIES "
            f"(missing={sorted(missing)!r}, extra={sorted(extra)!r})."
        )
    for st, defaults in TS_DEFAULTS_BY_SYSTEM_TYPE.items():
        policy = SYSTEM_TYPE_POLICIES[st]
        expected_align = not policy.neb_disable_alignment
        if defaults["neb_align_endpoints"] is not expected_align:
            raise RuntimeError(
                f"TS_DEFAULTS_BY_SYSTEM_TYPE[{st!r}]['neb_align_endpoints']="
                f"{defaults['neb_align_endpoints']!r} disagrees with "
                f"SystemPolicy.neb_disable_alignment={policy.neb_disable_alignment!r}."
            )
        if defaults["neb_interpolation_mic"] != policy.neb_force_mic:
            raise RuntimeError(
                f"TS_DEFAULTS_BY_SYSTEM_TYPE[{st!r}]['neb_interpolation_mic']="
                f"{defaults['neb_interpolation_mic']!r} disagrees with "
                f"SystemPolicy.neb_force_mic={policy.neb_force_mic!r}."
            )


_assert_ts_defaults_match_system_policies()


def get_ts_defaults(system_type: SystemType) -> dict[str, Any]:
    """Return a fresh copy of NEB knob defaults for one system type.

    Single source of truth read by :func:`get_ts_search_params` and
    :func:`scgo.utils.ts_runner_kwargs.coerce_ts_params_to_runner_kwargs`.
    """
    if system_type not in TS_DEFAULTS_BY_SYSTEM_TYPE:
        raise ValueError(
            f"Unsupported system_type={system_type!r}; expected one of "
            f"{sorted(TS_DEFAULTS_BY_SYSTEM_TYPE)!r}."
        )
    return dict(TS_DEFAULTS_BY_SYSTEM_TYPE[system_type])


def get_default_params() -> dict[str, Any]:
    """Return the default SCGO parameter dictionary."""
    return {
        "validate_with_hessian": False,
        "calculator": "MACE",
        "seed": None,  # Will be overridden by function parameter
        "calculator_kwargs": {"model_name": "mace_matpes_0"},
        "fmax_threshold": 0.05,
        "check_hessian": True,
        "imag_freq_threshold": 50.0,
        "n_trials": 1,
        "tag_final_minima": True,
        "connectivity_factor": 1.4,  # Default connectivity factor for cluster validation
        "fitness_strategy": "low_energy",  # Default: minimize energy
        "diversity_reference_db": None,  # For diversity strategy
        "diversity_max_references": 100,  # Performance limit
        "diversity_update_interval": 5,  # Update references every N iterations/generations
        "optimizer_params": {
            "simple": {
                "optimizer": "FIRE",
                "fmax": 0.05,
                "niter": 1,
                "niter_local_relaxation": "auto",
                "system_type": "gas_cluster",
            },
            "bh": {
                "optimizer": "FIRE",
                "temperature": 500 * 8.617e-5,  # 500K in eV
                "fmax": 0.05,
                "niter": "auto",
                "dr": 0.2,
                "move_fraction": 0.3,
                "niter_local_relaxation": "auto",
                "move_strategy": "random",
                "deduplicate": True,
                "energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
                "comparator_tol": DEFAULT_COMPARATOR_TOL,
                "comparator_pair_cor_max": DEFAULT_PAIR_COR_MAX,
                "comparator_n_top": None,
                "fitness_strategy": None,  # None = inherit from top-level
                "diversity_reference_db": None,  # For diversity strategy
                "diversity_max_references": 100,  # Performance limit
                "diversity_update_interval": 5,  # Update references every N iterations
                "system_type": "gas_cluster",
            },
            "ga": {
                "optimizer": "FIRE",
                "population_size": "auto",
                "niter": "auto",
                "niter_local_relaxation": "auto",
                "mutation_probability": 0.4,
                "offspring_fraction": 0.5,
                "fmax": 0.05,
                "vacuum": 10.0,
                "energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
                "use_adaptive_mutations": True,
                "stagnation_trigger": 4,
                "stagnation_full_trigger": 8,
                "recovery_window": 2,
                "aggressive_burst_multiplier": 1.8,
                "max_mutation_probability": 0.65,
                "early_stopping_niter": 10,  # Stop if no improvement after N generations
                "n_jobs_population_init": -2,  # Parallel batch init: -2 = all CPUs except one
                "n_jobs_offspring": -2,  # Parallel default aligned with n_jobs_population_init
                "batch_size": None,
                "relaxer": None,
                "fitness_strategy": None,  # None = inherit from top-level
                "diversity_reference_db": None,  # For diversity strategy
                "diversity_max_references": 100,  # Performance limit
                "diversity_update_interval": 5,  # Update references every N generations
                "system_type": "gas_cluster",
            },
        },
    }


def get_minimal_ga_params(
    seed: int | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Return compact GA-focused parameters (merged with defaults).

    Uses sequential population init and offspring work (``n_jobs_*`` set to 1) so
    runners stay easy to reason about; merge with :func:`initialize_params` to
    fill other keys from :func:`get_default_params`.
    """
    params = get_default_params()

    # Override GA-specific settings for faster/leaner runs
    params["optimizer_params"]["ga"].update(
        {
            "niter": "auto",
            "population_size": "auto",
            "mutation_probability": 0.4,
            "energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
            "niter_local_relaxation": "auto",
            "n_jobs_population_init": 1,  # Sequential for runners (explicit control)
            "n_jobs_offspring": 1,  # Match init: avoid parallel offspring when init is serial
        }
    )

    # Set model name if provided
    if model_name is not None:
        params["calculator_kwargs"]["model_name"] = model_name

    # Set seed if provided
    if seed is not None:
        params["seed"] = seed

    return params


def get_testing_params() -> dict[str, Any]:
    """Return fast, low-cost parameters for tests (EMT, fewer iterations)."""
    return {
        "validate_with_hessian": False,
        "calculator": "EMT",
        "seed": None,  # Will be overridden by function parameter
        "connectivity_factor": 1.4,
        "optimizer_params": {
            "simple": {
                "optimizer": "FIRE",
                "fmax": 0.05,
                "niter": 1,
                "niter_local_relaxation": 2,
                "system_type": "gas_cluster",
            },
            "bh": {
                "optimizer": "FIRE",
                "niter": 5,
                "dr": 0.2,
                "niter_local_relaxation": 2,
                "system_type": "gas_cluster",
            },
            "ga": {
                "optimizer": "FIRE",
                "population_size": 5,
                "offspring_fraction": 0.5,
                "niter": 2,
                "niter_local_relaxation": 2,
                "n_jobs_population_init": -2,  # Parallel for tests/benchmarks
                "system_type": "gas_cluster",
            },
        },
    }


def _get_base_ga_benchmark_params(seed: int) -> dict[str, Any]:
    """Return GA benchmark parameters derived from defaults."""
    params = get_default_params()
    params["seed"] = seed
    params["calculator_kwargs"]["default_dtype"] = "float32"

    # Customize GA parameters for benchmarking
    params["optimizer_params"]["ga"].update(
        {
            "fmax": 0.05,
            "niter_local_relaxation": 200,
            "niter": 10,
            "population_size": 50,
            "n_jobs_population_init": -2,  # Parallel for benchmarks
        },
    )

    return params


def _attach_fairchem_torchsim_relaxer(
    ga: dict[str, Any],
    calculator_kwargs: dict[str, Any],
    *,
    max_steps: int,
    autobatcher: bool | None = None,
    expected_max_atoms: int | None = None,
) -> None:
    """Set ``ga['relaxer']`` to a FairChem-backed :class:`TorchSimBatchRelaxer`."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    fmax_val = float(ga.get("fmax", 0.05))
    ga["relaxer"] = TorchSimBatchRelaxer(
        model_kind="fairchem",
        fairchem_model_name=calculator_kwargs["model_name"],
        fairchem_task_name=calculator_kwargs.get("task_name"),
        force_tol=fmax_val,
        optimizer_name="fire",
        max_steps=max_steps,
        dtype=None,  # TorchSim default per model; keep lazy/portable
        autobatcher=autobatcher,
        expected_max_atoms=expected_max_atoms,
    )


def get_uma_ga_benchmark_params(
    seed: int,
    *,
    model_name: str = "uma-s-1p2",
    uma_task: str = "oc25",
) -> dict[str, Any]:
    """GA benchmark parameters matching :func:`_get_base_ga_benchmark_params` but with UMA.

    Tuned for regression and profiling alongside the MACE TorchSim benchmark preset
    (:func:`get_torchsim_ga_params`): fixed local relaxation budget from the base
    preset (200 steps, not ``"auto"``), with autobatching and ``expected_max_atoms=600``
    for stable GPU memory behaviour. For general UMA runs with default GA
    ``"auto"`` local steps, use :func:`get_default_uma_params` instead.
    """
    params = _get_base_ga_benchmark_params(seed)
    params["calculator"] = "UMA"
    params["calculator_kwargs"] = {"model_name": model_name, "task_name": uma_task}

    ga = params["optimizer_params"]["ga"]
    niter_local = ga.get("niter_local_relaxation", 200)
    max_steps = 200 if niter_local == "auto" else int(niter_local)
    _attach_fairchem_torchsim_relaxer(
        ga,
        params["calculator_kwargs"],
        max_steps=max_steps,
        autobatcher=True,
        expected_max_atoms=600,
    )
    return params


def get_default_uma_params() -> dict[str, Any]:
    """Default SCGO parameters using the UMA calculator (fairchem-core).

    For typical campaigns with default GA settings: ``niter_local_relaxation`` is
    ``"auto"`` and the TorchSim relaxer uses 250 max steps in that case. Autobatcher
    and memory-probe defaults follow :class:`TorchSimBatchRelaxer` (``autobatcher``
    None: CUDA on, CPU off). Use :func:`get_uma_ga_benchmark_params` when you need
    the same structure as the MACE benchmark preset (fixed local steps, explicit
    autobatcher/expected_max_atoms).
    """
    params = get_default_params()
    params["calculator"] = "UMA"
    params["calculator_kwargs"] = {
        "model_name": "uma-s-1p2",
        "task_name": "oc25",
    }
    ga = params.get("optimizer_params", {}).get("ga", {})
    niter_local = ga.get("niter_local_relaxation", "auto")
    max_steps = 250 if niter_local == "auto" else int(niter_local)
    _attach_fairchem_torchsim_relaxer(
        ga,
        params["calculator_kwargs"],
        max_steps=max_steps,
        autobatcher=None,
        expected_max_atoms=None,
    )
    return params


def get_ts_search_params_uma(
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    model_name: str = "uma-s-1p2",
    uma_task: str | None = "oc25",
    seed: int | None = None,
) -> dict[str, Any]:
    """TS preset for UMA (FairChem); same NEB defaults as MACE, for ``scgo[uma]``."""
    return get_ts_search_params(
        calculator="UMA",
        calculator_kwargs={"model_name": model_name, "task_name": uma_task},
        system_type=system_type,
        surface_config=surface_config,
        seed=seed,
    )


def get_torchsim_ga_params(
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    seed: int | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Return GO params using TorchSim relaxer (requires ``scgo[mace]``).

    Mirrors :func:`get_ts_search_params` call style by requiring ``system_type``
    and accepting ``surface_config`` / ``seed`` explicitly.
    When ``model_name`` is set, it is written to ``calculator_kwargs`` and the
    :class:`~scgo.calculators.torchsim_helpers.TorchSimBatchRelaxer` uses the
    same MACE model name as the ASE calculator.
    """
    import torch

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    policy = get_system_policy(system_type)
    if policy.uses_surface and not isinstance(surface_config, SurfaceSystemConfig):
        raise ValueError(
            f"system_type={system_type!r} requires surface_config to be provided "
            "as a SurfaceSystemConfig when building go_params."
        )

    effective_seed = 0 if seed is None else int(seed)
    params = _get_base_ga_benchmark_params(effective_seed)
    if seed is None:
        params["seed"] = None
    if model_name is not None:
        params["calculator_kwargs"]["model_name"] = model_name

    mace_model = params["calculator_kwargs"].get("model_name", "mace_matpes_0")
    fmax_val = params["optimizer_params"]["ga"]["fmax"]
    niter_local = params["optimizer_params"]["ga"]["niter_local_relaxation"]

    params["optimizer_params"]["ga"].update(
        {
            "relaxer": TorchSimBatchRelaxer(
                force_tol=fmax_val,
                optimizer_name="fire",
                mace_model_name=mace_model,
                seed=seed,
                max_steps=niter_local,
                dtype=torch.float32,
                autobatcher=True,
                expected_max_atoms=600,
            ),
        },
    )
    for algo in ("simple", "bh", "ga"):
        params["optimizer_params"][algo]["system_type"] = system_type
    if policy.uses_surface:
        params["surface_config"] = surface_config
        for algo in ("simple", "bh", "ga"):
            params["optimizer_params"][algo]["surface_config"] = surface_config

    return params


def get_diversity_params(
    reference_db_glob: str = "**/*.db",
    max_references: int = 100,
    update_interval: int = 5,
) -> dict[str, Any]:
    """Return params for diversity-based optimization (reference DB, intervals).

    ``reference_db_glob`` must match at least one database with reference
    structures when you run; there is no runtime check that the glob is non-empty.
    """
    params = get_default_params()
    params["fitness_strategy"] = "diversity"
    params["diversity_reference_db"] = reference_db_glob
    params["diversity_max_references"] = max_references
    params["diversity_update_interval"] = update_interval

    # Diversity strategy works better with larger populations
    # Keep auto settings but note they will scale appropriately

    return params


def get_high_energy_params() -> dict[str, Any]:
    """Return params that bias exploration toward high-energy structures.

    Sets top-level ``fitness_strategy`` to ``high_energy`` (used by BH and GA).
    Basin hopping additionally uses a higher temperature. GA hyperparameters are
    otherwise unchanged—override ``optimizer_params['ga']`` if you need stronger
    exploration there.
    """
    params = get_default_params()
    params["fitness_strategy"] = "high_energy"

    # Increase temperature for BH to accept high-energy moves
    # Default is 500K, increase to 1000K for better high-energy exploration
    params["optimizer_params"]["bh"]["temperature"] = (
        1000 * BOLTZMANN_K_EV_PER_K
    )  # 1000K

    return params


def get_ts_search_params(
    calculator: str = "MACE",
    calculator_kwargs: dict[str, Any] | None = None,
    *,
    system_type: SystemType,
    surface_config: SurfaceSystemConfig | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """TS-only settings (NEB, calculator, pairing). Not merged with GO defaults.

    For EMT or other non-TorchSim calculators, set ``use_torchsim=False`` on the
    returned dict before running.
    `system_type` is used to shape technical defaults.
    For surface system types, `surface_config` is required and stored in the
    returned dictionary so TS loading/validation always receives explicit slab
    context (no guessing).
    If ``seed`` is set, it is stored in the returned dict; :func:`run_go_ts` / ``run_ts_*``
    require it to be consistent with ``go_params['seed']`` and the ``seed=`` run argument.
    The ``connectivity_factor`` key sets the global connectivity threshold for cluster
    validation (default 1.4).
    """
    policy = get_system_policy(system_type)
    if policy.uses_surface and not isinstance(surface_config, SurfaceSystemConfig):
        raise ValueError(
            f"system_type={system_type!r} requires surface_config to be provided "
            "as a SurfaceSystemConfig when building ts_params."
        )

    if calculator_kwargs is None:
        calc_u = str(calculator).strip().upper()
        calculator_kwargs = {"model_name": "mace_matpes_0"} if calc_u == "MACE" else {}

    params: dict[str, Any] = {
        "calculator": calculator,
        "calculator_kwargs": dict(calculator_kwargs),
        "connectivity_factor": 1.4,
        "max_pairs": None,
        "energy_gap_threshold": 2.0,
        "similarity_tolerance": DEFAULT_COMPARATOR_TOL,
        "similarity_pair_cor_max": 0.1,
        "use_torchsim": True,
        "torchsim_batch_size": 5,
        "use_parallel_neb": False,
        "dedupe_minima": True,
        "minima_energy_tolerance": DEFAULT_ENERGY_TOLERANCE,
    }
    params.update(get_ts_defaults(system_type))

    if policy.uses_surface:
        params["surface_config"] = surface_config

    if seed is not None:
        params["seed"] = int(seed)

    return params
