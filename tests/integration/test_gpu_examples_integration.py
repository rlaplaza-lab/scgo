"""GPU integration tests mirroring examples/ for Kaggle CI (dual-suite).

Each base example case is expanded over both MACE and UPET calculators, so the
Kaggle ``mace`` and ``upet`` suites (each installing only its own extra) both
exercise every system type. The test is decorated with **both**
``requires_mace`` and ``requires_upet`` so each suite selects the whole dual
set. The calculator-specific GO preset is selected *inside* the test function
(not at module import), because the opposite suite lacks that extra installed
and a top-level import would error / skip the module.

Budgets are the shared low-effort presets:
``get_low_effort_torchsim_ga_params`` / ``get_low_effort_upet_ga_params`` /
``get_low_effort_ts_search_params`` — ~25% of the production GA budget and a
~25%-with-a-floor NEB step budget. The MACE examples build their params from the
same ``get_low_effort_torchsim_ga_params`` function, so that path cannot drift
from ``examples/example_*.py``; the UPET mirror stays in lockstep via the dual
matrix here. Per-case deltas are limited to ``max_pairs`` (the dominant TS cost
lever) and ``connectivity_factor``.

Slabs match the examples exactly (``slab_layers=3``, ``slab_repeat_xy=3``), so
the defected / N-doped cells are physically meaningful rather than
self-interacting at ~4.9 Å.

Every case passes a ``barrier_range`` and ``require_ts_candidates=True`` (the
"trial of fire"): ``assert_e2e_go_ts_summary`` switches onto
``assert_ts_result_valid`` (any reported saddle must be an interior image with
correctly ordered endpoints and a sane barrier) **and** demands at least one
successful saddle, rejecting any OOM / never-ran band. This closes the silent
swap gap: previously only ``surface_cluster`` required a saddle, so the other
cases could pass with zero qualifying pairs even if GPU memory pressure dropped
every NEB band. ``max_pairs`` is tuned per case so each still finds a saddle
within the Kaggle timeout.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from ase import Atoms

from scgo import (
    get_cluster_formula,
    get_low_effort_torchsim_ga_params,
    get_low_effort_ts_search_params,
    get_low_effort_upet_ga_params,
    make_defected_graphite_surface_config,
    make_graphite_surface_config,
    make_n_doped_graphite_surface_config,
    parse_composition_arg,
    run_go_ts,
)
from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import (
    SystemType,
    build_adsorbate_definition_from_inputs,
    get_system_policy,
)
from tests.constants import MLIP_BARRIER_EV
from tests.test_utils import assert_e2e_go_ts_summary

SEED = 42

CONNECTIVITY = 1.8
# Slab geometry mirrors examples/example_*.py exactly. Smaller cells (repeat 2)
# put a vacancy / two N dopants in a ~8-atom top layer at ~4.9 Å, which
# self-interacts across the periodic image and is not a meaningful system.
SLAB_LAYERS = 3
SLAB_REPEAT_XY = 3

# Generic MLIP barrier band: any interior saddle must land inside it. This is a
# physically-plausible guard (rejects negative / absurd barriers), NOT an EMT
# accuracy reference — the GPU matrix runs MACE + UPET, a different PES than EMT.
BARRIER_RANGE_EV = MLIP_BARRIER_EV


def _adsorbates_oh(*, n: int = 1) -> list[Atoms]:
    out: list[Atoms] = []
    for i in range(n):
        shift = float(2.2 * i)
        out.append(
            Atoms(
                symbols=["O", "H"],
                positions=[[shift, 0.0, 0.0], [shift, 0.0, 0.96]],
            )
        )
    return out


@dataclass(frozen=True)
class GpuExampleCase:
    system_type: SystemType
    composition: str | list[str] = "Pt5"
    surface_config: SurfaceSystemConfig | None = None
    adsorbates: list[Atoms] | None = None
    connectivity_factor: float | None = None
    freeze_adsorbate_internal_geometry: bool = False
    # Dominant TS cost lever, so it stays per-case instead of living in the
    # shared low-effort preset (which leaves max_pairs uncapped).
    max_pairs: int = 2
    extra_ts: dict = field(default_factory=dict)
    expected_mobile_atoms: int = 5
    n_core_mobile: int = 5
    adsorbate_fragment_lengths: list[int] | None = None
    check_supported_binding: bool = True
    # Physics bar for every saddle this case reports: passing a range switches
    # assert_e2e_go_ts_summary onto assert_ts_result_valid (interior TS image,
    # endpoint ordering, barrier inside the band).
    barrier_range: tuple[float, float] = BARRIER_RANGE_EV
    # Calculator to drive this case. Each base case is expanded over both MACE
    # and UPET (see _all_cases), so both Kaggle suites exercise every type.
    calculator: str = "MACE"
    calculator_kwargs: dict = field(default_factory=dict)
    # True = "trial of fire": assert_e2e_go_ts_summary additionally demands at
    # least one *successful* saddle and rejects any OOM / never-ran band. Set for
    # every case now (closes the silent-swap gap): all 6 system types × both
    # calculators must produce a saddle within the tuned max_pairs budget.
    require_ts_candidates: bool = True


def _graphite_config() -> SurfaceSystemConfig:
    return make_graphite_surface_config(
        slab_layers=SLAB_LAYERS,
        slab_repeat_xy=SLAB_REPEAT_XY,
    )


def _defected_graphite_config() -> SurfaceSystemConfig:
    return make_defected_graphite_surface_config(
        slab_layers=SLAB_LAYERS,
        slab_repeat_xy=SLAB_REPEAT_XY,
        n_vacancies=1,
        seed=SEED,
    )


def _n_doped_graphite_config() -> SurfaceSystemConfig:
    return make_n_doped_graphite_surface_config(
        slab_layers=SLAB_LAYERS,
        slab_repeat_xy=SLAB_REPEAT_XY,
        n_dopants=2,
        seed=SEED,
    )


GPU_EXAMPLE_CASES = [
    # example_pt5_gas.py. Raised to max_pairs=4 so this case reliably finds a
    # saddle under the ~25% budget (was 2, where it often left zero on-disk pairs).
    GpuExampleCase(
        system_type="gas_cluster",
        max_pairs=4,
    ),
    # example_pt5_graphite.py. max_pairs=6 (>4) so more bands than
    # parallel_neb_max_bands=4 are produced and the surface chunking path
    # actually runs on the T4. This is the heaviest case.
    GpuExampleCase(
        system_type="surface_cluster",
        surface_config=_graphite_config(),
        connectivity_factor=CONNECTIVITY,
        max_pairs=6,
    ),
    # example_pt5_oh_gas.py
    GpuExampleCase(
        system_type="gas_cluster_adsorbate",
        adsorbates=_adsorbates_oh(n=1),
        connectivity_factor=CONNECTIVITY,
        freeze_adsorbate_internal_geometry=True,
        max_pairs=2,
        expected_mobile_atoms=7,
        adsorbate_fragment_lengths=[2],
    ),
    # example_pt5_2oh_graphite.py
    GpuExampleCase(
        system_type="surface_cluster_adsorbate",
        surface_config=_graphite_config(),
        adsorbates=_adsorbates_oh(n=2),
        connectivity_factor=CONNECTIVITY,
        freeze_adsorbate_internal_geometry=True,
        max_pairs=2,
        expected_mobile_atoms=9,
        adsorbate_fragment_lengths=[2, 2],
    ),
    # example_defected_graphite.py. Raised to max_pairs=2 (was 1) so a saddle
    # is found rather than ending with zero qualifying pairs.
    GpuExampleCase(
        system_type="surface",
        composition=[],
        surface_config=_defected_graphite_config(),
        connectivity_factor=CONNECTIVITY,
        max_pairs=2,
        expected_mobile_atoms=0,
        n_core_mobile=0,
        check_supported_binding=False,
    ),
    # example_n_doped_graphite.py. Raised to max_pairs=2 (was 1) so a saddle
    # is found rather than ending with zero qualifying pairs.
    GpuExampleCase(
        system_type="surface_adsorbate",
        composition=[],
        surface_config=_n_doped_graphite_config(),
        adsorbates=_adsorbates_oh(n=1),
        connectivity_factor=CONNECTIVITY,
        freeze_adsorbate_internal_geometry=True,
        max_pairs=2,
        expected_mobile_atoms=2,
        n_core_mobile=0,
        adsorbate_fragment_lengths=[2],
    ),
]

# Each base case is expanded over both calculators so the example matrix covers
# MACE + UPET. The MACE path uses the shared low-effort TorchSim preset (the same
# one examples/ use); UPET uses its low-effort mirror. UMA is intentionally
# excluded: HuggingFace auth for fairchem weights is unavailable on Kaggle.
_CALCULATOR_VARIANTS = (
    ("MACE", {}),
    ("UPET", {"model_name": "pet-mad-s", "version": "1.5.0"}),
)


def _all_cases() -> list[GpuExampleCase]:
    """Expand the 6 base cases over MACE + UPET into the 12-case dual matrix."""
    out: list[GpuExampleCase] = []
    for base in GPU_EXAMPLE_CASES:
        for calc, kwargs in _CALCULATOR_VARIANTS:
            out.append(
                replace(
                    base,
                    calculator=calc,
                    calculator_kwargs=dict(kwargs),
                )
            )
    return out


ALL_GPU_EXAMPLE_CASES = _all_cases()


def _build_go_params(case: GpuExampleCase) -> dict:
    # Select the calculator-specific low-effort GO preset *inside* the test
    # (not at module import): the opposite Kaggle suite has only its own extra
    # installed, so a top-level import of the other preset would error/skip.
    if case.calculator == "UPET":
        go_params = get_low_effort_upet_ga_params(
            system_type=case.system_type,
            surface_config=case.surface_config,
            seed=SEED,
            **case.calculator_kwargs,
        )
    else:
        go_params = get_low_effort_torchsim_ga_params(
            system_type=case.system_type,
            surface_config=case.surface_config,
            seed=SEED,
        )
    if case.connectivity_factor is not None:
        go_params["connectivity_factor"] = case.connectivity_factor
    if case.freeze_adsorbate_internal_geometry:
        go_params["freeze_adsorbate_internal_geometry"] = True
    return go_params


def _expected_formula(case: GpuExampleCase) -> str:
    """Match run_go_ts: core composition plus adsorbate symbols when present."""
    if isinstance(case.composition, str):
        core = parse_composition_arg(case.composition)
    else:
        core = list(case.composition)
    if case.adsorbates is None:
        if core:
            return get_cluster_formula(core)
        if case.surface_config is not None:
            return case.surface_config.name or "surface"
        return ""
    _ads_def, _fragments, full_mobile = build_adsorbate_definition_from_inputs(
        system_type=case.system_type,
        composition=core,
        adsorbates=case.adsorbates,
        context="test_run_go_ts_gpu_example_smoke",
    )
    return get_cluster_formula(full_mobile)


def _build_ts_params(case: GpuExampleCase) -> dict:
    ts_params = get_low_effort_ts_search_params(
        case.calculator,
        case.calculator_kwargs or None,
        system_type=case.system_type,
        surface_config=case.surface_config,
        seed=SEED,
    )
    ts_params["max_pairs"] = case.max_pairs
    if case.connectivity_factor is not None:
        ts_params["connectivity_factor"] = case.connectivity_factor
    ts_params.update(case.extra_ts)
    return ts_params


@pytest.mark.parametrize(
    "case", ALL_GPU_EXAMPLE_CASES, ids=lambda c: f"{c.calculator}-{c.system_type}"
)
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_cuda
@pytest.mark.requires_mace
@pytest.mark.requires_upet
def test_run_go_ts_gpu_example_smoke(tmp_path: Path, case: GpuExampleCase) -> None:
    """End-to-end GO+TS for each example system type under MACE and UPET.

    Carrying **both** ``requires_mace`` and ``requires_upet`` makes this test
    selectable by each Kaggle suite (which installs only its own extra), so both
    calculators are exercised. The calculator-specific GO preset is chosen inside
    the test, not at import, to stay compatible with the opposite suite.

    Despite carrying both ``requires_mace`` and ``requires_upet`` (so each
    Kaggle suite selects the whole matrix), only the calculator whose extra is
    installed in the active kernel can actually run. The guards below skip the
    opposite calculator's variants instead of erroring on a missing import.
    """
    pytest.importorskip("torch_sim")
    pytest.importorskip(case.calculator.lower())  # "mace" or "upet"
    output_dir = tmp_path / f"gpu_{case.calculator}_{case.system_type}"
    go_params = _build_go_params(case)
    summary = run_go_ts(
        case.composition,
        go_params=go_params,
        ts_params=_build_ts_params(case),
        seed=SEED,
        verbosity=0,
        output_dir=output_dir,
        system_type=case.system_type,
        surface_config=case.surface_config,
        adsorbates=case.adsorbates,
        log_summary=False,
    )

    assert_e2e_go_ts_summary(
        summary,
        expected_formula=_expected_formula(case),
        expected_mobile_atoms=case.expected_mobile_atoms,
        output_dir=output_dir,
        surface_config=case.surface_config,
        n_core_mobile=case.n_core_mobile,
        adsorbate_fragment_lengths=case.adsorbate_fragment_lengths,
        connectivity_factor=go_params.get("connectivity_factor"),
        check_supported_binding=(
            case.check_supported_binding
            and case.surface_config is not None
            and get_system_policy(case.system_type).needs_supported_deposit_validation
        ),
        require_ts_candidates=case.require_ts_candidates,
        barrier_range=case.barrier_range,
    )
