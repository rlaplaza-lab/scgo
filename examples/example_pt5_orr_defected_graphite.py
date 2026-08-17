#!/usr/bin/env python3
"""Pt5 and ORR intermediates (O, OH, OOH) on vacancy graphite: GO via ``run_go``.

Four separate searches on a monovacancy graphite slab
(:func:`~scgo.make_hopg_5x5_defected_graphite_surface_config`):

- bare Pt5 (``system_type="surface_cluster"``, no ``adsorbates``)
- Pt5+O, Pt5+OH, Pt5+OOH (``system_type="surface_cluster_adsorbate"``)

That is the usual ORR protocol (clean cluster plus ``*O``, ``*OH``, ``*OOH``),
not co-adsorption of all three fragments.

Requires ``scgo[mace]``. Pass the same ``surface_config`` to the preset builder
and ``run_go`` as a top-level value (they must agree when both are set).
See ``docs/source/parameters.rst`` (*Parameter resolution*) for merge rules.
Params come from the reduced-budget (~25% of production)
:func:`~scgo.param_presets.get_low_effort_torchsim_ga_params`, which keeps the
production calculator but shrinks the GA budget. This example is GO-only
(no ``run_go_ts`` / ``ts_params``). Swap in
:func:`~scgo.param_presets.get_torchsim_ga_params` for a full-strength search.

Output: ``results/pt5_orr_defected_graphite_mace/`` with sibling searches trees
``Pt5_defected_graphite_searches/``, ``Pt5_O_defected_graphite_searches/``,
``Pt5_OH_defected_graphite_searches/``, and
``Pt5_O2H_defected_graphite_searches/`` (OOH path-key formula is ``O2H``).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from ase import Atoms

from scgo import (
    SurfaceSystemConfig,
    get_low_effort_torchsim_ga_params,
    get_system_path_key,
    make_hopg_5x5_defected_graphite_surface_config,
    parse_composition_arg,
    run_go,
)

COMPOSITION = "Pt5"
SEED = 42
BARE_SYSTEM_TYPE = "surface_cluster"
ADSORBATE_SYSTEM_TYPE = "surface_cluster_adsorbate"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_STEM = "pt5_orr_defected_graphite"


def _resolve_output_stem() -> str:
    """Prefer ``SCGO_EXAMPLE_OUTPUT_STEM`` for clean end-to-end runs."""
    return (
        os.environ.get("SCGO_EXAMPLE_OUTPUT_STEM", OUTPUT_STEM).strip() or OUTPUT_STEM
    )


def _o_fragment() -> Atoms:
    return Atoms("O", positions=[[0.0, 0.0, 0.0]])


def _oh_fragment() -> Atoms:
    return Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]])


def _ooh_fragment() -> Atoms:
    """Hydroperoxyl template: binding O first, then O–O 1.45 Å, O–H 0.96 Å, 105°."""
    r_oo = 1.45
    r_oh = 0.96
    theta = math.radians(105.0)
    return Atoms(
        symbols=["O", "O", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [r_oo, 0.0, 0.0],
            [
                r_oo + r_oh * math.cos(math.pi - theta),
                r_oh * math.sin(math.pi - theta),
                0.0,
            ],
        ],
    )


ORR_ADSORBATES: tuple[Atoms, ...] = (
    _o_fragment(),
    _oh_fragment(),
    _ooh_fragment(),
)


def _build_go_params(
    surface_config: SurfaceSystemConfig,
    *,
    system_type: str,
    freeze_adsorbate: bool = False,
) -> dict:
    go_params = get_low_effort_torchsim_ga_params(
        system_type=system_type,
        surface_config=surface_config,
        seed=SEED,
    )
    go_params["connectivity_factor"] = 1.8
    go_params["optimizer_params"]["ga"].update(
        write_timing_json=True,
        detailed_timing=True,
    )
    go_params[
        "n_jobs"
    ] = -2  # one switch parallelizes population init, offspring, and validation
    if freeze_adsorbate:
        go_params["freeze_adsorbate_internal_geometry"] = True
    return go_params


def _searches_dir(
    campaign_root: Path,
    surface_config: SurfaceSystemConfig,
    fragment: Atoms | None = None,
) -> Path:
    core = parse_composition_arg(COMPOSITION)
    ads_def = None
    if fragment is not None:
        symbols = list(fragment.get_chemical_symbols())
        ads_def = {
            "core_symbols": core,
            "adsorbate_symbols": symbols,
            "adsorbate_fragment_lengths": [len(symbols)],
        }
    path_key = get_system_path_key(
        core,
        adsorbate_definition=ads_def,
        surface_name=surface_config.name,
    )
    return campaign_root / f"{path_key}_searches"


def main() -> None:
    surface_config = make_hopg_5x5_defected_graphite_surface_config(seed=SEED)
    campaign_root = DEFAULT_OUTPUT_ROOT / f"{_resolve_output_stem()}_mace"
    cases: list[tuple[str, Atoms | None]] = [(BARE_SYSTEM_TYPE, None)]
    cases.extend((ADSORBATE_SYSTEM_TYPE, frag) for frag in ORR_ADSORBATES)
    for system_type, fragment in cases:
        run_go(
            COMPOSITION,
            params=_build_go_params(
                surface_config,
                system_type=system_type,
                freeze_adsorbate=fragment is not None,
            ),
            seed=SEED,
            verbosity=1,
            output_dir=_searches_dir(campaign_root, surface_config, fragment),
            surface_config=surface_config,
            system_type=system_type,
            adsorbates=fragment,
        )


if __name__ == "__main__":
    main()
