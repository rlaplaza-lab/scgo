"""Consistency of structural validation across algorithms and NEB pre-screens.

CPU-only tests (no CUDA/MACE/UMA/UPET). They exercise the shared
``validate_minimum_structure`` gateway, the new bare-cluster connectivity check,
and the now-universal NEB path/energy pre-screen gates.
"""

from __future__ import annotations

import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from scgo.algorithms.ga_common import validate_structure_for_ga_storage
from scgo.exceptions import SCGOValidationError
from scgo.system_types import (
    AdsorbateDefinition,
    _validate_adsorbate_tag_partition,
    validate_minimum_structure,
    validate_structure_for_system_type,
)
from scgo.ts_search.transition_state import (
    find_transition_state,
    validate_initial_neb_energy_profile,
)


def _disconnected_gas_cluster() -> Atoms:
    """Two isolated Pt atoms far apart — not a single connected component."""
    atoms = Atoms(
        "Pt2",
        positions=[[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    atoms.calc = EMT()
    return atoms


def _connected_gas_cluster() -> Atoms:
    """Two Pt atoms at a bonded separation."""
    atoms = Atoms(
        "Pt2",
        positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    atoms.calc = EMT()
    return atoms


# --- Task 1: bare-cluster connectivity check ---------------------------------


def test_bare_gas_cluster_connectivity_rejected() -> None:
    # Rejected both via the public gateway and the shared minimum-structure
    # helper (both route through the same consolidated policy).
    with pytest.raises(SCGOValidationError):
        validate_structure_for_system_type(
            _disconnected_gas_cluster(), system_type="gas_cluster"
        )
    with pytest.raises(SCGOValidationError):
        validate_minimum_structure(
            _disconnected_gas_cluster(), system_type="gas_cluster"
        )


def test_bare_gas_cluster_connectivity_accepted_when_connected() -> None:
    # A genuinely connected dimer must not raise.
    validate_structure_for_system_type(
        _connected_gas_cluster(), system_type="gas_cluster"
    )


# --- Task 2: shared helper + per-algorithm routing ---------------------------


def test_ga_storage_rejects_fragmented_bare_cluster() -> None:
    err = validate_structure_for_ga_storage(
        _disconnected_gas_cluster(),
        surface_mode=False,
        n_slab=0,
        system_type="gas_cluster",
        surface_config=None,
    )
    assert err is not None
    # Sanity: a connected cluster passes the same storage gate.
    assert (
        validate_structure_for_ga_storage(
            _connected_gas_cluster(),
            surface_mode=False,
            n_slab=0,
            system_type="gas_cluster",
            surface_config=None,
        )
        is None
    )


def test_bh_ga_simple_share_validation_helper() -> None:
    """All three optimizers resolve the same rejection semantics."""
    from scgo.algorithms.basinhopping_go import validate_minimum_structure as bh_v
    from scgo.algorithms.geneticalgorithm_go_torchsim import (
        validate_minimum_structure as ga_v,
    )
    from scgo.algorithms.simple_go import validate_minimum_structure as simple_v

    assert bh_v is ga_v is simple_v is validate_minimum_structure


# --- Task 3: run_trials final structural gate --------------------------------


def _connected_cu3() -> Atoms:
    """Cu3 triangle — stable and connected under EMT relaxation."""
    atoms = Atoms(
        "Cu3",
        positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.25, 2.165, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    return atoms


def test_run_trials_final_gate_drops_fragmented_candidate(
    tmp_path, rng, monkeypatch
) -> None:
    from scgo.minima_search.core import run_trials

    connected = _connected_cu3()
    connected.calc = None
    disconnected = _disconnected_gas_cluster()
    disconnected.calc = None

    injected = [(0.0, connected), (5.0, disconnected)]

    def _fake_filter(unfiltered, **kwargs):
        return list(injected)

    monkeypatch.setattr("scgo.minima_search.core.filter_unique_minima", _fake_filter)

    results = run_trials(
        composition=["Cu", "Cu", "Cu"],
        global_optimizer="bh",
        global_optimizer_kwargs={
            "niter": 1,
            "niter_local_relaxation": 3,
            "system_type": "gas_cluster",
        },
        output_dir=str(tmp_path / "trials_gate"),
        rng=rng,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=False,
        verbosity=0,
    )

    # The fragmented candidate must be dropped by the final structural gate.
    assert len(results) == 1
    surviving = results[0][1]
    # Surviving minimum is the injected connected Cu3 (3 atoms).
    assert len(surviving) == 3


# --- Task 4b/4c: NEB energy profile gate for bare gas ------------------------


def test_validate_initial_neb_energy_profile_bare_gas_prominence() -> None:
    # Interior max prominence 0.05 eV < 0.10 eV floor -> rejected.
    with pytest.raises(SCGOValidationError):
        validate_initial_neb_energy_profile(
            [0.0, 0.05, 0.0],
            reference_reactant_energy=0.0,
            reference_product_energy=0.0,
            min_saddle_prominence=0.10,
            max_spurious_barrier=8.0,
        )


def test_find_transition_state_ase_runs_energy_prescreen(
    tmp_path, rng, h2_reactant, h2_product, monkeypatch
) -> None:
    """Gas TS pre-screen forwards the gas presets (not the function defaults)."""
    import scgo.ts_search.transition_state_run as run_mod

    calls: list[dict] = []

    def _fake_find(*args, **kwargs):
        calls.append(kwargs)
        atoms = args[0]
        return {
            "status": "success",
            "pair_id": "0_1",
            "neb_converged": True,
            "n_images": 3,
            "spring_constant": 0.1,
            "reactant_energy": 0.0,
            "product_energy": 0.0,
            "ts_energy": 0.5,
            "barrier_height": 0.5,
            "error": None,
            "transition_state": atoms.copy(),
            "reactant_structure": atoms.copy(),
            "product_structure": atoms.copy(),
        }

    monkeypatch.setattr(run_mod, "find_transition_state", _fake_find)
    monkeypatch.setattr(run_mod, "select_structure_pairs", lambda *a, **k: [(0, 1)])

    def _load(_dir, _comp, **_k):
        # Bonded H2 minima (both well inside the 1.4 connectivity threshold) so
        # the per-endpoint gate passes and find_transition_state is reached.
        a = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        a.center(vacuum=5.0)
        a.calc = EMT()
        b = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.80, 0.0, 0.0]])
        b.center(vacuum=5.0)
        b.calc = EMT()
        return {"H2": [(0.0, a), (0.5, b)]}

    monkeypatch.setattr(run_mod, "load_minima_by_composition", _load)

    run_mod.run_transition_state_search(
        ["H", "H"],
        system_type="gas_cluster",
        output_dir=str(tmp_path / "gas_prescreen"),
        params={"calculator": "EMT"},
        seed=0,
        verbosity=0,
        dedupe_minima=False,
        neb_prescreen_clash_distance=None,
        min_saddle_prominence=None,
        neb_max_spurious_barrier=None,
    )

    assert calls, "find_transition_state was not called for gas_cluster"
    assert calls[0]["neb_cfg"].min_saddle_prominence == 0.10
    assert calls[0]["neb_cfg"].neb_prescreen_clash_distance == 1.0
    assert calls[0]["neb_cfg"].neb_max_spurious_barrier == 8.0


def test_find_transition_state_ase_rejects_discontinuous_band(
    tmp_path, rng, h2_reactant, h2_product, monkeypatch
) -> None:
    import scgo.ts_search.transition_state as ts_mod

    def _bad_energies(images):
        return [0.0, 100.0, 0.0]

    monkeypatch.setattr(ts_mod, "evaluate_neb_image_energies_ase", _bad_energies)

    result = find_transition_state(
        h2_reactant,
        h2_product,
        EMT(),
        str(tmp_path),
        "pair_ase_reject",
        rng=rng,
        n_images=3,
        neb_steps=2,
        use_torchsim=False,
        system_type="gas_cluster",
        verbosity=0,
        neb_max_spurious_barrier=8.0,
        min_saddle_prominence=0.10,
        max_endpoint_mismatch=1.25,
    )

    # The discontinuous band is rejected by the energy-profile gate (recorded as
    # a failed result rather than raising upstream).
    assert "energy profile" in (result.get("error") or "")


# --- Task 4d: end-to-end system-type NEB pre-screen preset forwarding --------


def test_run_transition_state_search_forwards_system_type_prescreen_defaults(
    tmp_path, rng, monkeypatch
) -> None:
    """Authoritative test: presets flow from system type to find_transition_state.

    Gas uses the loose pre-screen (clash 1.0 / prominence 0.10 / barrier 8.0);
    surface types use the strict pre-screen (0.7 / 0.40 / 8.0). The values are
    asserted where they are *forwarded* (not the function-side defaults).
    """
    from ase.build import fcc111

    import scgo.ts_search.transition_state_run as run_mod
    from scgo.surface import make_surface_config
    from scgo.utils.helpers import copy_atoms

    captured: list[dict] = []

    def _fake_find(*args, **kwargs):
        captured.append(kwargs)
        atoms = args[0]
        return {
            "status": "success",
            "pair_id": "0_1",
            "neb_converged": True,
            "n_images": 3,
            "spring_constant": 0.1,
            "reactant_energy": 0.0,
            "product_energy": 0.0,
            "ts_energy": 0.5,
            "barrier_height": 0.5,
            "error": None,
            "transition_state": atoms.copy(),
            "reactant_structure": atoms.copy(),
            "product_structure": atoms.copy(),
        }

    monkeypatch.setattr(run_mod, "find_transition_state", _fake_find)
    # Bypass pair selection / comparator so the test focuses on preset forwarding.
    monkeypatch.setattr(run_mod, "select_structure_pairs", lambda *a, **k: [(0, 1)])
    # Force a stable formula key so the patched minima loader matches regardless
    # of the (large) surface composition.
    monkeypatch.setattr(run_mod, "get_cluster_formula", lambda comp: "H2")

    def _load(_dir, _comp, **_k):
        # Bonded H2 minima (inside the 1.4 connectivity threshold) so the
        # per-endpoint gate passes and find_transition_state is reached.
        a = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        a.center(vacuum=5.0)
        a.calc = EMT()
        b = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.80, 0.0, 0.0]])
        b.center(vacuum=5.0)
        b.calc = EMT()
        return {"H2": [(0.0, a), (0.5, b)]}

    monkeypatch.setattr(run_mod, "load_minima_by_composition", _load)

    def _wrap_prepare(atoms_i, atoms_j, neb_cfg):
        return copy_atoms(atoms_i), copy_atoms(atoms_j)

    slab = fcc111("Pt", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
    surf = make_surface_config(slab)

    def _surface_min(x_offset: float):
        z_top = slab.get_positions()[:, 2].max()
        pt = Atoms("Pt", positions=[[x_offset, x_offset, z_top + 1.5]])
        return slab.copy() + pt

    cases = [
        ("gas_cluster", None, None, 1.0, 0.10, 8.0),
        (
            "surface_cluster",
            surf,
            None,
            0.7,
            0.40,
            8.0,
        ),
    ]

    for system_type, surface_config, ads_def, exp_clash, exp_prom, exp_bar in cases:
        captured.clear()
        # Surface endpoints need a slab prefix; bypass real surface validation.
        monkeypatch.setattr(
            run_mod,
            "prepare_neb_endpoints",
            _wrap_prepare
            if surface_config is not None
            else run_mod.prepare_neb_endpoints,
        )
        if surface_config is not None:
            monkeypatch.setattr(
                run_mod,
                "load_minima_by_composition",
                lambda _d, _c, **_k: {
                    "H2": [
                        (0.0, _surface_min(0.5)),
                        (0.5, _surface_min(1.0)),
                    ]
                },
            )

        run_mod.run_transition_state_search(
            ["H", "H"] if surface_config is None else ["Pt"],
            system_type=system_type,
            output_dir=str(tmp_path / system_type),
            params={"calculator": "EMT"},
            seed=0,
            verbosity=0,
            dedupe_minima=False,
            neb_prescreen_clash_distance=None,
            min_saddle_prominence=None,
            neb_max_spurious_barrier=None,
            surface_config=surface_config,
            adsorbate_definition=ads_def,
        )

        assert captured, f"find_transition_state not called for {system_type}"
        kw = captured[0]
        assert kw["neb_cfg"].neb_prescreen_clash_distance == exp_clash, system_type
        assert kw["neb_cfg"].min_saddle_prominence == exp_prom, system_type
        assert kw["neb_cfg"].neb_max_spurious_barrier == exp_bar, system_type


def test_adsorbate_tag_partition_accepts_correct_tags():
    ads_def = AdsorbateDefinition(
        core_symbols=["Pt", "Pt"],
        adsorbate_symbols=["O", "O"],
        adsorbate_fragment_lengths=[2],
    )
    atoms = Atoms(
        "Pt2O2",
        positions=[
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.8, 0.0, 0.0],
        ],
    )
    atoms.set_tags([0, 0, 1, 1])
    _validate_adsorbate_tag_partition(atoms, 0, ads_def)
    validate_minimum_structure(
        atoms,
        system_type="gas_cluster_adsorbate",
        adsorbate_definition=ads_def,
    )


def test_adsorbate_tag_partition_rejects_mistagged():
    ads_def = AdsorbateDefinition(
        core_symbols=["Pt", "Pt"],
        adsorbate_symbols=["O", "O"],
        adsorbate_fragment_lengths=[2],
    )
    atoms = Atoms(
        "Pt2O2",
        positions=[
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.8, 0.0, 0.0],
        ],
    )
    atoms.set_tags([0, 1, 0, 1])
    with pytest.raises(SCGOValidationError, match="tag partition"):
        _validate_adsorbate_tag_partition(atoms, 0, ads_def)
    with pytest.raises(SCGOValidationError, match="tag partition"):
        validate_minimum_structure(
            atoms,
            system_type="gas_cluster_adsorbate",
            adsorbate_definition=ads_def,
        )
