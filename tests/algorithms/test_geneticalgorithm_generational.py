import math
import os
import pickle

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

import scgo.algorithms.geneticalgorithm_go_torchsim as ga_mod
from scgo.algorithms import ga_go
from scgo.database import get_connection
from scgo.database.metadata import get_metadata
from scgo.utils.rng_helpers import create_child_rng, ensure_rng
from tests.test_utils import MockRelaxer


def test_ga_go_generational_smoke(tmp_path, rng):
    calc = EMT()
    relaxer = MockRelaxer(max_steps=1)
    minima = ga_go(
        composition=["Pt", "Pt", "Pt"],
        output_dir=str(tmp_path / "ga_go_gen"),
        calculator=calc,
        relaxer=relaxer,
        niter=1,
        population_size=3,
        niter_local_relaxation=1,
        batch_size=2,
        rng=rng,
    )

    assert isinstance(minima, list)


def test_ga_go_accepts_optimizer(tmp_path, rng):
    from ase.optimize import LBFGS

    calc = EMT()
    relaxer = MockRelaxer(max_steps=1)
    minima = ga_go(
        composition=["Pt", "Pt", "Pt"],
        output_dir=str(tmp_path / "ga_go_opt"),
        calculator=calc,
        relaxer=relaxer,
        niter=1,
        population_size=3,
        niter_local_relaxation=1,
        batch_size=2,
        optimizer=LBFGS,
        rng=rng,
    )
    assert isinstance(minima, list)


def test_ga_go_optimizer_default_is_fire():
    import inspect

    from ase.optimize import FIRE

    sig_ts = inspect.signature(ga_go)
    assert sig_ts.parameters["optimizer"].default is FIRE


def test_ga_go_offspring_fraction_creates_expected_offspring(
    tmp_path, rng, monkeypatch
):
    calc = EMT()
    relaxer = MockRelaxer(max_steps=1)
    counter = {"i": 0}

    def fake_create_pairing(atoms_template, n_to_optimize, rng_arg, **kwargs):
        class Pairing:
            def get_new_individual(self, parents):
                i = counter["i"]
                counter["i"] += 1
                a = Atoms(
                    symbols=atoms_template.get_chemical_symbols(),
                    positions=[[i * 0.17, 0, 0] for _ in range(n_to_optimize)],
                )
                return a, f"fake:label{i}"

        return Pairing()

    monkeypatch.setattr(ga_mod, "create_ga_pairing", fake_create_pairing)

    population_size = 4
    offs_frac = 0.5
    expected_offspring = math.ceil(population_size * offs_frac)

    outdir = tmp_path / "ga_go_off"
    minima = ga_go(
        composition=["Pt"] * 3,
        output_dir=str(outdir),
        calculator=calc,
        relaxer=relaxer,
        niter=1,
        population_size=population_size,
        offspring_fraction=offs_frac,
        niter_local_relaxation=1,
        batch_size=None,
        rng=rng,
    )

    assert isinstance(minima, list)

    db_file = outdir / "ga_go.db"
    with get_connection(str(db_file)) as da:
        rows = da.get_all_relaxed_candidates()
        gen0 = [a for a in rows if get_metadata(a, "generation") == 0]

    unique_confids = {a.info.get("confid") for a in gen0}
    assert len(unique_confids) - population_size == expected_offspring


def test_ga_go_parallel_offspring_deterministic(tmp_path):
    calc = EMT()
    kwargs = {
        "composition": ["Pt"] * 3,
        "calculator": calc,
        "relaxer": MockRelaxer(max_steps=1),
        "niter": 1,
        "population_size": 3,
        "offspring_fraction": 0.34,
        "niter_local_relaxation": 1,
        "batch_size": None,
        "verbosity": 0,
        "clean": True,
        "previous_search_glob": ".__scgo_no_prior_runs__/**/*.db",
    }
    minima_single = ga_go(
        output_dir=str(tmp_path / "torchsim_single_worker"),
        rng=create_child_rng(ensure_rng(1234)),
        n_jobs_offspring=1,
        **kwargs,
    )
    minima_parallel = ga_go(
        output_dir=str(tmp_path / "torchsim_parallel_worker"),
        rng=create_child_rng(ensure_rng(1234)),
        n_jobs_offspring=2,
        **kwargs,
    )
    assert len(minima_single) == len(minima_parallel)
    energies_single = [float(e) for e, _ in minima_single]
    energies_parallel = [float(e) for e, _ in minima_parallel]
    np.testing.assert_allclose(energies_single, energies_parallel, atol=1e-12, rtol=0.0)
    for (_, a_single), (_, a_parallel) in zip(
        minima_single, minima_parallel, strict=True
    ):
        np.testing.assert_allclose(
            a_single.get_positions(),
            a_parallel.get_positions(),
            atol=1e-12,
            rtol=0.0,
        )


def test_ga_go_parallel_offspring_deterministic_adaptive_pt4(tmp_path):
    if (os.cpu_count() or 1) < 2:
        pytest.skip("Requires >=2 CPUs to validate parallel offspring behavior")

    calc = EMT()
    kwargs = {
        "composition": ["Pt"] * 4,
        "calculator": calc,
        "relaxer": MockRelaxer(max_steps=1),
        "niter": 1,
        "population_size": 4,
        "offspring_fraction": 0.5,
        "niter_local_relaxation": 2,
        "batch_size": None,
        "verbosity": 0,
        "use_adaptive_mutations": True,
        "clean": True,
        "previous_search_glob": ".__scgo_no_prior_runs__/**/*.db",
    }
    minima_single = ga_go(
        output_dir=str(tmp_path / "adaptive_single_worker"),
        rng=create_child_rng(ensure_rng(271828)),
        n_jobs_offspring=1,
        **kwargs,
    )
    minima_parallel = ga_go(
        output_dir=str(tmp_path / "adaptive_parallel_worker"),
        rng=create_child_rng(ensure_rng(271828)),
        n_jobs_offspring=2,
        **kwargs,
    )
    assert len(minima_single) == len(minima_parallel)
    for (e_single, _), (e_parallel, _) in zip(
        minima_single, minima_parallel, strict=True
    ):
        np.testing.assert_allclose(e_single, e_parallel, atol=1e-12, rtol=0.0)


class _RecordingRelaxer:
    """Records confid order passed to relax_batch."""

    def __init__(self, max_steps: int | None = None):
        self.max_steps = max_steps
        self.confid_order: list[int] = []

    def relax_batch(self, batch: list[Atoms]):
        for atoms in batch:
            self.confid_order.append(int(atoms.info.get("confid", -1)))
        return [(float(i) * 0.1, a.copy()) for i, a in enumerate(batch)]


def test_relax_unrelaxed_candidates_sorted_by_confid(tmp_path, rng):
    relaxer = _RecordingRelaxer(max_steps=1)
    ga_go(
        composition=["Pt"] * 3,
        output_dir=str(tmp_path / "sorted_relax"),
        calculator=EMT(),
        relaxer=relaxer,
        niter=1,
        population_size=3,
        offspring_fraction=0.67,
        niter_local_relaxation=1,
        batch_size=2,
        n_jobs_offspring=2,
        rng=rng,
        verbosity=0,
        clean=True,
        previous_search_glob=".__scgo_no_prior_runs__/**/*.db",
    )
    assert relaxer.confid_order
    assert relaxer.confid_order == sorted(relaxer.confid_order)


def test_offspring_build_context_picklable(rng):
    from ase.calculators.emt import EMT
    from ase_ga.utilities import closest_distances_generator, get_all_atom_types

    from scgo.algorithms.ga_common import create_mutation_operators
    from scgo.algorithms.geneticalgorithm_go_torchsim import (
        OffspringBuildContext,
        _picklable_atoms_copy,
    )
    from scgo.utils.mutation_weights import get_adaptive_mutation_config

    composition = ["Pt", "Pt", "Pt"]
    atoms_template = Atoms(
        symbols=composition,
        positions=[[0, 0, 0]] * 3,
        cell=[10, 10, 10],
        pbc=False,
    )
    atoms_template.calc = EMT()
    all_atom_types = get_all_atom_types(atoms_template, [78])
    blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)
    operators_list, name_map = create_mutation_operators(
        composition=composition,
        n_to_optimize=3,
        blmin=blmin,
        rng=rng,
        use_adaptive=True,
    )
    adaptive_config = get_adaptive_mutation_config(
        composition=composition,
        current_generation=0,
        total_generations=1,
        use_adaptive=True,
        generations_without_improvement=0,
    )
    ctx = OffspringBuildContext(
        atoms_template=_picklable_atoms_copy(atoms_template),
        n_to_optimize=3,
        composition=composition,
        blmin=blmin,
        system_type="gas_cluster",
        n_slab=0,
        slab_for_pairing=None,
        surface_normal_axis=2,
        adsorbate_definition=None,
        connectivity_factor=None,
        allow_cluster_fragmentation=False,
        allow_adsorbate_surface_detachment=False,
        surface_config=None,
        adaptive_config=adaptive_config,
        current_mutation_probability=0.3,
        operators_list=operators_list,
        name_map=name_map,
    )
    pickle.loads(pickle.dumps(ctx))


def test_ga_go_parallel_offspring_handles_worker_failures(tmp_path, rng, monkeypatch):
    calc = EMT()
    relaxer = MockRelaxer(max_steps=1)
    base_factory = ga_mod.create_ga_pairing
    call_counter = {"n": 0}

    def flaky_pairing_factory(*args, **kwargs):
        pairing = base_factory(*args, **kwargs)
        base_get = pairing.get_new_individual

        def wrapped_get(parents):
            call_counter["n"] += 1
            if call_counter["n"] % 4 == 0:
                raise RuntimeError("synthetic crossover failure")
            return base_get(parents)

        pairing.get_new_individual = wrapped_get  # type: ignore[assignment]
        return pairing

    monkeypatch.setattr(ga_mod, "create_ga_pairing", flaky_pairing_factory)

    minima = ga_go(
        composition=["Pt"] * 3,
        output_dir=str(tmp_path / "ga_go_worker_failures"),
        calculator=calc,
        relaxer=relaxer,
        niter=1,
        population_size=4,
        offspring_fraction=0.5,
        niter_local_relaxation=1,
        batch_size=None,
        n_jobs_offspring=2,
        rng=rng,
        verbosity=0,
    )
    assert isinstance(minima, list)


def test_ga_persisted_unconstrained_rows_are_centered(tmp_path, rng):
    calc = EMT()
    outdir_ase = tmp_path / "ga_center_ase"
    ga_go(
        composition=["Pt", "Pt", "Pt"],
        output_dir=str(outdir_ase),
        calculator=calc,
        rng=rng,
        niter=1,
        population_size=3,
        niter_local_relaxation=1,
    )

    with get_connection(str(outdir_ase / "ga_go.db")) as da:
        rows_ase = da.get_all_relaxed_candidates()
    assert rows_ase
    for row in rows_ase:
        bbox_center = 0.5 * (
            row.get_positions().min(axis=0) + row.get_positions().max(axis=0)
        )
        np.testing.assert_allclose(
            bbox_center,
            np.diag(row.get_cell()) / 2.0,
            atol=1e-6,
        )

    outdir_ts = tmp_path / "ga_center_torchsim"
    ga_go(
        composition=["Pt", "Pt", "Pt"],
        output_dir=str(outdir_ts),
        calculator=calc,
        relaxer=MockRelaxer(max_steps=1),
        niter=1,
        population_size=3,
        niter_local_relaxation=1,
        batch_size=2,
        rng=rng,
    )
    with get_connection(str(outdir_ts / "ga_go.db")) as da:
        rows_ts = da.get_all_relaxed_candidates()
    assert rows_ts
    for row in rows_ts:
        bbox_center = 0.5 * (
            row.get_positions().min(axis=0) + row.get_positions().max(axis=0)
        )
        np.testing.assert_allclose(
            bbox_center,
            np.diag(row.get_cell()) / 2.0,
            atol=0.75,
        )
