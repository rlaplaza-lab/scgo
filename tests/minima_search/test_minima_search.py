"""Tests for scgo.minima_search core orchestration."""

import json
import os

import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.io import read

import scgo.minima_search.core as main_mod
from scgo.database.metadata import add_metadata
from scgo.exceptions import SCGOValidationError
from scgo.minima_search import run_trials, scgo
from scgo.utils.helpers import ensure_directory_exists
from scgo.utils.ts_provenance import TS_OUTPUT_SCHEMA_VERSION
from tests.test_utils import create_test_atoms, setup_test_atoms


class TestRequireCalculator:
    """Tests for _require_calculator function."""

    def test_require_calculator_with_none(self):
        """Test that None calculator raises ValueError."""
        with pytest.raises(
            ValueError, match="calculator_for_global_optimization is required"
        ):
            main_mod._require_calculator(None)

    def test_require_calculator_with_calculator(self):
        """Test that provided calculator is returned unchanged."""
        provided_calc = EMT()
        calc = main_mod._require_calculator(provided_calc)
        assert calc is provided_calc


class TestValidateCalculatorCompatibility:
    """Tests for calculator interface validation."""

    def test_valid_calculator(self):
        """Test validation passes for valid calculator."""
        calc = EMT()
        is_valid, msg = main_mod._validate_calculator_compatibility(calc)
        assert is_valid is True
        assert "compatible" in msg.lower()

    def test_calculator_missing_method(self):
        """Test validation fails for calculator missing required methods."""

        class BadCalculator:
            """Calculator missing get_forces method."""

            def get_potential_energy(self):
                return 0.0

        calc = BadCalculator()
        is_valid, msg = main_mod._validate_calculator_compatibility(calc)
        assert is_valid is False
        assert "missing" in msg.lower()
        assert "get_forces" in msg

    def test_calculator_custom_required_methods(self):
        """Test validation with custom required methods list."""
        calc = EMT()

        # Should pass with custom list that calculator has
        is_valid, msg = main_mod._validate_calculator_compatibility(
            calc, required_methods=["get_potential_energy"]
        )
        assert is_valid is True

        # Should fail with method calculator doesn't have
        is_valid, msg = main_mod._validate_calculator_compatibility(
            calc, required_methods=["nonexistent_method"]
        )
        assert is_valid is False


class TestScgoFunction:
    """Tests for scgo() function - single GO run orchestration."""

    def test_scgo_with_bh_optimizer(self, tmp_path, rng):
        """Test scgo() with basin hopping optimizer."""
        composition = ["Pt", "Pt", "Pt"]
        output_dir = str(tmp_path / "test_bh")
        optimizer_kwargs = {
            "niter": 2,
            "niter_local_relaxation": 3,
            "system_type": "gas_cluster",
        }

        results = scgo(
            composition=composition,
            global_optimizer="bh",
            global_optimizer_kwargs=optimizer_kwargs,
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            verbosity=0,
        )

        assert isinstance(results, list)
        # Should create output directory
        assert os.path.exists(output_dir)

    def test_scgo_with_ga_optimizer(self, tmp_path, rng):
        """Test scgo() with genetic algorithm optimizer."""
        composition = ["Pt", "Pt", "Pt"]
        output_dir = str(tmp_path / "test_ga")
        optimizer_kwargs = {
            "niter": 2,
            "population_size": 3,
            "niter_local_relaxation": 3,
            "system_type": "gas_cluster",
        }

        results = scgo(
            composition=composition,
            global_optimizer="ga",
            global_optimizer_kwargs=optimizer_kwargs,
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            verbosity=0,
        )

        assert isinstance(results, list)
        assert os.path.exists(output_dir)

    def test_scgo_with_simple_optimizer(self, tmp_path, rng):
        """Test scgo() with simple optimizer."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "test_simple")
        optimizer_kwargs = {"niter": 1, "system_type": "gas_cluster"}

        results = scgo(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs=optimizer_kwargs,
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            verbosity=0,
        )

        assert isinstance(results, list)

    def test_scgo_requires_system_type(self, tmp_path, rng):
        with pytest.raises(SCGOValidationError, match="system_type must be set"):
            scgo(
                composition=["Pt", "Pt"],
                global_optimizer="simple",
                global_optimizer_kwargs={"niter": 1},
                output_dir=str(tmp_path / "missing_system_type"),
                rng=rng,
                calculator_for_global_optimization=EMT(),
                verbosity=0,
            )

    def test_scgo_surface_bh_is_supported(self, tmp_path, rng, monkeypatch):
        slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
        surface_config = main_mod.SurfaceSystemConfig(
            slab=slab,
            adsorption_height_min=1.0,
            adsorption_height_max=2.5,
        )

        captured: dict[str, object] = {}

        def _fake_bh_go(*, atoms, **kwargs):
            captured["atoms"] = atoms
            captured["kwargs"] = kwargs
            return []

        monkeypatch.setattr(main_mod, "bh_go", _fake_bh_go)
        monkeypatch.setitem(main_mod._ALGORITHM_REGISTRY["bh"], "function", _fake_bh_go)

        results = scgo(
            composition=["Pt", "O", "H"],
            global_optimizer="bh",
            global_optimizer_kwargs={
                "niter": 1,
                "niter_local_relaxation": 1,
                "system_type": "surface_cluster_adsorbate",
                "surface_config": surface_config,
                "adsorbate_definition": {
                    "core_symbols": ["Pt"],
                    "adsorbate_symbols": ["O", "H"],
                },
                "adsorbate_fragment_template": Atoms(
                    symbols=["O", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]]
                ),
            },
            output_dir=str(tmp_path / "surface_bh"),
            rng=rng,
            calculator_for_global_optimization=EMT(),
            verbosity=0,
        )
        assert results == []
        assert len(captured["atoms"]) > len(slab)

    def test_scgo_gas_adsorbate_bh_strips_init_only_kwargs(
        self, tmp_path, rng, monkeypatch
    ):
        captured: dict[str, object] = {}

        def _fake_bh_go(*, atoms, **kwargs):
            captured["atoms"] = atoms
            captured["kwargs"] = kwargs
            return []

        monkeypatch.setattr(main_mod, "bh_go", _fake_bh_go)
        monkeypatch.setitem(main_mod._ALGORITHM_REGISTRY["bh"], "function", _fake_bh_go)

        ads_def = {
            "core_symbols": ["Pt", "Pt"],
            "adsorbate_symbols": ["O"],
        }
        frag = Atoms(symbols=["O"], positions=[[0.0, 0.0, 0.0]])

        results = scgo(
            composition=["Pt", "Pt", "O"],
            global_optimizer="bh",
            global_optimizer_kwargs={
                "niter": 1,
                "niter_local_relaxation": 1,
                "system_type": "gas_cluster_adsorbate",
                "adsorbate_definition": ads_def,
                "adsorbate_fragment_template": frag,
                "vacuum": 12.0,
                "init_mode": "smart",
                "max_hierarchical_attempts": 5,
                "previous_search_glob": "**/*.db",
            },
            output_dir=str(tmp_path / "gas_bh"),
            rng=rng,
            calculator_for_global_optimization=EMT(),
            verbosity=0,
        )
        assert results == []
        assert len(captured["atoms"]) == 3
        bh_kwargs = captured["kwargs"]
        assert bh_kwargs["adsorbate_definition"] == ads_def
        assert "adsorbate_fragment_template" not in bh_kwargs
        assert "vacuum" not in bh_kwargs
        assert "init_mode" not in bh_kwargs
        assert "max_hierarchical_attempts" not in bh_kwargs
        assert "previous_search_glob" not in bh_kwargs

    def test_scgo_gas_adsorbate_empty_core_is_noop(self, tmp_path, rng):
        results = scgo(
            composition=["O", "H"],
            global_optimizer="ga",
            global_optimizer_kwargs={
                "niter": 1,
                "population_size": 2,
                "system_type": "gas_cluster_adsorbate",
                "adsorbate_definition": {
                    "core_symbols": [],
                    "adsorbate_symbols": ["O", "H"],
                },
                "adsorbate_fragment_template": Atoms(
                    symbols=["O", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]]
                ),
            },
            output_dir=str(tmp_path / "gas_empty_core_noop"),
            rng=rng,
            calculator_for_global_optimization=EMT(),
            verbosity=0,
        )
        assert results == []

    def test_scgo_unknown_optimizer(self, tmp_path, rng):
        """Test scgo() raises error for unknown optimizer."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "test_unknown")

        with pytest.raises(SCGOValidationError, match="Unknown global_optimizer"):
            scgo(
                composition=composition,
                global_optimizer="unknown",
                global_optimizer_kwargs={"system_type": "gas_cluster"},
                output_dir=output_dir,
                rng=rng,
                calculator_for_global_optimization=EMT(),
                verbosity=0,
            )

    def test_scgo_invalid_calculator(self, tmp_path, rng):
        """Test scgo() validates calculator interface requirements."""

        class BadCalculator:
            """Calculator missing required methods."""

        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "test_bad_calc")

        with pytest.raises(SCGOValidationError, match="Calculator validation failed"):
            scgo(
                composition=composition,
                global_optimizer="bh",
                global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
                output_dir=output_dir,
                rng=rng,
                calculator_for_global_optimization=BadCalculator(),
                verbosity=0,
            )

    def test_scgo_creates_output_directory(self, tmp_path, rng):
        """Test scgo() creates output directory if it doesn't exist."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "new_dir" / "subdir")

        scgo(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            verbosity=0,
        )

        assert os.path.exists(output_dir)

    def test_scgo_adds_provenance(self, tmp_path, rng):
        """Test scgo() adds provenance metadata to results."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "test_provenance")
        run_id = "test_run_123"

        results = scgo(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
            output_dir=output_dir,
            rng=rng,
            run_id=run_id,
            calculator_for_global_optimization=EMT(),
            verbosity=0,
        )

        for _, atoms in results:
            assert "provenance" in atoms.info
            assert atoms.info["provenance"]["run_id"] == run_id

    def test_scgo_empty_composition(self, tmp_path, rng):
        """Test scgo() raises error for empty composition."""
        composition = []
        output_dir = str(tmp_path / "test_empty")

        with pytest.raises(SCGOValidationError, match="Composition cannot be empty"):
            scgo(
                composition=composition,
                global_optimizer="simple",
                global_optimizer_kwargs={"niter": 1},
                output_dir=output_dir,
                rng=rng,
                verbosity=0,
            )


class TestRunTrials:
    """Tests for run_trials() function - single run orchestration."""

    def test_run_trials_single_run(self, tmp_path, rng):
        """Test run_trials() with a single datetime-tagged run."""
        composition = ["Pt", "Pt", "Pt"]
        output_dir = str(tmp_path / "trials_test")

        results = run_trials(
            composition=composition,
            global_optimizer="bh",
            global_optimizer_kwargs={
                "niter": 2,
                "niter_local_relaxation": 3,
                "system_type": "gas_cluster",
            },
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            verbosity=0,
        )

        assert isinstance(results, list)
        assert os.path.exists(output_dir)

    def test_run_trials_creates_db_at_run_root(self, tmp_path, rng):
        """Test run_trials() places database directly under run_*/."""
        composition = ["Pt", "Pt", "Pt"]
        output_dir = str(tmp_path / "trials_multi")

        run_trials(
            composition=composition,
            global_optimizer="bh",
            global_optimizer_kwargs={
                "niter": 2,
                "niter_local_relaxation": 3,
                "system_type": "gas_cluster",
            },
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            verbosity=0,
        )

        run_dirs = [d for d in os.listdir(output_dir) if d.startswith("run_")]
        assert len(run_dirs) == 1
        run_dir = os.path.join(output_dir, run_dirs[0])
        assert os.path.exists(os.path.join(run_dir, "bh_go.db"))
        assert not os.path.exists(os.path.join(run_dir, "trial_1"))

    def test_run_trials_missing_system_type_raises(self, tmp_path, rng):
        """Test run_trials() requires system_type in global_optimizer_kwargs."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "trials_missing_st")

        with pytest.raises(SCGOValidationError, match="system_type must be set"):
            run_trials(
                composition=composition,
                global_optimizer="bh",
                global_optimizer_kwargs={"niter": 1},
                output_dir=output_dir,
                rng=rng,
                verbosity=0,
            )

    def test_run_trials_creates_run_directory(self, tmp_path, rng):
        """Test run_trials() creates run-specific directory."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "trials_run_dir")

        run_trials(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            verbosity=0,
        )

        # Should create run_* directory
        run_dirs = [d for d in os.listdir(output_dir) if d.startswith("run_")]
        assert len(run_dirs) == 1

    def test_run_trials_with_run_id(self, tmp_path, rng):
        """Test run_trials() uses provided run_id."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "trials_custom_id")
        custom_run_id = "custom_run_123"

        run_trials(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
            output_dir=output_dir,
            rng=rng,
            run_id=custom_run_id,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            verbosity=0,
        )

        # Should create directory with custom run_id
        run_dir = os.path.join(output_dir, custom_run_id)
        assert os.path.exists(run_dir)

    def test_run_trials_clean_mode(self, tmp_path, rng):
        """Test run_trials() with clean=True ignores previous runs."""
        composition = ["Pt", "Pt"]
        output_dir = str(tmp_path / "trials_clean")

        # First run
        run_trials(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            verbosity=0,
        )

        # Second run with clean=True should start fresh
        results = run_trials(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
            output_dir=output_dir,
            rng=rng,
            clean=True,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            verbosity=0,
        )

        assert isinstance(results, list)

    def test_run_trials_with_ga(self, tmp_path, rng):
        """Test run_trials() with genetic algorithm."""
        composition = ["Pt", "Pt", "Pt"]
        output_dir = str(tmp_path / "trials_ga")

        results = run_trials(
            composition=composition,
            global_optimizer="ga",
            global_optimizer_kwargs={
                "niter": 2,
                "population_size": 3,
                "niter_local_relaxation": 3,
                "n_jobs_population_init": -2,  # Parallel for tests
                "system_type": "gas_cluster",
            },
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            verbosity=0,
        )

        assert isinstance(results, list)

    def test_run_trials_no_minima_found(self, tmp_path, rng):
        """Test run_trials() returns empty list when no minima found."""
        composition = ["Pt"]
        output_dir = str(tmp_path / "trials_no_minima")

        # Use very short run that might not find minima
        results = run_trials(
            composition=composition,
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            verbosity=0,
        )

        # Should return list (may be empty)
        assert isinstance(results, list)


def _slab_pt_adsorbate_pair(*, mobile_xy=(0.1, 0.1), wrap_x=False):
    """Build reference and x-wrapped slab+Pt adsorbate minima with surface metadata."""
    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    slab.pbc = [True, True, False]
    n_slab = len(slab)
    z0 = float(slab.get_positions()[:, 2].max()) + 1.5
    ref = slab.copy() + Atoms("Pt", positions=[[mobile_xy[0], mobile_xy[1], z0]])
    x_mob = slab.cell[0, 0] - mobile_xy[0] if wrap_x else mobile_xy[0]
    wrapped = slab.copy() + Atoms("Pt", positions=[[x_mob, mobile_xy[1], z0]])
    for atoms in (ref, wrapped):
        atoms.pbc = slab.pbc
        add_metadata(
            atoms,
            run_id="run_test",
            system_type="surface_cluster",
            n_slab_atoms=n_slab,
            raw_score=0.0,
        )
    return ref, wrapped, n_slab


class TestRunTrialsSurfaceAlignment:
    """Slab final minima are aligned to the lowest-energy minimum before write."""

    def test_resolve_surface_alignment_defaults(self):
        kwargs = _resolve_surface_alignment_kwargs(
            {"system_type": "surface_cluster", "surface_config": object()}
        )
        assert kwargs is not None
        assert kwargs["enable_cell_remap"] is True
        assert kwargs["enable_lattice_rotation"] is True
        assert kwargs["max_lattice_shift"] == 1

    def test_resolve_surface_alignment_gas_returns_none(self):
        assert _resolve_surface_alignment_kwargs({"system_type": "gas_cluster"}) is None

    def test_resolve_surface_alignment_reads_params(self):
        from scgo.surface.config import SurfaceSystemConfig

        slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
        cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
        kwargs = _resolve_surface_alignment_kwargs(
            {
                "system_type": "surface_cluster",
                "surface_config": cfg,
                "neb_surface_cell_remap": False,
                "neb_surface_lattice_rotation": True,
                "neb_surface_max_lattice_shift": 3,
            }
        )
        assert kwargs is not None
        assert kwargs["enable_cell_remap"] is False
        assert kwargs["enable_lattice_rotation"] is True
        assert kwargs["max_lattice_shift"] == 3

    def test_run_trials_aligns_slab_final_minima_to_best(
        self, tmp_path, rng, monkeypatch
    ):
        ref, wrapped, _n_slab = _slab_pt_adsorbate_pair(wrap_x=True)
        align_calls = 0

        def _fake_scgo(**_kwargs):
            return [(-1.0, ref), (-0.5, wrapped)]

        monkeypatch.setattr(main_mod, "scgo", _fake_scgo)

        orig_align = main_mod._align_slab_minimum_to_reference

        def _spy_align(reference, candidate, **kwargs):
            nonlocal align_calls
            align_calls += 1
            orig_align(reference, candidate, **kwargs)

        monkeypatch.setattr(main_mod, "_align_slab_minimum_to_reference", _spy_align)

        from scgo.surface.config import SurfaceSystemConfig

        slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
        cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)
        output_dir = str(tmp_path / "slab_align")

        run_trials(
            composition=["Pt"],
            global_optimizer="simple",
            global_optimizer_kwargs={
                "niter": 1,
                "system_type": "surface_cluster",
                "surface_config": cfg,
            },
            output_dir=output_dir,
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            tag_final_minima=False,
            verbosity=0,
        )

        assert align_calls == 2
        xyz_dir = os.path.join(output_dir, "final_unique_minima")
        written = sorted(f for f in os.listdir(xyz_dir) if f.endswith(".xyz"))
        assert len(written) == 2
        best_written = read(os.path.join(xyz_dir, written[0]))
        second_written = read(os.path.join(xyz_dir, written[1]))
        disp = second_written.get_positions() - best_written.get_positions()
        assert abs(float(disp[-1, 0])) < 0.5

    def test_run_trials_forwards_alignment_knobs(self, tmp_path, rng, monkeypatch):
        ref, wrapped, n_slab = _slab_pt_adsorbate_pair(wrap_x=True)
        captured: dict[str, int] = {}

        def _fake_scgo(**_kwargs):
            return [(-1.0, ref), (-0.5, wrapped)]

        monkeypatch.setattr(main_mod, "scgo", _fake_scgo)

        from scgo.ts_search import transition_state as ts_mod

        orig_pbc = ts_mod._align_product_surface_pbc

        def _spy_pbc(reactant, product_positions, **kwargs):
            captured["max_lattice_shift"] = kwargs.get("max_lattice_shift", -1)
            return orig_pbc(reactant, product_positions, **kwargs)

        monkeypatch.setattr(ts_mod, "_align_product_surface_pbc", _spy_pbc)

        from scgo.surface.config import SurfaceSystemConfig

        slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
        cfg = SurfaceSystemConfig(slab=slab, fix_all_slab_atoms=True)

        run_trials(
            composition=["Pt"],
            global_optimizer="simple",
            global_optimizer_kwargs={
                "niter": 1,
                "system_type": "surface_cluster",
                "surface_config": cfg,
                "neb_surface_max_lattice_shift": 2,
            },
            output_dir=str(tmp_path / "slab_knobs"),
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            tag_final_minima=False,
            verbosity=0,
        )

        assert captured["max_lattice_shift"] == 2
        assert n_slab > 0

    def test_run_trials_gas_skips_slab_alignment(self, tmp_path, rng, monkeypatch):
        atoms = create_test_atoms(["Pt", "Pt"])
        add_metadata(atoms, run_id="run_test", system_type="gas_cluster")
        align_calls = 0

        def _fake_scgo(**_kwargs):
            return [(-1.0, atoms)]

        monkeypatch.setattr(main_mod, "scgo", _fake_scgo)

        def _spy_align(*_args, **_kwargs):
            nonlocal align_calls
            align_calls += 1

        monkeypatch.setattr(main_mod, "_align_slab_minimum_to_reference", _spy_align)

        run_trials(
            composition=["Pt", "Pt"],
            global_optimizer="simple",
            global_optimizer_kwargs={"niter": 1, "system_type": "gas_cluster"},
            output_dir=str(tmp_path / "gas_no_align"),
            rng=rng,
            calculator_for_global_optimization=EMT(),
            validate_with_hessian=False,
            tag_final_minima=False,
            verbosity=0,
        )

        assert align_calls == 0


def _resolve_surface_alignment_kwargs(kwargs):
    return main_mod._resolve_surface_alignment_kwargs(kwargs)


class TestWriteResultsSummary:
    """Tests for _write_results_summary function."""

    def test_write_results_summary_creates_file(self, tmp_path):
        """Test _write_results_summary creates summary file."""
        output_dir = str(tmp_path / "summary_test")
        ensure_directory_exists(output_dir)

        # Create some dummy results
        atoms1 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        setup_test_atoms(atoms1)
        atoms2 = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]])
        setup_test_atoms(atoms2)

        results = [(-10.0, atoms1), (-15.0, atoms2)]

        sample_params = {"global_optimizer": "bh"}
        main_mod._write_results_summary(
            output_dir=output_dir,
            final_minima=results,
            composition_str="Pt5",
            run_id="test_run_123",
            params=sample_params,
        )

        summary_file = os.path.join(output_dir, "results_summary.json")
        assert os.path.exists(summary_file)

        # Verify content
        with open(summary_file) as f:
            summary = json.load(f)

        assert "composition" in summary
        assert summary["composition"] == "Pt5"
        assert "total_unique_minima" in summary
        assert summary["total_unique_minima"] == 2
        assert summary["params"] == sample_params
        assert summary["run_metadata_relpath"] == "test_run_123/metadata.json"
        assert summary["schema_version"] == TS_OUTPUT_SCHEMA_VERSION
        assert isinstance(summary.get("scgo_version"), str) and summary["scgo_version"]
        assert isinstance(summary.get("python_version"), str)
        assert isinstance(summary.get("created_at"), str)

    def test_write_results_summary_empty_results(self, tmp_path):
        """Test _write_results_summary handles empty results."""
        output_dir = str(tmp_path / "summary_empty")
        ensure_directory_exists(output_dir)

        main_mod._write_results_summary(
            output_dir=output_dir,
            final_minima=[],
            composition_str="Pt2",
            run_id="test_run_empty",
            params=None,
        )

        summary_file = os.path.join(output_dir, "results_summary.json")
        assert os.path.exists(summary_file)

        with open(summary_file) as f:
            summary = json.load(f)

        assert summary["total_unique_minima"] == 0
        assert summary["params"] is None
        assert summary["run_metadata_relpath"] == "test_run_empty/metadata.json"
        assert summary["schema_version"] == TS_OUTPUT_SCHEMA_VERSION
        assert isinstance(summary.get("scgo_version"), str) and summary["scgo_version"]


def test_scgo_ga_delegates_to_ga_go(monkeypatch, rng, tmp_path):
    """Unified GA path in scgo() calls ga_go."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    called = {"ga": False}

    def fake_ga_go(**kwargs):
        called["ga"] = True
        return [(-1.0, atoms.copy())]

    monkeypatch.setattr(main_mod, "ga_go", fake_ga_go)

    results = scgo(
        composition=["H", "H"],
        global_optimizer="ga",
        global_optimizer_kwargs={
            "niter": 1,
            "population_size": 2,
            "system_type": "gas_cluster",
        },
        output_dir=str(tmp_path / "ga_delegate"),
        rng=rng,
        calculator_for_global_optimization=EMT(),
        verbosity=0,
    )

    assert called["ga"] is True
    assert isinstance(results, list)


def test_sanitize_global_optimizer_kwargs_for_metadata_surface_config():
    """surface_config must not embed ASE Atoms in JSON metadata."""
    from ase.build import fcc111

    from scgo.surface.config import SurfaceSystemConfig

    slab = fcc111("Pt", size=(2, 2, 1), vacuum=6.0, orthogonal=True)
    cfg = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=2.0,
    )
    raw = {"niter": 1, "surface_config": cfg, "relaxer": object()}
    clean = main_mod._sanitize_global_optimizer_kwargs_for_metadata(raw)
    assert "relaxer" not in clean
    assert isinstance(clean["surface_config"], dict)
    assert clean["surface_config"]["present"] is True
    assert clean["surface_config"]["n_slab_atoms"] == len(slab)
    assert clean["surface_config"]["slab_chemical_symbols"] == list(
        slab.get_chemical_symbols()
    )
    assert clean["surface_config"]["surface_normal_axis"] == 2
    assert clean["surface_config"]["fix_all_slab_atoms"] is True
    assert clean["surface_config"]["n_fix_bottom_slab_layers"] is None
    assert clean["surface_config"]["n_relax_top_slab_layers"] is None
    assert clean["surface_config"]["adsorption_height_min"] == 1.0
    assert clean["surface_config"]["adsorption_height_max"] == 2.0
    assert clean["surface_config"]["comparator_use_mic"] is False
    assert clean["surface_config"]["cluster_init_vacuum"] == 8.0
    assert clean["surface_config"]["init_mode"] == "smart"
    assert clean["surface_config"]["max_placement_attempts"] == 200


def test_run_trials_passes_hessian_params_to_is_true_minimum(
    tmp_path, monkeypatch, rng
):
    """Preset Hessian knobs must reach is_true_minimum when validation is enabled."""
    from ase.calculators.emt import EMT

    captured: dict[str, object] = {}

    def _fake_is_true_minimum(*, atoms, calculator, **kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(main_mod, "is_true_minimum", _fake_is_true_minimum)

    def _fake_scgo(*_args, **_kwargs):
        atoms = Atoms("Pt2", positions=[[0, 0, 0], [0, 0, 2.5]])
        atoms.calc = EMT()
        energy = float(atoms.get_potential_energy())
        return [(energy, atoms)]

    monkeypatch.setattr(main_mod, "scgo", _fake_scgo)
    monkeypatch.setattr(
        main_mod, "filter_unique_minima", lambda candidates, **_: candidates
    )

    outdir = str(tmp_path / "searches")
    run_trials(
        composition=["Pt", "Pt"],
        global_optimizer="bh",
        global_optimizer_kwargs={
            "niter": 1,
            "system_type": "gas_cluster",
        },
        output_dir=outdir,
        calculator_for_global_optimization=EMT(),
        validate_with_hessian=True,
        check_hessian=False,
        fmax_threshold=0.02,
        imag_freq_threshold=25.0,
        rng=rng,
        clean=True,
    )

    assert captured.get("check_hessian") is False
    assert captured.get("fmax_threshold") == 0.02
    assert captured.get("imag_freq_threshold") == 25.0
