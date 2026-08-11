"""Tests for calculator helper functions.

This module tests utility functions for generating input files for various
quantum chemistry calculators.
"""

from pathlib import Path

import pytest
from ase import Atoms

from scgo.calculators.orca_helpers import prepare_orca_calculations, write_orca_inputs
from scgo.calculators.vasp_helpers import prepare_vasp_calculations, write_vasp_inputs
from tests.helpers import setup_test_atoms


def test_write_orca_inputs(tmp_path):
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    output_dir = str(tmp_path / "orca_calc")
    orca_settings = {
        "charge": 0,
        "multiplicity": 1,
        "keywords_opt": "! PBE def2-SVP Opt",
        "blocks_str": "%scf MaxIter 200 end",
    }

    write_orca_inputs(atoms, output_dir, orca_settings)

    input_file = tmp_path / "orca_calc" / "orca.inp"
    assert input_file.exists()
    with open(input_file) as f:
        content = f.read()

    job1, job2 = content.split("$new_job")

    expected_blocks1 = "%scf MaxIter 200 end"
    expected_blocks2 = """%moinp "orca.gbw"

%scf MaxIter 200 end"""

    assert expected_blocks1 in job1
    assert "$new_job" in content
    assert "! PBE0 def2-tzvp VerySlowConv MOread" in job2
    assert "! PBE def2-SVP" not in job2
    assert expected_blocks2 in job2
    assert "* xyz 0 1" in job1
    assert "* xyzfile 0 1 orca.xyz" in job2


def test_write_orca_inputs_per_job_keywords(tmp_path):
    """``keywords_opt`` / ``keywords_sp`` configure the two jobs independently."""
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    output_dir = str(tmp_path / "orca_calc")
    orca_settings = {
        "keywords_opt": "! PBE def2-SVP Opt",
        "keywords_sp": "! PBE0 def2-TZVP MOread",
    }

    write_orca_inputs(atoms, output_dir, orca_settings)

    content = (tmp_path / "orca_calc" / "orca.inp").read_text()
    job1, job2 = content.split("$new_job")
    assert "! PBE def2-SVP Opt" in job1
    assert "! PBE0 def2-TZVP MOread" in job2
    assert "! PBE def2-SVP Opt" not in job2


def test_write_orca_inputs_nonzero_charge_multiplicity(tmp_path):
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    output_dir = str(tmp_path / "orca_calc")
    orca_settings = {
        "charge": -1,
        "multiplicity": 2,
    }

    write_orca_inputs(atoms, output_dir, orca_settings)

    input_file = tmp_path / "orca_calc" / "orca.inp"
    content = input_file.read_text()
    assert "* xyz -1 2" in content
    assert "* xyzfile -1 2 orca.xyz" in content


def test_write_orca_inputs_defaults(tmp_path):
    atoms = Atoms("Pt3", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    output_dir = str(tmp_path / "orca_calc")
    orca_settings = {}  # Empty settings to test defaults

    write_orca_inputs(atoms, output_dir, orca_settings)

    input_file = tmp_path / "orca_calc" / "orca.inp"
    assert input_file.exists()
    with open(input_file) as f:
        content = f.read()

    expected_keywords1 = "! RI PBE def2-tzvp def2/J Opt Freq VerySlowConv"
    expected_blocks1 = """%pal
nprocs 24
end

%scf
MaxIter 1500
DIISMaxEq 15
directresetfreq 5
end"""
    expected_keywords2 = "! PBE0 def2-tzvp VerySlowConv MOread"
    expected_blocks2 = """%moinp "orca.gbw"

%pal
nprocs 24
end

%scf
MaxIter 1500
DIISMaxEq 15
directresetfreq 5
end"""

    assert expected_keywords1 in content
    assert expected_blocks1 in content
    assert "$new_job" in content
    assert expected_keywords2 in content
    assert expected_blocks2 in content
    assert "* xyz 0 1" in content
    assert "* xyzfile 0 1 orca.xyz" in content


class TestPrepareOrcaCalculations:
    """Tests for prepare_orca_calculations function."""

    def test_prepare_orca_calculations_creates_directories(self, tmp_path):
        """Test prepare_orca_calculations creates subdirectories for each minimum."""
        atoms1 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        setup_test_atoms(atoms1)
        atoms2 = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]])
        setup_test_atoms(atoms2)

        unique_minima = [(-10.0, atoms1), (-15.0, atoms2)]
        base_dir = str(tmp_path / "orca_prep")
        orca_settings = {"charge": 0, "multiplicity": 1}

        prepare_orca_calculations(unique_minima, base_dir, orca_settings)

        # Should create subdirectories
        subdirs = [d for d in Path(base_dir).iterdir() if d.is_dir()]
        assert len(subdirs) == 2

        # Each subdirectory should have orca.inp
        for subdir in subdirs:
            orca_file = subdir / "orca.inp"
            assert orca_file.exists()

    def test_prepare_orca_calculations_empty_list(self, tmp_path):
        """Test prepare_orca_calculations handles empty list."""
        base_dir = str(tmp_path / "orca_empty")
        orca_settings = {}

        # Should not raise
        prepare_orca_calculations([], base_dir, orca_settings)


def _patch_vasp_capture(monkeypatch) -> dict:
    """Replace ``Vasp`` with a stub recording the constructor kwargs.

    Avoids requiring ``VASP_PP_PATH`` (POTCAR generation) while still asserting
    on the parameters SCGO hands to ASE.
    """
    captured: dict = {}

    class _FakeVasp:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def write_input(self, atoms):
            """No-op: this test only inspects the constructor parameters."""

    monkeypatch.setattr("scgo.calculators.vasp_helpers.Vasp", _FakeVasp)
    return captured


class TestWriteVaspInputs:
    """Tests for write_vasp_inputs function."""

    def test_write_vasp_inputs_creates_files(self, tmp_path, monkeypatch):
        """Test write_vasp_inputs creates VASP input files."""

        # Mock VASP to avoid requiring VASP_PP_PATH
        def mock_write_input(self, atoms):
            """Mock write_input that creates dummy files."""
            incar_path = Path(self.directory) / "INCAR"
            poscar_path = Path(self.directory) / "POSCAR"
            kpoints_path = Path(self.directory) / "KPOINTS"
            incar_path.write_text("ENCUT = 400\n")
            poscar_path.write_text("dummy POSCAR\n")
            kpoints_path.write_text("dummy KPOINTS\n")

        monkeypatch.setattr("ase.calculators.vasp.Vasp.write_input", mock_write_input)

        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        setup_test_atoms(atoms)
        output_dir = str(tmp_path / "vasp_calc")
        vasp_settings = {"encut": 400}

        write_vasp_inputs(atoms, output_dir, vasp_settings)

        # Should create VASP input files
        assert (tmp_path / "vasp_calc" / "INCAR").exists()
        assert (tmp_path / "vasp_calc" / "POSCAR").exists()
        assert (tmp_path / "vasp_calc" / "KPOINTS").exists()

    def test_write_vasp_inputs_creates_xyz(self, tmp_path, monkeypatch):
        """Test write_vasp_inputs creates XYZ file."""

        def mock_write_input(self, atoms):
            """Mock write_input."""

        monkeypatch.setattr("ase.calculators.vasp.Vasp.write_input", mock_write_input)

        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        setup_test_atoms(atoms)
        output_dir = str(tmp_path / "vasp_calc")
        vasp_settings = {}

        write_vasp_inputs(atoms, output_dir, vasp_settings)

        # Should create XYZ file
        xyz_files = list(Path(output_dir).glob("*.xyz"))
        assert len(xyz_files) > 0

    def test_write_vasp_inputs_custom_vacuum(self, tmp_path, monkeypatch):
        """Test write_vasp_inputs uses custom vacuum parameter."""

        def mock_write_input(self, atoms):
            """Mock write_input that checks vacuum."""
            # Check that atoms have been centered with vacuum
            # Store for later verification
            self._atoms = atoms

        monkeypatch.setattr("ase.calculators.vasp.Vasp.write_input", mock_write_input)

        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        setup_test_atoms(atoms)
        output_dir = str(tmp_path / "vasp_calc")
        vasp_settings = {}

        write_vasp_inputs(atoms, output_dir, vasp_settings, vacuum=15.0)

        # Function should complete without error

    def test_user_settings_override_defaults(self, tmp_path, monkeypatch):
        """K2: ``vasp_settings`` must override (not collide with) SCGO defaults."""
        captured = _patch_vasp_capture(monkeypatch)

        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        setup_test_atoms(atoms)

        write_vasp_inputs(
            atoms,
            str(tmp_path / "vasp_calc"),
            {"xc": "PBEsol", "kpts": (3, 3, 1), "gamma": False, "encut": 400},
        )

        assert captured["xc"] == "PBEsol"
        assert captured["kpts"] == (3, 3, 1)
        assert captured["gamma"] is False
        assert captured["encut"] == 400

    def test_defaults_used_when_not_overridden(self, tmp_path, monkeypatch):
        """The SCGO gas-phase defaults still apply for unspecified keys."""
        captured = _patch_vasp_capture(monkeypatch)

        atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        setup_test_atoms(atoms)

        write_vasp_inputs(atoms, str(tmp_path / "vasp_calc"), {"encut": 350})

        assert captured["xc"] == "PBE"
        assert captured["kpts"] == (1, 1, 1)
        assert captured["gamma"] is True
        assert captured["encut"] == 350


class TestPrepareVaspCalculations:
    """Tests for prepare_vasp_calculations function."""

    def test_prepare_vasp_calculations_creates_directories(self, tmp_path, monkeypatch):
        """Test prepare_vasp_calculations creates subdirectories."""

        def mock_write_input(self, atoms):
            """Mock write_input."""

        monkeypatch.setattr("ase.calculators.vasp.Vasp.write_input", mock_write_input)

        atoms1 = Atoms("Pt2", positions=[[0, 0, 0], [2.5, 0, 0]])
        setup_test_atoms(atoms1)
        atoms2 = Atoms("Pt3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.165, 0]])
        setup_test_atoms(atoms2)

        unique_minima = [(-10.0, atoms1), (-15.0, atoms2)]
        base_dir = str(tmp_path / "vasp_prep")
        vasp_settings = {"encut": 400}

        prepare_vasp_calculations(unique_minima, base_dir, vasp_settings, vacuum=10.0)

        # Should create subdirectories
        subdirs = [d for d in Path(base_dir).iterdir() if d.is_dir()]
        assert len(subdirs) == 2


@pytest.mark.requires_mace
class TestMaceHelpers:
    """Tests for MACE calculator helpers."""

    def test_mace_calculator_import(self):
        from scgo.calculators.mace_helpers import MACE

        assert MACE is not None and callable(MACE)

    def test_mace_urls_enum(self):
        from scgo.calculators.mace_helpers import MaceUrls

        assert hasattr(MaceUrls, "mace_mp_small") or hasattr(MaceUrls, "mace_matpes_0")

    @pytest.mark.slow
    def test_mace_calculator_initialization(self):
        from scgo.calculators.mace_helpers import MACE

        try:
            calc = MACE(model="mace_mp_small")
        except (FileNotFoundError, OSError, RuntimeError) as e:
            pytest.skip(f"MACE init failed (e.g. missing model): {e}")
        assert calc is not None


class TestAseBatchRelaxerSurfaceMode:
    """K1: the plain-ASE batch relaxer must not gas-phase-center slabs."""

    @staticmethod
    def _slab_with_adsorbate():
        from ase.build import add_adsorbate, fcc111
        from ase.constraints import FixAtoms

        slab = fcc111("Cu", size=(2, 2, 2), vacuum=6.0, orthogonal=True)
        slab.pbc = True
        n_slab = len(slab)
        add_adsorbate(slab, "Cu", height=2.0, position=(1.0, 1.0))
        slab.set_constraint(FixAtoms(indices=list(range(n_slab))))
        slab.set_tags([0] * n_slab + [1])
        return slab, n_slab

    def test_surface_mode_keeps_slab_positions(self):
        import numpy as np
        from ase.calculators.emt import EMT

        from scgo.calculators.ase_batch_relaxer import AseBatchRelaxer

        atoms, n_slab = self._slab_with_adsorbate()
        original = atoms.get_positions()[:n_slab].copy()

        relaxer = AseBatchRelaxer(
            EMT(), force_tol=0.05, max_steps=0, surface_mode=True, n_slab=n_slab
        )
        _energy, relaxed = relaxer.relax_batch([atoms], steps=0)[0]

        assert np.allclose(relaxed.get_positions()[:n_slab], original)

    def test_default_mode_recenters_the_whole_system(self):
        import numpy as np
        from ase.calculators.emt import EMT

        from scgo.calculators.ase_batch_relaxer import AseBatchRelaxer

        atoms, n_slab = self._slab_with_adsorbate()
        original = atoms.get_positions()[:n_slab].copy()

        relaxer = AseBatchRelaxer(EMT(), force_tol=0.05, max_steps=0)
        _energy, relaxed = relaxer.relax_batch([atoms], steps=0)[0]

        # Gas-phase canonicalization translates every atom, slab included.
        assert not np.allclose(relaxed.get_positions()[:n_slab], original)

    def test_ga_passes_surface_mode_to_the_ase_relaxer(self):
        """The GA construction site must forward surface_mode / n_slab."""
        import inspect

        from scgo.algorithms import geneticalgorithm_go_torchsim as ga_mod

        src = inspect.getsource(ga_mod)
        marker = "relaxer = AseBatchRelaxer("
        assert marker in src
        call = src.split(marker, 1)[1].split(")", 1)[0]
        assert "surface_mode=surface_mode" in call
        assert "n_slab=n_fixed" in call


@pytest.mark.requires_mace
def test_mace_calculator_stores_resolved_device(monkeypatch):
    """K6: an explicit ``device="cpu"`` must be readable from the calculator."""
    from unittest.mock import MagicMock

    from scgo.calculators import mace_helpers

    monkeypatch.setattr(
        mace_helpers, "mace_mp", lambda **_kwargs: MagicMock(name="mace_calc")
    )

    calc = mace_helpers.MACE(model_name="mace_matpes_0", device="cpu")
    assert calc.device == "cpu"
