"""Tests for calculator helper functions."""

import pytest


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

    def test_ga_passes_surface_mode_to_the_ase_relaxer(
        self, monkeypatch, surface_config_pt111, tmp_path
    ):
        """The GA construction site must forward surface_mode / n_slab by behaviour."""
        from ase.calculators.emt import EMT
        from numpy.random import default_rng

        from scgo.algorithms import ga_go

        captured: dict = {}
        sentinel = type("RelaxerProbe", (Exception,), {})

        class _SpyRelaxer:
            def __init__(self, calculator, **kwargs):
                captured.update(kwargs)
                raise sentinel()

        # Patch the name the GA module actually binds when building the relaxer.
        monkeypatch.setattr(
            "scgo.algorithms.geneticalgorithm_go_torchsim.AseBatchRelaxer",
            _SpyRelaxer,
        )

        with pytest.raises(sentinel):
            ga_go(
                ["Pt", "Pt"],
                output_dir=str(tmp_path / "relaxer_probe"),
                calculator=EMT(),
                rng=default_rng(42),
                system_type="surface_cluster",
                surface_config=surface_config_pt111,
                niter=1,
                population_size=2,
                niter_local_relaxation=5,
                batch_size=2,
                offspring_fraction=0.5,
                early_stopping_niter=0,
                n_jobs_population_init=1,
            )

        assert captured.get("surface_mode") is True
        assert captured.get("n_slab") == len(surface_config_pt111.slab)


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
