"""TorchSim GA path with `surface_config` (MockRelaxer, no GPU required)."""

from __future__ import annotations

import numpy as np
import pytest
from ase.calculators.emt import EMT

from scgo.algorithms import ga_go
from scgo.calculators.mace_helpers import MACE
from scgo.database import get_connection
from scgo.database.metadata import get_metadata
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.deposition import slab_surface_extreme
from tests.test_utils import MockRelaxer


class PartiallyDisconnectingRelaxer:
    """Relaxer that disconnects exactly one adsorbate candidate across the whole run."""

    def __init__(self) -> None:
        self._disconnect_remaining = 1

    def relax_batch(self, batch):
        results = []
        for i, atoms in enumerate(batch):
            relaxed = atoms.copy()
            if self._disconnect_remaining > 0 and len(relaxed) >= 2:
                pos = relaxed.get_positions()
                # Push mobile atoms far from slab to trigger connectivity failure.
                pos[-2:, 2] += 10.0
                relaxed.set_positions(pos)
                self._disconnect_remaining -= 1
            results.append((float(i) * 0.1, relaxed))
        return results


def test_ga_go_surface_config_mock_relaxer(pt_slab_small, tmp_path, rng):
    """Exercise TorchSim batching + slab constraints without CUDA or MACE."""
    slab = pt_slab_small
    surface_config = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=2.8,
        fix_all_slab_atoms=True,
        comparator_use_mic=False,
        max_placement_attempts=400,
    )
    out = tmp_path / "surface_ga_torchsim"
    out.mkdir(parents=True, exist_ok=True)

    minima = ga_go(
        composition=["Pt", "Pt"],
        output_dir=str(out),
        calculator=EMT(),
        relaxer=MockRelaxer(max_steps=1),
        niter=1,
        population_size=3,
        offspring_fraction=0.5,
        niter_local_relaxation=20,
        batch_size=2,
        verbosity=0,
        rng=rng,
        system_type="surface_cluster",
        surface_config=surface_config,
    )

    assert isinstance(minima, list)
    assert len(minima) >= 1
    _e, best = minima[0]
    n_slab = len(slab)
    assert len(best) == n_slab + 2
    np.testing.assert_allclose(
        best.get_positions()[:n_slab].mean(axis=0),
        slab.get_positions().mean(axis=0),
        atol=1e-6,
    )
    z_top = slab_surface_extreme(slab, 2, upper=True)
    ads_z = best.get_positions()[n_slab:, 2]
    assert np.min(ads_z) > z_top - 0.2


def test_ga_go_disconnected_rows_persist_but_are_ineligible(
    pt_slab_small, tmp_path, rng
):
    slab = pt_slab_small
    surface_config = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=2.8,
        fix_all_slab_atoms=True,
        comparator_use_mic=False,
        max_placement_attempts=400,
    )
    out = tmp_path / "surface_ga_torchsim_disconnected_ineligible"
    out.mkdir(parents=True, exist_ok=True)

    minima = ga_go(
        composition=["Pt", "Pt"],
        output_dir=str(out),
        calculator=EMT(),
        relaxer=PartiallyDisconnectingRelaxer(),
        niter=1,
        population_size=4,
        offspring_fraction=0.5,
        niter_local_relaxation=10,
        batch_size=2,
        verbosity=0,
        rng=rng,
        system_type="surface_cluster",
        surface_config=surface_config,
    )

    assert isinstance(minima, list)
    assert len(minima) >= 1
    for _energy, atoms in minima:
        assert bool(get_metadata(atoms, "ga_eligible", default=True))

    with get_connection(str(out / "ga_go.db")) as da:
        rows = da.get_all_relaxed_candidates()
    assert rows
    assert any(not bool(get_metadata(row, "ga_eligible", default=True)) for row in rows)


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_ga_go_surface_config_mace_cuda(pt_slab_small, tmp_path, rng):
    """Optional real GPU path: MACE + CUDA when available (conda scgo on a GPU box)."""
    slab = pt_slab_small
    surface_config = SurfaceSystemConfig(
        slab=slab,
        adsorption_height_min=1.0,
        adsorption_height_max=2.8,
        fix_all_slab_atoms=True,
        comparator_use_mic=False,
        max_placement_attempts=400,
    )
    out = tmp_path / "surface_ga_torchsim_cuda"
    out.mkdir(parents=True, exist_ok=True)

    calc = MACE(model_name="small", device="cuda")
    minima = ga_go(
        composition=["Pt", "Pt"],
        output_dir=str(out),
        calculator=calc,
        niter=1,
        population_size=3,
        offspring_fraction=0.5,
        niter_local_relaxation=30,
        batch_size=2,
        verbosity=0,
        rng=rng,
        system_type="surface_cluster",
        surface_config=surface_config,
    )

    assert isinstance(minima, list)
    assert len(minima) >= 1
    _e, best = minima[0]
    assert len(best) == len(slab) + 2
