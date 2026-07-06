"""TorchSim GA path with `surface_config` (MockRelaxer, no GPU required)."""

from __future__ import annotations

import pytest
from ase.calculators.emt import EMT

from scgo.algorithms import ga_go
from scgo.database import get_connection
from scgo.database.metadata import get_metadata


class PartiallyDisconnectingRelaxer:
    """Relaxer that disconnects exactly one adsorbate candidate across the whole run."""

    def __init__(self) -> None:
        self._disconnect_remaining = 1

    def relax_batch(self, batch):
        results = []
        for i, atoms in enumerate(batch):
            relaxed = atoms.copy()
            # Disconnect exactly one candidate in a multi-structure batch so at
            # least one relaxed structure remains GA-eligible for this test.
            if (
                self._disconnect_remaining > 0
                and len(batch) > 1
                and i == len(batch) - 1
                and len(relaxed) >= 2
            ):
                pos = relaxed.get_positions()
                # Push mobile atoms far from slab to trigger connectivity failure.
                pos[-2:, 2] += 10.0
                relaxed.set_positions(pos)
                self._disconnect_remaining -= 1
            results.append((float(i) * 0.1, relaxed))
        return results


def test_ga_go_disconnected_rows_persist_but_are_ineligible(
    surface_config_pt111, tmp_path, rng
):
    out = tmp_path / "surface_ga_torchsim_disconnected_ineligible"
    out.mkdir(parents=True, exist_ok=True)

    minima = ga_go(
        composition=["Pt", "Pt"],
        output_dir=str(out),
        calculator=EMT(),
        relaxer=PartiallyDisconnectingRelaxer(),
        niter=1,
        population_size=12,
        offspring_fraction=0.5,
        niter_local_relaxation=10,
        batch_size=2,
        verbosity=0,
        rng=rng,
        system_type="surface_cluster",
        surface_config=surface_config_pt111,
    )

    assert isinstance(minima, list)
    assert len(minima) >= 1
    for _energy, atoms in minima:
        assert bool(get_metadata(atoms, "ga_eligible", default=True))

    with get_connection(str(out / "ga_go.db")) as da:
        rows = da.get_all_relaxed_candidates()
    assert rows
    assert any(not bool(get_metadata(row, "ga_eligible", default=True)) for row in rows)


@pytest.mark.requires_mace
@pytest.mark.requires_cuda
@pytest.mark.slow
def test_ga_go_surface_config_mace_cuda(surface_config_pt111, tmp_path, rng):
    """Optional real GPU path: MACE + CUDA when available (conda scgo on a GPU box)."""
    from scgo.calculators.mace_helpers import MACE

    slab = surface_config_pt111.slab
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
        surface_config=surface_config_pt111,
    )

    assert isinstance(minima, list)
    assert len(minima) >= 1
    _e, best = minima[0]
    assert len(best) == len(slab) + 2
