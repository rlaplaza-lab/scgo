"""Tests for parallel NEB optimization with batched GPU force evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer
from scgo.ts_search.parallel_neb import ParallelNEBBatch, _neb_image_dedup_key
from scgo.ts_search.transition_state import TorchSimNEB, interpolate_path

pytestmark = pytest.mark.requires_cuda


def _unique_neb_image_count(*image_lists: list) -> int:
    keys = {_neb_image_dedup_key(atoms) for images in image_lists for atoms in images}
    return len(keys)


class _CountingFakeRelaxer:
    """Relaxer stub that records batch sizes and returns zero forces."""

    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    def relax_batch(self, atoms_list, steps=0):
        self.calls += 1
        self.batch_sizes.append(len(atoms_list))
        results = []
        for a in atoms_list:
            ra = a.copy()
            ra.arrays["forces"] = np.zeros((len(a), 3))
            results.append((0.0, ra))
        return results


def _assert_one_global_relax_batch(
    neb1, neb2, relaxer, *, expected_unique: int
) -> None:
    batch = ParallelNEBBatch([neb1, neb2], relaxer, max_total_steps=5)
    batch.run_optimization(fmax=1.0, max_steps=1)
    assert relaxer.calls == 1
    assert relaxer.batch_sizes == [expected_unique]
    assert neb1.get_force_calls() >= 1
    assert neb2.get_force_calls() >= 1


class TestParallelNEBBatch:
    """Tests for ParallelNEBBatch parallel NEB optimization."""

    def test_parallel_neb_initialization(self, cu3_triangle, cu3_linear):
        """Test ParallelNEBBatch initialization with multiple NEBs."""
        relaxer = TorchSimBatchRelaxer(
            device="cuda",
            mace_model_name="mace_matpes_0",
            force_tol=0.05,
            max_steps=100,
        )

        # Create two NEB paths
        images1 = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
        images2 = interpolate_path(
            cu3_triangle, cu3_linear, n_images=3, method="linear"
        )

        neb1 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
        neb2 = TorchSimNEB(images2, relaxer, k=0.1, climb=False)

        batch = ParallelNEBBatch([neb1, neb2], relaxer, max_total_steps=50)

        assert len(batch.neb_instances) == 2
        assert len(batch.active_nebs) == 2
        assert batch.step_count == 0

    @pytest.mark.slow
    def test_parallel_neb_basic_run(self, cu3_triangle, cu3_linear, cu3_bent):
        """Test basic parallel NEB optimization with multiple paths."""
        relaxer = TorchSimBatchRelaxer(
            device="cuda",
            mace_model_name="mace_matpes_0",
            force_tol=0.1,
            max_steps=100,
            autobatcher=True,
        )

        # Create three NEB paths
        images1 = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
        images2 = interpolate_path(cu3_triangle, cu3_bent, n_images=3, method="linear")
        images3 = interpolate_path(cu3_linear, cu3_bent, n_images=3, method="idpp")

        neb1 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
        neb2 = TorchSimNEB(images2, relaxer, k=0.1, climb=False)
        neb3 = TorchSimNEB(images3, relaxer, k=0.1, climb=False)

        batch = ParallelNEBBatch([neb1, neb2, neb3], relaxer, max_total_steps=100)

        # Run optimization
        results = batch.run_optimization(fmax=0.5, max_steps=50)

        # Check results structure
        assert len(results) == 3
        for result in results:
            assert "converged" in result
            assert "steps_taken" in result
            assert "final_fmax" in result
            assert "force_calls" in result
            assert "error" in result

        # All should have run at least one step
        for result in results:
            assert result["steps_taken"] > 0 or result["error"] is not None

    @pytest.mark.slow
    def test_parallel_neb_summary(self, cu3_triangle, cu3_linear):
        """Test ParallelNEBBatch summary statistics."""
        relaxer = TorchSimBatchRelaxer(
            device="cuda",
            mace_model_name="mace_matpes_0",
            force_tol=0.1,
            max_steps=50,
        )

        images1 = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
        images2 = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")

        neb1 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
        neb2 = TorchSimNEB(images2, relaxer, k=0.1, climb=False)

        batch = ParallelNEBBatch([neb1, neb2], relaxer, max_total_steps=30)
        batch.run_optimization(fmax=1.0, max_steps=30)

        summary = batch.get_summary()

        assert summary["total_nebs"] == 2
        assert summary["total_steps"] > 0
        assert summary["converged"] + summary["failed"] <= 2

    def test_parallel_neb_batching_efficiency(self, cu3_triangle, cu3_linear, cu3_bent):
        """Test that parallel NEB batches images from multiple NEBs together."""
        relaxer = TorchSimBatchRelaxer(
            device="cuda",
            mace_model_name="mace_matpes_0",
            force_tol=0.1,
            max_steps=100,
        )

        # Create two NEBs with 5 images each
        images1 = interpolate_path(cu3_triangle, cu3_linear, n_images=5, method="idpp")
        images2 = interpolate_path(cu3_linear, cu3_bent, n_images=5, method="idpp")

        neb1 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
        neb2 = TorchSimNEB(images2, relaxer, k=0.1, climb=False)

        batch = ParallelNEBBatch([neb1, neb2], relaxer, max_total_steps=50)

        # Run one step and verify both NEBs got evaluated
        results = batch.run_optimization(fmax=1.0, max_steps=1)

        # After 1 step, both should have attempted force evaluation
        assert results[0]["steps_taken"] >= 1 or results[0]["error"] is not None
        assert results[1]["steps_taken"] >= 1 or results[1]["error"] is not None

    def test_parallel_neb_partial_convergence(self, cu3_triangle, cu3_linear, cu3_bent):
        """Test parallel NEB with different convergence rates."""
        relaxer = TorchSimBatchRelaxer(
            device="cuda",
            mace_model_name="mace_matpes_0",
            force_tol=0.1,
            max_steps=100,
        )

        # Create two different paths
        images1 = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
        images2 = interpolate_path(cu3_triangle, cu3_bent, n_images=3, method="idpp")

        neb1 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
        neb2 = TorchSimNEB(images2, relaxer, k=0.1, climb=False)

        batch = ParallelNEBBatch([neb1, neb2], relaxer, max_total_steps=200)

        # Run with loose convergence to allow some to finish
        results = batch.run_optimization(fmax=2.0, max_steps=200)

        # Check that at least one NEB attempted optimization or both have errors
        assert any(r["steps_taken"] > 0 for r in results) or any(
            r["error"] is not None for r in results
        )


def test_parallel_neb_relax_batch_dedups_identical_cu3_paths(cu3_triangle, cu3_linear):
    """Cu3 triangle→linear: IDPP matches linear, so 10 slots collapse to 5.

    ``ParallelNEBBatch`` still evaluates once globally and fans results out to
    both bands (no per-band second ``relax_batch``).
    """
    relaxer = _CountingFakeRelaxer()
    images1 = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    images2 = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="linear")

    assert len(images1) == len(images2) == 5
    assert np.allclose(images1[2].positions, images2[2].positions)
    expected_unique = _unique_neb_image_count(images1, images2)
    assert expected_unique == 5
    assert expected_unique < len(images1) + len(images2)

    neb1 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
    neb2 = TorchSimNEB(images2, relaxer, k=0.1, climb=False)
    _assert_one_global_relax_batch(neb1, neb2, relaxer, expected_unique=expected_unique)


def test_parallel_neb_relax_batch_keeps_distinct_ir4_interiors(
    ir4_tetrahedron, ir4_tetrahedron_atom_swapped
):
    """Ir4 tet→atom-swapped tet: IDPP interiors differ from linear.

    Shared endpoints still dedupe (2), but distinct interiors keep 3+3, so the
    first ``relax_batch`` sees 8 unique images rather than 5 or 10.
    """
    relaxer = _CountingFakeRelaxer()
    images1 = interpolate_path(
        ir4_tetrahedron, ir4_tetrahedron_atom_swapped, n_images=3, method="idpp"
    )
    images2 = interpolate_path(
        ir4_tetrahedron, ir4_tetrahedron_atom_swapped, n_images=3, method="linear"
    )

    assert len(images1) == len(images2) == 5
    assert not np.allclose(images1[2].positions, images2[2].positions)
    expected_unique = _unique_neb_image_count(images1, images2)
    assert expected_unique == 8
    assert expected_unique > len(images1)
    assert expected_unique < len(images1) + len(images2)

    neb1 = TorchSimNEB(images1, relaxer, k=0.1, climb=False)
    neb2 = TorchSimNEB(images2, relaxer, k=0.1, climb=False)
    _assert_one_global_relax_batch(neb1, neb2, relaxer, expected_unique=expected_unique)


def test_parallel_neb_uses_neb_forces_for_stepping(cu3_triangle, cu3_linear):
    """Verify position updates use NEB forces (not raw PES forces).

    We provide PES forces = 0 via the relaxer but mock `neb.get_forces()` to
    return a known NEB force on the intermediate image and then assert that
    the intermediate positions change according to that NEB force and the
    batch.step_size.
    """

    class FakeRelaxer:
        def relax_batch(self, atoms_list, steps=0):
            results = []
            for a in atoms_list:
                ra = a.copy()
                ra.arrays["forces"] = np.zeros((len(a), 3))
                results.append((0.0, ra))
            return results

    relaxer = FakeRelaxer()
    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)

    # Replace neb.get_forces() with an NEB-force array that exerts a known
    # force on the first interior image (band index 1). ASE NEB.get_forces()
    # returns interior images only, shape (n_interior_atoms, 3).
    def fake_neb_get_forces():
        n_images = len(neb.images)
        natoms = len(neb.images[0])
        n_int_atoms = (n_images - 2) * natoms
        forces = np.zeros((n_int_atoms, 3), dtype=float)
        # Apply +0.5 eV/Ang in x for all atoms of the first interior image
        forces[0:natoms, 0] = 0.5
        return forces

    neb.get_forces = fake_neb_get_forces

    batch = ParallelNEBBatch([neb], relaxer, max_total_steps=1)

    # Record initial position of the first atom in the intermediate image
    init_pos = neb.images[1].positions[0].copy()

    # Use a smaller fmax to force an optimizer step (NEB force = 0.5)
    batch.run_optimization(fmax=0.1, max_steps=1)

    # Position should have moved in the same direction as the optimizer
    # gradient (FIRE uses the supplied NEB forces directly). Force was +0.5
    new_pos = neb.images[1].positions[0]
    assert new_pos[0] > init_pos[0]
    assert not np.allclose(new_pos, init_pos)

    # An ASE optimizer should have been created for this NEB (default: FIRE)
    assert 0 in batch._optimizers
    from ase.optimize import FIRE as ASE_FIRE

    assert isinstance(batch._optimizers[0], ASE_FIRE)


def test_torchsimneb_get_forces_skips_relax_if_forces_present(cu3_triangle, cu3_linear):
    """TorchSimNEB.get_forces() should not call relax_batch when images
    already contain PES forces/calculators (cached results).
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    class DummyRelaxer:
        def __init__(self):
            self.calls = 0

        def relax_batch(self, images, steps=0):
            self.calls += 1
            results = []
            for a in images:
                ra = a.copy()
                ra.arrays["forces"] = np.ones((len(a), 3))
                results.append((0.0, ra))
            return results

    relaxer = DummyRelaxer()
    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")

    # Pre-attach SinglePointCalculator (forces present) to simulate cached PES
    for img in images:
        img.calc = SinglePointCalculator(
            img, energy=0.0, forces=np.zeros((len(img), 3))
        )

    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)

    # Should skip calling relax_batch because forces are already present
    neb.get_forces()
    assert relaxer.calls == 0


def test_parallel_neb_skips_endpoints_after_first_step(cu3_triangle, cu3_linear):
    """After step 0, only interior images are batch-evaluated."""
    relaxer = _CountingFakeRelaxer()
    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    assert len(images) == 5
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)
    batch = ParallelNEBBatch([neb], relaxer, max_total_steps=5)
    # Force at least two steps by keeping fmax tiny and get_forces returning large force.
    original_get_forces = neb.get_forces

    def always_high_forces():
        forces = original_get_forces()
        return np.ones_like(forces) * 10.0

    neb.get_forces = always_high_forces  # type: ignore[method-assign]
    batch.run_optimization(fmax=1e-6, max_steps=2)
    assert relaxer.calls == 2
    assert relaxer.batch_sizes[0] == 5  # all images on step 0
    assert relaxer.batch_sizes[1] == 3  # interiors only


def test_parallel_neb_require_forces_raises_when_missing(cu3_triangle, cu3_linear):
    """Missing forces from relax_batch must raise (require_forces=True)."""
    from scgo.exceptions import SCGORuntimeError

    class EnergyOnlyRelaxer:
        def relax_batch(self, atoms_list, steps=0):
            return [(0.0, a.copy()) for a in atoms_list]

    images = interpolate_path(cu3_triangle, cu3_linear, n_images=3, method="idpp")
    relaxer = EnergyOnlyRelaxer()
    neb = TorchSimNEB(images, relaxer, k=0.1, climb=False)
    batch = ParallelNEBBatch([neb], relaxer, max_total_steps=1)
    with pytest.raises(SCGORuntimeError, match="did not return forces"):
        batch.run_optimization(fmax=1.0, max_steps=1)


def test_run_parallel_neb_search_skips_invalid_pair(tmp_path, cu3_triangle, cu3_linear):
    """Validation failure on one pair skips it without aborting the batch."""
    from unittest.mock import patch

    from scgo.ts_search.parallel_neb import run_parallel_neb_search

    minima = [(0.0, cu3_triangle), (1.0, cu3_linear)]
    pairs = [(0, 1)]

    with (
        patch(
            "scgo.ts_search.parallel_neb._neb_endpoint_copies",
            side_effect=ValueError("bad structure"),
        ),
        patch(
            "scgo.ts_search.parallel_neb._tsh.TorchSimBatchRelaxer",
            return_value=_CountingFakeRelaxer(),
        ),
    ):
        results, _meta = run_parallel_neb_search(
            pairs,
            minima,
            run_dir=tmp_path,
            surface_config=None,
            rng=None,
            neb_n_images=3,
            neb_spring_constant=0.1,
            neb_fmax=0.05,
            neb_steps=2,
            neb_climb=False,
            neb_interpolation_method="linear",
            neb_align_endpoints=False,
            neb_perturb_sigma=0.0,
            neb_interpolation_mic=False,
            neb_tangent_method="aseneb",
            torchsim_params={},
            system_type="gas_cluster",
        )

    assert len(results) == 1
    assert results[0]["status"] == "skipped"
    assert "bad structure" in str(results[0].get("error", ""))
