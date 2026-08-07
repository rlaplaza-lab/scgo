"""Parallel NEB batch runner that batches GPU force evaluations."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.optimize import FIRE

from scgo.calculators import torchsim_helpers as _tsh
from scgo.exceptions import SCGOValidationError
from scgo.utils.logging import get_logger
from scgo.utils.run_helpers import cleanup_torch_cuda
from scgo.utils.ts_runner_kwargs import NebRunConfig

from .neb_endpoints import prepare_neb_endpoints
from .transition_state import (
    TorchSimNEB,
    _detach_calc,
    _finalize_neb_result,
    attach_minima_traceability,
    attach_singlepoint_from_relax_output,
    evaluate_neb_image_energies,
    interpolate_path,
    make_ts_result,
    neb_max_atom_force,
    neb_uses_two_stage_climb,
    save_neb_result,
    validate_initial_neb_energy_profile,
    validate_initial_neb_path,
)

if TYPE_CHECKING:
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

logger = get_logger(__name__)


def _neb_image_dedup_key(atoms: Atoms) -> tuple:
    """Hashable key for deduplicating NEB images across bands."""
    return (
        tuple(atoms.get_chemical_symbols()),
        tuple(np.round(atoms.get_positions().ravel(), 6)),
    )


class ParallelNEBBatch:
    """Coordinate multiple TorchSimNEB instances and run batched evaluations."""

    def __init__(
        self,
        neb_instances: list[TorchSimNEB],
        relaxer: TorchSimBatchRelaxer,
        max_total_steps: int = 1000,
        optimizer: type = FIRE,
    ):
        """Initialize with NEBs, relaxer, max steps, and ASE optimizer (default FIRE)."""
        self.neb_instances = neb_instances
        self.relaxer = relaxer
        self.max_total_steps = max_total_steps
        self.optimizer_cls = optimizer

        self.active_nebs = list(range(len(neb_instances)))
        self.converged_nebs: dict[int, bool] = {}
        self.failed_nebs: dict[int, str] = {}
        self.step_count = 0

        # Per-NEB optimizer instances (created lazily). Uses ASE optimizers
        # (default: FIRE) so stepping respects NEB forces / spring terms.
        self._optimizers: dict[int, Any] = {}

    def run_optimization(  # noqa: C901
        self,
        fmax: float = 0.05,
        max_steps: int = 500,
    ) -> list[dict[str, Any]]:
        """Optimize NEBs using batched evaluations; return per-NEB summaries."""
        if not self.neb_instances:
            logger.error("No NEB instances provided to run_optimization")
            return []

        results: list[dict[str, Any]] = [
            {
                "converged": False,
                "steps_taken": 0,
                "final_fmax": None,
                "error": None,
                "force_calls": None,
            }
            for _ in self.neb_instances
        ]

        step_cap = min(self.max_total_steps, int(max_steps))
        while self.active_nebs and self.step_count < step_cap:
            unique_images: list[Atoms] = []
            unique_index: dict[tuple, int] = {}
            neb_image_map: list[tuple[int, int, int]] = []
            # After step 0, endpoints keep cached SinglePoint energy/forces.
            evaluate_endpoints = self.step_count == 0

            for neb_idx in self.active_nebs:
                neb = self.neb_instances[neb_idx]
                n_img = len(neb.images)
                for img_idx, atoms in enumerate(neb.images):
                    is_endpoint = img_idx == 0 or img_idx == n_img - 1
                    if is_endpoint and not evaluate_endpoints:
                        continue
                    key = _neb_image_dedup_key(atoms)
                    if key not in unique_index:
                        unique_index[key] = len(unique_images)
                        unique_images.append(atoms)
                    unique_slot = unique_index[key]
                    neb_image_map.append((neb_idx, img_idx, unique_slot))

            if not unique_images:
                break

            logger.debug(
                f"Step {self.step_count}: Evaluating {len(unique_images)} unique images "
                f"({len(neb_image_map)} total slots) from {len(self.active_nebs)} active NEBs"
                f"{'' if evaluate_endpoints else ' (interiors only)'}"
            )

            try:
                unique_results = self.relaxer.relax_batch(unique_images, steps=0)
            except (RuntimeError, ValueError) as e:
                kind = (
                    "Invalid input"
                    if isinstance(e, ValueError)
                    else "Batched force evaluation"
                )
                logger.error("%s failed: %s", kind, e)
                for neb_idx in self.active_nebs:
                    self.failed_nebs[neb_idx] = str(e)
                    results[neb_idx]["error"] = str(e)
                break

            for neb_idx in self.active_nebs:
                self.neb_instances[neb_idx]._force_calls += 1

            for neb_idx, img_idx, unique_slot in neb_image_map:
                energy, relaxed_atoms = unique_results[unique_slot]
                atoms = self.neb_instances[neb_idx].images[img_idx]
                attach_singlepoint_from_relax_output(
                    atoms, energy, relaxed_atoms, require_forces=True
                )

            still_active: list[int] = []
            for neb_idx in self.active_nebs:
                neb = self.neb_instances[neb_idx]
                try:
                    neb_forces = neb.get_forces()
                    max_force = neb_max_atom_force(neb_forces)

                    results[neb_idx]["final_fmax"] = max_force
                    results[neb_idx]["steps_taken"] = self.step_count + 1

                    if not np.isfinite(max_force):
                        msg = (
                            "NEB forces are non-finite "
                            f"(fmax={max_force!r}); refusing optimizer step"
                        )
                        self.failed_nebs[neb_idx] = msg
                        results[neb_idx]["error"] = msg
                        logger.debug("NEB %d step failed: %s", neb_idx, msg)
                    elif max_force < fmax:
                        results[neb_idx]["converged"] = True
                        self.converged_nebs[neb_idx] = True
                        logger.debug(
                            f"NEB {neb_idx} finished: converged, fmax={max_force:.6f}"
                        )
                    else:
                        if neb_idx not in self._optimizers:
                            self._optimizers[neb_idx] = self.optimizer_cls(
                                neb, logfile=None, trajectory=None
                            )
                        self._optimizers[neb_idx].step()
                        still_active.append(neb_idx)
                except (RuntimeError, ValueError) as e:
                    logger.debug("NEB %d step failed: %s", neb_idx, e)
                    self.failed_nebs[neb_idx] = str(e)
                    results[neb_idx]["error"] = str(e)

            self.active_nebs = still_active
            self.step_count += 1

            if not self.active_nebs:
                break

        for neb_idx in range(len(self.neb_instances)):
            if neb_idx not in self.converged_nebs and neb_idx not in self.failed_nebs:
                steps = results[neb_idx]["steps_taken"] or 0
                results[neb_idx]["error"] = (
                    f"NEB did not converge after {steps} steps"
                    if steps
                    else "NEB not processed"
                )

        # Final FIRE.step() invalidates SinglePoint caches on moved images.
        # Refresh PES at the final geometries so barrier finalize can read energies.
        self._refresh_pes_after_optimization()

        logger.info(
            f"Parallel NEB batch complete: {self.step_count} steps, "
            f"{len(self.converged_nebs)} converged, {len(self.failed_nebs)} failed"
        )

        return results

    def _refresh_pes_after_optimization(self) -> None:
        """Re-evaluate all NEB images at their final positions (no optimizer step)."""
        unique_images: list[Atoms] = []
        unique_index: dict[tuple, int] = {}
        neb_image_map: list[tuple[int, int, int]] = []

        for neb_idx, neb in enumerate(self.neb_instances):
            if neb_idx in self.failed_nebs:
                continue
            for img_idx, atoms in enumerate(neb.images):
                key = _neb_image_dedup_key(atoms)
                if key not in unique_index:
                    unique_index[key] = len(unique_images)
                    unique_images.append(atoms)
                neb_image_map.append((neb_idx, img_idx, unique_index[key]))

        if not unique_images:
            return

        try:
            unique_results = self.relaxer.relax_batch(unique_images, steps=0)
        except (RuntimeError, ValueError) as e:
            logger.warning(
                "Final NEB PES refresh failed (%s); finalize will use cached energies if present",
                e,
            )
            return

        for neb_idx, img_idx, unique_slot in neb_image_map:
            energy, relaxed_atoms = unique_results[unique_slot]
            atoms = self.neb_instances[neb_idx].images[img_idx]
            attach_singlepoint_from_relax_output(
                atoms, energy, relaxed_atoms, require_forces=True
            )

    def get_summary(self) -> dict[str, int]:
        """Return counts of total, converged and failed NEBs."""
        return {
            "total_nebs": len(self.neb_instances),
            "converged": len(self.converged_nebs),
            "failed": len(self.failed_nebs),
            "total_steps": self.step_count,
        }


def run_parallel_neb_search(  # noqa: C901
    pairs: list[tuple[int, int]],
    minima: list[tuple[float, Atoms]],
    *,
    neb_cfg: NebRunConfig,
    run_dir: Path,
    rng: np.random.Generator | None,
    parallel_neb_max_bands: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run all pairs through ParallelNEBBatch. Returns (results, timing meta).

    ``parallel_neb_max_bands`` limits how many bands share one force batch
    (``None`` = all). Surface presets pass ``1`` so large slab cells stay under
    GPU memory while still using the parallel NEB runner.
    """
    t_parallel0 = perf_counter()
    torchsim_params = neb_cfg.torchsim_params or {}
    relaxer = _tsh.TorchSimBatchRelaxer(**torchsim_params)
    neb_steps_i = int(neb_cfg.neb_steps)
    system_type = neb_cfg.system_type

    neb_instances: list[TorchSimNEB] = []
    # Parallel to neb_instances: (pair_index_in_results, i, j)
    neb_meta: list[tuple[int, int, int]] = []
    # Per-band two-stage flag (endpoint-max IDPP → climb from step 0).
    neb_two_stage: list[bool] = []
    pair_results: list[dict[str, Any] | None] = [None] * len(pairs)

    def _make_pair_ts_result(
        pair_id: str,
        *,
        react_e: float,
        prod_e: float,
        error: str | None = None,
    ) -> dict[str, Any]:
        return make_ts_result(
            pair_id=pair_id,
            n_images=neb_cfg.neb_n_images,
            spring_constant=neb_cfg.neb_spring_constant,
            use_torchsim=True,
            fmax=neb_cfg.neb_fmax,
            neb_steps=neb_cfg.neb_steps,
            interpolation_method=neb_cfg.neb_interpolation_method,
            climb=neb_cfg.neb_climb,
            align_endpoints=neb_cfg.neb_align_endpoints,
            perturb_sigma=neb_cfg.neb_perturb_sigma,
            neb_interpolation_mic=neb_cfg.neb_interpolation_mic,
            neb_tangent_method=neb_cfg.neb_tangent_method,
            use_parallel_neb=True,
            reactant_energy=react_e,
            product_energy=prod_e,
            error=error,
        )

    def _record_skipped_pair(
        pair_ord: int,
        pair_id: str,
        i: int,
        j: int,
        react_e: float,
        prod_e: float,
        error: str,
    ) -> None:
        skipped = _make_pair_ts_result(
            pair_id, react_e=react_e, prod_e=prod_e, error=error
        )
        skipped["status"] = "skipped"
        skipped["system_type"] = system_type
        attach_minima_traceability(skipped, minima, i, j)
        pair_dir = run_dir / f"pair_{pair_id}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        save_neb_result(skipped, str(pair_dir), pair_id)
        pair_results[pair_ord] = skipped

    for pair_ord, (i, j) in enumerate(pairs):
        pair_id = f"{i}_{j}"
        react_e = float(minima[i][0])
        prod_e = float(minima[j][0])
        try:
            react_ep, prod_ep = prepare_neb_endpoints(
                minima[i][1],
                minima[j][1],
                neb_cfg,
            )
        except (ValueError, SCGOValidationError) as e:
            logger.warning(
                "Skipping pair %s due to structure validation error: %s", pair_id, e
            )
            _record_skipped_pair(pair_ord, pair_id, i, j, react_e, prod_e, str(e))
            continue

        images = interpolate_path(
            react_ep,
            prod_ep,
            n_images=neb_cfg.neb_n_images,
            method=neb_cfg.neb_interpolation_method,
            mic=neb_cfg.neb_interpolation_mic,
            align_endpoints=neb_cfg.neb_align_endpoints,
            perturb_sigma=neb_cfg.neb_perturb_sigma,
            rng=rng,
            system_type=system_type,
            n_slab=neb_cfg.n_slab,
            n_core_mobile=neb_cfg.n_core_mobile,
            n_adsorbate_mobile=neb_cfg.n_adsorbate_mobile,
            adsorbate_fragment_lengths=neb_cfg.adsorbate_fragment_lengths,
            neb_surface_cell_remap=neb_cfg.neb_surface_cell_remap,
            neb_surface_lattice_rotation=neb_cfg.neb_surface_lattice_rotation,
            neb_surface_max_lattice_shift=neb_cfg.neb_surface_max_lattice_shift,
        )
        band_energies: list[float] | None = None
        try:
            validate_initial_neb_path(
                images,
                n_slab=neb_cfg.n_slab,
                mic=neb_cfg.neb_interpolation_mic,
                max_endpoint_mismatch=neb_cfg.max_endpoint_mismatch,
            )
            if neb_cfg.max_endpoint_mismatch is not None:
                band_energies = evaluate_neb_image_energies(images, relaxer)
                validate_initial_neb_energy_profile(
                    band_energies,
                    reference_reactant_energy=react_e,
                    reference_product_energy=prod_e,
                )
        except SCGOValidationError as e:
            logger.warning("Skipping pair %s: %s", pair_id, e)
            _record_skipped_pair(pair_ord, pair_id, i, j, react_e, prod_e, str(e))
            continue
        pair_two_stage = neb_uses_two_stage_climb(
            neb_cfg.neb_climb, neb_steps_i, initial_energies=band_energies
        )
        neb_instances.append(
            TorchSimNEB(
                images,
                relaxer,
                k=neb_cfg.neb_spring_constant,
                climb=bool(neb_cfg.neb_climb) and not pair_two_stage,
                method=neb_cfg.neb_tangent_method,
            )
        )
        neb_two_stage.append(pair_two_stage)
        if band_energies is not None:
            react_e = float(band_energies[0])
            prod_e = float(band_energies[-1])
        result = _make_pair_ts_result(pair_id, react_e=react_e, prod_e=prod_e)
        result["system_type"] = system_type
        pair_results[pair_ord] = result
        neb_meta.append((pair_ord, i, j))

    if neb_instances:
        t_batch0 = perf_counter()
        batch_results = [
            {
                "converged": False,
                "final_fmax": None,
                "steps_taken": 0,
                "error": None,
            }
            for _ in neb_instances
        ]
        band_cap = (
            int(parallel_neb_max_bands)
            if parallel_neb_max_bands is not None and int(parallel_neb_max_bands) > 0
            else len(neb_instances)
        )
        if band_cap < len(neb_instances):
            logger.info(
                "Parallel NEB concurrency capped at %d band(s) "
                "(%d total; avoids GPU OOM on large cells)",
                band_cap,
                len(neb_instances),
            )

        def _chunk_indices(indices: list[int]) -> list[list[int]]:
            if not indices:
                return []
            return [indices[i : i + band_cap] for i in range(0, len(indices), band_cap)]

        # Single-stage climb bands (typical endpoint-max IDPP adsorbate paths).
        single_idx = [i for i, ts in enumerate(neb_two_stage) if not ts]
        two_idx = [i for i, ts in enumerate(neb_two_stage) if ts]
        for chunk in _chunk_indices(single_idx):
            chunk_nebs = [neb_instances[i] for i in chunk]
            batch = ParallelNEBBatch(chunk_nebs, relaxer, max_total_steps=neb_steps_i)
            chunk_results = batch.run_optimization(
                fmax=neb_cfg.neb_fmax, max_steps=neb_steps_i
            )
            for local_i, neb_i in enumerate(chunk):
                batch_results[neb_i] = chunk_results[local_i]
            del batch
            cleanup_torch_cuda(logger=logger)
        for chunk in _chunk_indices(two_idx):
            # Interior-max IDPP: relax without climb, then climb (always).
            chunk_nebs = [neb_instances[i] for i in chunk]
            stage1_cap = neb_steps_i // 2
            batch = ParallelNEBBatch(chunk_nebs, relaxer, max_total_steps=stage1_cap)
            stage1_results = batch.run_optimization(
                fmax=neb_cfg.neb_fmax, max_steps=stage1_cap
            )
            del batch
            cleanup_torch_cuda(logger=logger)
            for neb in chunk_nebs:
                neb.climb = True
            climb_local = [
                i
                for i, summary in enumerate(stage1_results)
                if not summary.get("error")
            ]
            if climb_local:
                steps1_vals = [
                    int(stage1_results[i].get("steps_taken") or 0) for i in climb_local
                ]
                stage2_steps = max(
                    neb_steps_i // 2,
                    neb_steps_i - max(steps1_vals),
                    1,
                )
                stage2_nebs = [chunk_nebs[i] for i in climb_local]
                batch2 = ParallelNEBBatch(
                    stage2_nebs, relaxer, max_total_steps=stage2_steps
                )
                stage2_results = batch2.run_optimization(
                    fmax=neb_cfg.neb_fmax, max_steps=stage2_steps
                )
                del batch2
                cleanup_torch_cuda(logger=logger)
                for local_i, s1_i in enumerate(climb_local):
                    s2 = stage2_results[local_i]
                    s1 = stage1_results[s1_i]
                    steps1 = int(s1.get("steps_taken") or 0)
                    steps2 = int(s2.get("steps_taken") or 0)
                    stage1_results[s1_i] = {
                        "converged": bool(s2.get("converged", False)),
                        "final_fmax": s2.get("final_fmax", s1.get("final_fmax")),
                        "steps_taken": steps1 + steps2,
                        "error": s2.get("error") or s1.get("error"),
                    }
            for local_i, neb_i in enumerate(chunk):
                batch_results[neb_i] = stage1_results[local_i]
        neb_batch_s = perf_counter() - t_batch0
    else:
        batch_results = []
        neb_batch_s = 0.0

    wall_total = perf_counter() - t_parallel0
    n_active = max(1, len(neb_instances))
    neb_each = neb_batch_s / n_active
    wall_each = wall_total / max(1, len(pairs))

    for neb_idx, (pair_ord, i, j) in enumerate(neb_meta):
        neb = neb_instances[neb_idx]
        summary = batch_results[neb_idx]
        stored_result = pair_results[pair_ord]
        assert stored_result is not None
        result = stored_result
        result["neb_converged"] = bool(summary.get("converged", False))
        result["error"] = summary.get("error")
        result["final_fmax"] = summary.get("final_fmax")
        result["force_calls"] = neb.get_force_calls()
        result["steps_taken"] = summary.get("steps_taken")

        # Batch failures (e.g. CUDA OOM) leave only GO endpoint energies on the
        # band; finalize would overwrite the real error with endpoint-as-TS.
        batch_never_ran = (
            result.get("error")
            and (result.get("force_calls") or 0) == 0
            and not result.get("steps_taken")
        )
        if batch_never_ran:
            result["status"] = "failed"
            result["neb_converged"] = False
            logger.warning(
                "Parallel NEB batch failed before any steps for pair %s: %s",
                result.get("pair_id"),
                result.get("error"),
            )
        else:
            try:
                _finalize_neb_result(result, neb.images, logger=logger)
            except (RuntimeError, SCGOValidationError) as e:
                # SCGORuntimeError is a RuntimeError subclass; catch RuntimeError
                # (plus SCGOValidationError) so a missing-energy finalize cannot
                # abort the whole parallel batch.
                result["status"] = "failed"
                result["error"] = str(e)
                _detach_calc(result.get("transition_state"))

        if result["neb_converged"] and result.get("status") != "success":
            logger.warning(
                "Parallel NEB converged but no usable TS for pair %s; marking failed",
                result.get("pair_id"),
            )

        attach_minima_traceability(result, minima, i, j)
        pair_id = str(result["pair_id"])
        pair_dir = run_dir / f"pair_{pair_id}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        save_neb_result(result, str(pair_dir), pair_id)
        result["timings_s"] = {
            "total_wall_s": wall_each,
            "neb_optimization_s": neb_each,
            "cpu_non_relax_s": max(0.0, wall_each - neb_each),
        }

    meta = {
        "neb_batch_optimization_s": neb_batch_s,
        "parallel_wall_s": wall_total,
    }
    assert all(r is not None for r in pair_results)
    return pair_results, meta  # type: ignore[return-value]
