"""Parallel NEB batch runner that batches GPU force evaluations."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.optimize import FIRE

from scgo.calculators import torchsim_helpers as _tsh
from scgo.surface.config import SurfaceSystemConfig
from scgo.surface.constraints import attach_slab_constraints_from_surface_config
from scgo.system_types import (
    AdsorbateDefinition,
    SystemType,
    validate_structure_for_system_type,
)
from scgo.utils.helpers import extract_energy_from_atoms
from scgo.utils.logging import get_logger

from .transition_state import (
    TorchSimNEB,
    _detach_calc,
    _finalize_neb_result,
    attach_minima_traceability,
    attach_singlepoint_from_relax_output,
    interpolate_path,
    make_ts_result,
    neb_max_atom_force,
    save_neb_result,
)

if TYPE_CHECKING:
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

logger = get_logger(__name__)


def _neb_endpoint_energy(minimum: tuple[float, Atoms]) -> float:
    energy, atoms = minimum
    extracted = extract_energy_from_atoms(atoms)
    return float(extracted if extracted is not None else energy)


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
        self._optimizers: dict[int, object] = {}

    def run_optimization(
        self,
        fmax: float = 0.05,
        max_steps: int = 500,
    ) -> list[dict[str, Any]]:
        """Optimize NEBs using batched evaluations; return per-NEB summaries."""
        if not self.neb_instances:
            logger.error("No NEB instances provided to run_optimization")
            return []

        results = [
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

                    if max_force < fmax:
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

        logger.info(
            f"Parallel NEB batch complete: {self.step_count} steps, "
            f"{len(self.converged_nebs)} converged, {len(self.failed_nebs)} failed"
        )

        return results

    def get_summary(self) -> dict[str, int]:
        """Return counts of total, converged and failed NEBs."""
        return {
            "total_nebs": len(self.neb_instances),
            "converged": len(self.converged_nebs),
            "failed": len(self.failed_nebs),
            "total_steps": self.step_count,
        }


def _neb_endpoint_copies(
    atoms_i: Atoms,
    atoms_j: Atoms,
    surface_config: SurfaceSystemConfig | None,
    system_type: SystemType,
    n_slab: int = 0,
    adsorbate_definition: AdsorbateDefinition | None = None,
    connectivity_factor: float | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
) -> tuple[Atoms, Atoms]:
    """Copy minima endpoints, optionally re-attaching surface FixAtoms constraints."""
    react = atoms_i.copy()
    prod = atoms_j.copy()
    if surface_config is not None:
        attach_slab_constraints_from_surface_config(react, surface_config)
        attach_slab_constraints_from_surface_config(prod, surface_config)
    validate_structure_for_system_type(
        react,
        system_type=system_type,
        surface_config=surface_config,
        n_slab=n_slab,
        adsorbate_definition=adsorbate_definition,
        connectivity_factor=connectivity_factor,
        allow_cluster_fragmentation=allow_cluster_fragmentation,
        allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
        enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
    )
    validate_structure_for_system_type(
        prod,
        system_type=system_type,
        surface_config=surface_config,
        n_slab=n_slab,
        adsorbate_definition=adsorbate_definition,
        connectivity_factor=connectivity_factor,
        allow_cluster_fragmentation=allow_cluster_fragmentation,
        allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
        enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
    )
    return react, prod


def run_parallel_neb_search(
    pairs: list[tuple[int, int]],
    minima: list[tuple[float, Atoms]],
    *,
    run_dir: Path,
    surface_config: SurfaceSystemConfig | None,
    rng: np.random.Generator | None,
    neb_n_images: int,
    neb_spring_constant: float,
    neb_fmax: float,
    neb_steps: int,
    neb_climb: bool,
    neb_interpolation_method: str,
    neb_align_endpoints: bool,
    neb_perturb_sigma: float,
    neb_interpolation_mic: bool,
    neb_tangent_method: str,
    neb_surface_cell_remap: bool = True,
    neb_surface_lattice_rotation: bool = True,
    neb_surface_max_lattice_shift: int = 1,
    torchsim_params: dict[str, Any],
    system_type: SystemType,
    n_slab: int = 0,
    n_core_mobile: int | None = None,
    n_adsorbate_mobile: int | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
    connectivity_factor: float | None = None,
    allow_cluster_fragmentation: bool = False,
    allow_adsorbate_surface_detachment: bool = False,
    enforce_adsorbate_subgraph_integrity: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run all pairs through ParallelNEBBatch. Returns (results, timing meta)."""
    t_parallel0 = perf_counter()
    relaxer = _tsh.TorchSimBatchRelaxer(**(torchsim_params or {}))

    neb_instances: list[TorchSimNEB] = []
    # Parallel to neb_instances: (pair_index_in_results, i, j)
    neb_meta: list[tuple[int, int, int]] = []
    pair_results: list[dict[str, Any] | None] = [None] * len(pairs)

    for pair_ord, (i, j) in enumerate(pairs):
        pair_id = f"{i}_{j}"
        react_e = _neb_endpoint_energy(minima[i])
        prod_e = _neb_endpoint_energy(minima[j])
        try:
            react_ep, prod_ep = _neb_endpoint_copies(
                minima[i][1],
                minima[j][1],
                surface_config,
                system_type,
                n_slab=n_slab,
                adsorbate_definition=adsorbate_definition,
                connectivity_factor=connectivity_factor,
                allow_cluster_fragmentation=allow_cluster_fragmentation,
                allow_adsorbate_surface_detachment=allow_adsorbate_surface_detachment,
                enforce_adsorbate_subgraph_integrity=enforce_adsorbate_subgraph_integrity,
            )
        except ValueError as e:
            logger.warning(
                "Skipping pair %s due to structure validation error: %s", pair_id, e
            )
            skipped = make_ts_result(
                pair_id=pair_id,
                n_images=neb_n_images,
                spring_constant=neb_spring_constant,
                use_torchsim=True,
                fmax=neb_fmax,
                neb_steps=neb_steps,
                interpolation_method=neb_interpolation_method,
                climb=neb_climb,
                align_endpoints=neb_align_endpoints,
                perturb_sigma=neb_perturb_sigma,
                neb_interpolation_mic=neb_interpolation_mic,
                neb_tangent_method=neb_tangent_method,
                use_parallel_neb=True,
                reactant_energy=react_e,
                product_energy=prod_e,
                error=str(e),
            )
            skipped["status"] = "skipped"
            skipped["system_type"] = system_type
            attach_minima_traceability(skipped, minima, i, j)
            pair_dir = run_dir / f"pair_{pair_id}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            save_neb_result(skipped, str(pair_dir), pair_id)
            pair_results[pair_ord] = skipped
            continue

        images = interpolate_path(
            react_ep,
            prod_ep,
            n_images=neb_n_images,
            method=neb_interpolation_method,
            mic=neb_interpolation_mic,
            align_endpoints=neb_align_endpoints,
            perturb_sigma=neb_perturb_sigma,
            rng=rng,
            system_type=system_type,
            n_slab=n_slab,
            n_core_mobile=n_core_mobile,
            n_adsorbate_mobile=n_adsorbate_mobile,
            neb_surface_cell_remap=neb_surface_cell_remap,
            neb_surface_lattice_rotation=neb_surface_lattice_rotation,
            neb_surface_max_lattice_shift=neb_surface_max_lattice_shift,
        )
        neb_instances.append(
            TorchSimNEB(
                images,
                relaxer,
                k=neb_spring_constant,
                climb=neb_climb,
                method=neb_tangent_method,
            )
        )
        result = make_ts_result(
            pair_id=pair_id,
            n_images=neb_n_images,
            spring_constant=neb_spring_constant,
            use_torchsim=True,
            fmax=neb_fmax,
            neb_steps=neb_steps,
            interpolation_method=neb_interpolation_method,
            climb=neb_climb,
            align_endpoints=neb_align_endpoints,
            perturb_sigma=neb_perturb_sigma,
            neb_interpolation_mic=neb_interpolation_mic,
            neb_tangent_method=neb_tangent_method,
            use_parallel_neb=True,
            reactant_energy=react_e,
            product_energy=prod_e,
        )
        result["system_type"] = system_type
        pair_results[pair_ord] = result
        neb_meta.append((pair_ord, i, j))

    neb_steps_i = int(neb_steps)
    if neb_instances:
        batch = ParallelNEBBatch(neb_instances, relaxer, max_total_steps=neb_steps_i)
        t_batch0 = perf_counter()
        batch_results = batch.run_optimization(fmax=neb_fmax, max_steps=neb_steps_i)
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
        result = pair_results[pair_ord]
        assert result is not None
        result["neb_converged"] = bool(summary.get("converged", False))
        result["error"] = summary.get("error")
        result["final_fmax"] = summary.get("final_fmax")
        result["force_calls"] = neb.get_force_calls()
        result["steps_taken"] = summary.get("steps_taken")

        try:
            _finalize_neb_result(result, neb.images, logger=logger)
        except RuntimeError as e:
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
