"""Utilities for integrating TorchSim batched relaxations with SCGO.

This module wraps the TorchSim high-level optimization API so SCGO can relax
multiple candidate structures in a single batched call.

Important:
- Imports for optional stacks (TorchSim, MACE, FairChem) are **lazy** so SCGO can
  be imported in minimal environments without pulling MLIP dependencies.
- TorchSim can run with multiple model families. SCGO supports MACE, FairChem/UMA,
  and UPET/metatomic via TorchSim model wrappers.
"""

from __future__ import annotations

import functools
import json
import logging
import time
import warnings
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.constraints import FixAtoms as ASEFixAtoms

from scgo.database.metadata import update_metadata
from scgo.exceptions import (
    SCGORuntimeError,
    SCGOValidationError,
)
from scgo.utils.helpers import ensure_float64_forces
from scgo.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "MemoryScalerCache",
    "TorchSimBatchRelaxer",
    "build_torchsim_fixatoms_from_ase_batch",
    "collect_ase_fixatoms_indices",
    "get_global_memory_scaler_cache",
]


class MemoryScalerCache:
    """Disk-backed cache for TorchSim ``max_memory_scaler`` (GPU probing takes ~70s).

    Essential for performance: without caching, each first run in a cluster size
    forces expensive memory estimation via forward passes. Saves ~70s per campaign.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        cache_file: str = "memory_scaler_cache.json",
    ):
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "scgo" / "torchsim"
        self._cache_dir = Path(cache_dir)
        self._cache_path = self._cache_dir / cache_file
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load cache from disk if it exists."""
        if not self._cache_path.exists():
            return {}
        try:
            with open(self._cache_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load memory scaler cache: %s", exc)
            return {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)
        except OSError as exc:
            logger.warning("Failed to save memory scaler cache: %s", exc)

    def _make_key(
        self,
        n_atoms: int,
        model_name: str,
        memory_scales_with: str,
        device: str,
    ) -> str:
        """Create a cache key from parameters (n_atoms binned to nearest 5)."""
        atom_bin = ((n_atoms + 4) // 5) * 5
        return f"{model_name}|{memory_scales_with}|{device}|atoms_{atom_bin}"

    def get(
        self,
        n_atoms: int,
        model_name: str,
        memory_scales_with: str,
        device: str,
    ) -> float | None:
        """Get cached max_memory_scaler if available."""
        key = self._make_key(n_atoms, model_name, memory_scales_with, device)
        return self._cache.get(key)

    def set(
        self,
        n_atoms: int,
        model_name: str,
        memory_scales_with: str,
        device: str,
        value: float,
    ) -> None:
        """Cache a max_memory_scaler value to disk."""
        key = self._make_key(n_atoms, model_name, memory_scales_with, device)
        self._cache[key] = value
        self._save_cache()

    def delete(
        self,
        n_atoms: int,
        model_name: str,
        memory_scales_with: str,
        device: str,
    ) -> None:
        """Remove one cached scaler entry (e.g. when it is too tight for a batch)."""
        key = self._make_key(n_atoms, model_name, memory_scales_with, device)
        if key not in self._cache:
            return
        del self._cache[key]
        self._save_cache()

    def clear(self) -> None:
        """Clear the cache."""
        self._cache = {}
        if self._cache_path.exists():
            self._cache_path.unlink()


# Global cache instance shared across all TorchSimBatchRelaxer instances
_GLOBAL_MEMORY_SCALER_CACHE = MemoryScalerCache()


def collect_ase_fixatoms_indices(atoms: Atoms) -> list[int]:
    """Return sorted unique indices constrained by ASE :class:`ase.constraints.FixAtoms`.

    Other ASE constraint types are ignored (not represented in TorchSim today).
    """
    out: list[int] = []
    for c in atoms.constraints:
        if isinstance(c, ASEFixAtoms):
            out.extend(int(i) for i in c.index)
    return sorted(set(out))


def _patch_torchsim_constraint_device_mismatch() -> None:
    """Monkey-patch TorchSim ``IndexedConstraint.select_sub_constraint``.

    Upstream ``torch_sim.state._split_state`` builds a CPU ``atom_idx`` tensor
    while a GPU-backed ``FixAtoms`` keeps its indices on CUDA, triggering a
    ``RuntimeError`` inside ``torch.isin`` (device mismatch). The fix aligns
    ``atom_idx`` with ``self.atom_idx`` before the ``isin`` call.
    """
    from torch_sim.constraints import AtomConstraint  # type: ignore

    if getattr(AtomConstraint, "_scgo_device_patch", False):
        return

    def select_sub_constraint(self, atom_idx, sys_idx):  # noqa: ARG001
        if hasattr(atom_idx, "device") and atom_idx.device != self.atom_idx.device:
            atom_idx = atom_idx.to(self.atom_idx.device)
        mask = torch.isin(self.atom_idx, atom_idx)
        masked_indices = self.atom_idx[mask]
        new_atom_idx = masked_indices - atom_idx.min()
        if len(new_atom_idx) == 0:
            return None
        return type(self)(new_atom_idx)

    AtomConstraint.select_sub_constraint = select_sub_constraint
    AtomConstraint._scgo_device_patch = True


_TORCHSIM_WARNINGS_REGISTERED = False


def _register_torchsim_warning_filters() -> None:
    """Suppress known upstream TorchSim/warp warnings (not actionable in SCGO)."""
    global _TORCHSIM_WARNINGS_REGISTERED  # noqa: PLW0603
    if _TORCHSIM_WARNINGS_REGISTERED:
        return
    warnings.filterwarnings(
        "ignore",
        message=r"The \.grad attribute of a Tensor that is not a leaf Tensor",
        category=UserWarning,
        module=r"warp\._src\.torch",
    )
    _TORCHSIM_WARNINGS_REGISTERED = True


def build_torchsim_fixatoms_from_ase_batch(
    atoms_list: Sequence[Atoms],
    device: object,
) -> object | None:
    """Map per-structure ASE ``FixAtoms`` to one TorchSim ``FixAtoms`` (global indices).

    :func:`torch_sim.initialize_state` (and the lower-level
    :func:`torch_sim.io.atoms_to_state`) do not read ``atoms.constraints``, so
    SCGO builds the TorchSim constraint explicitly here and attaches it to the
    resulting ``SimState`` before calling ``ts.optimize``.

    Args:
        atoms_list: One or more ASE systems in batch order (same order as
            :func:`torch_sim.initialize_state`).
        device: ``torch.device`` for the index tensor (match compute device).

    Returns:
        A ``torch_sim.constraints.FixAtoms`` instance, or ``None`` if nothing to fix.
    """
    # Lazy import: do not require TorchSim until needed.
    from torch_sim.constraints import FixAtoms as TSFixAtoms  # type: ignore

    _patch_torchsim_constraint_device_mismatch()

    merged: list[int] = []
    offset = 0
    for atoms in atoms_list:
        merged.extend(offset + idx for idx in collect_ase_fixatoms_indices(atoms))
        offset += len(atoms)
    if not merged:
        return None
    idx_t = torch.tensor(merged, device=device, dtype=torch.long)
    return TSFixAtoms(atom_idx=idx_t)


def _load_default_mace_model(
    *,
    device,
    dtype,
    mace_model_name: str = "mace_matpes_0",
    compute_forces: bool = True,
    compute_stress: bool = False,
):
    """Create a TorchSim MACE model given a canonical model identifier."""
    from scgo.utils.mlip_extras import clear_torch_force_no_weights_only_load_env

    clear_torch_force_no_weights_only_load_env()
    # Lazy imports: only required for the MACE TorchSim path.
    from mace.calculators.foundations_models import mace_mp  # type: ignore
    from torch_sim.models.mace import MaceModel  # type: ignore

    from scgo.calculators.mace_helpers import (
        MaceUrls,
        _ensure_torch_load_mace_checkpoints,
    )

    _ensure_torch_load_mace_checkpoints()
    model_selector = getattr(MaceUrls, mace_model_name, mace_model_name)
    raw_model = mace_mp(
        model=model_selector,
        return_raw_model=True,
        default_dtype=str(dtype).removeprefix("torch."),
        device=device,
    )
    return MaceModel(
        model=raw_model,
        device=device,
        dtype=dtype,
        compute_forces=compute_forces,
        compute_stress=compute_stress,
    )


def _ensure_torchsim_mace_wrapper(
    model: object, device: object, dtype: object
) -> object:
    """Wrap a raw ASE/MACE ``ScaleShiftMACE`` for :func:`torch_sim.optimize``.

    ``ga_go`` reuses the calculator's loaded weights via
    :func:`try_extract_torchsim_model_from_mace_calculator`, which returns the
    inner torch module. TorchSim expects a model exposing ``.device`` and
    ``.dtype`` (e.g. :class:`torch_sim.models.mace.MaceModel`).
    """
    if hasattr(model, "device") and hasattr(model, "dtype"):
        return model
    mod = getattr(type(model), "__module__", "") or ""
    name = type(model).__name__
    if "mace" not in mod.lower() and "MACE" not in name:
        return model
    from torch_sim.models.mace import MaceModel  # type: ignore

    return MaceModel(
        model=model,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
    )


def _load_default_fairchem_model(
    *,
    device,
    dtype,
    fairchem_model_name: str,
    fairchem_task_name: str | None,
    compute_stress: bool = False,
):
    """Create a TorchSim FairChem model for UMA checkpoints."""
    from torch_sim.models.fairchem import FairChemModel  # type: ignore

    return FairChemModel(
        model=fairchem_model_name,
        task_name=fairchem_task_name,
        device=device,
        dtype=dtype,
        compute_stress=compute_stress,
    )


def _parse_upet_model_and_size(model_name: str) -> tuple[str, str]:
    """Split a UPET model id (e.g. ``pet-mad-s``) into base model and size."""
    if "-" not in model_name:
        raise SCGOValidationError(
            f"Invalid UPET model_name {model_name!r}; expected form 'pet-mad-s'."
        )
    model, size = model_name.rsplit("-", 1)
    return model, size


def _prepare_atoms_for_metatomic_torchsim(atoms: Atoms) -> Atoms:
    """Return a copy safe for metatomic/vesin when PBC is disabled.

    Metatomic neighbor lists require zero cell vectors along non-periodic
    directions (gas-phase clusters still use a finite ASE box for spacing).
    """
    prepared = atoms.copy()
    if not any(prepared.pbc):
        prepared.cell[:] = 0.0
    return prepared


def _restore_ase_cell_from_reference(relaxed: Atoms, reference: Atoms) -> None:
    """Restore SCGO storage cell/PBC after a metatomic TorchSim relaxation."""
    relaxed.cell = reference.cell.copy()
    relaxed.pbc = reference.pbc


def _load_default_upet_model(
    *,
    device,
    dtype,
    upet_model_name: str,
    upet_version: str | None,
    upet_checkpoint_path: str | None = None,
    upet_non_conservative: bool = False,
    compute_stress: bool = False,
):
    """Create a TorchSim MetatomicModel for UPET checkpoints."""
    import metatomic_torchsim._neighbors as _mt_neighbors  # type: ignore
    from metatomic_torchsim import MetatomicModel  # type: ignore
    from upet import get_upet

    # nvalchemiops CUDA NL can fail for non-cubic gas-phase cells (float max_neighbors
    # in metatomic-torchsim); vesin handles these systems reliably.
    _mt_neighbors.HAS_NVALCHEMIOPS = False

    if upet_checkpoint_path:
        atomistic = get_upet(checkpoint_path=upet_checkpoint_path)
    else:
        model, size = _parse_upet_model_and_size(upet_model_name)
        atomistic = get_upet(
            model=model,
            size=size,
            version=upet_version or "latest",
        )

    return MetatomicModel(
        atomistic,
        device=device,
        non_conservative=upet_non_conservative,
        compute_stress=compute_stress,
    )


def _steps_taken_from_optimize_state(state: Any) -> int | None:
    """Best-effort extraction of optimizer steps from a TorchSim state object."""
    for attr in ("n_steps", "step", "steps"):
        val = getattr(state, attr, None)
        if val is None:
            continue
        try:
            if hasattr(val, "detach"):
                val = val.detach().cpu()
            if hasattr(val, "max"):
                return int(val.max().item())
            if hasattr(val, "__len__") and len(val) > 0:
                return int(max(int(v) for v in val))
            return int(val)
        except (TypeError, ValueError):
            continue
    return None


@dataclass(eq=False)
class TorchSimBatchRelaxer:
    """Batched relaxer that offloads geometry optimization to :func:`torch_sim.optimize`.

    ASE :class:`ase.constraints.FixAtoms` on input structures are translated to
    TorchSim's internal ``FixAtoms`` before optimization, since
    :func:`torch_sim.initialize_state` does not import ``atoms.constraints``.

    Parameters
    ----------
    device:
        Optional torch device. Defaults to CUDA when available, otherwise CPU.
    dtype:
        Torch dtype. Defaults to ``torch.float64`` for parity with the ASE MACE
        wrapper; override to ``torch.float32`` for speed at the cost of accuracy.
    model:
        Optional TorchSim model implementing ``ModelInterface``. If omitted, a
        MACE foundation model specified by ``mace_model_name`` is loaded.
    mace_model_name:
        Name of the TorchSim ``MaceUrls`` member to load when ``model`` is not
        provided (default: ``"mace_matpes_0"``).
    optimizer_name:
        Name of TorchSim optimizer (e.g., "fire"), resolved to ``ts.Optimizer.*``.
    force_tol:
        Force convergence threshold (eV/Å) passed to
        :func:`torch_sim.generate_force_convergence_fn`. ``None`` uses the
        torch-sim default energy-based convergence.
    autobatcher:
        Whether to use :class:`torch_sim.InFlightAutoBatcher` when calling
        :func:`torch_sim.optimize`. ``None`` (the default) enables it on CUDA and
        disables it on CPU, matching the torch-sim recommendation that
        autobatching is "generally not supported on CPUs". ``True``/``False``
        force the choice; passing ``True`` on CPU triggers a one-time warning
        and coerces back to ``False``. Only :class:`InFlightAutoBatcher` is
        accepted by :func:`torch_sim.optimize`.
    memory_scales_with, max_memory_scaler:
        Advanced knobs forwarded to :class:`torch_sim.InFlightAutoBatcher` when
        autobatching is active.
    expected_max_atoms:
        Optional atom count (e.g. ``cluster_size * population_size``) used both
        to warm the on-disk memory-scaler cache at init time and to cap the
        autobatcher's GPU probe via ``max_atoms_to_try`` (see
        :class:`torch_sim.InFlightAutoBatcher`). Without this cap, the probe
        can geometrically climb to its 500k-atom default and OOM small GPUs.
        Recommended for GA/BH campaigns with known population sizes.
    max_atoms_to_try:
        Explicit override for the autobatcher's probe cap. Defaults to
        ``expected_max_atoms`` when that is set; otherwise falls back to
        torch-sim's default (500,000). Always pass a tight value on GPUs
        with limited memory.
    init_kwargs:
        Extra kwargs forwarded to the torch-sim optimizer init function via
        the ``init_kwargs`` argument of :func:`torch_sim.optimize`.
    optimizer_kwargs:
        Extra kwargs forwarded to the torch-sim optimizer step function via
        ``**optimizer_kwargs`` of :func:`torch_sim.optimize`.
    runner_kwargs:
        Extra keyword arguments forwarded directly to
        :func:`torch_sim.optimize` (overrides anything set above).

    """

    device: object | None = None
    dtype: object | None = None
    model: object | None = None
    model_kind: str = "mace"  # "mace", "fairchem", or "upet"
    mace_model_name: str = "mace_matpes_0"
    fairchem_model_name: str | None = None
    fairchem_task_name: str | None = None
    upet_model_name: str | None = None
    upet_version: str | None = None
    upet_checkpoint_path: str | None = None
    upet_non_conservative: bool = False
    optimizer_name: str = "fire"
    force_tol: float | None = 0.05
    max_steps: int | None = 100
    # Autobatching: None -> enable on CUDA / disable on CPU (matches docs).
    autobatcher: bool | None = None
    memory_scales_with: str = "n_atoms_x_density"
    max_memory_scaler: float | None = None
    max_memory_padding: float = 1.05
    expected_max_atoms: int | None = (
        None  # Probe memory upfront with this atom count (cluster_size * pop_size)
    )
    # Hard cap on the InFlightAutoBatcher GPU probe. None -> fall back to
    # expected_max_atoms (if set) or torch-sim's 500k default.
    max_atoms_to_try: int | None = None
    init_kwargs: dict | None = None
    optimizer_kwargs: dict | None = None  # forwarded as **optimizer_kwargs to step-fn
    runner_kwargs: dict | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self._torch = torch
        # Lazy import: only require TorchSim when actually instantiating the relaxer.
        import torch_sim as ts  # type: ignore

        _register_torchsim_warning_filters()
        self._ts = ts
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.dtype is None:
            # Match ASE MACE wrapper default of float64 for parity
            self.dtype = torch.float64

        # Optional seeding (do not force deterministic algorithms to avoid CuBLAS constraints)
        if self.seed is not None:
            torch.manual_seed(self.seed)

        if isinstance(self.optimizer_name, str):
            try:
                self.optimizer = getattr(ts.Optimizer, self.optimizer_name.lower())
            except AttributeError as exc:
                available = [x for x in dir(ts.Optimizer) if not x.startswith("_")]
                raise SCGOValidationError(
                    f"Unknown TorchSim optimizer '{self.optimizer_name}'. "
                    f"Available: {available}",
                ) from exc
        else:
            self.optimizer = self.optimizer_name

        if self.model is None:
            mk = str(self.model_kind or "mace").strip().lower()
            if mk == "mace":
                self.model = _load_default_mace_model(
                    device=self.device,
                    dtype=self.dtype,
                    mace_model_name=self.mace_model_name,
                )
            elif mk in ("fairchem", "uma"):
                if not self.fairchem_model_name:
                    raise SCGOValidationError(
                        "TorchSimBatchRelaxer(model_kind='fairchem') requires fairchem_model_name"
                    )
                self.model = _load_default_fairchem_model(
                    device=self.device,
                    dtype=self.dtype,
                    fairchem_model_name=str(self.fairchem_model_name),
                    fairchem_task_name=self.fairchem_task_name,
                )
            elif mk in ("upet", "metatomic"):
                if not self.upet_model_name and not self.upet_checkpoint_path:
                    raise SCGOValidationError(
                        "TorchSimBatchRelaxer(model_kind='upet') requires "
                        "upet_model_name or upet_checkpoint_path"
                    )
                self.model = _load_default_upet_model(
                    device=self.device,
                    dtype=self.dtype,
                    upet_model_name=str(self.upet_model_name or ""),
                    upet_version=self.upet_version,
                    upet_checkpoint_path=self.upet_checkpoint_path,
                    upet_non_conservative=self.upet_non_conservative,
                )
            else:
                raise SCGOValidationError(
                    f"Unknown model_kind {self.model_kind!r}; "
                    "expected 'mace', 'fairchem', or 'upet'"
                )
        else:
            self.model = _ensure_torchsim_mace_wrapper(
                self.model, self.device, self.dtype
            )
        self._patch_model_for_cuda()

        # Store device string for cache key (e.g., "cuda" or "cpu")
        self._device_str = str(self.device).split(":")[0]
        self.last_batch_relax_steps: list[int] = []

        self._runner_kwargs = dict(self.runner_kwargs or {})

        # Resolve autobatcher policy: only InFlightAutoBatcher is accepted by
        # ts.optimize; on CPU torch-sim recommends disabling it altogether.
        on_cpu = str(self.device).split(":")[0] == "cpu"
        if self.autobatcher is None:
            use_autobatcher = not on_cpu
        else:
            use_autobatcher = bool(self.autobatcher)
            if use_autobatcher and on_cpu:
                logger.warning(
                    "TorchSim autobatching is not supported on CPU; disabling "
                    "the autobatcher. Pass autobatcher=False to avoid this warning."
                )
                use_autobatcher = False
        if use_autobatcher and "autobatcher" not in self._runner_kwargs:
            # Cap the autobatcher's probe at the actual workload so small GPUs
            # don't get pushed toward the 500k-atom default. Prefer the explicit
            # knob, then expected_max_atoms; leave unset to inherit torch-sim's
            # default when the caller can't give us a bound.
            probe_cap = self.max_atoms_to_try
            if probe_cap is None and self.expected_max_atoms is not None:
                probe_cap = int(self.expected_max_atoms)
            autobatcher_kwargs: dict = {
                "model": self.model,
                "memory_scales_with": self.memory_scales_with,
                "max_memory_scaler": self.max_memory_scaler,
                "max_memory_padding": self.max_memory_padding,
            }
            if probe_cap is not None:
                autobatcher_kwargs["max_atoms_to_try"] = probe_cap
            self._runner_kwargs["autobatcher"] = self._ts.InFlightAutoBatcher(
                **autobatcher_kwargs
            )

        if self.init_kwargs and "init_kwargs" not in self._runner_kwargs:
            self._runner_kwargs["init_kwargs"] = dict(self.init_kwargs)
        # ts.optimize forwards **optimizer_kwargs to the step function; flatten them in.
        if self.optimizer_kwargs:
            for key, value in self.optimizer_kwargs.items():
                self._runner_kwargs.setdefault(key, value)
        if self.force_tol is not None and "convergence_fn" not in self._runner_kwargs:
            self._runner_kwargs["convergence_fn"] = ts.generate_force_convergence_fn(
                force_tol=self.force_tol,
                include_cell_forces=False,
            )
        # Cap iterations; default 100 matches ASE GA niter_local_relaxation default
        if "max_steps" not in self._runner_kwargs and self.max_steps is not None:
            self._runner_kwargs["max_steps"] = self.max_steps

        # Probe memory upfront if expected_max_atoms provided (avoids runtime probing cost)
        if self.expected_max_atoms is not None and self.max_memory_scaler is None:
            self._warm_autobatcher_memory_scaler(self.expected_max_atoms)

    def _memory_scaler_cache_key(self, n_atoms: int) -> dict[str, Any]:
        return {
            "n_atoms": n_atoms,
            "model_name": self._cache_model_name(),
            "memory_scales_with": self.memory_scales_with,
            "device": self._device_str,
        }

    def _get_cached_memory_scaler(self, n_atoms: int) -> float | None:
        return _GLOBAL_MEMORY_SCALER_CACHE.get(**self._memory_scaler_cache_key(n_atoms))

    def _apply_cached_memory_scaler(self, n_atoms: int) -> bool:
        cached_scaler = self._get_cached_memory_scaler(n_atoms)
        if cached_scaler is None:
            return False
        autobatcher = self._runner_kwargs.get("autobatcher")
        if autobatcher is None:
            return False
        autobatcher.max_memory_scaler = cached_scaler
        return True

    def _invalidate_memory_scaler_cache(self, n_atoms: int) -> None:
        _GLOBAL_MEMORY_SCALER_CACHE.delete(**self._memory_scaler_cache_key(n_atoms))

    def _recreate_autobatcher(self) -> None:
        if "autobatcher" not in self._runner_kwargs:
            return
        probe_cap = self.max_atoms_to_try
        if probe_cap is None and self.expected_max_atoms is not None:
            probe_cap = int(self.expected_max_atoms)
        autobatcher_kwargs: dict = {
            "model": self.model,
            "memory_scales_with": self.memory_scales_with,
            "max_memory_scaler": self.max_memory_scaler,
            "max_memory_padding": self.max_memory_padding,
        }
        if probe_cap is not None:
            autobatcher_kwargs["max_atoms_to_try"] = probe_cap
        self._runner_kwargs["autobatcher"] = self._ts.InFlightAutoBatcher(
            **autobatcher_kwargs
        )

    def _reset_autobatcher_memory_scaler(self) -> None:
        self.max_memory_scaler = None
        self._recreate_autobatcher()

    @staticmethod
    def _is_max_metric_value_error(exc: BaseException) -> bool:
        return isinstance(exc, ValueError) and "max_metric" in str(exc)

    def _persist_autobatcher_scaler(self, n_atoms: int) -> None:
        """Persist the current autobatcher's ``max_memory_scaler`` to the disk cache.

        No-op when the autobatcher is not active or has not produced a scaler.
        """
        autobatcher = self._runner_kwargs.get("autobatcher")
        if autobatcher is None:
            return
        scaler = getattr(autobatcher, "max_memory_scaler", None)
        if not scaler:
            return
        _GLOBAL_MEMORY_SCALER_CACHE.set(
            value=float(scaler),
            **self._memory_scaler_cache_key(n_atoms),
        )

    def _warm_autobatcher_memory_scaler(self, n_atoms: int) -> None:
        """Pre-populate the InFlight autobatcher's ``max_memory_scaler``.

        Uses the on-disk cache when present; otherwise runs a one-step dummy
        optimization so torch-sim's autobatcher probes GPU memory, then stores
        the resulting scaler on disk so subsequent processes skip probing.

        No-ops when the autobatcher is not active (e.g. CPU runs) or when the
        user already supplied ``max_memory_scaler``.
        """
        autobatcher = self._runner_kwargs.get("autobatcher")
        if autobatcher is None or self.max_memory_scaler is not None:
            return

        if self._apply_cached_memory_scaler(n_atoms):
            logger.info("Used cached memory scaler from disk (avoided probing)")
            return

        try:
            # Build a dummy system of the requested size to trigger torch-sim's
            # memory estimation (see autobatching tutorial).
            dummy = bulk("Cu", "fcc", a=3.61, cubic=True)
            while len(dummy) < n_atoms:
                dummy = dummy.repeat((2, 2, 2))
            dummy = dummy[:n_atoms]
            dummy.center(vacuum=3.0)

            logger.info(
                f"Probing GPU memory with {n_atoms} atoms (cluster_size * population)..."
            )
            initial_time = time.time()
            _ = self._ts.optimize(
                system=[dummy],
                model=self.model,
                optimizer=self.optimizer,
                max_steps=1,
                **{k: v for k, v in self._runner_kwargs.items() if k != "max_steps"},
            )
            probe_time = time.time() - initial_time

            if getattr(autobatcher, "max_memory_scaler", None):
                self._persist_autobatcher_scaler(n_atoms)
                logger.info(
                    f"Memory probing complete ({probe_time:.2f}s). "
                    f"Scaler cached for {n_atoms} atoms."
                )
        except (
            RuntimeError,
            ValueError,
            OSError,
            AttributeError,
            torch.cuda.OutOfMemoryError,
        ) as e:
            self._reset_autobatcher_memory_scaler()
            logger.warning(
                f"Memory probing failed (non-fatal): {e}. Will retry on first relax_batch()."
            )

    def relax_batch(
        self, atoms_list: Sequence[Atoms], steps: int | None = None
    ) -> list[tuple[float, Atoms]]:
        """Relax a batch of ASE ``Atoms`` objects using TorchSim.

        Args:
            atoms_list: List of Atoms objects to relax.
            steps: Optional override for max_steps. Set to 0 for single-point calculation.

        Returns:
            A list of ``(energy, atoms)`` with matching order to the input
        list. Energies are converted to Python floats in eV.
        """
        if not atoms_list:
            return []

        max_atoms_in_batch = max(len(atoms) for atoms in atoms_list)
        for attempt in range(2):
            try:
                return self._relax_batch_once(
                    atoms_list, steps=steps, max_atoms_in_batch=max_atoms_in_batch
                )
            except ValueError as exc:
                if attempt == 0 and self._is_max_metric_value_error(exc):
                    logger.warning(
                        "Cached or probed max_memory_scaler too tight (%s); "
                        "invalidating cache and re-estimating.",
                        exc,
                    )
                    self._invalidate_memory_scaler_cache(max_atoms_in_batch)
                    self._reset_autobatcher_memory_scaler()
                    continue
                raise
        raise SCGORuntimeError("relax_batch retry loop exited without returning")

    def _relax_batch_once(
        self,
        atoms_list: Sequence[Atoms],
        *,
        steps: int | None,
        max_atoms_in_batch: int,
    ) -> list[tuple[float, Atoms]]:
        # Try to apply cached memory scaler to avoid expensive re-probing (~70s per new cluster size)
        if self.max_memory_scaler is None and "autobatcher" in self._runner_kwargs:
            self._apply_cached_memory_scaler(max_atoms_in_batch)

        runner_kwargs = self._runner_kwargs.copy()
        if steps is not None:
            runner_kwargs["max_steps"] = steps

        atoms_seq = list(atoms_list)
        reference_atoms = list(atoms_list)
        if self._uses_metatomic_model():
            atoms_seq = [
                _prepare_atoms_for_metatomic_torchsim(atoms) for atoms in atoms_seq
            ]

        # torch_sim.initialize_state ignores ASE constraints; map FixAtoms -> TorchSim.
        ts_fix = build_torchsim_fixatoms_from_ase_batch(atoms_seq, self.device)
        if ts_fix is not None:
            system_in = self._ts.initialize_state(
                atoms_seq,
                self.device,
                self.dtype,
            )
            system_in.constraints = ts_fix
        else:
            system_in = atoms_seq

        # `steps=0` is our single-point mode (endpoint energies and batched force
        # evaluations in NEB/TS paths). We intentionally stay on ts.optimize rather
        # than ts.static because we need the final SimState with positions/forces;
        # ts.static returns property dicts only.
        max_steps_now = runner_kwargs.get("max_steps", self.max_steps)
        if max_steps_now == 0:
            logger.debug(
                "Running TorchSim single-point evaluation via optimize(max_steps=0)."
            )
        optimize_kwargs = runner_kwargs
        with warnings.catch_warnings() if max_steps_now == 0 else nullcontext():
            if max_steps_now == 0:
                warnings.filterwarnings(
                    "ignore",
                    message="All systems have reached the maximum number of steps",
                )
            state = self._ts.optimize(  # type: ignore[call-arg]
                system=system_in,
                model=self.model,
                optimizer=self.optimizer,
                **optimize_kwargs,
            )

        # Cache the memory scaler if we computed a new estimate (avoid ~70s re-probing)
        if self.max_memory_scaler is None:
            self._persist_autobatcher_scaler(max_atoms_in_batch)

        batch_steps = _steps_taken_from_optimize_state(state)
        self.last_batch_relax_steps = (
            [batch_steps] * len(atoms_list) if batch_steps is not None else []
        )
        if batch_steps is not None and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "TorchSim relax_batch: %d structures, steps_taken=%d (max_steps=%s)",
                len(atoms_list),
                batch_steps,
                runner_kwargs.get("max_steps", self.max_steps),
            )

        energies_tensor = getattr(state, "energy", None)
        if energies_tensor is None:
            raise SCGORuntimeError(
                "TorchSim optimize did not return energy information"
            )

        energies = [float(val) for val in energies_tensor.detach().cpu().tolist()]

        forces_tensor = getattr(state, "forces", None)
        forces_list = None
        if forces_tensor is not None:
            forces_np = forces_tensor.detach().cpu().numpy()  # Shape: (total_atoms, 3)

        relaxed_atoms = state.to_atoms()
        if len(relaxed_atoms) != len(energies):
            raise SCGORuntimeError(
                "TorchSim returned mismatched counts for atoms and energies",
            )

        # Split forces by number of atoms per structure
        if forces_tensor is not None:
            forces_list = []
            offset = 0
            for atoms in relaxed_atoms:
                n_atoms = len(atoms)
                struct_forces = forces_np[
                    offset : offset + n_atoms
                ]  # Shape: (n_atoms, 3)
                forces_list.append(struct_forces)
                offset += n_atoms

            if offset != forces_np.shape[0]:
                raise SCGORuntimeError(
                    f"Forces shape mismatch: expected {offset} total atoms, "
                    f"got {forces_np.shape[0]} forces"
                )

        results: list[tuple[float, Atoms]] = []
        for idx, (energy, relaxed) in enumerate(
            zip(energies, relaxed_atoms, strict=True)
        ):
            if self._uses_metatomic_model():
                _restore_ase_cell_from_reference(relaxed, reference_atoms[idx])
            if forces_list is not None:
                relaxed.arrays["forces"] = np.asarray(
                    forces_list[idx], dtype=np.float64
                )
            elif "forces" in relaxed.arrays or relaxed.calc is not None:
                ensure_float64_forces(relaxed)

            relaxed.info.setdefault("key_value_pairs", {})

            update_metadata(
                relaxed,
                potential_energy=energy,
                raw_score=-energy,
            )
            if batch_steps is not None:
                update_metadata(relaxed, relaxation_steps=batch_steps)
            results.append((energy, relaxed))
        return results

    def _uses_metatomic_model(self) -> bool:
        mk = str(self.model_kind or "mace").strip().lower()
        return mk in ("upet", "metatomic")

    def _cache_model_name(self) -> str:
        mk = str(self.model_kind or "mace").strip().lower()
        if mk == "mace":
            return str(self.mace_model_name)
        if mk in ("fairchem", "uma"):
            return str(self.fairchem_model_name or "fairchem")
        if mk in ("upet", "metatomic"):
            if self.upet_checkpoint_path:
                return str(self.upet_checkpoint_path)
            ver = self.upet_version or "latest"
            return f"{self.upet_model_name}-v{ver}"
        return str(mk)

    def __deepcopy__(self, memo):  # pragma: no cover - deepcopy helper
        """Treat the relaxer as a singleton under ``deepcopy``.

        ``TorchSimBatchRelaxer`` holds a live PyTorch model and caches that are
        not safely picklable (modules, CUDA tensors, runner kwargs with module
        references). Callers typically deepcopy parameter dicts containing the
        relaxer for bookkeeping — short-circuiting to ``self`` gives them a
        usable reference without attempting to clone the model.
        """
        memo[id(self)] = self
        return self

    def _patch_model_for_cuda(self) -> None:
        """Ensure TorchSim models handle CUDA atomic numbers safely."""
        setup_fn = getattr(self.model, "setup_from_system_idx", None)
        if setup_fn is None or getattr(type(self.model), "_scgo_setup_patched", False):
            return

        @functools.wraps(setup_fn)
        def patched_setup(atomic_numbers, system_idx):
            original_device = None
            if hasattr(atomic_numbers, "is_cuda") and atomic_numbers.is_cuda:
                original_device = atomic_numbers.device
                atomic_numbers = atomic_numbers.cpu()
            result = setup_fn(atomic_numbers, system_idx)
            if original_device is not None and hasattr(self.model, "atomic_numbers"):
                self.model.atomic_numbers = self.model.atomic_numbers.to(
                    original_device,
                )
            return result

        self.model.setup_from_system_idx = patched_setup  # type: ignore[assignment]
        type(self.model)._scgo_setup_patched = True


def get_global_memory_scaler_cache() -> MemoryScalerCache:
    """Return the process-wide :class:`MemoryScalerCache` used by default."""
    return _GLOBAL_MEMORY_SCALER_CACHE
