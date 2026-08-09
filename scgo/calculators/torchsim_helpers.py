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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.constraints import FixAtoms as ASEFixAtoms

from scgo.calculators.torch_device import resolve_torch_device
from scgo.exceptions import (
    SCGORuntimeError,
    SCGOValidationError,
)
from scgo.metadata.atoms import set_tags
from scgo.utils.helpers import copy_atoms, ensure_float64_forces
from scgo.utils.logging import get_logger
from scgo.utils.run_helpers import cleanup_torch_cuda

logger = get_logger(__name__)

_DEFAULT_UPET_VERSION = "1.5.0"

#: Cache namespace used when the caller does not name the probe geometry.
DEFAULT_GEOMETRY_TAG = "default"
#: Probe-shape discriminators appended to ``geometry_tag`` in the cache key.
_PROBE_KIND_BULK = "bulk"
_PROBE_KIND_ATOMS = "atoms"
_PROBE_KIND_BUILDER = "builder"

__all__ = [
    "DEFAULT_GEOMETRY_TAG",
    "MemoryScalerCache",
    "TorchSimBatchRelaxer",
    "build_torchsim_fixatoms_from_ase_batch",
    "build_torchsim_relaxer",
    "collect_ase_fixatoms_indices",
    "get_global_memory_scaler_cache",
]


class MemoryScalerCache:
    """Disk-backed cache for TorchSim ``max_memory_scaler`` (GPU probing takes ~70s).

    Essential for performance: without caching, each first run in a cluster size
    forces expensive memory estimation via forward passes. Saves ~70s per campaign.

    Entries are keyed by ``geometry_tag`` in addition to atom count, because the
    ``n_atoms_x_density`` metric depends on *geometry*, not just size: a dense
    bulk probe and a sparse slab+vacuum probe with the same atom count produce
    very different scalers and must never share a cache slot.
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
        geometry_tag: str = DEFAULT_GEOMETRY_TAG,
    ) -> str:
        """Create a cache key from parameters (n_atoms binned to nearest 5)."""
        atom_bin = ((n_atoms + 4) // 5) * 5
        tag = str(geometry_tag or DEFAULT_GEOMETRY_TAG)
        return f"{model_name}|{memory_scales_with}|{device}|{tag}|atoms_{atom_bin}"

    def get(
        self,
        n_atoms: int,
        model_name: str,
        memory_scales_with: str,
        device: str,
        geometry_tag: str = DEFAULT_GEOMETRY_TAG,
    ) -> float | None:
        """Get cached max_memory_scaler if available."""
        key = self._make_key(
            n_atoms, model_name, memory_scales_with, device, geometry_tag
        )
        return self._cache.get(key)

    def set(
        self,
        n_atoms: int,
        model_name: str,
        memory_scales_with: str,
        device: str,
        value: float,
        geometry_tag: str = DEFAULT_GEOMETRY_TAG,
    ) -> None:
        """Cache a max_memory_scaler value to disk."""
        key = self._make_key(
            n_atoms, model_name, memory_scales_with, device, geometry_tag
        )
        self._cache[key] = value
        self._save_cache()

    def delete(
        self,
        n_atoms: int,
        model_name: str,
        memory_scales_with: str,
        device: str,
        geometry_tag: str = DEFAULT_GEOMETRY_TAG,
    ) -> None:
        """Remove one cached scaler entry (e.g. when it is too tight for a batch)."""
        key = self._make_key(
            n_atoms, model_name, memory_scales_with, device, geometry_tag
        )
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

    Remove once TorchSim fixes CPU/CUDA ``atom_idx`` handling in constraints
    (see CHANGELOG maintainer notes for related upstream workarounds).
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
    upet_model_name: str,
    upet_version: str | None,
    upet_checkpoint_path: str | None = None,
    upet_non_conservative: bool = False,
    compute_stress: bool = False,
):
    """Create a TorchSim MetatomicModel for UPET checkpoints."""
    from metatomic_torchsim import MetatomicModel  # type: ignore
    from upet import get_upet

    from scgo.calculators.upet_helpers import disable_metatomic_nvalchemiops

    # nvalchemiops CUDA NL can fail for non-cubic gas-phase cells (float max_neighbors
    # in metatomic-torchsim); vesin handles these systems reliably.
    disable_metatomic_nvalchemiops()

    if upet_checkpoint_path:
        atomistic = get_upet(checkpoint_path=upet_checkpoint_path)
    else:
        model, size = _parse_upet_model_and_size(upet_model_name)
        atomistic = get_upet(
            model=model,
            size=size,
            version=upet_version or _DEFAULT_UPET_VERSION,
        )

    return MetatomicModel(
        atomistic,
        device=device,
        non_conservative=upet_non_conservative,
        compute_stress=compute_stress,
    )


def _coerce_step_count(val: Any) -> int | None:
    """Convert a scalar/tensor step count to int, or None on failure."""
    try:
        if hasattr(val, "detach"):
            val = val.detach().cpu()
        if hasattr(val, "item"):
            return int(val.item())
        return int(val)
    except (TypeError, ValueError):
        return None


def _steps_taken_from_optimize_state(state: Any) -> int | None:
    """Extract optimizer steps from a TorchSim state when available.

    FIRE ``OptimState`` does not expose a step count (the runner keeps a local
    counter). Prefer ``n_iter`` (BFGS/LBFGS), then ``n_steps``.
    """
    for attr in ("n_iter", "n_steps"):
        val = getattr(state, attr, None)
        if val is None:
            continue
        steps = _coerce_step_count(val)
        if steps is not None:
            return steps
    return None


def build_torchsim_relaxer(
    calculator: Any,
    *,
    fmax: float,
    max_steps: int,
    expected_max_atoms: int,
    torchsim_params: dict[str, Any] | None = None,
    dtype: Any | None = None,
) -> TorchSimBatchRelaxer:
    """Build a TorchSim relaxer from a live MLIP ASE calculator.

    Cascades UMA/FairChem → UPET → MACE: classify the calculator, extract a
    shared TorchSim model when possible (clearing ``calculator._inner`` so the
    ASE wrapper does not hold a duplicate), then construct
    :class:`TorchSimBatchRelaxer`. Optional ``torchsim_params`` override any
    constructed kwargs. Preset builders that already know ``model_kind`` /
    model names construct :class:`TorchSimBatchRelaxer` directly.

    Args:
        dtype: Optional torch dtype (e.g. ``torch.float32``). Pass ``None`` to
            keep :class:`TorchSimBatchRelaxer`'s default (``torch.float64``, for
            parity with the ASE MACE wrapper). ``torch.float32`` enables much
            faster FP32/TF32 GPU kernels at the cost of some numerical accuracy.
    """
    from scgo.utils.torchsim_policy import (
        is_uma_like_calculator,
        is_upet_like_calculator,
    )

    base: dict[str, Any] = {
        "force_tol": fmax,
        "max_steps": max_steps,
        "expected_max_atoms": expected_max_atoms,
        "max_atoms_to_try": expected_max_atoms,
    }
    if dtype is not None:
        # Override the relaxer default (float64) for faster FP32/TF32 kernels.
        # ``None`` leaves the model default untouched (parity for non-preset users).
        base["dtype"] = dtype

    if is_uma_like_calculator(calculator):
        from scgo.calculators.uma_helpers import (
            infer_uma_model_name_from_calculator,
            try_extract_torchsim_model_from_uma_calculator,
        )

        model_name = infer_uma_model_name_from_calculator(calculator)
        if not model_name:
            raise SCGOValidationError(
                "Cannot infer UMA model_name from calculator for TorchSim relaxer."
            )
        shared_model = try_extract_torchsim_model_from_uma_calculator(calculator)
        if shared_model is None:
            logger.warning(
                "Could not extract live FairChem predictor from UMA "
                "calculator; TorchSim will reload the checkpoint."
            )
        base.update(
            {
                "model_kind": "fairchem",
                "fairchem_model_name": str(model_name),
                "fairchem_task_name": getattr(calculator, "task_name", None),
                "model": shared_model,
            }
        )
        if shared_model is not None and hasattr(calculator, "_inner"):
            calculator._inner = None
    elif is_upet_like_calculator(calculator):
        from scgo.calculators.upet_helpers import (
            infer_upet_model_name_from_calculator,
            try_extract_torchsim_model_from_upet_calculator,
        )

        model_name = infer_upet_model_name_from_calculator(calculator)
        if not model_name and not getattr(calculator, "checkpoint_path", None):
            raise SCGOValidationError(
                "Cannot infer UPET model_name from calculator for TorchSim relaxer."
            )
        shared_model = try_extract_torchsim_model_from_upet_calculator(calculator)
        if shared_model is None:
            logger.warning(
                "Could not extract live AtomisticModel from UPET "
                "calculator; TorchSim will reload the checkpoint."
            )
        base.update(
            {
                "model_kind": "upet",
                "upet_model_name": str(model_name) if model_name else None,
                "upet_version": getattr(calculator, "version", None),
                "upet_checkpoint_path": getattr(calculator, "checkpoint_path", None),
                "upet_non_conservative": getattr(calculator, "non_conservative", False),
                "model": shared_model,
            }
        )
        if shared_model is not None and hasattr(calculator, "_inner"):
            calculator._inner = None
    else:
        from scgo.calculators.mace_helpers import (
            infer_mace_model_name_from_calculator,
            try_extract_torchsim_model_from_mace_calculator,
        )

        mace_model_name = infer_mace_model_name_from_calculator(calculator)
        if not mace_model_name:
            raise SCGOValidationError(
                "Cannot infer MACE model_name from calculator for TorchSim relaxer."
            )
        shared_model = try_extract_torchsim_model_from_mace_calculator(calculator)
        base.update(
            {
                "mace_model_name": str(mace_model_name),
                "model": shared_model,
            }
        )

    if torchsim_params:
        base.update(torchsim_params)
    return TorchSimBatchRelaxer(**base)


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
    probe_atoms:
        Optional representative single structure used to warm the memory
        scaler instead of the default dense bulk-Cu dummy. The
        ``n_atoms_x_density`` metric depends on geometry, so probing a dense
        block for a sparse slab+vacuum workload (NEB bands) yields a scaler
        that does not describe the real batches. Pass one *actual* workload
        structure (e.g. a minimum from the TS pool); torch-sim grows the probe
        batch itself up to ``max_atoms_to_try``.
    probe_builder:
        Optional ``callable(n_atoms) -> Atoms`` alternative to ``probe_atoms``
        for callers that want to synthesize the probe lazily. Takes precedence
        over ``probe_atoms``.
    geometry_tag:
        Cache namespace for the probed scaler. Distinguishes probe shapes (e.g.
        ``"neb-surface_cluster"`` vs ``"ga-gas_cluster"``) so gas and surface
        campaigns never read each other's on-disk scaler. The resolved key also
        records whether the probe was bulk / caller-supplied atoms / a builder,
        so a stray tag cannot alias two different probe geometries.
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
    upet_version: str | None = _DEFAULT_UPET_VERSION
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
    # Warm-probe geometry. None -> dense bulk-Cu dummy (legacy GO behaviour).
    probe_atoms: Atoms | None = None
    probe_builder: Callable[[int], Atoms] | None = None
    geometry_tag: str | None = None
    init_kwargs: dict | None = None
    optimizer_kwargs: dict | None = None  # forwarded as **optimizer_kwargs to step-fn
    runner_kwargs: dict | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        # Lazy import: only require TorchSim when actually instantiating the relaxer.
        import torch_sim as ts  # type: ignore

        _register_torchsim_warning_filters()
        self._ts = ts
        if self.device is None:
            self.device = torch.device(
                resolve_torch_device(None, allow_mps=True, backend_name="TorchSim")
            )
        else:
            self.device = torch.device(
                resolve_torch_device(
                    self.device, allow_mps=True, backend_name="TorchSim"
                )
            )
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
        self._sync_device_dtype_from_model()
        self._patch_model_for_cuda()

        # Store device string for cache key (e.g., "cuda" or "cpu")
        self._device_str = str(self.device).split(":")[0]
        self.last_batch_relax_steps: list[int] = []
        self._resolved_geometry_tag = self._resolve_geometry_tag()

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
            self._runner_kwargs["autobatcher"] = self._build_autobatcher()

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

    def _probe_kind(self) -> str:
        """Which probe geometry :meth:`_build_probe_atoms` will produce."""
        if self.probe_builder is not None:
            return _PROBE_KIND_BUILDER
        if self.probe_atoms is not None:
            return _PROBE_KIND_ATOMS
        return _PROBE_KIND_BULK

    def _resolve_geometry_tag(self) -> str:
        """Cache namespace for the probed scaler: ``<caller tag>|<probe kind>``.

        The probe kind is always part of the key so a caller-supplied tag that
        travels further than its probe (e.g. shared ``torchsim_params``) cannot
        make a dense bulk probe masquerade as a sparse slab one.
        """
        tag = str(self.geometry_tag).strip() if self.geometry_tag else ""
        return f"{tag or DEFAULT_GEOMETRY_TAG}|{self._probe_kind()}"

    def _memory_scaler_cache_key(self, n_atoms: int) -> dict[str, Any]:
        return {
            "n_atoms": n_atoms,
            "model_name": self._cache_model_name(),
            "memory_scales_with": self.memory_scales_with,
            "device": self._device_str,
            "geometry_tag": getattr(
                self, "_resolved_geometry_tag", DEFAULT_GEOMETRY_TAG
            ),
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
        """Drop the cached scaler for ``n_atoms`` and for the warm-probe bucket.

        ``relax_batch`` only knows the largest structure in the failing batch,
        while the warm probe cached its scaler under ``expected_max_atoms``.
        Deleting both keys makes the "cached scaler too tight" retry actually
        clear the entry that produced the bad value.
        """
        buckets = {int(n_atoms)}
        if self.expected_max_atoms is not None:
            buckets.add(int(self.expected_max_atoms))
        for bucket in buckets:
            _GLOBAL_MEMORY_SCALER_CACHE.delete(**self._memory_scaler_cache_key(bucket))

    def _build_autobatcher(self) -> object:
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
        return self._ts.InFlightAutoBatcher(**autobatcher_kwargs)

    def _recreate_autobatcher(self) -> None:
        if "autobatcher" not in self._runner_kwargs:
            return
        self._runner_kwargs["autobatcher"] = self._build_autobatcher()

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

    def _build_probe_atoms(self, n_atoms: int) -> tuple[Atoms, str]:
        """Return ``(probe_structure, description)`` for the memory warm-probe.

        Preference order: ``probe_builder`` → ``probe_atoms`` → dense bulk Cu.

        For the caller-supplied cases a *single representative* structure is
        returned rather than an ``n_atoms``-sized block: torch-sim's
        ``determine_max_batch_size`` replicates it geometrically up to
        ``max_atoms_to_try`` (== ``expected_max_atoms``), so the resulting
        scaler is exactly "how many of these real structures fit", which is the
        quantity the binning autobatcher needs. Sizing the probe by atom count
        instead (the legacy bulk dummy) measures a geometry the workload never
        sees and yields a scaler that does not transfer.
        """
        if self.probe_builder is not None:
            return self.probe_builder(int(n_atoms)), "caller-supplied probe builder"
        if self.probe_atoms is not None:
            probe = copy_atoms(self.probe_atoms)
            probe.calc = None
            # Constraints are irrelevant for a forward pass and would only add
            # device-placement work inside torch-sim.
            probe.set_constraint()
            return probe, f"workload geometry ({len(probe)} atoms/structure)"
        # Legacy fallback: dense bulk Cu sized by atom count (GO/cluster path).
        dummy = bulk("Cu", "fcc", a=3.61, cubic=True)
        while len(dummy) < n_atoms:
            dummy = dummy.repeat((2, 2, 2))
        dummy = dummy[:n_atoms]
        dummy.center(vacuum=3.0)
        return dummy, f"dense bulk-Cu dummy ({n_atoms} atoms)"

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
            probe, probe_desc = self._build_probe_atoms(n_atoms)
            logger.info(
                "Probing GPU memory with %s, capped at %d atoms/batch...",
                probe_desc,
                n_atoms,
            )
            initial_time = time.time()
            _ = self._ts.optimize(
                system=[probe],
                model=self.model,
                optimizer=self.optimizer,
                max_steps=1,
                **{k: v for k, v in self._runner_kwargs.items() if k != "max_steps"},
            )
            probe_time = time.time() - initial_time

            if getattr(autobatcher, "max_memory_scaler", None):
                self._persist_autobatcher_scaler(n_atoms)
                logger.info(
                    "Memory probing complete (%.2fs). max_memory_scaler=%.1f "
                    "cached for %d atoms (%s).",
                    probe_time,
                    float(autobatcher.max_memory_scaler),
                    n_atoms,
                    self._resolved_geometry_tag,
                )
        except (
            RuntimeError,
            ValueError,
            OSError,
            AttributeError,
            torch.cuda.OutOfMemoryError,
        ) as e:
            self._reset_autobatcher_memory_scaler()
            # A failed probe can leave the allocator fragmented (or holding the
            # partially built probe batch); hand the real workload a clean GPU.
            cleanup_torch_cuda(logger=logger)
            logger.warning(
                f"Memory probing failed (non-fatal): {e}. Will retry on first relax_batch()."
            )

    def relax_batch(
        self, atoms_list: Sequence[Atoms], steps: int | None = None
    ) -> list[tuple[float, Atoms]]:
        """Relax a batch of ASE ``Atoms`` objects using TorchSim.

        Args:
            atoms_list: List of Atoms objects to relax.
            steps: Optional override for max_steps. Set to 0 for a true single-point
                evaluation via ``torch_sim.static`` (used by NEB/TS force paths).

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

    def _prepare_batch_atoms(
        self, atoms_list: Sequence[Atoms]
    ) -> tuple[list[Atoms], list[Atoms], object]:
        """Prepare ASE atoms for TorchSim (metatomic cell + optional FixAtoms).

        Returns:
            ``(atoms_seq, reference_atoms, system_in)`` where ``system_in`` is either
            the ASE list or a ``SimState`` with TorchSim ``FixAtoms`` attached.
        """
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
        return atoms_seq, reference_atoms, system_in

    def _effective_max_memory_scaler(self) -> float | None:
        """Best known ``max_memory_scaler`` for this relaxer, or ``None``.

        ``self.max_memory_scaler`` is only set when the *user* supplied one; the
        value produced by the warm probe (or restored from the disk cache) lives
        on the InFlight autobatcher instance held in ``_runner_kwargs``. Both are
        the same quantity, so single-point batching reads whichever is known.
        """
        if self.max_memory_scaler:
            return float(self.max_memory_scaler)
        runner_kwargs = getattr(self, "_runner_kwargs", None) or {}
        autobatcher = runner_kwargs.get("autobatcher")
        scaler = getattr(autobatcher, "max_memory_scaler", None)
        return float(scaler) if scaler else None

    def _static_autobatcher(self, *, n_structures: int, max_atoms: int) -> object:
        """Resolve the ``autobatcher`` argument for a ``ts.static`` call.

        When a ``max_memory_scaler`` is already known and we are on GPU, build a
        :class:`torch_sim.BinningAutoBatcher` seeded with it. ``ts.static`` only
        accepts ``BinningAutoBatcher | bool`` (the InFlight batcher used by
        ``ts.optimize`` is rejected), and seeding the scaler means the batcher
        bins without re-probing. This is what keeps the fused NEB force batch
        from running as one monolithic ``torch.cat`` forward pass.

        Falls back to :meth:`_static_autobatcher_arg` (a plain bool) on CPU or
        when no scaler is known.
        """
        on_cpu = str(self.device).split(":")[0] == "cpu"
        if not on_cpu:
            scaler = self._effective_max_memory_scaler()
            if scaler:
                logger.debug(
                    "Binning %d single-point structures with max_memory_scaler=%.1f",
                    n_structures,
                    scaler,
                )
                return self._ts.BinningAutoBatcher(
                    model=self.model,
                    memory_scales_with=self.memory_scales_with,
                    max_memory_scaler=scaler,
                    max_memory_padding=self.max_memory_padding,
                )
        return self._static_autobatcher_arg(
            n_structures=n_structures, max_atoms=max_atoms
        )

    def _static_autobatcher_arg(self, *, n_structures: int, max_atoms: int) -> bool:
        """Fallback ``ts.static`` autobatcher flag when no scaler is known.

        ``ts.static(autobatcher=True)`` builds a fresh ``BinningAutoBatcher`` that
        re-probes GPU memory on *every* call (unlike the cached InFlight
        autobatcher used by ``optimize``). That probe climbs to thousands of
        atoms and can burn hours during NEB force loops.

        Default: no autobatcher for single-point (NEB batches are modest). Only
        enable when the caller explicitly set ``autobatcher=True`` *and* the
        batch is large enough that packing may matter.
        """
        on_cpu = str(self.device).split(":")[0] == "cpu"
        if on_cpu:
            return False
        if self.autobatcher is not True:
            # None (default) or False → skip probing BinningAutoBatcher.
            return False
        # Explicit opt-in: still skip tiny batches where one forward is fine.
        return n_structures * max_atoms >= 256

    def _results_from_static_props(
        self,
        props: Sequence[dict[str, Any]],
        *,
        atoms_seq: Sequence[Atoms],
        reference_atoms: Sequence[Atoms],
    ) -> list[tuple[float, Atoms]]:
        """Map ``ts.static`` property dicts onto ASE ``(energy, atoms)`` results."""
        if len(props) != len(reference_atoms):
            raise SCGORuntimeError(
                "TorchSim static returned mismatched counts for atoms and energies"
            )

        self.last_batch_relax_steps = [0] * len(reference_atoms)
        results: list[tuple[float, Atoms]] = []
        for idx, prop in enumerate(props):
            energy_t = prop.get("potential_energy")
            if energy_t is None:
                raise SCGORuntimeError(
                    "TorchSim static did not return potential_energy"
                )
            energy = float(energy_t.detach().cpu().reshape(-1)[0])

            # Isolate nested info so raw_score writes cannot corrupt caller atoms
            # (ASE Atoms.copy() shares key_value_pairs dicts).
            out = copy_atoms(atoms_seq[idx])
            if self._uses_metatomic_model():
                _restore_ase_cell_from_reference(out, reference_atoms[idx])

            forces_t = prop.get("forces")
            if forces_t is not None:
                forces = np.asarray(forces_t.detach().cpu().numpy(), dtype=np.float64)
                if forces.shape[0] != len(out):
                    raise SCGORuntimeError(
                        f"Forces shape mismatch for structure {idx}: "
                        f"expected {len(out)} atoms, got {forces.shape[0]}"
                    )
                out.arrays["forces"] = forces
            elif "forces" in out.arrays or out.calc is not None:
                ensure_float64_forces(out)

            set_tags(
                out,
                potential_energy=energy,
                raw_score=-energy,
                relaxation_steps=0,
            )
            results.append((energy, out))
        return results

    def _single_point_batch(
        self,
        atoms_list: Sequence[Atoms],
        *,
        max_atoms_in_batch: int,
    ) -> list[tuple[float, Atoms]]:
        """True single-point PES eval via ``ts.static`` (no FIRE / no max_steps warn).

        Used for NEB/TS force evaluations and endpoint energies. Unlike
        ``optimize(max_steps=0)``, positions are not updated and forces match the
        input geometry (ASE calculator semantics).
        """
        atoms_seq, reference_atoms, system_in = self._prepare_batch_atoms(atoms_list)
        logger.debug("Running TorchSim single-point evaluation via static().")
        props = self._ts.static(  # type: ignore[call-arg]
            system=system_in,
            model=self.model,
            autobatcher=self._static_autobatcher(
                n_structures=len(atoms_seq),
                max_atoms=max_atoms_in_batch,
            ),
        )
        return self._results_from_static_props(
            props, atoms_seq=atoms_seq, reference_atoms=reference_atoms
        )

    def _relax_batch_once(
        self,
        atoms_list: Sequence[Atoms],
        *,
        steps: int | None,
        max_atoms_in_batch: int,
    ) -> list[tuple[float, Atoms]]:
        # Try to apply cached memory scaler to avoid expensive re-probing (~70s per new cluster size)
        # The batch bucket first (exact workload), then the warm-probe bucket
        # (``expected_max_atoms``): NEB force batches hold many small images, so
        # ``max_atoms_in_batch`` rarely matches the probe's cache key.
        if (
            self.max_memory_scaler is None
            and "autobatcher" in self._runner_kwargs
            and not self._apply_cached_memory_scaler(max_atoms_in_batch)
            and self.expected_max_atoms is not None
        ):
            self._apply_cached_memory_scaler(int(self.expected_max_atoms))

        runner_kwargs = self._runner_kwargs.copy()
        if steps is not None:
            runner_kwargs["max_steps"] = steps

        # `steps=0` is single-point mode (NEB/TS force evals and endpoint energies).
        # Use ts.static: optimize(max_steps=0) still takes one FIRE step, displaces
        # atoms, returns forces at the wrong geometry, and emits
        # "All systems have reached the maximum number of steps: 0".
        max_steps_now = runner_kwargs.get("max_steps", self.max_steps)
        if max_steps_now == 0:
            return self._single_point_batch(
                atoms_list, max_atoms_in_batch=max_atoms_in_batch
            )

        atoms_seq, reference_atoms, system_in = self._prepare_batch_atoms(atoms_list)

        state = self._ts.optimize(  # type: ignore[call-arg]
            system=system_in,
            model=self.model,
            optimizer=self.optimizer,
            **runner_kwargs,
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

            set_tags(
                relaxed,
                potential_energy=energy,
                raw_score=-energy,
            )
            if batch_steps is not None:
                set_tags(relaxed, relaxation_steps=batch_steps)
            results.append((energy, relaxed))
        return results

    def _uses_metatomic_model(self) -> bool:
        mk = str(self.model_kind or "mace").strip().lower()
        return mk in ("upet", "metatomic")

    def _sync_device_dtype_from_model(self) -> None:
        """Align SimState device/dtype with the loaded TorchSim model.

        MetatomicModel (UPET) chooses dtype from model capabilities and moves
        weights to ``device``. Mismatched SimState dtypes raise in
        ``MetatomicModel.forward``; syncing here keeps batched GPU relaxations
        on the model device with autobatching.
        """
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            self.device = model_device
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            self.dtype = model_dtype
        if self._uses_metatomic_model():
            logger.info(
                "UPET/TorchSim model ready: device=%s dtype=%s model_kind=%s",
                self.device,
                self.dtype,
                self.model_kind,
            )

    def _cache_model_name(self) -> str:
        mk = str(self.model_kind or "mace").strip().lower()
        if mk == "mace":
            return str(self.mace_model_name)
        if mk in ("fairchem", "uma"):
            return str(self.fairchem_model_name or "fairchem")
        if mk in ("upet", "metatomic"):
            if self.upet_checkpoint_path:
                return str(self.upet_checkpoint_path)
            ver = self.upet_version or _DEFAULT_UPET_VERSION
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
