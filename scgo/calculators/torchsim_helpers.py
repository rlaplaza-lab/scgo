"""Utilities for integrating TorchSim batched relaxations with SCGO.

This module wraps the TorchSim high-level optimization API so SCGO can relax
multiple candidate structures in a single batched call.

Important:
- Imports for optional stacks (TorchSim, MACE, FairChem, UPET) are **lazy** so
  SCGO can be imported in minimal environments without pulling MLIP dependencies.
- TorchSim can run with multiple model families. SCGO supports MACE, FairChem/UMA,
  and UPET/metatomic via TorchSim model wrappers.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from ase import Atoms
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

__all__ = [
    "TorchSimBatchRelaxer",
    "build_torchsim_fixatoms_from_ase_batch",
    "build_torchsim_relaxer",
    "collect_ase_fixatoms_indices",
]


#: Constraint type names already reported as dropped (warn once per process).
_WARNED_DROPPED_CONSTRAINTS: set[str] = set()


def _warn_dropped_ase_constraints(names: Sequence[str]) -> None:
    """Warn once per process for each ASE constraint type TorchSim cannot map."""
    new = sorted({name for name in names if name not in _WARNED_DROPPED_CONSTRAINTS})
    if not new:
        return
    _WARNED_DROPPED_CONSTRAINTS.update(new)
    logger.warning(
        "Ignoring ASE constraint(s) that TorchSim cannot represent: %s. Only "
        "FixAtoms is mapped to TorchSim; these constraints are NOT enforced "
        "during batched relaxation.",
        ", ".join(new),
    )


def collect_ase_fixatoms_indices(atoms: Atoms) -> list[int]:
    """Return sorted unique indices fixed by ASE :class:`ase.constraints.FixAtoms`.

    Negative indices (ASE allows ``FixAtoms(indices=[-1])``) are normalized to
    their positive equivalents so that batching them into global TorchSim
    indices cannot freeze the wrong atom.

    Other ASE constraint types cannot be represented in TorchSim today; they are
    dropped with a once-per-process warning listing the constraint type names.
    """
    n_atoms = len(atoms)
    out: list[int] = []
    dropped: list[str] = []
    for c in atoms.constraints:
        if not isinstance(c, ASEFixAtoms):
            dropped.append(type(c).__name__)
            continue
        if n_atoms == 0:
            logger.warning("Ignoring FixAtoms on an empty Atoms object")
            continue
        out.extend(int(i) % n_atoms for i in c.index)
    if dropped:
        _warn_dropped_ase_constraints(dropped)
    return sorted(set(out))


def _patch_torchsim_constraint_device_mismatch() -> None:
    """Monkey-patch TorchSim ``AtomConstraint.select_sub_constraint``.

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
    # Apply the torch.load(weights_only=False) compat shim BEFORE importing
    # ``mace``/``e3nn``: ``e3nn.o3._wigner`` unpickles ``constants.pt`` at import
    # time, and PyTorch >=2.6 rejects it under the default weights_only=True. The
    # shim (installed by importing mace_helpers / _ensure_torch_load_mace_checkpoints)
    # must be live before that import runs.
    from scgo.calculators.mace_helpers import (
        MaceUrls,
        _ensure_torch_load_mace_checkpoints,
    )
    from scgo.utils.mlip_extras import clear_torch_force_no_weights_only_load_env

    clear_torch_force_no_weights_only_load_env()
    _ensure_torch_load_mace_checkpoints()
    # Lazy imports: only required for the MACE TorchSim path. The shim above is
    # already active, so e3nn's constants.pt load succeeds under PyTorch >=2.6.
    from mace.calculators.foundations_models import mace_mp  # type: ignore
    from torch_sim.models.mace import MaceModel  # type: ignore

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
    """Wrap a raw ASE/MACE ``ScaleShiftMACE`` for :func:`torch_sim.optimize`.

    ``ga_go`` reuses the calculator's loaded weights via
    :func:`try_extract_torchsim_model_from_mace_calculator`, which returns the
    inner torch module. TorchSim expects a model exposing ``.device`` and
    ``.dtype`` (e.g. :class:`torch_sim.models.mace.MaceModel`).

    The module handed in is the **live** ASE calculator's module and
    ``MaceModel`` casts it to ``dtype`` in place, which would silently change
    the user's ASE calculator precision. Wrap a deep copy instead; if copying
    fails, fall back to the shared module with a warning.
    """
    if hasattr(model, "device") and hasattr(model, "dtype"):
        return model
    mod = getattr(type(model), "__module__", "") or ""
    name = type(model).__name__
    if "mace" not in mod.lower() and "MACE" not in name:
        return model
    from torch_sim.models.mace import MaceModel  # type: ignore

    try:
        model_for_wrapper = copy.deepcopy(model)
    except (TypeError, RuntimeError) as exc:
        logger.warning(
            "Could not deep-copy the live MACE module before wrapping it for "
            "TorchSim (%s); the ASE calculator shares this module and its "
            "weights may be cast to %s",
            exc,
            dtype,
        )
        model_for_wrapper = model

    return MaceModel(
        model=model_for_wrapper,
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
    directions (gas-phase clusters still use a finite ASE box for spacing). For
    partially-periodic systems (e.g. surface slabs with ``pbc=(True, True,
    False)``) only the non-periodic direction's cell vector must be zeroed; the
    periodic directions keep their full cell vectors. The geometry itself lives
    in the atom positions, so zeroing a non-periodic cell vector does not change
    the physics — it only satisfies the stricter validation introduced in
    metatomic 0.4.1+ (newer than this code was originally written against).
    """
    prepared = atoms.copy()
    if not any(prepared.pbc):
        prepared.cell[:] = 0.0
        return prepared
    cell = prepared.cell.copy()
    for axis in range(3):
        if not prepared.pbc[axis]:
            cell[axis, :] = 0.0
            cell[:, axis] = 0.0
    prepared.cell[:] = cell
    return prepared


def _restore_ase_cell_from_reference(relaxed: Atoms, reference: Atoms) -> None:
    """Restore SCGO storage cell/PBC after a metatomic TorchSim relaxation."""
    relaxed.cell = reference.cell.copy()
    relaxed.pbc = reference.pbc


def _reattach_input_metadata(relaxed: Atoms, source: Atoms) -> None:
    """Copy ``tags`` / ``constraints`` / ``info`` from the input onto an output.

    ``SimState.to_atoms()`` builds fresh :class:`ase.Atoms` from positions,
    numbers and cell only, so the optimize path would otherwise return
    structures without the integer ``tags`` array, without ASE constraints and
    without ``atoms.info`` — unlike the ``ts.static`` path, which copies the
    input. This aligns both output contracts.

    Nested ``info`` dicts are de-aliased (``copy_atoms``-style) so later
    ``set_tags`` writes cannot corrupt the caller's ``key_value_pairs``.
    """
    if len(relaxed) == len(source):
        relaxed.set_tags(source.get_tags())
        if source.constraints:
            relaxed.set_constraint(copy.deepcopy(source.constraints))
    for key, value in source.info.items():
        if key in relaxed.info:
            continue
        relaxed.info[key] = dict(value) if isinstance(value, dict) else value


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
    shared TorchSim model when possible (on the UMA/UPET paths this also clears
    ``calculator._inner`` so the ASE wrapper does not hold a duplicate), then
    construct :class:`TorchSimBatchRelaxer`. Optional ``torchsim_params``
    override any constructed kwargs. Preset builders that already know
    ``model_kind`` / model names construct :class:`TorchSimBatchRelaxer`
    directly.

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
                "calculator; TorchSim will reload the checkpoint"
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
                "calculator; TorchSim will reload the checkpoint"
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
        base.update(_filter_torchsim_params(torchsim_params))
    return TorchSimBatchRelaxer(**base)


def _filter_torchsim_params(torchsim_params: dict[str, Any] | None) -> dict[str, Any]:
    """Drop unknown ``torchsim_params`` keys (stale probe/geometry knobs) with a warning.

    Builds the keep-set from the :class:`TorchSimBatchRelaxer` dataclass fields so
    it stays correct as fields change. Unknown keys (e.g. the removed
    ``probe_atoms`` / ``geometry_tag``) are dropped rather than raising, since
    callers may carry stale presets.
    """
    if not torchsim_params:
        return {}
    known_fields = {f.name for f in dataclasses.fields(TorchSimBatchRelaxer)}
    filtered: dict[str, Any] = {}
    for key, value in torchsim_params.items():
        if key in known_fields:
            filtered[key] = value
        else:
            warnings.warn(
                f"Ignoring unknown torchsim_params key {key!r}; it is not a "
                f"TorchSimBatchRelaxer field and has been dropped.",
                DeprecationWarning,
                stacklevel=2,
            )
    return filtered


@dataclass(eq=False, repr=False)
class TorchSimBatchRelaxer:
    """Batched relaxer built on :func:`torch_sim.optimize` and :func:`torch_sim.static`.

    ASE :class:`ase.constraints.FixAtoms` on input structures are translated to
    TorchSim's internal ``FixAtoms`` before optimization, since
    :func:`torch_sim.initialize_state` does not import ``atoms.constraints``.

    Parameters
    ----------
    device:
        Optional torch device. Defaults to CUDA when available, then MPS,
        otherwise CPU.
    dtype:
        Torch dtype. Defaults to ``torch.float64`` for parity with the ASE MACE
        wrapper; override to ``torch.float32`` for speed at the cost of accuracy.
    model:
        Optional TorchSim model implementing ``ModelInterface``. If omitted, a
        model is loaded according to ``model_kind`` (``"mace"`` by default, from
        ``mace_model_name``).
    model_kind:
        Which stack to load when ``model`` is omitted: ``"mace"`` (default),
        ``"fairchem"``/``"uma"``, or ``"upet"``/``"metatomic"``.
    mace_model_name:
        Name of the SCGO ``MaceUrls`` member (or any ``mace_mp`` model name) to
        load when ``model`` is not provided (default: ``"mace_matpes_0"``).
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
        force the choice. On CPU torch-sim raises for autobatching, so the batcher
        is always built as ``False`` there. Only the :class:`InFlightAutoBatcher`
        is accepted by :func:`torch_sim.optimize`; a matching
        :class:`torch_sim.BinningAutoBatcher` is built for ``ts.static`` (NEB).
    memory_scales_with, max_memory_scaler:
        Advanced knobs forwarded to :class:`torch_sim.InFlightAutoBatcher` when
        autobatching is active.
    expected_max_atoms:
        Optional atom count (e.g. ``cluster_size * population_size``) used to cap
        the native autobatcher's GPU probe via ``max_atoms_to_try`` (see
        :class:`torch_sim.InFlightAutoBatcher`). Without this cap, the probe
        can geometrically climb to its 500k-atom default and OOM small GPUs.
        Recommended for GA/BH/TS campaigns with known population sizes. The probe
        runs on the real workload batches (no synthetic dummy is needed).
    max_atoms_to_try:
        Explicit override for the autobatcher's probe cap. Defaults to
        ``expected_max_atoms`` when that is set; otherwise falls back to
        torch-sim's default (500,000). Always pass a tight value on GPUs
        with limited memory.
    cutoff:
        Neighbor cutoff (Å). Only forwarded to the batchers when
        ``memory_scales_with == "n_edges"``; for the default
        ``"n_atoms_x_density"`` metric it must NOT be passed (torch-sim 0.6.0
        raises ``TypeError``). Safe to overestimate when ``n_edges`` is enabled.
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
    # Autobatching: None -> enabled unless the device is CPU (matches the docs).
    autobatcher: bool | None = None
    memory_scales_with: str = "n_atoms_x_density"
    max_memory_scaler: float | None = None
    max_memory_padding: float = 1.05
    expected_max_atoms: int | None = (
        None  # Probe memory upfront with this atom count (cluster_size * pop_size)
    )
    # Hard cap on the native InFlightAutoBatcher / BinningAutoBatcher GPU probe.
    # None -> fall back to expected_max_atoms (if set) or torch-sim's 500k default.
    max_atoms_to_try: int | None = None
    # Only forwarded to the batchers when memory_scales_with == "n_edges";
    # ignored (and must not be passed) for the default "n_atoms_x_density" metric.
    cutoff: float = 6.0
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

        # Optional seeding (deterministic algorithms are not forced, to avoid
        # CuBLAS constraints)
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

        self._on_cpu = str(self.device).split(":")[0] == "cpu"
        self.last_batch_relax_steps: list[int] = []
        self._runner_kwargs = dict(self.runner_kwargs or {})

        # Resolve autobatcher policy: build the native InFlight/Binning batchers
        # once on GPU; on CPU (or when autobatcher is explicitly False) disable
        # autobatching. torch-sim's autobatching is "generally not supported on
        # CPUs" and raises on CPU, so CPU must pass autobatcher=False.
        use_in_flight = (self.autobatcher is None and not self._on_cpu) or (
            self.autobatcher is True and not self._on_cpu
        )
        if use_in_flight:
            cap = int(self.max_atoms_to_try or self.expected_max_atoms or 50_000)
            kw: dict[str, Any] = {
                "model": self.model,
                "memory_scales_with": self.memory_scales_with,
                "max_memory_scaler": self.max_memory_scaler,  # None -> estimate on first use
                "max_atoms_to_try": cap,
                "max_memory_padding": self.max_memory_padding,
            }
            if self.memory_scales_with == "n_edges":
                kw["cutoff"] = self.cutoff
            self._optimize_batcher: Any = self._ts.InFlightAutoBatcher(**kw)  # type: ignore[call-arg]
            self._static_batcher: Any = self._ts.BinningAutoBatcher(**kw)  # type: ignore[call-arg]
            self._runner_kwargs["autobatcher"] = self._optimize_batcher
        else:
            self._runner_kwargs["autobatcher"] = False

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
        # ``max_steps`` is deliberately NOT baked into ``_runner_kwargs``: it is
        # resolved per call in ``_relax_batch_once`` so that mutating
        # ``relaxer.max_steps`` after construction (the GA sets it from
        # ``niter_local_relaxation``) actually takes effect. A caller-supplied
        # ``runner_kwargs['max_steps']`` seeds the field instead.
        runner_max_steps = self._runner_kwargs.pop("max_steps", None)
        if runner_max_steps is not None:
            self.max_steps = runner_max_steps

    def relax_batch(
        self, atoms_list: Sequence[Atoms], steps: int | None = None
    ) -> list[tuple[float, Atoms]]:
        """Relax a batch of ASE ``Atoms`` objects using TorchSim.

        Args:
            atoms_list: List of Atoms objects to relax.
            steps: Optional override for max_steps. Set to 0 for a true single-point
                evaluation via ``torch_sim.static`` (used by NEB/TS force paths).

        Returns:
            A list of ``(energy, atoms)`` in the same order as the input list.
            Energies are converted to Python floats in eV.
        """
        if not atoms_list:
            return []

        max_atoms_in_batch = max(len(atoms) for atoms in atoms_list)
        return self._relax_batch_once(
            atoms_list, steps=steps, max_atoms_in_batch=max_atoms_in_batch
        )

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
        logger.debug("Running TorchSim single-point evaluation via static()")
        # Use the persistent BinningAutoBatcher built in __post_init__ on GPU; fall
        # back to a plain ``False`` (no batching) on CPU or when autobatching was
        # explicitly disabled. ``_static_batcher`` only exists when native
        # batching was enabled, hence the getattr guard.
        static_batcher = getattr(self, "_static_batcher", None)

        def _static_arg():
            return (
                static_batcher
                if (static_batcher is not None and not self._on_cpu)
                else False
            )

        try:
            props = self._ts.static(  # type: ignore[call-arg]
                system=system_in,
                model=self.model,
                autobatcher=_static_arg(),
            )
        except ValueError as exc:
            if self._is_max_metric_value_error(exc):
                logger.warning(
                    "Cached/sticky max_memory_scaler too tight for this single-point "
                    "batch (%s); resetting batchers and re-probing GPU memory",
                    exc,
                )
                self._reset_and_reprobe()
                # One retry with a fresh, re-estimated Binning batcher.
                try:
                    props = self._ts.static(  # type: ignore[call-arg]
                        system=system_in,
                        model=self.model,
                        autobatcher=_static_arg(),
                    )
                except (torch.cuda.OutOfMemoryError, RuntimeError) as retry_exc:
                    if self._is_cuda_oom_error(retry_exc):
                        cleanup_torch_cuda(logger=logger)
                        raise SCGORuntimeError(
                            "TorchSim ran out of GPU memory during a NEB single-point "
                            "force evaluation after re-probing the autobatcher. Reduce "
                            "expected_max_atoms / max_atoms_to_try or the batch size."
                        ) from retry_exc
                    raise
            else:
                raise
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if self._is_cuda_oom_error(exc):
                # The native Binning batcher should catch OOM itself, but if a batch
                # still overflows, hand the next call a clean allocator and surface
                # a clear error.
                cleanup_torch_cuda(logger=logger)
                raise SCGORuntimeError(
                    "TorchSim ran out of GPU memory during a NEB single-point force "
                    "evaluation (ts.static). Reduce expected_max_atoms / "
                    "max_atoms_to_try or the batch size."
                ) from exc
            raise
        return self._results_from_static_props(
            props, atoms_seq=atoms_seq, reference_atoms=reference_atoms
        )

    @staticmethod
    def _is_max_metric_value_error(exc: BaseException) -> bool:
        """True for torch-sim's sticky-scaler overflow (``"max_metric"`` ValueError)."""
        return isinstance(exc, ValueError) and "max_metric" in str(exc)

    @staticmethod
    def _is_cuda_oom_error(exc: BaseException) -> bool:
        """True for a genuine GPU OOM: torch's error or cuBLAS/cuDNN text.

        cuBLAS/cuDNN sometimes raise a plain ``RuntimeError`` containing
        ``"out of memory"`` rather than ``torch.cuda.OutOfMemoryError``; both must
        be treated as memory degradation so the Kaggle OOM guard cannot be masked.
        """
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    def _reset_and_reprobe(self) -> None:
        """Drop the sticky native-batcher scalers and re-probe on the next call.

        The native InFlight/Binning ``max_memory_scaler`` is estimated once (on the
        first workload) and then stays sticky. A later batch whose
        ``n_atoms_x_density`` metric exceeds the cached scaler raises a hard
        ``ValueError`` ("... > max_metric ...") instead of being re-binned. Resetting
        the scalers to ``None`` makes torch-sim re-estimate memory rather than crash.
        The allocator is also freed so the re-probe starts on a clean GPU.
        """
        self.max_memory_scaler = None
        for batcher_attr in ("_optimize_batcher", "_static_batcher"):
            batcher = getattr(self, batcher_attr, None)
            if batcher is not None:
                batcher.max_memory_scaler = None
        cleanup_torch_cuda(logger=logger)

    def _relax_batch_once(
        self,
        atoms_list: Sequence[Atoms],
        *,
        steps: int | None,
        max_atoms_in_batch: int,
    ) -> list[tuple[float, Atoms]]:
        runner_kwargs = self._runner_kwargs.copy()
        # Resolve ``max_steps`` at call time so a post-construction
        # ``relaxer.max_steps = N`` (GA ``niter_local_relaxation``) is honoured.
        max_steps_now = steps if steps is not None else self.max_steps
        if max_steps_now is not None:
            runner_kwargs["max_steps"] = max_steps_now

        # `steps=0` is single-point mode (NEB/TS force evals and endpoint energies).
        # Use ts.static: optimize(max_steps=0) still takes one FIRE step, displaces
        # atoms, returns forces at the wrong geometry, and emits
        # "All systems have reached the maximum number of steps: 0".
        if max_steps_now == 0:
            return self._single_point_batch(
                atoms_list, max_atoms_in_batch=max_atoms_in_batch
            )

        atoms_seq, reference_atoms, system_in = self._prepare_batch_atoms(atoms_list)

        try:
            state = self._ts.optimize(  # type: ignore[call-arg]
                system=system_in,
                model=self.model,
                optimizer=self.optimizer,
                **runner_kwargs,
            )
        except (ValueError, torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if isinstance(exc, ValueError) and self._is_max_metric_value_error(exc):
                logger.warning(
                    "Cached/sticky max_memory_scaler too tight for this batch (%s); "
                    "resetting batchers and re-probing GPU memory",
                    exc,
                )
                self._reset_and_reprobe()
                # One retry with a fresh, re-estimated autobatcher.
                try:
                    state = self._ts.optimize(  # type: ignore[call-arg]
                        system=system_in,
                        model=self.model,
                        optimizer=self.optimizer,
                        **runner_kwargs,
                    )
                except (torch.cuda.OutOfMemoryError, RuntimeError) as retry_exc:
                    if self._is_cuda_oom_error(retry_exc):
                        cleanup_torch_cuda(logger=logger)
                        raise SCGORuntimeError(
                            "TorchSim relaxation ran out of GPU memory during "
                            "optimization after re-probing the autobatcher. The batch "
                            "is larger than the GPU can hold: reduce expected_max_atoms / "
                            "max_atoms_to_try or the batch size."
                        ) from retry_exc
                    raise
            elif self._is_cuda_oom_error(exc):
                # The native autobatcher should catch OOM itself, but if a batch
                # still overflows, hand the next call a clean allocator and surface
                # a clear error.
                cleanup_torch_cuda(logger=logger)
                raise SCGORuntimeError(
                    "TorchSim relaxation ran out of GPU memory during optimization. "
                    "The sticky max_memory_scaler may be too tight, or the batch is "
                    "larger than the GPU can hold: reduce expected_max_atoms / "
                    "max_atoms_to_try or the batch size."
                ) from exc
            else:
                raise

        batch_steps = _steps_taken_from_optimize_state(state)
        self.last_batch_relax_steps = (
            [batch_steps] * len(atoms_list) if batch_steps is not None else []
        )
        if batch_steps is not None and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "TorchSim relax_batch: %d structures, steps_taken=%d (max_steps=%s)",
                len(atoms_list),
                batch_steps,
                max_steps_now,
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
            # ``state.to_atoms()`` drops tags/constraints/info; restore them from
            # the input so optimize and static share one output contract.
            _reattach_input_metadata(relaxed, reference_atoms[idx])
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

    def __repr__(self) -> str:
        return (
            f"TorchSimBatchRelaxer(model_kind={self.model_kind!r}, "
            f"device={self.device!r}, max_steps={self.max_steps!r})"
        )

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
