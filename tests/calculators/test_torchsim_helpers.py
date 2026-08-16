"""Tests for TorchSim helper functionality.

These tests verify the TorchSimBatchRelaxer interface. torch-sim-atomistic is a
mandatory dependency, so the native autobatching API is always importable. GPU
tests are gated behind ``requires_cuda`` and run on the Kaggle GPU CI (see
.github/workflows/kaggle-gpu.yml).
"""

import pytest


@pytest.mark.requires_mace
def test_infer_mace_model_name_from_calculator():
    from ase.calculators.calculator import Calculator

    from scgo.calculators.mace_helpers import infer_mace_model_name_from_calculator

    class FakeMace(Calculator):
        implemented_properties: list[str] = []

        def __init__(self, model_name: str) -> None:
            super().__init__(name=f"MACE-{model_name}")
            self.model_name = model_name

    calc = FakeMace("small")
    assert infer_mace_model_name_from_calculator(calc) == "small"
    calc.model_name = "mace_matpes_0"
    assert infer_mace_model_name_from_calculator(calc) == "mace_matpes_0"


@pytest.mark.requires_uma
def test_infer_uma_model_name_from_calculator():
    from types import SimpleNamespace

    from ase.calculators.calculator import Calculator

    from scgo.calculators.uma_helpers import infer_uma_model_name_from_calculator

    class FakeUma(Calculator):
        implemented_properties: list[str] = []

        def __init__(self, model_name: str) -> None:
            super().__init__()
            self.model_name = model_name

    calc = FakeUma("uma-s-1p2")
    assert infer_uma_model_name_from_calculator(calc) == "uma-s-1p2"
    # ASE Calculator.name is a read-only property; name-prefix fallback uses
    # duck-typed objects (e.g. mocks) that expose a string name.
    assert (
        infer_uma_model_name_from_calculator(SimpleNamespace(name="UMA-uma-s-1p2"))
        == "uma-s-1p2"
    )


@pytest.mark.requires_upet
def test_infer_upet_model_name_from_calculator():
    from types import SimpleNamespace

    from ase.calculators.calculator import Calculator

    from scgo.calculators.upet_helpers import infer_upet_model_name_from_calculator

    class FakeUpet(Calculator):
        implemented_properties: list[str] = []

        def __init__(self, model_name: str) -> None:
            super().__init__()
            self.model_name = model_name

    calc = FakeUpet("pet-mad-s")
    assert infer_upet_model_name_from_calculator(calc) == "pet-mad-s"
    assert (
        infer_upet_model_name_from_calculator(
            SimpleNamespace(name="UPET-pet-mad-s-v1.5.0")
        )
        == "pet-mad-s"
    )


@pytest.mark.requires_mace
def test_try_extract_torchsim_model_from_mace_calculator():
    from unittest.mock import MagicMock

    from scgo.calculators.mace_helpers import (
        try_extract_torchsim_model_from_mace_calculator,
    )

    model = object()
    calc = MagicMock()
    calc._mace_calc = MagicMock(models=[model])
    assert try_extract_torchsim_model_from_mace_calculator(calc) is model


@pytest.mark.requires_uma
def test_try_extract_torchsim_model_from_uma_calculator():
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    import torch

    from scgo.calculators import uma_helpers as uh

    predictor = object()
    calc = MagicMock()
    calc._inner = MagicMock(predictor=predictor, task_name="oc25")
    calc.task_name = "oc25"

    class FakeFairChemModel(torch.nn.Module):
        """Mirrors the real ``FairChemModel``, which is an ``nn.Module``."""

    fairchem_mod = SimpleNamespace(FairChemModel=FakeFairChemModel)
    with patch.dict("sys.modules", {"torch_sim.models.fairchem": fairchem_mod}):
        result = uh.try_extract_torchsim_model_from_uma_calculator(calc)

    assert isinstance(result, FakeFairChemModel)
    assert result.predictor is predictor
    assert result.task_name == "oc25"
    assert result._compute_forces is True

    calc_missing = MagicMock(spec=["_inner"])
    calc_missing._inner = MagicMock(spec=[])
    with patch.dict("sys.modules", {"torch_sim.models.fairchem": fairchem_mod}):
        assert uh.try_extract_torchsim_model_from_uma_calculator(calc_missing) is None


@pytest.mark.requires_upet
def test_try_extract_torchsim_model_from_upet_calculator():
    from types import ModuleType
    from unittest.mock import MagicMock, patch

    from scgo.calculators import upet_helpers as uh

    atomistic = object()
    meta_calc = MagicMock()
    meta_calc.model = atomistic
    meta_calc.device = "cpu"

    upet_inner = MagicMock()
    upet_inner.calculator = meta_calc
    upet_inner.non_conservative = False

    calc = MagicMock()
    calc._inner = upet_inner
    calc.non_conservative = False
    calc.device = None

    fake_metatomic = MagicMock(return_value="wrapped")
    neighbors_mod = ModuleType("metatomic_torchsim._neighbors")
    neighbors_mod.HAS_NVALCHEMIOPS = True
    root_mod = ModuleType("metatomic_torchsim")
    root_mod.MetatomicModel = fake_metatomic
    root_mod._neighbors = neighbors_mod

    with patch.dict(
        "sys.modules",
        {
            "metatomic_torchsim": root_mod,
            "metatomic_torchsim._neighbors": neighbors_mod,
        },
    ):
        result = uh.try_extract_torchsim_model_from_upet_calculator(calc)

    assert result == "wrapped"
    fake_metatomic.assert_called_once()
    assert fake_metatomic.call_args[0][0] is atomistic
    assert neighbors_mod.HAS_NVALCHEMIOPS is False

    calc_missing = MagicMock(spec=["_inner"])
    calc_missing._inner = MagicMock(spec=[])
    with patch.dict(
        "sys.modules",
        {
            "metatomic_torchsim": root_mod,
            "metatomic_torchsim._neighbors": neighbors_mod,
        },
    ):
        assert uh.try_extract_torchsim_model_from_upet_calculator(calc_missing) is None


@pytest.mark.requires_mace
def test_torchsim_basic_initialization():
    """Test basic TorchSimBatchRelaxer initialization."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    # Since torch-sim-atomistic is now mandatory, this should work
    relaxer = TorchSimBatchRelaxer(
        device="cpu",
        mace_model_name="mace_matpes_0",
        force_tol=0.05,
        max_steps=10,
    )
    assert relaxer.device is not None
    assert relaxer.force_tol == pytest.approx(0.05, rel=1e-6)
    assert relaxer.max_steps == 10


@pytest.mark.requires_mace
def test_torchsim_optimizer_kwargs_flatten_into_runner_kwargs():
    """``optimizer_kwargs`` should be forwarded flat as **optimizer_kwargs to ts.optimize."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer(
        device="cpu",
        mace_model_name="mace_matpes_0",
        optimizer_kwargs={"dt_max": 0.1},
    )
    # Flattened into the resolved runner kwargs rather than nested.
    assert relaxer._runner_kwargs.get("dt_max") == pytest.approx(0.1)
    assert "optimizer_kwargs" not in relaxer._runner_kwargs
    assert "step_kwargs" not in relaxer._runner_kwargs


class _StubModel:
    """Minimal stand-in for a torch-sim model; avoids MACE/UMA downloads in unit tests."""


def test_torchsim_unknown_model_kind_raises():
    """Unknown model_kind must raise a single validation error."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer
    from scgo.exceptions import SCGOValidationError

    with pytest.raises(SCGOValidationError, match="Unknown model_kind"):
        TorchSimBatchRelaxer(device="cpu", model_kind="bogus")


def test_torchsim_autobatcher_default_off_on_cpu():
    """On CPU the default ``autobatcher=None`` disables autobatching (docs recommendation)."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer(
        device="cpu",
        model=_StubModel(),
    )
    # On CPU the native batchers are never built; the runner gets a plain False.
    assert relaxer._on_cpu is True
    assert relaxer._runner_kwargs["autobatcher"] is False
    assert not hasattr(relaxer, "_optimize_batcher")
    assert not hasattr(relaxer, "_static_batcher")


def test_torchsim_autobatcher_true_on_cpu_is_disabled():
    """Passing ``autobatcher=True`` on CPU must still disable autobatching (no batcher built)."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer(
        device="cpu",
        model=_StubModel(),
        autobatcher=True,
    )
    assert relaxer._on_cpu is True
    assert relaxer._runner_kwargs["autobatcher"] is False


def test_torchsim_autobatcher_true_on_mps_is_disabled():
    """MPS is not CUDA: native batchers must not be built (probe uses torch.cuda)."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer(
        device="mps",
        model=_StubModel(),
        autobatcher=True,
    )
    assert relaxer._on_cpu is False
    assert str(relaxer.device).startswith("mps")
    assert relaxer._runner_kwargs["autobatcher"] is False
    assert not hasattr(relaxer, "_optimize_batcher")
    assert not hasattr(relaxer, "_static_batcher")


def _make_fake_batcher_factory(captured: dict, key: str):
    """Return a fake batcher class that records its constructor kwargs under ``key``."""

    class _FakeAutoBatcher:
        def __init__(self, **kwargs):
            captured[key] = dict(kwargs)

    return _FakeAutoBatcher


def test_torchsim_native_batchers_built_on_gpu_with_probe_cap(monkeypatch):
    """On GPU the relaxer builds real InFlight + Binning batchers capped by ``expected_max_atoms``.

    Native torch-sim autobatching replaces the old disk-cache + warm-probe system:
    both batchers are built once per relaxer and reused (the Binning one protects
    NEB single-point force batches that previously ran as one monolithic forward).
    """
    import torch_sim as ts

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    captured: dict = {}
    monkeypatch.setattr(
        ts, "InFlightAutoBatcher", _make_fake_batcher_factory(captured, "inflight")
    )
    monkeypatch.setattr(
        ts, "BinningAutoBatcher", _make_fake_batcher_factory(captured, "binning")
    )

    stub_model = _StubModel()
    relaxer = TorchSimBatchRelaxer(
        device="cuda",  # triggers the native batcher construction path
        model=stub_model,
        expected_max_atoms=256,
        max_memory_padding=1.05,
        autobatcher=None,
    )
    assert relaxer._on_cpu is False
    # The optimize runner gets the InFlight batcher; the static path gets Binning.
    assert relaxer._runner_kwargs["autobatcher"] is relaxer._optimize_batcher
    assert relaxer._static_batcher is not None

    inflight = captured["inflight"]
    binning = captured["binning"]
    # Both batchers receive the same GPU-probe knobs.
    assert inflight["model"] is stub_model
    assert inflight["memory_scales_with"] == "n_atoms_x_density"
    assert inflight["max_atoms_to_try"] == 256
    assert inflight["max_memory_padding"] == pytest.approx(1.05)
    assert inflight["max_memory_scaler"] is None  # estimated lazily on first use
    # ``cutoff`` must NOT be forwarded for the default metric.
    assert "cutoff" not in inflight
    assert binning["max_atoms_to_try"] == 256


def test_torchsim_native_batchers_forward_cutoff_only_for_n_edges(monkeypatch):
    """``cutoff`` is only forwarded when ``memory_scales_with == "n_edges"`` (0.6.0 API)."""
    import torch_sim as ts

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    captured: dict = {}
    monkeypatch.setattr(
        ts, "InFlightAutoBatcher", _make_fake_batcher_factory(captured, "inflight")
    )
    monkeypatch.setattr(
        ts, "BinningAutoBatcher", _make_fake_batcher_factory(captured, "binning")
    )

    TorchSimBatchRelaxer(
        device="cuda",
        model=_StubModel(),
        expected_max_atoms=256,
        memory_scales_with="n_edges",
        cutoff=4.0,
        autobatcher=None,
    )
    assert captured["inflight"]["cutoff"] == pytest.approx(4.0)
    assert captured["binning"]["cutoff"] == pytest.approx(4.0)


def test_torchsim_autobatcher_probe_cap_explicit_override_is_honored(monkeypatch):
    """An explicit ``max_atoms_to_try`` wins over ``expected_max_atoms``."""
    import torch_sim as ts

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    captured: dict = {}
    monkeypatch.setattr(
        ts, "InFlightAutoBatcher", _make_fake_batcher_factory(captured, "inflight")
    )

    TorchSimBatchRelaxer(
        device="cuda",
        model=_StubModel(),
        expected_max_atoms=10_000,
        max_atoms_to_try=128,
        autobatcher=None,
    )
    assert captured["inflight"]["max_atoms_to_try"] == 128


def test_torchsim_autobatcher_probe_cap_defaults_to_fifty_thousand(monkeypatch):
    """With no ``expected_max_atoms``/``max_atoms_to_try`` we cap the probe at 50k atoms."""
    import torch_sim as ts

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    captured: dict = {}
    monkeypatch.setattr(
        ts, "InFlightAutoBatcher", _make_fake_batcher_factory(captured, "inflight")
    )

    TorchSimBatchRelaxer(
        device="cuda",
        model=_StubModel(),
        autobatcher=None,
    )
    assert captured["inflight"]["max_atoms_to_try"] == 50_000


@pytest.mark.requires_mace
def test_torchsim_optimizer_set_correctly():
    """Test that TorchSimBatchRelaxer sets optimizer correctly for different torch-sim versions.

    This test verifies that the optimizer is set correctly:
    - torch-sim 0.4.0+: ts.Optimizer.fire (enum)
    - torch-sim 0.3.0: ts.optimizers.fire (callable function)
    """
    import torch_sim as ts

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    # Create relaxer with default optimizer_name="fire"
    relaxer = TorchSimBatchRelaxer(
        device="cpu",
        mace_model_name="mace_matpes_0",
        optimizer_name="fire",
        force_tol=0.05,
        max_steps=10,
    )

    # Verify optimizer is set correctly
    assert relaxer.optimizer is not None

    # Current torch-sim (0.4.0+) uses the enum API.
    import enum

    assert relaxer.optimizer == ts.Optimizer.fire
    assert isinstance(relaxer.optimizer, enum.Enum)


def test_single_point_uses_native_binning_batcher_on_gpu(monkeypatch):
    """NEB single-point (``steps=0``) must bin ``ts.static`` via the native BinningAutoBatcher.

    This is the fix for NEB running unprotected: the static path now always uses
    the persistent ``_static_batcher`` built once in ``__post_init__`` on GPU.
    """
    import torch
    from ase import Atoms

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    relaxer._on_cpu = False
    relaxer.last_batch_relax_steps = []
    relaxer.device = torch.device("cuda")
    relaxer.dtype = torch.float64
    relaxer.model = object()
    relaxer.model_kind = "mace"
    relaxer.max_memory_padding = 1.05
    relaxer.memory_scales_with = "n_atoms_x_density"
    relaxer._runner_kwargs = {}

    class _FakeBinning:
        pass

    class _FakeTS:
        def static(self, **kwargs):
            captured["autobatcher"] = kwargs["autobatcher"]
            n = len(kwargs["system"]) if isinstance(kwargs["system"], list) else 1
            return [
                {
                    "potential_energy": torch.tensor([1.5]),
                    "forces": torch.zeros((1, 3)),
                }
                for _ in range(n)
            ]

        def optimize(self, **kwargs):
            raise AssertionError("optimize must not be used for steps=0")

        def initialize_state(self, *args, **kwargs):
            raise AssertionError("unused in this stub path")

    captured: dict = {}
    relaxer._ts = _FakeTS()
    relaxer._static_batcher = _FakeBinning()
    monkeypatch.setattr(
        "scgo.calculators.torchsim_helpers.build_torchsim_fixatoms_from_ase_batch",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(relaxer, "_uses_metatomic_model", lambda: False)

    relaxer.relax_batch([Atoms("H", positions=[[0.0, 0.0, 0.0]])], steps=0)
    assert captured["autobatcher"] is relaxer._static_batcher
    assert isinstance(captured["autobatcher"], _FakeBinning)


def test_single_point_unbatched_on_cpu(monkeypatch):
    """On CPU the single-point path passes ``autobatcher=False`` (no Binning batcher)."""
    import torch
    from ase import Atoms

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    relaxer._on_cpu = True
    relaxer.last_batch_relax_steps = []
    relaxer.device = torch.device("cpu")
    relaxer.dtype = torch.float64
    relaxer.model = object()
    relaxer.model_kind = "mace"
    relaxer.max_memory_padding = 1.05
    relaxer.memory_scales_with = "n_atoms_x_density"
    relaxer._runner_kwargs = {}

    class _FakeTS:
        def static(self, **kwargs):
            captured["autobatcher"] = kwargs["autobatcher"]
            n = len(kwargs["system"]) if isinstance(kwargs["system"], list) else 1
            return [
                {
                    "potential_energy": torch.tensor([1.5]),
                    "forces": torch.zeros((1, 3)),
                }
                for _ in range(n)
            ]

        def optimize(self, **kwargs):
            raise AssertionError("optimize must not be used for steps=0")

        def initialize_state(self, *args, **kwargs):
            raise AssertionError("unused in this stub path")

    captured: dict = {}
    relaxer._ts = _FakeTS()
    monkeypatch.setattr(
        "scgo.calculators.torchsim_helpers.build_torchsim_fixatoms_from_ase_batch",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(relaxer, "_uses_metatomic_model", lambda: False)

    relaxer.relax_batch([Atoms("H", positions=[[0.0, 0.0, 0.0]])], steps=0)
    assert captured["autobatcher"] is False


def test_build_torchsim_relaxer_uma_like_sets_fairchem_kind():
    """Factory cascade: UMA-like calc -> fairchem model_kind with shared model."""
    from unittest.mock import MagicMock, patch

    from scgo.calculators.torchsim_helpers import build_torchsim_relaxer

    shared_model = object()

    class FakeUMA:
        name = "UMA-uma-s-1p2"
        model_name = "uma-s-1p2"
        task_name = "oc25"

        def __init__(self) -> None:
            self._inner: object | None = object()

    calc = FakeUMA()
    captured: dict = {}

    def _fake_relaxer(**kwargs):
        captured.update(kwargs)
        return MagicMock(**kwargs)

    with (
        patch(
            "scgo.utils.torchsim_policy.is_uma_like_calculator",
            return_value=True,
        ),
        patch(
            "scgo.calculators.uma_helpers.try_extract_torchsim_model_from_uma_calculator",
            return_value=shared_model,
        ),
        patch(
            "scgo.calculators.torchsim_helpers.TorchSimBatchRelaxer",
            side_effect=_fake_relaxer,
        ),
    ):
        relaxer = build_torchsim_relaxer(
            calc,
            fmax=0.05,
            max_steps=50,
            expected_max_atoms=100,
        )

    assert relaxer is not None
    assert captured["model_kind"] == "fairchem"
    assert captured["fairchem_model_name"] == "uma-s-1p2"
    assert captured["fairchem_task_name"] == "oc25"
    assert captured["model"] is shared_model
    assert captured["force_tol"] == 0.05
    assert captured["max_steps"] == 50
    assert calc._inner is None


def test_filter_torchsim_params_drops_unknown_keys_with_warning():
    """Stale ``torchsim_params`` keys (probe/geometry) are dropped with a DeprecationWarning."""
    from scgo.calculators.torchsim_helpers import _filter_torchsim_params

    with pytest.warns(DeprecationWarning):
        filtered = _filter_torchsim_params(
            {
                "force_tol": 0.1,  # known field, kept
                "expected_max_atoms": 100,  # known field, kept
                "probe_atoms": object(),  # removed
                "geometry_tag": "neb-surface_cluster",  # removed
                "probe_builder": object(),  # removed
            }
        )

    # Known fields kept; unknown fields dropped.
    assert filtered["force_tol"] == pytest.approx(0.1)
    assert filtered["expected_max_atoms"] == 100
    assert "probe_atoms" not in filtered
    assert "geometry_tag" not in filtered
    assert "probe_builder" not in filtered


def test_filter_torchsim_params_none_returns_empty():
    """No ``torchsim_params`` yields an empty dict (no warning)."""
    from scgo.calculators.torchsim_helpers import _filter_torchsim_params

    assert _filter_torchsim_params(None) == {}
    assert _filter_torchsim_params({}) == {}


@pytest.mark.requires_mace
def test_ensure_torchsim_mace_wrapper_does_not_upcast_shared_module(monkeypatch):
    """K4: wrapping must not cast the live ASE calculator's weights in place."""
    import torch
    import torch_sim.models.mace as ts_mace

    from scgo.calculators.torchsim_helpers import _ensure_torchsim_mace_wrapper

    class _FakeMaceModel:
        """Mirrors ``MaceModel``: casts the module it is given, in place."""

        def __init__(self, *, model, device, dtype, compute_forces, compute_stress):
            self.model = model.to(dtype=dtype)
            self.device = device
            self.dtype = dtype
            self.compute_forces = compute_forces
            self.compute_stress = compute_stress

    monkeypatch.setattr(ts_mace, "MaceModel", _FakeMaceModel)

    source = _make_fake_mace_module()
    assert next(source.parameters()).dtype == torch.float32

    wrapper = _ensure_torchsim_mace_wrapper(source, torch.device("cpu"), torch.float64)

    assert isinstance(wrapper, _FakeMaceModel)
    assert next(wrapper.model.parameters()).dtype == torch.float64
    # The shared ASE calculator module must keep its own precision.
    assert next(source.parameters()).dtype == torch.float32
    assert wrapper.model is not source


@pytest.mark.requires_mace
def test_ensure_torchsim_mace_wrapper_falls_back_when_deepcopy_fails(
    monkeypatch, caplog
):
    """K4: an uncopyable module still gets wrapped (with a warning)."""
    import copy

    import torch
    import torch_sim.models.mace as ts_mace

    from scgo.calculators.torchsim_helpers import _ensure_torchsim_mace_wrapper

    class _FakeMaceModel:
        def __init__(self, *, model, device, dtype, compute_forces, compute_stress):
            self.model = model
            self.device = device
            self.dtype = dtype

    monkeypatch.setattr(ts_mace, "MaceModel", _FakeMaceModel)

    def _boom(_obj, _memo=None):
        raise TypeError("cannot deepcopy this module")

    monkeypatch.setattr(copy, "deepcopy", _boom)

    source = _make_fake_mace_module()
    with caplog.at_level("WARNING", logger="scgo.calculators.torchsim_helpers"):
        wrapper = _ensure_torchsim_mace_wrapper(
            source, torch.device("cpu"), torch.float64
        )

    assert wrapper.model is source
    assert any("deep-copy" in rec.message for rec in caplog.records)


def _make_fake_mace_module():
    """A tiny ``nn.Module`` whose class name triggers the MACE wrapper branch."""
    import torch

    class ScaleShiftMACE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    return ScaleShiftMACE().to(dtype=torch.float32)


def test_relaxer_max_steps_is_resolved_at_call_time(monkeypatch):
    """K5: mutating ``relaxer.max_steps`` after construction must take effect."""
    import torch
    from ase import Atoms

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer(
        device="cpu",
        model=_StubModel(),
        max_steps=3,
        force_tol=None,
    )
    # Never baked into the runner kwargs (that is what made mutation a no-op).
    assert "max_steps" not in relaxer._runner_kwargs

    captured: list[dict] = []

    class _FakeState:
        energy = torch.tensor([1.25])

        @staticmethod
        def to_atoms():
            return [Atoms("H", positions=[[0.0, 0.0, 0.0]])]

    class _FakeTS:
        def optimize(self, **kwargs):
            captured.append(kwargs)
            return _FakeState()

    relaxer._ts = _FakeTS()
    monkeypatch.setattr(
        "scgo.calculators.torchsim_helpers.build_torchsim_fixatoms_from_ase_batch",
        lambda *_a, **_k: None,
    )

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    relaxer.relax_batch([atoms])
    assert captured[-1]["max_steps"] == 3

    relaxer.max_steps = 7
    relaxer.relax_batch([atoms])
    assert captured[-1]["max_steps"] == 7

    # An explicit per-call override still wins over the field.
    relaxer.relax_batch([atoms], steps=2)
    assert captured[-1]["max_steps"] == 2


def test_relaxer_runner_kwargs_max_steps_seeds_the_field():
    """K5: a caller-supplied ``runner_kwargs['max_steps']`` keeps working."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer(
        device="cpu",
        model=_StubModel(),
        max_steps=100,
        runner_kwargs={"max_steps": 42},
    )
    assert "max_steps" not in relaxer._runner_kwargs
    assert relaxer.max_steps == 42


def test_relax_batch_preserves_tags_constraints_and_info(monkeypatch):
    """K9: the optimize path must return the same metadata as the static path."""
    import numpy as np
    import torch
    from ase import Atoms
    from ase.constraints import FixAtoms

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    def _make_input(shift: float) -> Atoms:
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74 + shift]])
        atoms.set_tags([0, 1])
        atoms.set_constraint(FixAtoms(indices=[0]))
        atoms.info["key_value_pairs"] = {"gaid": 11}
        atoms.info["confid"] = 3
        return atoms

    inputs = [_make_input(0.0), _make_input(0.1)]

    relaxer = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    relaxer.max_memory_scaler = None
    relaxer.max_steps = 1
    relaxer.expected_max_atoms = None
    relaxer._runner_kwargs = {}
    relaxer.device = torch.device("cpu")
    relaxer.dtype = torch.float64
    relaxer.model = object()
    relaxer.optimizer = object()
    relaxer.model_kind = "mace"
    relaxer.autobatcher = False
    relaxer.last_batch_relax_steps = []

    class _FakeState:
        energy = torch.tensor([-1.0, -2.0])
        forces = torch.zeros((4, 3))

        @staticmethod
        def to_atoms():
            # ``SimState.to_atoms`` returns bare Atoms: no tags, no constraints,
            # no info.
            return [
                Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
                for _ in range(2)
            ]

    class _FakeTS:
        def optimize(self, **_kwargs):
            return _FakeState()

    relaxer._ts = _FakeTS()
    monkeypatch.setattr(
        "scgo.calculators.torchsim_helpers.build_torchsim_fixatoms_from_ase_batch",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(relaxer, "_uses_metatomic_model", lambda: False)

    results = relaxer.relax_batch(inputs, steps=1)
    assert len(results) == 2

    for src, (_energy, out) in zip(inputs, results, strict=True):
        assert np.array_equal(out.get_tags(), src.get_tags())
        assert len(out.constraints) == 1
        assert isinstance(out.constraints[0], FixAtoms)
        assert list(out.constraints[0].index) == [0]
        assert out.constraints[0] is not src.constraints[0]
        assert out.info["confid"] == 3
        assert out.info["key_value_pairs"]["gaid"] == 11
        # The input must not be aliased by the tag writes.
        assert "raw_score" not in src.info["key_value_pairs"]

    # Mutating the output tags must not touch the input.
    _energy, first_out = results[0]
    first_out.set_tags([5, 5])
    assert np.array_equal(inputs[0].get_tags(), [0, 1])


def test_relax_batch_steps_zero_routes_to_static(monkeypatch):
    """``steps=0`` must use ``ts.static``, not ``optimize(max_steps=0)``."""
    import numpy as np
    import torch
    from ase import Atoms

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    relaxer.max_memory_scaler = None
    relaxer._runner_kwargs = {"max_steps": 100}
    relaxer.max_steps = 100
    relaxer._on_cpu = True
    relaxer._static_batcher = None
    relaxer.device = torch.device("cpu")
    relaxer.dtype = torch.float64
    relaxer.model = object()
    relaxer.optimizer = object()
    relaxer.model_kind = "mace"
    relaxer.autobatcher = False
    relaxer.last_batch_relax_steps = []

    class _FakeTS:
        def static(self, **kwargs):
            calls["static"] += 1
            assert "optimizer" not in kwargs
            n = len(kwargs["system"]) if isinstance(kwargs["system"], list) else 1
            return [
                {
                    "potential_energy": torch.tensor([1.5]),
                    "forces": torch.zeros((1, 3)),
                }
                for _ in range(n)
            ]

        def optimize(self, **kwargs):
            calls["optimize"] += 1
            raise AssertionError("optimize must not be used for steps=0")

        def initialize_state(self, *args, **kwargs):
            raise AssertionError("unused in this stub path")

    calls = {"static": 0, "optimize": 0}
    relaxer._ts = _FakeTS()
    monkeypatch.setattr(
        "scgo.calculators.torchsim_helpers.build_torchsim_fixatoms_from_ase_batch",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(relaxer, "_uses_metatomic_model", lambda: False)

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    results = relaxer.relax_batch([atoms], steps=0)
    assert calls["static"] == 1
    assert calls["optimize"] == 0
    assert len(results) == 1
    energy, out = results[0]
    assert energy == pytest.approx(1.5)
    assert np.allclose(out.get_positions(), [[0.0, 0.0, 0.0]])
    assert "forces" in out.arrays
    assert relaxer.last_batch_relax_steps == [0]


def test_single_point_unbatched_when_autobatcher_false_on_gpu(monkeypatch):
    """Explicit ``autobatcher=False`` on GPU (no ``_static_batcher``) still passes False."""
    import torch
    from ase import Atoms

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    relaxer._on_cpu = False  # GPU, but autobatching explicitly disabled
    relaxer.last_batch_relax_steps = []
    relaxer.device = torch.device("cuda")
    relaxer.dtype = torch.float64
    relaxer.model = object()
    relaxer.model_kind = "mace"
    relaxer.max_memory_padding = 1.05
    relaxer.memory_scales_with = "n_atoms_x_density"
    relaxer._runner_kwargs = {"autobatcher": False}  # no native batcher built
    # Note: ``_static_batcher`` is intentionally absent (autobatcher=False).

    class _FakeTS:
        def static(self, **kwargs):
            captured["autobatcher"] = kwargs["autobatcher"]
            n = len(kwargs["system"]) if isinstance(kwargs["system"], list) else 1
            return [
                {
                    "potential_energy": torch.tensor([1.5]),
                    "forces": torch.zeros((1, 3)),
                }
                for _ in range(n)
            ]

        def optimize(self, **kwargs):
            raise AssertionError("optimize must not be used for steps=0")

        def initialize_state(self, *args, **kwargs):
            raise AssertionError("unused in this stub path")

    captured: dict = {}
    relaxer._ts = _FakeTS()
    monkeypatch.setattr(
        "scgo.calculators.torchsim_helpers.build_torchsim_fixatoms_from_ase_batch",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(relaxer, "_uses_metatomic_model", lambda: False)

    relaxer.relax_batch([Atoms("H", positions=[[0.0, 0.0, 0.0]])], steps=0)
    assert captured["autobatcher"] is False


@pytest.mark.requires_cuda
@pytest.mark.requires_mace
def test_torchsim_native_autobatching_on_gpu():
    """Kaggle GPU gate: native torch-sim autobatching relaxes a tiny structure.

    Exercises the exact API the refactor relies on (built-in LennardJonesModel,
    no external weights) so the Kaggle GPU CI confirms the new autobatcher logic
    is wired correctly and produces finite energies without manual probing.
    """
    import torch
    import torch_sim as ts
    from ase.build import bulk
    from torch_sim.models.lennard_jones import LennardJonesModel

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    dev = torch.device("cuda")
    model = LennardJonesModel(
        sigma=3.405,
        epsilon=0.0104,
        cutoff=8.5,
        device=dev,
        compute_forces=True,
        compute_stress=True,
    )

    relaxer = TorchSimBatchRelaxer(
        model=model,
        model_kind="mace",
        force_tol=0.1,
        max_steps=3,
        expected_max_atoms=2_000,
        memory_scales_with="n_atoms_x_density",
        max_memory_padding=1.05,
        device=dev,
        dtype=torch.float64,
        autobatcher=None,
    )

    # Native batchers are built once and reused (the NEB single-point path relies
    # on the Binning batcher; optimize relies on the InFlight batcher).
    assert isinstance(relaxer._optimize_batcher, ts.InFlightAutoBatcher)
    assert isinstance(relaxer._static_batcher, ts.BinningAutoBatcher)
    # Scaler is estimated lazily on first use.
    assert relaxer._optimize_batcher.max_memory_scaler is None

    tiny = [bulk("Cu", "fcc", a=3.61, cubic=True).repeat((2, 2, 2))]
    out = relaxer.relax_batch(tiny, steps=3)
    assert len(out) == 1
    energy, atoms = out[0]
    assert float("nan") not in (energy,)
    assert torch.isfinite(torch.as_tensor(energy)).all()
    assert len(atoms) == len(tiny[0])


def test_autobatcher_none_on_cpu_disables_native_batchers():
    """``autobatcher=None`` on CPU builds no native batchers; single-point stays unbatched.

    This guards the CPU path that the hardening plan calls out as the one that would
    raise if a native InFlight/Binning batcher were ever constructed on CPU. The
    optimize and static runners must receive a plain ``False`` (no autobatching).
    """
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer(
        device="cpu",
        model=_StubModel(),
        autobatcher=None,
    )
    assert relaxer._on_cpu is True
    # No native InFlight/Binning batchers are built on CPU.
    assert not hasattr(relaxer, "_optimize_batcher")
    assert not hasattr(relaxer, "_static_batcher")
    # The optimize/static runners receive a plain ``False`` (no autobatching).
    assert relaxer._runner_kwargs["autobatcher"] is False


def test_relax_batch_retries_on_max_metric_sticky_scaler(monkeypatch):
    """A too-tight sticky scaler must re-probe once, not hard-crash.

    torch-sim's InFlight/Binning ``max_memory_scaler`` is estimated once and then
    stays sticky; a later larger batch raises ``ValueError("... > max_metric ...")``
    instead of being re-binned. The relaxer must reset the scalers and retry the
    call rather than surfacing an uncaught error that ``parallel_neb`` would treat
    as an ordinary "search failed".
    """
    import torch
    from ase import Atoms

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    relaxer._on_cpu = True
    relaxer.max_memory_scaler = None
    relaxer._runner_kwargs = {"max_steps": 100}
    relaxer.max_steps = 100
    relaxer.device = torch.device("cpu")
    relaxer.dtype = torch.float64
    relaxer.model = object()
    relaxer.optimizer = object()
    relaxer.model_kind = "mace"
    relaxer.autobatcher = False
    relaxer.last_batch_relax_steps = []
    relaxer.memory_scales_with = "n_atoms_x_density"
    relaxer.max_memory_padding = 1.05

    calls = {"count": 0}

    class _FakeState:
        energy = torch.tensor([1.0])

        @staticmethod
        def to_atoms():
            return [Atoms("H", positions=[[0.0, 0.0, 0.0]])]

    class _FakeTS:
        def optimize(self, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise ValueError(
                    "Max metric of system with index 0 in states: 914.0 is greater "
                    "than max_metric 605.0, please set a larger max_metric"
                )
            return _FakeState()

    relaxer._ts = _FakeTS()
    monkeypatch.setattr(
        "scgo.calculators.torchsim_helpers.build_torchsim_fixatoms_from_ase_batch",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(relaxer, "_uses_metatomic_model", lambda: False)

    class _Batcher:
        def __init__(self) -> None:
            self.max_memory_scaler = 605.0

    relaxer._optimize_batcher = _Batcher()
    relaxer._static_batcher = _Batcher()

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    results = relaxer.relax_batch([atoms])
    assert calls["count"] == 2
    assert len(results) == 1
    assert relaxer._optimize_batcher.max_memory_scaler is None
    assert relaxer._static_batcher.max_memory_scaler is None
    assert relaxer.max_memory_scaler is None


def test_single_point_retries_on_max_metric_sticky_scaler(monkeypatch):
    """``ts.static`` must re-probe once on a sticky max_metric ValueError."""
    import torch
    from ase import Atoms

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    relaxer._on_cpu = True
    relaxer.max_memory_scaler = None
    relaxer._runner_kwargs = {}
    relaxer.max_steps = 100
    relaxer.device = torch.device("cpu")
    relaxer.dtype = torch.float64
    relaxer.model = object()
    relaxer.model_kind = "mace"
    relaxer.last_batch_relax_steps = []

    calls = {"count": 0}

    class _Batcher:
        def __init__(self) -> None:
            self.max_memory_scaler = 605.0

    class _FakeTS:
        def static(self, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise ValueError(
                    "Max metric of system with index 0 in states: 914.0 is greater "
                    "than max_metric 605.0, please set a larger max_metric"
                )
            n = len(kwargs["system"]) if isinstance(kwargs["system"], list) else 1
            return [
                {
                    "potential_energy": torch.tensor([1.5]),
                    "forces": torch.zeros((1, 3)),
                }
                for _ in range(n)
            ]

        def optimize(self, **kwargs):
            raise AssertionError("optimize must not be used for steps=0")

    relaxer._ts = _FakeTS()
    relaxer._static_batcher = _Batcher()
    monkeypatch.setattr(
        "scgo.calculators.torchsim_helpers.build_torchsim_fixatoms_from_ase_batch",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(relaxer, "_uses_metatomic_model", lambda: False)

    results = relaxer.relax_batch([Atoms("H", positions=[[0.0, 0.0, 0.0]])], steps=0)
    assert calls["count"] == 2
    assert len(results) == 1
    assert relaxer._static_batcher.max_memory_scaler is None


def test_patch_model_for_cuda_wraps_each_instance():
    """A second model of the same class must still receive the CUDA setup wrap."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    class _Model:
        def setup_from_system_idx(self, atomic_numbers, system_idx):
            self.atomic_numbers = atomic_numbers
            return "ok"

    first = _Model()
    second = _Model()
    relaxer_a = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    relaxer_a.model = first
    relaxer_a._patch_model_for_cuda()
    relaxer_b = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    relaxer_b.model = second
    relaxer_b._patch_model_for_cuda()

    assert first._scgo_setup_patched is True
    assert second._scgo_setup_patched is True
    assert first.setup_from_system_idx is not _Model.setup_from_system_idx
    assert second.setup_from_system_idx is not _Model.setup_from_system_idx
    assert not getattr(_Model, "_scgo_setup_patched", False)


def _make_lj_relaxer(max_steps=10, force_tol=0.05):
    """CPU TorchSim relaxer backed by the built-in LennardJonesModel (no weights)."""
    import torch
    from torch_sim.models.lennard_jones import LennardJonesModel

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    model = LennardJonesModel(
        sigma=3.405,
        epsilon=0.0104,
        cutoff=8.5,
        device=torch.device("cpu"),
        compute_forces=True,
        compute_stress=False,
    )
    return TorchSimBatchRelaxer(
        model=model,
        model_kind="mace",
        device=torch.device("cpu"),
        force_tol=force_tol,
        max_steps=max_steps,
    )


def test_torchsim_fixbondlengths_preserved_during_relaxation():
    """P3: FixBondLengths bond length is held within 1e-3 A through relaxation."""
    from ase import Atoms
    from ase.constraints import FixBondLengths

    relaxer = _make_lj_relaxer(max_steps=10)

    init_length = 1.5
    atoms = Atoms(
        "H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, init_length]], cell=[20, 20, 20]
    )
    atoms.set_constraint(FixBondLengths([(0, 1)]))

    results = relaxer.relax_batch([atoms])
    assert len(results) == 1
    _energy, relaxed = results[0]

    final_length = relaxed.get_distance(0, 1)
    assert abs(final_length - init_length) <= 1e-3


def test_torchsim_fixbondlengths_survives_upet_zeroed_slab_cell():
    """UPET cell prep zeros vacuum; FIRE + FixBondLengths must still relax.

    Reproduces the Kaggle full-suite crash: ``model_kind='upet'`` zeros the
    non-periodic lattice vector, then ``ts.optimize`` inverts that cell in
    ``TorchSimFixBondLengths.adjust_forces``.
    """
    import torch
    from ase import Atoms
    from ase.constraints import FixBondLengths
    from torch_sim.models.lennard_jones import LennardJonesModel

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    model = LennardJonesModel(
        sigma=3.405,
        epsilon=0.0104,
        cutoff=8.5,
        device=torch.device("cpu"),
        compute_forces=True,
        compute_stress=False,
    )
    relaxer = TorchSimBatchRelaxer(
        model=model,
        model_kind="upet",
        device=torch.device("cpu"),
        force_tol=0.05,
        max_steps=10,
    )

    init_length = 0.96
    atoms = Atoms(
        "OH",
        positions=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0 + init_length]],
        cell=[10.0, 10.0, 20.0],
        pbc=(True, True, False),
    )
    atoms.set_constraint(FixBondLengths([(0, 1)]))

    results = relaxer.relax_batch([atoms])
    assert len(results) == 1
    _energy, relaxed = results[0]
    assert abs(relaxed.get_distance(0, 1, mic=True) - init_length) <= 1e-3
    # Storage cell/PBC restored after metatomic prep.
    assert list(relaxed.pbc) == [True, True, False]
    assert relaxed.cell[2, 2] == pytest.approx(20.0)


def test_torchsim_fixbondlengths_preserved_in_batch():
    """P3: both FixAtoms and FixBondLengths survive in a batched relax."""
    import torch
    from ase import Atoms
    from ase.constraints import FixAtoms, FixBondLengths

    relaxer = _make_lj_relaxer(max_steps=10)

    # s0: H2 with a fixed bond (atom 1 is free to rotate/translate).
    s0 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]], cell=[20, 20, 20])
    s0.set_constraint(FixBondLengths([(0, 1)]))
    # s1: H2 with atom 0 fixed (atom 1 must move toward the LJ well).
    s1 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]], cell=[20, 20, 20])
    fixed_pos = s1.positions[0].copy()
    s1.set_constraint(FixAtoms(indices=[0]))

    results = relaxer.relax_batch([s0, s1])
    assert len(results) == 2

    _e0, r0 = results[0]
    assert abs(r0.get_distance(0, 1) - 1.3) <= 1e-3

    _e1, r1 = results[1]
    # Fixed atom must not move; the free atom is pulled to the LJ equilibrium.
    moved = torch.linalg.norm(
        torch.as_tensor(r1.positions[0]) - torch.as_tensor(fixed_pos)
    ).item()
    assert moved <= 1e-6
    assert r1.get_distance(0, 1) > 1.5


def test_torchsim_fixatoms_atom_does_not_move_during_relaxation():
    """P3: FixAtoms-fixed atoms are stationary after relaxation."""
    import torch
    from ase import Atoms
    from ase.constraints import FixAtoms

    relaxer = _make_lj_relaxer(max_steps=10)

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]], cell=[20, 20, 20])
    fixed_before = atoms.positions[0].copy()
    atoms.set_constraint(FixAtoms(indices=[0]))

    _energy, relaxed = relaxer.relax_batch([atoms])[0]

    moved = torch.linalg.norm(
        torch.as_tensor(relaxed.positions[0]) - torch.as_tensor(fixed_before)
    ).item()
    assert moved <= 1e-6
    # The free atom should have relaxed toward the LJ minimum (~3.8 A).
    assert relaxed.get_distance(0, 1) > 1.5


def test_torchsim_061_exposes_detach_state_graph():
    """Pinned torch-sim 0.6.1 detaches FIRE states before the autobatcher split."""
    torch_sim = pytest.importorskip("torch_sim")
    from importlib.metadata import version

    assert version("torch-sim-atomistic") == "0.6.1"
    assert hasattr(torch_sim, "detach_state_graph") or hasattr(
        torch_sim.state, "detach_state_graph"
    )


def test_relax_batch_fixatoms_with_graph_carrying_forces():
    """Graph-carrying forces + FixAtoms must not crash ts.optimize split/pop.

    Production GPU GO failed with SplitWithSizesBackward0 when FIRE states still
    had requires_grad and FixAtoms wrote inplace after autobatcher split.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_sim")
    from ase import Atoms
    from ase.constraints import FixAtoms
    from torch_sim.models.interface import ModelInterface

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    class GraphForcesModel(ModelInterface):
        def __init__(self, device, dtype) -> None:
            super().__init__()
            self._device = torch.device(device)
            self._dtype = dtype
            self._compute_forces = True
            self._compute_stress = False
            self._memory_scales_with = "n_atoms"

        def forward(self, state, **_kwargs):
            pos = state.positions.detach().to(dtype=self._dtype).requires_grad_(True)
            per_atom = pos.pow(2).sum(dim=-1)
            energy = torch.zeros(state.n_systems, device=pos.device, dtype=pos.dtype)
            energy = energy.index_add(0, state.system_idx, per_atom)
            (forces,) = torch.autograd.grad(energy.sum(), pos, create_graph=True)
            return {"energy": energy, "forces": -forces}

    def _h2(xshift: float) -> Atoms:
        atoms = Atoms(
            "H2",
            positions=[[xshift, 0.0, 0.0], [xshift, 0.0, 0.74]],
        )
        atoms.center(vacuum=4.0)
        atoms.set_constraint(FixAtoms(indices=[0]))
        return atoms

    relaxer = TorchSimBatchRelaxer(
        model=GraphForcesModel(device="cpu", dtype=torch.float64),
        device="cpu",
        dtype=torch.float64,
        force_tol=1.0,
        max_steps=2,
        autobatcher=False,
    )
    results = relaxer.relax_batch([_h2(0.0), _h2(1.0)], steps=2)
    assert len(results) == 2
    for energy, atoms in results:
        assert isinstance(energy, float)
        assert len(atoms) == 2


def test_autobatcher_memory_estimation_prints_suppressed_with_summary(
    monkeypatch, capfd
):
    """Suppress torch-sim memory prints; one INFO summary per kind, re-announce after reset."""
    import torch
    from ase import Atoms

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer
    from scgo.utils.logging import configure_logging

    configure_logging(1)
    relaxer = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    relaxer._on_cpu = False
    relaxer.max_memory_scaler = None
    relaxer.max_atoms_to_try = 256
    relaxer.expected_max_atoms = 256
    relaxer._runner_kwargs = {"max_steps": 10}
    relaxer.max_steps = 10
    relaxer.device = torch.device("cuda")
    relaxer.dtype = torch.float64
    relaxer.model = object()
    relaxer.optimizer = object()
    relaxer.model_kind = "mace"
    relaxer.last_batch_relax_steps = []
    relaxer._announced_autobatcher_kinds = set()

    class _Batcher:
        max_memory_scaler = None

    class _FakeState:
        energy = torch.tensor([1.0])

        @staticmethod
        def to_atoms():
            return [Atoms("H", positions=[[0.0, 0.0, 0.0]])]

    class _FakeTS:
        def optimize(self, **kwargs):
            print(
                "Model Memory Estimation: Running forward pass on state with 1 atoms."
            )
            relaxer._optimize_batcher.max_memory_scaler = 42.5
            return _FakeState()

        def static(self, **kwargs):
            print(
                "Model Memory Estimation: Running forward pass on state with 1 atoms."
            )
            relaxer._static_batcher.max_memory_scaler = 18.0
            return [
                {
                    "potential_energy": torch.tensor([1.5]),
                    "forces": torch.zeros((1, 3)),
                }
            ]

    relaxer._ts = _FakeTS()
    relaxer._optimize_batcher = _Batcher()
    relaxer._static_batcher = _Batcher()
    monkeypatch.setattr(
        "scgo.calculators.torchsim_helpers.build_torchsim_fixatoms_from_ase_batch",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(relaxer, "_uses_metatomic_model", lambda: False)
    monkeypatch.setattr(
        "scgo.calculators.torchsim_helpers.cleanup_torch_cuda",
        lambda **_k: None,
    )

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    relaxer.relax_batch([atoms])
    out = capfd.readouterr().out
    assert "Model Memory Estimation" not in out
    assert out.count("TorchSim autobatcher (relax)") == 1
    assert "max_memory_scaler=42.5" in out

    relaxer.relax_batch([atoms])
    out = capfd.readouterr().out
    assert "Model Memory Estimation" not in out
    assert "TorchSim autobatcher (relax)" not in out

    relaxer._reset_and_reprobe()
    relaxer.relax_batch([atoms])
    assert capfd.readouterr().out.count("TorchSim autobatcher (relax)") == 1

    relaxer.relax_batch([atoms], steps=0)
    out = capfd.readouterr().out
    assert "Model Memory Estimation" not in out
    assert out.count("TorchSim autobatcher (single-point)") == 1
