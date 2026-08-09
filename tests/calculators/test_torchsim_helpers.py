"""Tests for TorchSim helper functionality.

These tests verify the TorchSimBatchRelaxer interface. Since torch-sim-atomistic
is now a mandatory dependency, these tests expect it to be available.
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


def test_torchsim_import_success():
    """TorchSim and PyTorch are available as core dependencies."""
    import torch
    import torch_sim as ts

    assert torch is not None
    assert ts is not None


def test_torchsim_batch_relaxer_import():
    """Test that TorchSimBatchRelaxer can be imported."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    assert TorchSimBatchRelaxer is not None


def test_geneticalgorithm_go_torchsim_import():
    """Test that ga_go can be imported."""
    from scgo.algorithms import ga_go

    assert ga_go is not None


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


def test_memory_scaler_cache_basic():
    """Test basic MemoryScalerCache functionality."""
    import tempfile

    from scgo.calculators.torchsim_helpers import MemoryScalerCache

    # Create a temporary cache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = MemoryScalerCache(cache_dir=tmpdir)

        # Test set and get
        cache.set(
            n_atoms=32,
            model_name="mace_matpes_0",
            memory_scales_with="n_atoms",
            device="cuda",
            value=100.0,
        )

        # Retrieve the cached value
        cached = cache.get(
            n_atoms=32,
            model_name="mace_matpes_0",
            memory_scales_with="n_atoms",
            device="cuda",
        )
        assert cached == pytest.approx(100.0, rel=1e-6)

        # Test that similar n_atoms values use the same bin
        cached_similar = cache.get(
            n_atoms=33,  # Should bin to same value as 32 (both -> 35)
            model_name="mace_matpes_0",
            memory_scales_with="n_atoms",
            device="cuda",
        )
        assert cached_similar == pytest.approx(100.0, rel=1e-6)

        # Test that different parameters return None
        cached_different = cache.get(
            n_atoms=32,
            model_name="large",  # Different model
            memory_scales_with="n_atoms",
            device="cuda",
        )
        assert cached_different is None


def test_memory_scaler_cache_persistence():
    """Test that MemoryScalerCache persists to disk."""
    import tempfile
    from pathlib import Path

    from scgo.calculators.torchsim_helpers import MemoryScalerCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.json"

        # Create cache and add value
        cache1 = MemoryScalerCache(cache_dir=tmpdir, cache_file="test_cache.json")
        cache1.set(
            n_atoms=50,
            model_name="mace_matpes_0",
            memory_scales_with="n_atoms_x_density",
            device="cpu",
            value=250.5,
        )

        # Verify file was created
        assert cache_path.exists()

        # Create new cache instance and verify it loads the persisted value
        cache2 = MemoryScalerCache(cache_dir=tmpdir, cache_file="test_cache.json")
        cached = cache2.get(
            n_atoms=50,
            model_name="mace_matpes_0",
            memory_scales_with="n_atoms_x_density",
            device="cpu",
        )
        assert cached == pytest.approx(250.5, rel=1e-6)


def test_memory_scaler_cache_clear():
    """Test clearing the cache."""
    import tempfile

    from scgo.calculators.torchsim_helpers import MemoryScalerCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = MemoryScalerCache(cache_dir=tmpdir)

        # Add some values
        cache.set(32, "medium", "n_atoms", "cuda", 100.0)
        cache.set(64, "medium", "n_atoms", "cuda", 200.0)

        # Clear the cache
        cache.clear()

        # Verify values are gone
        assert cache.get(32, "medium", "n_atoms", "cuda") is None
        assert cache.get(64, "medium", "n_atoms", "cuda") is None


def test_get_global_memory_scaler_cache():
    """Test accessing the global cache."""
    from scgo.calculators.torchsim_helpers import get_global_memory_scaler_cache

    cache = get_global_memory_scaler_cache()
    assert cache is not None
    assert hasattr(cache, "get")
    assert hasattr(cache, "set")
    assert hasattr(cache, "clear")


def test_torchsim_step_kwargs_removed():
    """``step_kwargs`` has been removed from TorchSimBatchRelaxer (never forwarded by ts.optimize).

    The field was replaced by ``optimizer_kwargs``; passing the old name should raise TypeError.
    """
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    with pytest.raises(TypeError):
        TorchSimBatchRelaxer(
            device="cpu",
            mace_model_name="mace_matpes_0",
            step_kwargs={"dt_max": 0.1},  # type: ignore[call-arg]
        )


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


@pytest.mark.requires_mace
def test_torchsim_autobatcher_default_off_on_cpu():
    """On CPU the default ``autobatcher=None`` disables autobatching (docs recommendation)."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer(
        device="cpu",
        mace_model_name="mace_matpes_0",
    )
    assert "autobatcher" not in relaxer._runner_kwargs


class _StubModel:
    """Minimal stand-in for a torch-sim model; avoids MACE/UMA downloads in unit tests."""


def test_torchsim_autobatcher_probe_capped_by_expected_max_atoms(monkeypatch):
    """``expected_max_atoms`` should cap the autobatcher's GPU probe (``max_atoms_to_try``).

    Without this cap the probe can geometrically climb to 500k atoms and OOM
    small GPUs. We must never probe more atoms than the workload demands.
    """
    import torch_sim as ts

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    captured: dict = {}

    class _FakeAutoBatcher:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ts, "InFlightAutoBatcher", _FakeAutoBatcher)

    TorchSimBatchRelaxer(
        device="cuda",  # triggers the autobatcher construction path
        model=_StubModel(),
        expected_max_atoms=256,
        autobatcher=True,
    )
    assert captured.get("max_atoms_to_try") == 256


def test_torchsim_autobatcher_probe_cap_explicit_override_is_honored(monkeypatch):
    """An explicit ``max_atoms_to_try`` wins over ``expected_max_atoms``."""
    import torch_sim as ts

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    captured: dict = {}

    class _FakeAutoBatcher:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ts, "InFlightAutoBatcher", _FakeAutoBatcher)

    TorchSimBatchRelaxer(
        device="cuda",
        model=_StubModel(),
        expected_max_atoms=10_000,
        max_atoms_to_try=128,
        autobatcher=True,
    )
    assert captured.get("max_atoms_to_try") == 128


def test_torchsim_autobatcher_probe_cap_defaults_to_torchsim_when_unset(monkeypatch):
    """Without ``expected_max_atoms``/``max_atoms_to_try`` we inherit torch-sim's default."""
    import torch_sim as ts

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    captured: dict = {}

    class _FakeAutoBatcher:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ts, "InFlightAutoBatcher", _FakeAutoBatcher)

    TorchSimBatchRelaxer(
        device="cuda",
        model=_StubModel(),
        autobatcher=True,
    )
    assert "max_atoms_to_try" not in captured


@pytest.mark.requires_mace
def test_torchsim_autobatcher_true_on_cpu_warns_and_coerces(caplog):
    """Passing ``autobatcher=True`` on CPU logs warning and disables autobatching."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer(
        device="cpu",
        mace_model_name="mace_matpes_0",
        autobatcher=True,
    )
    assert any("autobatching" in rec.message.lower() for rec in caplog.records)
    assert "autobatcher" not in relaxer._runner_kwargs


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

    # Check which API version we're using
    if hasattr(ts, "Optimizer"):
        # 0.4.0+ uses enum API
        assert relaxer.optimizer == ts.Optimizer.fire
        # Verify it's an enum (may be StrEnum which is also a str, but that's OK)
        import enum

        assert isinstance(relaxer.optimizer, enum.Enum)
    elif hasattr(ts, "optimizers"):
        # 0.3.0 uses callable function API
        assert relaxer.optimizer == ts.optimizers.fire
        # Verify it's callable
        assert callable(relaxer.optimizer)
        # Verify it's not a string
        assert not isinstance(relaxer.optimizer, str)
    else:
        pytest.fail("Neither ts.Optimizer nor ts.optimizers found")


def test_torchsim_relax_batch_retries_after_max_metric_error(monkeypatch):
    """Stale cached scalers should trigger one retry with a fresh autobatcher."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    relaxer.max_memory_scaler = None
    relaxer._runner_kwargs = {"autobatcher": object()}
    calls = {"count": 0}

    def fake_once(atoms_list, *, steps, max_atoms_in_batch):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError(
                "Max metric of system with index 0 in states: 914.0 is greater "
                "than max_metric 605.0, please set a larger max_metric"
            )
        return [(0.0, atoms_list[0])]

    invalidated: list[int] = []

    monkeypatch.setattr(relaxer, "_relax_batch_once", fake_once)
    monkeypatch.setattr(
        relaxer,
        "_invalidate_memory_scaler_cache",
        lambda n_atoms: invalidated.append(n_atoms),
    )
    monkeypatch.setattr(relaxer, "_reset_autobatcher_memory_scaler", lambda: None)

    from ase import Atoms

    atoms = Atoms("H", positions=[[0, 0, 0]])
    results = relaxer.relax_batch([atoms])
    assert calls["count"] == 2
    assert invalidated == [1]
    assert len(results) == 1


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


def test_static_autobatcher_disabled_by_default():
    """NEB single-point must not enable probing BinningAutoBatcher by default."""
    import torch

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    relaxer.device = torch.device("cuda")
    relaxer.autobatcher = None
    assert relaxer._static_autobatcher_arg(n_structures=84, max_atoms=7) is False
    relaxer.autobatcher = True
    assert relaxer._static_autobatcher_arg(n_structures=10, max_atoms=7) is False
    assert relaxer._static_autobatcher_arg(n_structures=50, max_atoms=10) is True


class _FakeBinningAutoBatcher:
    """Records the kwargs ``ts.static`` would receive as its autobatcher."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _stub_static_relaxer(device: str, *, scaler_on_autobatcher=None):
    """Build a ``TorchSimBatchRelaxer`` skeleton for ``_single_point_batch`` tests."""
    import torch

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    relaxer.max_memory_scaler = None
    relaxer.max_memory_padding = 1.05
    relaxer.memory_scales_with = "n_atoms_x_density"
    relaxer.expected_max_atoms = None
    relaxer.max_steps = 100
    relaxer.device = torch.device(device)
    relaxer.dtype = torch.float64
    relaxer.model = object()
    relaxer.optimizer = object()
    relaxer.model_kind = "mace"
    relaxer.autobatcher = None
    relaxer.last_batch_relax_steps = []
    relaxer._runner_kwargs = {"max_steps": 100}
    if scaler_on_autobatcher is not None:
        # Mirrors production: the warm probe leaves its scaler on the InFlight
        # autobatcher instance, not on the relaxer field.
        autobatcher = type(
            "_InFlight", (), {"max_memory_scaler": scaler_on_autobatcher}
        )()
        relaxer._runner_kwargs["autobatcher"] = autobatcher
    return relaxer


def _install_fake_ts(relaxer, monkeypatch, captured: dict):
    """Give ``relaxer`` a fake ``torch_sim`` module recording ``static`` kwargs."""
    import torch

    class _FakeTS:
        BinningAutoBatcher = _FakeBinningAutoBatcher

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

    relaxer._ts = _FakeTS()
    monkeypatch.setattr(
        "scgo.calculators.torchsim_helpers.build_torchsim_fixatoms_from_ase_batch",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(relaxer, "_uses_metatomic_model", lambda: False)
    # Keep the developer's on-disk scaler cache out of the assertion: this test
    # is about which autobatcher ``ts.static`` receives, not cache lookups.
    monkeypatch.setattr(relaxer, "_apply_cached_memory_scaler", lambda _n: False)


def test_single_point_uses_binning_autobatcher_when_scaler_known(monkeypatch):
    """T2: the NEB force path must bin ``ts.static`` with the cached scaler.

    Without this, the whole fused NEB batch runs as one ``torch.cat`` forward
    pass and OOMs on a 16 GB GPU. ``ts.static`` only accepts
    ``BinningAutoBatcher | bool`` — the InFlight batcher used by ``ts.optimize``
    is rejected — so a fresh binning batcher seeded with the known scaler is the
    only way to reuse the probe result here.
    """
    from ase import Atoms

    relaxer = _stub_static_relaxer("cuda", scaler_on_autobatcher=1234.0)
    captured: dict = {}
    _install_fake_ts(relaxer, monkeypatch, captured)

    relaxer.relax_batch([Atoms("H", positions=[[0.0, 0.0, 0.0]])], steps=0)

    batcher = captured["autobatcher"]
    assert isinstance(batcher, _FakeBinningAutoBatcher)
    assert batcher.kwargs["max_memory_scaler"] == pytest.approx(1234.0)
    assert batcher.kwargs["model"] is relaxer.model
    assert batcher.kwargs["memory_scales_with"] == "n_atoms_x_density"


def test_single_point_stays_unbatched_without_a_scaler(monkeypatch):
    """No known scaler → keep the single-pass behaviour (never probe per call)."""
    from ase import Atoms

    relaxer = _stub_static_relaxer("cuda")
    captured: dict = {}
    _install_fake_ts(relaxer, monkeypatch, captured)

    relaxer.relax_batch([Atoms("H", positions=[[0.0, 0.0, 0.0]])], steps=0)
    assert captured["autobatcher"] is False


def test_single_point_stays_unbatched_on_cpu(monkeypatch):
    """CPU runs never build a BinningAutoBatcher, even with a known scaler."""
    from ase import Atoms

    relaxer = _stub_static_relaxer("cpu", scaler_on_autobatcher=1234.0)
    captured: dict = {}
    _install_fake_ts(relaxer, monkeypatch, captured)

    relaxer.relax_batch([Atoms("H", positions=[[0.0, 0.0, 0.0]])], steps=0)
    assert captured["autobatcher"] is False


def test_memory_scaler_cache_key_separates_geometry_tags():
    """T1: gas and surface probes of the same size must not share a cache slot."""
    import tempfile

    from scgo.calculators.torchsim_helpers import MemoryScalerCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = MemoryScalerCache(cache_dir=tmpdir)
        common = {
            "n_atoms": 4000,
            "model_name": "mace_matpes_0",
            "memory_scales_with": "n_atoms_x_density",
            "device": "cuda",
        }
        cache.set(**common, geometry_tag="neb-gas_cluster|atoms", value=100.0)
        cache.set(**common, geometry_tag="neb-surface_cluster|atoms", value=7.0)

        assert cache.get(**common, geometry_tag="neb-gas_cluster|atoms") == 100.0
        assert cache.get(**common, geometry_tag="neb-surface_cluster|atoms") == 7.0
        # An untagged (legacy-style) lookup must miss rather than alias either.
        assert cache.get(**common) is None


def test_geometry_tag_records_the_probe_shape():
    """A tag that outlives its probe must not alias a bulk probe onto a real one."""
    from ase import Atoms

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    bulk_probe = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    bulk_probe.geometry_tag = "neb-surface_cluster"
    bulk_probe.probe_atoms = None
    bulk_probe.probe_builder = None
    assert bulk_probe._resolve_geometry_tag() == "neb-surface_cluster|bulk"

    real_probe = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    real_probe.geometry_tag = "neb-surface_cluster"
    real_probe.probe_atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    real_probe.probe_builder = None
    assert real_probe._resolve_geometry_tag() == "neb-surface_cluster|atoms"

    untagged = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    untagged.geometry_tag = None
    untagged.probe_atoms = None
    untagged.probe_builder = None
    assert untagged._resolve_geometry_tag() == "default|bulk"


def test_probe_atoms_is_used_verbatim_as_the_warm_probe():
    """T1: the probe must be the real workload geometry, not a dense bulk block.

    torch-sim replicates the probe itself up to ``max_atoms_to_try``, so passing
    one representative structure measures "how many of these fit", which is
    exactly what the binning autobatcher needs.
    """
    import numpy as np
    from ase import Atoms
    from ase.constraints import FixAtoms

    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    slab_like = Atoms(
        "H4",
        positions=np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]]
        ),
        cell=[8.0, 8.0, 20.0],
        pbc=True,
    )
    slab_like.set_constraint(FixAtoms(indices=[0, 1]))

    relaxer = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    relaxer.probe_atoms = slab_like
    relaxer.probe_builder = None

    probe, desc = relaxer._build_probe_atoms(4000)
    assert len(probe) == 4  # one representative structure, not 4000 atoms
    assert np.allclose(probe.get_cell(), slab_like.get_cell())
    assert probe.constraints == []
    assert probe.calc is None
    assert "4 atoms/structure" in desc


def test_probe_falls_back_to_bulk_dummy_without_probe_atoms():
    """Backwards compatibility: the GO/cluster path keeps the bulk-Cu dummy."""
    from scgo.calculators.torchsim_helpers import TorchSimBatchRelaxer

    relaxer = TorchSimBatchRelaxer.__new__(TorchSimBatchRelaxer)
    relaxer.probe_atoms = None
    relaxer.probe_builder = None

    probe, desc = relaxer._build_probe_atoms(64)
    assert len(probe) == 64
    assert set(probe.get_chemical_symbols()) == {"Cu"}
    assert "bulk-Cu" in desc


def test_build_torchsim_relaxer_uma_like_sets_fairchem_kind():
    """Factory cascade: UMA-like calc → fairchem model_kind with shared model."""
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


def _make_fake_mace_module():
    """A tiny ``nn.Module`` whose class name triggers the MACE wrapper branch."""
    import torch

    class ScaleShiftMACE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    return ScaleShiftMACE().to(dtype=torch.float32)


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
