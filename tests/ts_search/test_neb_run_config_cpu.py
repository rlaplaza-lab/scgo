"""CPU-only recurrence guard for ``NebRunConfig`` required-field drift.

This module deliberately carries NO ``requires_cuda``/``requires_mace`` marker
so it runs on the CPU CI fast suite. It build-exercises the shared
``_gas_neb_cfg`` helper in ``neb_fixtures`` to catch the class of break where a new *required*
field is added to ``NebRunConfig`` but the test helper is left behind (that
drift only surfaced on Kaggle previously because the MACE/CUDA-gated tests are
deselected by the CPU CI expression).
"""

from __future__ import annotations

from scgo.initialization.initialization_config import CLASH_HARD_CAP_FACTOR
from tests.ts_search.neb_fixtures import _gas_neb_cfg


def test_neb_run_config_builds_with_all_required_fields():
    """Helper must supply every required ``NebRunConfig`` field (gas defaults)."""
    cfg = _gas_neb_cfg()
    assert cfg.neb_prescreen_clash_distance == CLASH_HARD_CAP_FACTOR
    assert cfg.min_saddle_prominence == 0.10
    assert cfg.neb_max_spurious_barrier == 8.0


def test_neb_run_config_surface_override_uses_surface_defaults():
    """Switching system_type must pick surface-family required-field defaults."""
    cfg = _gas_neb_cfg(system_type="surface_cluster")
    assert cfg.neb_prescreen_clash_distance == CLASH_HARD_CAP_FACTOR
    assert cfg.min_saddle_prominence == 0.40
    assert cfg.neb_max_spurious_barrier == 8.0
