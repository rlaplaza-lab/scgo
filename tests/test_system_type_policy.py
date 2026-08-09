"""Regression tests for :func:`scgo.system_types.get_system_policy` lookups.

An unknown ``system_type`` used to leak a bare ``KeyError`` from the policy
table instead of a user-facing ``SCGOValidationError``.
"""

from __future__ import annotations

import pytest

from scgo.exceptions import SCGOValidationError
from scgo.system_types import SYSTEM_TYPE_POLICIES, get_system_policy


@pytest.mark.parametrize(
    "system_type",
    ["not_a_type", "", "GAS_CLUSTER", "gas cluster", None, 3],
)
def test_get_system_policy_rejects_unknown_system_type(system_type):
    with pytest.raises(SCGOValidationError) as exc_info:
        get_system_policy(system_type)
    assert "system_type" in str(exc_info.value)


@pytest.mark.parametrize("system_type", sorted(SYSTEM_TYPE_POLICIES))
def test_get_system_policy_returns_policy_for_known_types(system_type):
    assert get_system_policy(system_type).system_type == system_type


def test_get_system_policy_rejects_unhashable_system_type():
    with pytest.raises(SCGOValidationError):
        get_system_policy(["gas_cluster"])
