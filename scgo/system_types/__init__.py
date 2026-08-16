"""Canonical system-type definitions and validation helpers."""

from __future__ import annotations

from scgo.system_types.composition import (
    AdsorbateDefinition,
    AdsorbateFragmentInput,
    AdsorbatesInput,
    as_adsorbate_definition,
    build_adsorbate_definition_from_inputs,
    extract_adsorbate_definition_from_params,
    flatten_adsorbate_symbols,
    normalize_adsorbates_input,
    resolve_adsorbate_fragments,
    resolve_adsorbate_run_composition,
    resolve_mobile_composition,
    resolve_search_mobile_composition,
    validate_adsorbate_definition,
    validate_composition_against_adsorbate,
)
from scgo.system_types.connectivity_factor import (
    ConnectivityFactorInput,
    NormalizedConnectivityFactor,
    normalize_connectivity_factor,
)
from scgo.system_types.params import (
    CalculatorKwargs,
    GLOptimizerParams,
    OptimizerSlotParams,
)
from scgo.system_types.policy import (
    SYSTEM_TYPE_POLICIES,
    SystemPolicy,
    SystemType,
    get_system_policy,
    resolve_connectivity_factor,
    resolve_neb_mic,
    resolve_structure_mic,
    select_scgo_minima_algorithm,
    slab_is_search_target,
    uses_surface,
    validate_system_type_settings,
)
from scgo.system_types.validation import (
    _validate_adsorbate_tag_partition,
    validate_connectivity_policy,
    validate_minimum_structure,
    validate_mobile_symbols_match_adsorbate_definition,
    validate_structure_for_system_type,
)

__all__ = [
    "AdsorbateDefinition",
    "AdsorbateFragmentInput",
    "AdsorbatesInput",
    "CalculatorKwargs",
    "ConnectivityFactorInput",
    "GLOptimizerParams",
    "NormalizedConnectivityFactor",
    "OptimizerSlotParams",
    "SYSTEM_TYPE_POLICIES",
    "SystemPolicy",
    "SystemType",
    "_validate_adsorbate_tag_partition",
    "as_adsorbate_definition",
    "build_adsorbate_definition_from_inputs",
    "extract_adsorbate_definition_from_params",
    "flatten_adsorbate_symbols",
    "get_system_policy",
    "normalize_adsorbates_input",
    "normalize_connectivity_factor",
    "resolve_adsorbate_fragments",
    "resolve_adsorbate_run_composition",
    "resolve_connectivity_factor",
    "resolve_mobile_composition",
    "resolve_neb_mic",
    "resolve_search_mobile_composition",
    "resolve_structure_mic",
    "select_scgo_minima_algorithm",
    "slab_is_search_target",
    "uses_surface",
    "validate_adsorbate_definition",
    "validate_composition_against_adsorbate",
    "validate_connectivity_policy",
    "validate_minimum_structure",
    "validate_mobile_symbols_match_adsorbate_definition",
    "validate_structure_for_system_type",
    "validate_system_type_settings",
]
