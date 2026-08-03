"""Component-aware path-key resolution for GO / TS / campaign directories.

:func:`get_system_path_key` is the pure formatter; :func:`resolve_run_path_key`
is the single place that decides how composition + params/system_type map to
a path key.
"""

from __future__ import annotations

from typing import Any

from scgo.surface.config import SurfaceSystemConfig
from scgo.system_types import (
    AdsorbateDefinition,
    SystemType,
    extract_adsorbate_definition_from_params,
    get_system_policy,
)
from scgo.utils.helpers import get_system_path_key

__all__ = [
    "resolve_run_path_key",
]


def _surface_name_for_path(
    system_type: SystemType | None,
    surface_config: SurfaceSystemConfig | None,
) -> str | None:
    """Return surface path-key segment when the system uses a surface."""
    if system_type is None or surface_config is None:
        return None
    if not get_system_policy(system_type).uses_surface:
        return None
    return surface_config.name


def resolve_run_path_key(
    composition: list[str],
    *,
    system_type: SystemType | None = None,
    adsorbate_definition: AdsorbateDefinition | None = None,
    surface_config: SurfaceSystemConfig | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """Resolve component-aware path key for searches / TS / campaign dirs."""
    ads_def = adsorbate_definition
    if ads_def is None and params is not None:
        ads_def = extract_adsorbate_definition_from_params(params)
        if ads_def is None:
            raw = params.get("adsorbate_definition")
            if isinstance(raw, dict):
                ads_def = raw  # type: ignore[assignment]
    sc = surface_config
    if sc is None and params is not None:
        raw_sc = params.get("surface_config")
        if isinstance(raw_sc, SurfaceSystemConfig):
            sc = raw_sc
    return get_system_path_key(
        composition,
        adsorbate_definition=ads_def,  # type: ignore[arg-type]
        surface_name=_surface_name_for_path(system_type, sc),
    )
