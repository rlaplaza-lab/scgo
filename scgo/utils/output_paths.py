"""Campaign output directory layout for global optimization and TS search.

Every runner resolves a single **campaign root**. GO artifacts live in
``{root}/{path_key}_searches/`` and TS artifacts in
``{root}/{path_key}_ts_results/`` as siblings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_ROOT_SUFFIXES: tuple[str, ...] = ("_searches", "_ts_results")


def formula_searches_dir(root: str | Path, formula: str) -> Path:
    """Return ``{root}/{path_key}_searches`` (``formula`` is the path key)."""
    return Path(root) / f"{formula}_searches"


def formula_ts_results_dir(root: str | Path, formula: str) -> Path:
    """Return ``{root}/{path_key}_ts_results`` (``formula`` is the path key)."""
    return Path(root) / f"{formula}_ts_results"


def calculator_slug_from_go_params(go_params: dict[str, Any] | None) -> str:
    """Return the lowercase calculator slug used in default campaign-root names."""
    c = str((go_params or {}).get("calculator", "MACE")).strip().upper()
    if c in ("MACE", "UMA"):
        return c.lower()
    return c.lower() or "calc"


def resolve_campaign_root_from_args(
    output_dir: str | Path | None,
    *,
    output_root: str | Path | None = None,
    output_stem: str | None = None,
    path_key: str,
    calc_slug: str | None = None,
) -> Path:
    """Resolve the single campaign root shared by every runner.

    Resolution order:

    - ``output_dir`` given: a ``*_searches`` / ``*_ts_results`` path resolves to
      its parent; anything else is the campaign root itself.
    - ``output_root`` or ``output_stem`` given: build the ``go_ts``-style default
      ``{output_root or ./scgo_runs}/{output_stem or path_key}_{calc_slug}``.
    - neither: the current working directory.
    """
    if output_dir is not None:
        candidate = Path(output_dir).expanduser().resolve()
        if candidate.name.endswith(_ROOT_SUFFIXES):
            return candidate.parent
        return candidate
    if output_root is not None or output_stem is not None:
        base = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else (Path.cwd() / "scgo_runs").resolve()
        )
        stem = output_stem or path_key
        slug = calc_slug or calculator_slug_from_go_params(None)
        return (base / f"{stem}_{slug}").resolve()
    return Path.cwd().resolve()


def resolve_minima_dir(
    campaign_root: str | Path,
    formula: str,
    *,
    searches_dir: str | Path | None = None,
) -> Path:
    """Return the directory containing GO ``run_*/`` minima databases.

    When ``searches_dir`` is provided, minima are read from that path.
    """
    if searches_dir is not None:
        return Path(searches_dir).expanduser().resolve()
    return formula_searches_dir(campaign_root, formula)


def resolve_ts_campaign_paths(
    output_dir: str | Path | None,
    path_key_formula: str,
    *,
    searches_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Return ``(campaign_root, minima_dir, ts_results_root)`` for TS search.

    ``path_key_formula`` is the component-aware path key used for sibling
    ``{key}_searches`` and ``{key}_ts_results`` directory names
    (nanoparticle, adsorbate fragments, and surface name; slab chemical
    symbols are never included).
    """
    explicit_searches = (
        Path(searches_dir).expanduser().resolve() if searches_dir is not None else None
    )
    if explicit_searches is not None:
        campaign_root = explicit_searches.parent
        minima_dir = explicit_searches
    else:
        campaign_root = resolve_campaign_root_from_args(
            output_dir, path_key=path_key_formula
        )
        candidate = (
            Path(output_dir).expanduser().resolve() if output_dir is not None else None
        )
        if candidate is not None and candidate.name.endswith("_searches"):
            minima_dir = candidate
        else:
            minima_dir = formula_searches_dir(campaign_root, path_key_formula)

    ts_results_root = formula_ts_results_dir(campaign_root, path_key_formula)
    return campaign_root, minima_dir, ts_results_root


def resolve_go_searches_dir(
    output_dir: str | Path | None,
    formula: str,
) -> Path:
    """Return the GO ``{path_key}_searches/`` directory for ``run_go``.

    When ``output_dir`` is provided, it is the searches directory itself.
    When ``output_dir`` is ``None``, use ``{path_key}_searches`` under CWD.
    """
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return formula_searches_dir(Path.cwd().resolve(), formula)


def resolve_go_campaign_searches_dir(
    campaign_parent: str | Path | None,
    formula: str,
) -> Path | None:
    """Return ``{parent}/{path_key}_searches`` for ``run_go_campaign``.

    When ``campaign_parent`` is ``None``, return ``None`` so ``run_go`` applies
    its own default searches path.
    """
    if campaign_parent is None:
        return None
    return formula_searches_dir(Path(campaign_parent).expanduser(), formula)


def resolve_go_ts_pipeline_paths(
    campaign_root: str | Path,
    formula: str,
) -> tuple[Path, Path]:
    """Return ``(searches_dir, ts_results_dir)`` under a GO+TS campaign root.

    ``formula`` is the component-aware path key
    (e.g. ``Pt5`` or ``Pt5_OH_OH_graphite``).
    """
    root = Path(campaign_root)
    return (
        formula_searches_dir(root, formula),
        formula_ts_results_dir(root, formula),
    )
