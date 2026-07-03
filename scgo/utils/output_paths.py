"""Campaign output directory layout for global optimization and TS search."""

from __future__ import annotations

from pathlib import Path


def formula_searches_dir(root: str | Path, formula: str) -> Path:
    """Return ``{root}/{formula}_searches``."""
    return Path(root) / f"{formula}_searches"


def formula_ts_results_dir(root: str | Path, formula: str) -> Path:
    """Return ``{root}/{formula}_ts_results``."""
    return Path(root) / f"{formula}_ts_results"


def resolve_campaign_root(
    output_dir: str | Path | None,
    *,
    formula: str | None = None,
) -> Path:
    """Resolve campaign root from ``output_dir``.

    When ``output_dir`` is ``None``, use the current working directory.
    When ``output_dir`` ends with ``_searches``, use its parent as the
    campaign root. Otherwise treat ``output_dir`` as the campaign root.
    """
    _ = formula  # reserved for future path disambiguation
    if output_dir is None:
        return Path.cwd().resolve()
    candidate = Path(output_dir).expanduser().resolve()
    if candidate.name.endswith("_searches"):
        return candidate.parent
    return candidate


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

    ``path_key_formula`` is the cluster/mobile formula used for sibling
    ``{formula}_searches`` and ``{formula}_ts_results`` directory names
    (without slab symbols for surface runs).
    """
    explicit_searches = (
        Path(searches_dir).expanduser().resolve() if searches_dir is not None else None
    )
    if explicit_searches is not None:
        campaign_root = explicit_searches.parent
        minima_dir = explicit_searches
    elif output_dir is not None:
        candidate = Path(output_dir).expanduser().resolve()
        if candidate.name.endswith("_searches"):
            minima_dir = candidate
            campaign_root = candidate.parent
        else:
            campaign_root = candidate
            minima_dir = formula_searches_dir(campaign_root, path_key_formula)
    else:
        campaign_root = Path.cwd().resolve()
        minima_dir = formula_searches_dir(campaign_root, path_key_formula)

    ts_results_root = formula_ts_results_dir(campaign_root, path_key_formula)
    return campaign_root, minima_dir, ts_results_root
