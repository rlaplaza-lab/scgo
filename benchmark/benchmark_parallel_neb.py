#!/usr/bin/env python3
"""Compare serial vs parallel NEB wall time on existing Pt5 minima.

Requires a completed ``Pt5_searches`` run with final unique minima. Runs TS search
twice (serial then parallel) with the same ``max_pairs``, renames each
``Pt5_ts_results`` output folder, and writes ``parallel_neb_benchmark.json``.

Usage::

    python -m benchmark.benchmark_parallel_neb \\
        --searches-dir /path/to/results/pt5_mace_mace_matpes_0/Pt5_searches \\
        --max-pairs 5
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from scgo.ts_search.transition_state_run import run_transition_state_search
from scgo.utils.logging import get_logger
from scgo.utils.output_paths import formula_ts_results_dir

logger = get_logger(__name__)

PT5_FORMULA = "Pt5"


def _run_variant(
    *,
    searches_dir: Path,
    label: str,
    use_parallel_neb: bool,
    max_pairs: int,
    seed: int,
) -> dict:
    campaign_root = searches_dir.parent
    t0 = time.perf_counter()
    results = run_transition_state_search(
        composition=["Pt"] * 5,
        system_type="gas_cluster",
        output_dir=str(campaign_root),
        searches_dir=str(searches_dir),
        seed=seed,
        verbosity=1,
        max_pairs=max_pairs,
        use_parallel_neb=use_parallel_neb,
        write_timing_json=True,
        params={"calculator": "MACE"},
    )
    wall_s = time.perf_counter() - t0
    neb_sum = sum(
        float((r.get("timings_s") or {}).get("neb_optimization_s", 0.0))
        for r in results
    )
    default_result = formula_ts_results_dir(campaign_root, PT5_FORMULA)
    renamed = default_result.with_name(f"{PT5_FORMULA}_ts_results_{label}")
    if default_result.exists():
        if renamed.exists():
            shutil.rmtree(renamed)
        shutil.move(default_result, renamed)
    return {
        "label": label,
        "use_parallel_neb": use_parallel_neb,
        "wall_s": wall_s,
        "neb_sum_s": neb_sum,
        "n_results": len(results),
        "n_success": sum(1 for r in results if r.get("status") == "success"),
        "result_dir": str(renamed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--searches-dir",
        type=Path,
        required=True,
        help="Path to Pt5_searches/ with completed GO minima",
    )
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.searches_dir.is_dir():
        raise SystemExit(f"Missing {args.searches_dir}")

    serial = _run_variant(
        searches_dir=args.searches_dir,
        label="serial",
        use_parallel_neb=False,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    parallel = _run_variant(
        searches_dir=args.searches_dir,
        label="parallel",
        use_parallel_neb=True,
        max_pairs=args.max_pairs,
        seed=args.seed + 1,
    )
    summary = {"serial": serial, "parallel": parallel}
    out_path = args.searches_dir.parent / "parallel_neb_benchmark.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote parallel NEB benchmark results to %s", out_path)


if __name__ == "__main__":
    main()
