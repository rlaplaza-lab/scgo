# Changelog

## 0.5.2

### Added

- Verbosity-level logging for GA runs with v1 phase headers and aggregated
  initialization/generation summaries, v2 per-individual detail. New
  :func:`~scgo.configure_logging` helper and
  :class:`~scgo.utils.phase_logging.InitDiagnosticsCollector` for batched
  initialization messages. Standardized %-style logging across runners and
  TS code paths.
- Typed parameter dicts: :class:`~scgo.system_types.GLOptimizerParams` and
  :class:`~scgo.system_types.TSParams` TypedDicts for GO and TS parameters,
  with :class:`~scgo.system_types.CalculatorKwargs` and
  :class:`~scgo.system_types.OptimizerSlotParams` for nested configuration.

### Changed

- Adsorbate/core partition reconciliation now routes through all runner paths
  via centralized ``resolve_adsorbate_run_composition``, sharing the same
  core/adsorbate stripping logic across gas and surface runs, ``run_go``,
  campaigns, GO+TS, and TS entry points.
- Simplified adsorbate/core reconciliation logic: use list-based stripping,
  drop redundant count checks, consolidate test coverage.
- Deduplicated candidate-discovery path filtering via shared path relevance
  helper, cleaning up parse/filter branches while preserving unparseable-path
  accounting.
- Hardened initialization fallback chains with coherent seed+growth behavior,
  magic-number tolerance for near templates, aligned radii usage in placement,
  and targeted logging/regression tests to prevent silent skips.
- Improved initialization logging: grouped seed-sampling failures into single
  INFO summaries with specific reasons; compact, consistently formatted placement
  error messages for large runs.
- Hardened database operations: production retries for reads, connection opens,
  structure extraction, and count queries via unified ``retry_on_lock`` /
  ``database_retry`` machinery; IMMEDIATE isolation for final-minima tagging;
  backoff on transient lock/I/O OperationalErrors; retry actual SQLite open
  during setup; log stamp failures instead of suppressing them.
- Aligned database retry logic: ``database_retry`` now only backs off on
  transient lock/I/O OperationalErrors, matching ``retry_on_lock`` and
  ``retry_transaction``; shared retried ``DataConnection`` factory between
  ``setup_database`` and ``get_connection``.
- Hardened composition parsing with explicit errors for empty and unknown
  symbols; expanded regression tests covering ``HO2Ru9W2`` adsorbate resolution
  and edge cases.
- Made compact formula parsing unambiguous: use ASE ``Formula`` with required
  chemical capitalization for multi-element strings; allow lowercase only for
  unambiguous single-element forms (``pt3``); reject ambiguous cases (``ho2``,
  ``cu``, ``pt3au``) with actionable errors; comma-separated symbols remain the
  fully unambiguous input format.
- Validation and configuration failures across SCGO now raise typed exceptions
  (``SCGOValidationError``, ``SCGORuntimeError``, etc.) instead of bare
  ``ValueError`` / ``RuntimeError``. Downstream code should catch
  ``SCGOValidationError`` (or ``SCGOError``) rather than ``ValueError``.
- ``SCGOValidationError`` is logged at ERROR when logging is configured.
- Preset dicts are documented as ``GLOptimizerParams`` and ``TSParams`` TypedDicts;
  default GO params template is cached via ``@cache``.

### Fixed

- MACE import on PyTorch 2.6+: patch ``torch.load`` before ``mace``/e3nn import so
  checkpoint and constants loading no longer fails with ``weights_only`` unpickling errors.
- Fix lowercase compact formula parsing by normalizing all-lowercase strings
  (e.g., ``pt3`` → ``Pt3``) before calling ASE ``Formula``, preserving case-
  sensitive ``HO2``-style formulas unchanged.
- Fix ``parse_composition_arg`` docstring for Sphinx ``-W`` builds by removing
  indented bullet continuation that docutils treated as invalid RST.
- Fix adsorbate/core partition reconciliation for oxide campaigns by deriving
  ``core_symbols`` from full mobile formulas when preset cores disagree,
  updating ``adsorbate_definition`` in place, and deep-copying preset definitions
  per campaign composition.
- SQLite connection handle leaks in database setup and configuration paths.

## 0.5.1

### Fixed

- ASE icosahedron/decahedron/octahedron templates for HCP elements by passing an
  explicit lattice constant (structures are still rescaled to covalent bond length).
- Compact formula parsing for hydrogen–oxide strings such as ``HO2Ru9W2`` (via ASE
  ``Formula`` instead of mis-reading ``Ho`` as holmium).
- Gas/surface adsorbate runs with a preset ``adsorbate_definition``: reconcile
  campaign composition to ``core_symbols + adsorbate_symbols`` when counts match
  but symbol order differs, when only the core formula is supplied, or when the
  full mobile formula requires re-deriving ``core_symbols`` by stripping known
  ``adsorbate_symbols`` (oxide campaigns such as ``HO2Ru9W2``). Applies to gas and
  surface adsorbate system types across all runner entry points.

### Changed

- Template discovery failures no longer emit per-attempt debug noise for expected
  ASE lattice-guess misses.

## 0.5.0

### Added

- Manual Kaggle GPU workflow for CUDA/MACE integration tests on T4 hardware.
- GPU example integration tests aligned with real example workloads.
- SQLite PRAGMA debug logging for easier HPC filesystem troubleshooting.

### Changed

- Refactored runner/database workflow to reduce repeated overhead and unify
  discovery, streaming, and candidate-loading paths.
- Fail-fast validation at API boundaries; reduced silent defensive fallbacks.
- Strengthened physics assertions, reproducibility checks, and CI strictness.
- Dual MACE/UMA CI matrix with marker-based test partitioning.
- Capped NumPy below 2.5 and aligned Kaggle GPU dependency installs with CI.
- Corrected algorithm selection docs: 3-atom adsorbate systems use GA, not BH.
- Docs version fallback now reads from ``scgo.__version__`` instead of a stale literal.

### Fixed

- SQLite connection handle leaks in ``setup_database`` and DB configuration paths.
- Concurrent SQLite write stress test stability in CI.
- Reference run provenance and streaming warning behavior.
- TorchSim warnings API usage and raw MACE model wrapping for ``optimize()``.
- Kaggle runner resilience (conda detection, source tarball, log redaction, CUDA torch).
- Empty GA population crash and surface ``run_go`` e2e test stability.
- Cross-fragment adsorbate bonding rejection in integrity checks.
- Adsorption height checks and CI disk cleanup for UMA installs.

## 0.4.1

### Fixed

- Adsorbate partition overlap handling and ``source_db_relpath`` provenance fields.

### Documentation

- Minor documentation fixes following the 0.4.0 release.

## 0.4.0

### Changed

- Flattened GO runs to datetime-tagged `run_*` directories (removed `trial_*` layer).
- Run IDs and `metadata.json` timestamps now use UTC.
- Timing JSON (`timing.json`, `go_ts_timing.json`) includes structured provenance headers,
  `run_id`, and `timing_schema_version`.
- `go_ts_timing.json` links to per-run GO/TS timing files via `current_*_run_id` and
  `*_run_timing_relpath` fields.
- TS `results_summary.json` handles skipped pairs without KeyError.
- `get_provenance()` reads `provenance` and `key_value_pairs` in addition to `metadata`.
- Database discovery warns on unresolved `run_id` paths instead of silently skipping.

### Documentation

- Updated quickstart output layout, provenance fields, and timing schema.
- Corrected algorithm selection rules in `parameters.rst`.
