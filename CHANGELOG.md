# Changelog

## 0.5.1

### Fixed

- ASE icosahedron/decahedron/octahedron templates for HCP elements by passing an
  explicit lattice constant (structures are still rescaled to covalent bond length).
- Compact formula parsing for hydrogen–oxide strings such as ``HO2Ru9W2`` (via ASE
  ``Formula`` instead of mis-reading ``Ho`` as holmium).
- Gas/surface adsorbate runs with a preset ``adsorbate_definition``: reconcile
  campaign composition to ``core_symbols + adsorbate_symbols`` when counts match
  but symbol order differs, or when only the core formula is supplied.

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
