# Changelog

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
