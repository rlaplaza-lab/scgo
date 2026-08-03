# Changelog

## 0.6.5

### Added

- ``build_torchsim_relaxer`` factory for shared UMA → UPET → MACE TorchSim
  relaxer construction from a live calculator (GA). Presets still construct
  ``TorchSimBatchRelaxer`` directly when ``model_kind`` is already known.
- ``validate_and_resolve_run_context`` shared BH/GA preamble
  (policy, connectivity factor, fitness strategy).

### Changed

- Path-key resolution consolidated in ``scgo.utils.path_keys.resolve_run_path_key``
  (importable from ``runner_params``); GO/TS/minima search use the same helper.
- ``run_go_campaign`` result dict keys are always ``path_key`` (including failed
  compositions); gas-cluster keys still match the formula.
- GA ``create_mutation_operators`` uses a shared partitioned-mutation helper for
  flattening / breathing / in-plane slide core/_ads variants (names/weights unchanged).
- Drop unused aliases: ``retry_with_backoff`` (use ``database_retry``),
  ``assert_adsorption_height_in_bounds``, Kaggle ``_install_scgo_mace``, and
  streaming ``_relaxed_rows_where_clause``.

### Fixed

- Bare ``surface`` / ``surface_adsorbate`` empty-core composition is accepted by
  ``run_go`` / ``run_go_ts`` / GA / TS (examples with ``COMPOSITION=[]``).
- Slab-search TS uses fixed-bottom ``n_slab`` (not full slab length) so mobile
  top layers remain comparable.
- Adsorbate-only deposition on planar graphite: planar site fallback when the
  3D convex hull is empty, and skip whole-slab connectivity checks that reject
  van der Waals stacked layers.
- Penalty-energy path attaches a ``SinglePointCalculator`` so later energy/force
  reads do not hit a broken calculator.
- BH ``_move_atoms``: single tag groups displace rigidly; empty movable sets log
  ``Moved_atoms: none``; adsorbate-scaled moves no longer throttle core; single-
  atom descriptions are bracketed for ASE DB compatibility.
- ``iter_databases_minima(max_structures=0)`` yields nothing (``0`` is not treated
  as unlimited).

### Maintainer notes

- Upstream shims still required (Phase 5 — do not remove until triggers fire):
  TorchSim constraint device patch; ``HAS_NVALCHEMIOPS = False``; Kaggle
  ``vesin`` force-install; ``max_steps=0`` warning filters; ``standardmutations``
  re-export; ``pytest.ini`` filters. Timing ``"trials"`` guard is intentional
  API protection, not an upstream shim.

## 0.6.4

### Added

- ``surface`` and ``surface_adsorbate`` system types: GA/BH search mobile
  top slab layers (bottom layers fixed), with optional adsorbates and no
  cluster core. Includes slab search partition helpers, defected/N-doped
  graphite presets, and examples.

### Changed

- On-disk path keys for searches, TS results, XYZ prefixes, and default
  campaign stems are component-aware: nanoparticle, each adsorbate fragment,
  then surface name (e.g. ``Pt5_OH_OH_graphite``). Chemical composition
  matching still uses ASE-style formulas (``H2O2Pt5``).
- ``SurfaceSystemConfig.name`` (default ``"slab"``) supplies the surface
  path-key segment; ``make_graphite_surface_config`` sets ``name="graphite"``.

### Fixed

- Ruff import sorting / formatting leftovers from the surface-search merge
  so the GitHub Actions lint job passes on main.

## 0.6.3

### Fixed

- UPET/UMA TS preset tests expect ``use_parallel_neb=True``, matching the
  0.6.2 default (fixes GitHub Actions UPET CI jobs).

## 0.6.2

### Changed

- NEB plumbing: serial and parallel runners share ``NebRunConfig`` and
  ``prepare_neb_endpoints`` (copy / FixAtoms / validate). Public
  ``run_transition_state_search`` kwargs are unchanged.
- BH surface post-relax framing is owned by ``perform_local_relaxation``
  (``surface_mode`` / ``n_slab``); diversity scoring uses mobile composition.
  GA soft-fail storage validation wraps shared
  ``canonicalize_and_validate_for_storage``.
- Structure MIC reads go through ``resolve_structure_mic`` /
  ``resolve_neb_mic``. Pair selection takes explicit ``use_mic`` (scoring
  regime stays ``surface_aware``). Empty-core adsorbate NEB dims use shared
  ``resolve_neb_mobile_dims``.
- TS presets: ``neb_fmax`` / ``torchsim_fmax`` are shared at ``0.20`` for
  every system type; ``use_parallel_neb=True`` is the default everywhere
  (including the low-level ``run_transition_state_search`` signature).
  Surface types set ``parallel_neb_max_bands=1`` so large slab cells do not
  GPU-OOM; the parallel runner is still used, with bands chunked
  one-at-a-time (and CUDA cache cleared between chunks). Bare surface NEB
  step budget rises to ``2000``. Supersedes the 0.6.1 surface-adsorbate
  ``neb_fmax=0.25`` / serial-NEB defaults. Removed unused
  ``torchsim_batch_size`` from TS presets (OOM safety is band concurrency).
- Adsorbate TS pair oversample is ``min(max_pairs * 10, max(max_pairs, 50))``;
  pair selection reuses one structure comparator and avoids full Atoms
  slices for core-RMS.

### Fixed

- Parallel NEB refuses FIRE steps when band fmax is non-finite (e.g. ASE
  ``improvedtangent`` on a flat energy profile), marking the band failed
  instead of propagating NaN geometries.
- Unify structure MIC / surface-awareness across GO, GA, BH, and TS: Pure
  comparator honors ``mic`` literally under PBC; ``SurfaceSystemConfig``
  defaults ``comparator_use_mic=True`` (GO/GA/BH via ``resolve_structure_mic``).
  TS pair scoring uses ``uses_surface`` for the scoring regime and
  ``resolve_neb_mic`` (``neb_force_mic``) for geometry / minima dedupe — not
  the comparator knob. BH uses mobile ``n_top`` and skips COM recenter on
  surfaces; empty-core adsorbate enables blockwise NEB dims; core-RMS pair
  gate is permutation-invariant; GO final XYZ alignment forwards
  ``n_core_mobile``.

## 0.6.1

### Fixed

- TorchSim NEB/TS single-point force evaluations now use ``torch_sim.static``
  instead of ``optimize(max_steps=0)``. The old path still took one FIRE step
  (wrong forces at displaced geometries) and spammed
  ``All systems have reached the maximum number of steps: 0`` via torch_sim's
  logger in production. Batched NEB spring/climb/tangent physics remain ASE's.
  Single-point calls default to ``autobatcher=False`` so TorchSim does not
  re-probe GPU memory on every NEB force evaluation.
- Parallel/serial TorchSim NEB finalize no longer fails with
  ``The property "energy" is not available`` after a final FIRE step: PES is
  refreshed at the final geometries, and energies are also cached in atoms
  metadata when attaching SinglePoint results.
- ASE ``Atoms.copy()`` shallow-shares nested ``info`` dicts; TorchSim
  single-points writing ``raw_score`` were corrupting minima reused by later
  NEB pairs (multi-eV bogus product energies). Endpoint/path copies now isolate
  nested metadata, and static result mapping uses ``copy_atoms``.
- Surface-adsorbate NEBs no longer apply free in-plane Kabsch rotation (breaks
  adsorbate–slab registry). Cell remap / MIC remain on. Pre-NEB band checks
  reject aligned endpoint energy drift ``> 0.5`` eV vs canonical minima, and
  one-sided interior maxima with prominence ``< 0.40`` eV (slides that CI-NEB
  collapses to an endpoint).

### Changed

- Adsorbate TS presets (`gas_cluster_adsorbate` / `surface_cluster_adsorbate`)
  now use climbing NEB, spring ``0.5``, ``neb_steps=4000``, 7 images, a tighter
  ``energy_gap_threshold`` (``0.75`` eV), and a hard ``max_endpoint_mismatch``
  pair gate. Gas adsorbates use ``neb_fmax=0.20`` with ``use_parallel_neb=True``;
  surface adsorbates use ``neb_fmax=0.25`` with serial NEB (avoids GPU OOM on
  large slab cells and matches attainable MLIP force convergence). Fragment-wise
  adsorbate matching, core-anchored endpoint alignment, and pre-NEB
  clash/discontinuity rejection improve path quality for multi-fragment
  adsorbates. Surface-adsorbate presets set
  ``neb_surface_lattice_rotation=False``; pair selection also skips tiny
  adsorbate hops (``max_diff < 0.20`` Å) that are usually barrierless slides.
- Parallel NEB no longer overwrites batch failures (e.g. CUDA OOM with
  ``force_calls=0``) as ``endpoint as TS`` during finalize.
- Pre-NEB path/energy rejects in parallel NEB are recorded as ``skipped``
  (consistent with structure-validation skips and the serial path), not
  ``failed``.
- Provenance ``scgo_version`` now reads the in-tree version
  (``scgo._version``) so editable checkouts are not stuck on stale
  ``dist-info`` after a bump.
- Adsorbate NEBs reject IDPP bands with absurdly high barriers
  (``> 8`` eV; likely discontinuous) before optimization, and use two-stage
  CI-NEB (relax without climb, then climb). Stage 2 always runs and keeps at
  least half the step budget.
- Parallel two-stage CI-NEB always runs the climb stage after no-climb
  relaxation when used, even if stage 1 already met ``fmax``.
- Two-stage climb is skipped for endpoint-max / barrierless IDPP bands
  and for soft interior maxima (IDPP barrier ``< 1.0`` eV); climb starts
  immediately. A no-climb pre-relax was collapsing those MEPs so finalize
  reported ``endpoint as TS`` for adsorbate OH hops.
- Finalize rejects NEB results with barrier ``> 8`` eV (same discontinuous
  threshold as the pre-NEB IDPP energy gate).
- Adsorbate pair selection now prefers activated hops (moderate endpoint
  mismatch and core RMS) over near-isomer slides, oversamples candidates, and
  re-ranks by IDPP profile so NEB budgets favor robust interior maxima (and
  skip endpoint-max slides when any robust bands exist).
- TS minima deduplication for adsorbate systems uses core+adsorbate mobile
  count (matching GA ``n_to_optimize``), not core-only length.

## 0.6.0

### Added

- UPET MLIP backend (``[upet]`` extra) via metatomic-TorchSim, with CI matrix
  coverage alongside MACE and UMA, plus Kaggle GPU suites for MACE/UPET.
- Height aliases on surface and cluster-adsorbate configs: surface accepts
  ``height_*`` as aliases for ``adsorption_height_*``; adsorbate configs accept
  ``adsorption_height_*`` as aliases for ``height_*``. Conflicting values raise
  ``SCGOValidationError``.
- Shared helpers: :mod:`scgo.calculators.torch_device`,
  :mod:`scgo.utils.config_aliases`, :mod:`scgo.utils.combine_atoms`.
- GO top-level parameter allowlist (including ``validation_n_jobs``); unexpected
  keys raise ``SCGOValidationError`` with the expected set.

### Changed

- Split the large ``runner_api`` module into focused modules
  (``runner_composition``, ``runner_params``, ``runner_go``, ``runner_ts``) while
  keeping the public ``scgo.runner_api`` / ``scgo`` import surface stable via
  re-exports (including names used by test monkeypatches).
- Split ASE GA ``standardmutations`` into
  :mod:`scgo.ase_ga_patches.mutations` (one module per family); the old import
  path remains a thin re-export.
- Unsupported Torch devices warn once and raise ``SCGOValidationError`` instead
  of silently falling back to CPU (MACE / UMA / UPET / TorchSim paths).
- ``SCGOValidationError`` no longer logs at ERROR on construction. Runner API
  entry points log validation failures at the prepare boundary; campaign and
  pair handlers catch ``SCGOValidationError`` and continue where appropriate.
- Top-level ``surface_config`` in ``go_params`` / ``ts_params`` is allowed and
  fanned into optimizer slots; only ``system_type`` remains rejected in params
  (use the run-function argument). Adsorbate placement knobs stay in
  ``go_params``.
- Surface slab constraint attachment preserves non-``FixAtoms`` constraints
  (e.g. ``FixBondLength``). Multi-fragment hierarchical placement keeps sites
  on the original metal core.
- Parallel NEB skips re-evaluating endpoints after step 0 and uses a clearer
  max-atom force metric; force attachment requires forces.

### Fixed

- Restore auto GA scaling in the TorchSim preset.
- Align concurrent DB stress tests with production retry policy.
- Handle ``SCGOValidationError`` in growth, GA, and initialization fallbacks
  (and in GO campaign / TS pair error paths).

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
- ``SCGOValidationError`` is logged at ERROR when logging is configured
  (construct-time logging in 0.5.2; superseded in 0.6.0 by runner-boundary
  logging).
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
