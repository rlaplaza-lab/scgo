Validation and Constraints
==========================

SCGO checks two related things: which atoms may move (constraints), and which
structures are legal (validation). Outcome toggles control how strict the
surface search is when a structure splits or detaches.

What constraints and validation do
----------------------------------

- **Constraints** define what atoms may move: slab fixation, core freezing, and
  adsorbate rigidity.
- **Validation** rejects structures that violate the allowed geometry: clashes,
  connectivity, penetration, and fragmentation rules.
- **Outcome toggles** decide how permissive the surface search can be when a
  structure is partially detached or fragmented.

A relaxed structure is only legal relative to the constraints that defined the
search. Read them together.

What counts as connected
------------------------

Gas systems (``gas_cluster`` and ``gas_cluster_adsorbate``) must form a single
connected component. A fragmented bare cluster or a gas adsorbate detached from
its core is rejected.

Surface systems (``surface_cluster`` and the ``*_adsorbate`` variants) require
connectivity and slab contact for each mobile subgroup, through
:func:`~scgo.system_types.validate_connectivity_policy`.

The bare ``surface`` type skips the supported-deposit connectivity gate. It
still checks the slab-prefix layout through
:func:`~scgo.surface.partition.validate_slab_search_config`.

All global optimizers use the same validation rule set when a ``system_type`` is
supplied. Placement uses looser steric floors than the final check, so the
search can explore broadly while illegal candidates are still dropped before the
final set.

Outcome toggles
---------------

Defaults come from :doc:`/parameters` and
:class:`~scgo.system_types.GLOptimizerParams`.

.. list-table::
   :widths: 38 10 30 32
   :header-rows: 1

   * - Toggle
     - Default
     - Gas (``gas_cluster_adsorbate``)
     - Surface (``surface_cluster_adsorbate``)
   * - ``allow_cluster_fragmentation``
     - ``False``
     - n/a (no slab)
     - Multiple core/mixed subgroups OK, each must touch slab
   * - ``allow_adsorbate_surface_detachment``
     - ``False``
     - n/a
     - Adsorbate-only subgroups on slab allowed
   * - ``enforce_adsorbate_subgraph_integrity``
     - ``True``
     - Fragments kept connected
     - Fragments kept connected
   * - ``freeze_adsorbate_internal_geometry``
     - ``False``
     - Rigid fragment (``FixBondLengths``)
     - Rigid fragment (``FixBondLengths``)
   * - ``validate_combined_structure`` (``ClusterAdsorbateConfig``)
     - ``True``
     - Pre/post-relax combined check
     - Pre/post-relax combined check

.. note::

   ``allow_cluster_fragmentation`` and ``allow_adsorbate_surface_detachment`` are
   surface-only. Gas paths ignore them because there is no slab. Surface
   defaults require a single connected mobile component touching the slab.
   When ``allow_adsorbate_surface_detachment`` is true, the genetic algorithm
   also registers ``fragment_detach``, which moves one intact fragment from the
   cluster onto a nearby slab site; that operator is absent when the flag is false.

How validation runs during global optimization
----------------------------------------------

Every candidate minimum goes through
:func:`~scgo.system_types.validate_minimum_structure` (a thin wrapper over
:func:`~scgo.system_types.validate_structure_for_system_type`):

- Basin Hopping validates each relaxed trial. The initial seed is checked
  softly: a failure is logged as a warning and the run continues (later moves
  and the final gate still reject disconnected minima).
- The genetic algorithm validates each child before relaxation and uses the same
  helper for storage.
- ``simple_go`` validates its single relaxed structure when a ``system_type`` is
  given and returns an empty list if it is invalid.

After global optimization,
:func:`~scgo.minima_search.core.run_trials` applies the same structural gate to
deduplicated unique candidates before the Hessian/vibration check. See
:doc:`/uniqueness`. Surface candidates are checked against the prepared slab
search config when the slab is the search target, and the frozen bottom-layer
prefix is passed as the deposit boundary (``n_slab_deposit``) so a detached /
migrated search-mobile sheet is rejected exactly like at GA storage time. The
basin-hopping inline gates apply the same deposit-aware boundary to the relaxed
seed and every trial.

.. note::

   For slab-search adsorbate types (``surface_adsorbate``, and any
   ``*_adsorbate`` type where the slab top layers are the search core), metal
   cores are rejected up front: the mobile core **is** the supported slab top.
   Pass adsorbates only (``core_symbols=[]``).

The bond cutoff resolves the same way at every gate: explicit
``connectivity_factor``, then ``ClusterAdsorbateConfig``, then
``SurfaceSystemConfig``, then ``1.2``. The value may be a float or a
per-element/pair dict. Clashes are a different scale: kept structures use
``0.7`` times the covalent sum, and nothing can go below the ``0.4`` hard cap.

Minima pair selection
---------------------

Before NEB, endpoints are chosen by
:func:`~scgo.ts_search.transition_state_io.select_structure_pairs`. Hard gates
(``energy_gap_threshold``, ``max_endpoint_mismatch``, ``pair_core_rms_max``) and
soft ranking (``pair_score_*``) are documented under **Pair selection** in
:doc:`/parameters`. Defaults come from
:func:`~scgo.pair_selection_defaults.pair_selection_param_defaults`.

``max_pairs`` is the NEB budget. Adsorbate searches may oversample the select
pool (then re-rank) via
:func:`~scgo.ts_search.transition_state_io.resolve_ts_pair_select_cap`. Bare
types do not. See **Budget and oversampling** in :doc:`/parameters`.

Pairing regimes per system type (intentional):

- ``surface_cluster`` uses the **bare-surface** regime: no core-RMS gate;
  distinctness and mismatch gates only.
- On ``surface_adsorbate`` runs with an inferred top-layer mobile block, the
  inferred block is gated by ``pair_core_rms_max`` (default ``2.0`` Å). Whole
  layer shifts can legitimately exceed this; raise the knob to relax.

.. note::

   The clash prescreen is a covalent scale and does not grow with the number
   of mobile atoms. ``max_endpoint_mismatch`` stays an absolute distance.

.. note::

   Optimized NEB bands are compared with plain Cartesian image differences;
   minimum-image distances apply at interpolation and at the gates only.

Surface NEB path preparation
----------------------------

For slab prefixes (``n_slab > 0``) or exactly two periodic axes,
:func:`~scgo.ts_search.transition_state.interpolate_path` uses surface PBC
alignment (not gas Kabsch):

- One shared in-plane lattice image per bonded mobile fragment, ranked by
  Cartesian displacement. Minimum-image scores cannot see that shift, and a
  per-atom wrap would split a fragment that straddles the cell. Lone adatoms
  still wrap by minimum image. Discrete slab-validated proper rotations then
  compete when the cell metrics allow, scored alongside Kabsch when
  ``neb_surface_lattice_rotation`` is on. Policy defaults keep Kabsch off for
  ``surface_cluster`` and adsorbate surface types; the runner AND-gates the
  knob so an explicit ``True`` cannot re-enable it for those types. In-plane
  Kabsch overlays both centers of mass and leaves the surface-normal offset.
- Moiety-aware unwrap of intact mobile fragments so ASE ``mic=True``
  interpolation does not split a bonded adsorbate across a cell boundary.
  Tiny bonded polishes keep ``mic=True``;
  :func:`~scgo.ts_search.neb_surface.unwrap_breaks_mic` disables MIC only when
  the unwrap exceeds half the shorter in-plane cell.
- If a symmetry copy's product single-point drifts more than ``0.1`` eV from
  the stored GO minimum, the band is skipped.

Gas clusters (no slab, including a 3D vacuum box) are unchanged. See
:mod:`scgo.ts_search.neb_surface`.

NEB pre-screen gates
--------------------

Before any NEB optimization,
:func:`~scgo.ts_search.transition_state.validate_initial_neb_path` runs for
every system type.
:func:`~scgo.ts_search.transition_state.validate_initial_neb_energy_profile`
runs only when ``max_endpoint_mismatch`` is set (bare ``gas_cluster`` leaves it
``None``):

- ``validate_initial_neb_path`` always checks that the aligned endpoints are
  actually distinct (maximum mobile displacement at least
  ``MIN_NEB_ENDPOINT_DISPLACEMENT_A`` = 0.3 Å; a label-swap or symmetry image of
  the same minimum has no transition path) and that interior images do not
  clash (any mobile pair closer than ``neb_prescreen_clash_distance`` times
  its covalent sum).
  An upper aligned endpoint-displacement gate also runs when
  ``max_endpoint_mismatch`` is set.
- ``validate_initial_neb_energy_profile`` runs only when
  ``max_endpoint_mismatch`` is set (barrier cap ``neb_max_spurious_barrier``;
  drift and ``min_saddle_prominence`` checks when endpoint energies are
  available).

Per-system-type defaults:

.. list-table::
   :widths: 30 22 22 22 22
   :header-rows: 1

   * - Preset
     - ``neb_prescreen_clash_distance``
     - ``min_saddle_prominence``
     - ``neb_max_spurious_barrier``
     - ``max_endpoint_mismatch``
   * - Bare gas
     - 0.4
     - 0.10
     - 8.0
     - ``None`` (clash always; energy-profile skipped)
   * - Surface cluster
     - 0.4
     - 0.40
     - 12.0
     - 2.5
   * - Bare surface
     - 0.4
     - 0.40
     - 8.0
     - 3.0
   * - Gas adsorbate
     - 0.4
     - 0.40
     - 8.0
     - 1.25
   * - Surface cluster adsorbate
     - 0.4
     - 0.40
     - 8.0
     - 1.5
   * - Surface adsorbate
     - 0.4
     - 0.40
     - 8.0
     - 3.0

``neb_prescreen_clash_distance`` is a covalent scale, the same 0.4 for every
preset. Prominence and the barrier cap are in eV. ``max_endpoint_mismatch``
is in Å.

Constraint model
----------------

Slab fixation
~~~~~~~~~~~~~

Configured via :class:`~scgo.surface.config.SurfaceSystemConfig`:

- ``fix_all_slab_atoms=True`` (default),
- ``n_relax_top_slab_layers=N``, or
- ``n_fix_bottom_slab_layers=L-N``.

Applied by :func:`~scgo.surface.constraints.attach_slab_constraints`. Existing
non-``FixAtoms`` constraints (for example adsorbate ``FixBondLengths``) are kept
when slab fixation is refreshed.

Core freeze
~~~~~~~~~~~

``fix_core`` in
:func:`~scgo.cluster_adsorbate.relax.relax_metal_cluster_with_adsorbate`:
when ``True``, it freezes core indices ``0..n_core-1`` via ``FixAtoms``
(gas-phase only; periodic cores keep their cell).

Adsorbate rigidity
~~~~~~~~~~~~~~~~~~

``freeze_adsorbate_internal_geometry`` triggers
:func:`~scgo.cluster_adsorbate.constraints.attach_adsorbate_internal_geometry_constraints`,
which appends one multi-pair ``FixBondLengths`` per fragment. Those constraints
survive slab re-freezing.

Tunables and floors
-------------------

Placement and steric room
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``height_min`` / ``height_max``: ``None`` derives the steric floor through
  the bond cutoff. Explicit values are intersected with that window.
- ``blmin_ratio`` and ``structure_min_distance_factor``: ``0.7``. Placement
  does not go below this. The ``0.4`` hard cap is the prefilter and the NEB
  prescreen only.

Connectivity and legal topology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Distances are ``scale * (r_i + r_j)`` from the ASE Cordero radii.

- ``CONNECTIVITY_FACTOR``: ``1.2``. Bonded when ``d <= threshold``. Precedence
  is above. A float, an element dict (missing symbols use ``1.2``), or a pair
  key ``"Pt-C"`` / ``("Pt", "C")`` that overrides the element form.
- ``BLMIN_RATIO_DEFAULT``: ``0.7``. Steric floor for growth, seeds, GA
  offspring, deposition, adsorbate placement, and stored minima. Clash when
  ``d < threshold``.
- ``CLASH_HARD_CAP_FACTOR``: ``0.4``. Prefilter and NEB image prescreen.
  :func:`~scgo.system_types.connectivity_factor.effective_clash_factor` raises
  anything smaller to this value.
- ``BOND_LENGTH_SCALE``: ``1.0``. Placement targets and template
  nearest-neighbor lengths are the covalent sum.

Example: ``{"Pt": 1.2, "C": 1.2, "Pt-C": 1.8}``. The same pair form loosens
H–H. At 1.2, H2 is bonded (inclusive at 0.744 Å) and fcc second neighbors are
not. bcc Fe's second neighbor stays inside that cutoff.

Surface contact and penetration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Penetration tolerance: ``0.1`` Å below the nominal slab top.
- Slab contact uses the same bond thresholds, hydrogen included. On a mixed
  top layer a mobile atom touches the slab when any top-layer atom is inside
  that pair's cutoff.
- The top-layer window is one covalent diameter of the largest slab species
  (``SURFACE_LAYER_THICKNESS_SCALE = 2.0``).

MIC semantics
-------------

Gas always uses ``use_mic=False``. Surface derives it from
``SurfaceSystemConfig.comparator_use_mic``, resolved by
:func:`~scgo.system_types.resolve_structure_mic`. It returns ``False`` for
non-surface systems and raises if the system is surface-based but
``surface_config`` is ``None``.
