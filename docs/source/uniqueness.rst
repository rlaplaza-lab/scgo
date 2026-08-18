Uniqueness (de-duplication)
===========================

SCGO treats two minima as the **same structure** only when both are true:

1. Their energies differ by at most ``energy_tolerance``.
2. Their **mobile** geometries match the Vilhelmsen–Hammer pair-correlation
   test (:class:`~scgo.utils.comparators.PureInteratomicDistanceComparator`).

Distinct isomers at similar energy are kept. Similar geometries at very
different energies are kept. See :doc:`/parameters` for the knob tables;
this page explains the rule and how to change it.

What is compared
----------------

Geometry fingerprints the **trailing mobile atoms** (``n_top``), not the full
cell. That slice is the search-mobile region from
:func:`~scgo.system_types.resolve_search_mobile_composition` (the same count
``run_go`` passes as ``search_mobile_count``):

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - System type
     - Mobile atoms in the fingerprint
   * - ``gas_cluster`` / ``gas_cluster_adsorbate``
     - Whole composition (core + adsorbate)
   * - ``surface_cluster`` / ``surface_cluster_adsorbate``
     - Deposited cluster (+ adsorbate); fixed slab ignored
   * - ``surface`` / ``surface_adsorbate``
     - Mobile top slab layers (+ adsorbate); fixed bottom layers ignored

MIC (minimum-image distances) follows
:func:`~scgo.system_types.resolve_structure_mic`: off for gas, and for surfaces
the ``SurfaceSystemConfig.comparator_use_mic`` flag.

Energy is a cheap prefilter
---------------------------

Pair correlation is more expensive than comparing two floats. Basin Hopping
end-of-run filtering and campaign-level
:func:`~scgo.utils.helpers.filter_unique_minima` **bin by energy** first
(bin width ``1.5 × energy_tolerance``) and only run geometry on neighbors that
could still match. That is an optimization, not a second uniqueness policy:
the decision remains energy **and** mobile geometry.

The GA population has no energy index, so
:class:`~scgo.utils.comparators.EnergyAndStructureComparator` applies the same
``|ΔE|`` gate inside ``looks_like`` before calling Pure.

Where it runs
-------------

All three use the same knobs from the **active** optimizer slot
(``optimizer_params["simple"|"bh"|"ga"]``):

- **GA in-search:** population duplicate gate on every candidate.
- **BH end-of-run:** when ``deduplicate=True`` (default).
- **Campaign filter:** always, in :func:`~scgo.minima_search.core.run_trials`,
  before the structural and Hessian gates. Simple GO has no in-algorithm
  uniqueness pass; it relies on this filter.

How to control it
-----------------

Set these on the optimizer slot you actually run (they share the same defaults):

.. list-table::
   :widths: 32 18 50
   :header-rows: 1

   * - Knob
     - Default
     - Role
   * - ``energy_tolerance``
     - ``0.02`` eV
     - Energy window and binning width
   * - ``comparator_tol``
     - ``0.015``
     - Cumulative pair-correlation tolerance (unitless)
   * - ``comparator_pair_cor_max``
     - ``0.7`` Å
     - Maximum single interatomic-distance difference
   * - ``comparator_n_top``
     - ``None``
     - Expert override of trailing mobile-atom count. Leave ``None`` so
       uniqueness uses the system-type mobile region.
   * - ``deduplicate`` (BH only)
     - ``True``
     - Run BH's end-of-run uniqueness pass. ``False`` skips it; campaign
       filtering still runs
   * - ``comparator_use_mic``
     - ``True`` on surfaces
     - MIC for periodic slabs (``SurfaceSystemConfig``)

Example::

   from scgo.param_presets import get_default_params

   params = get_default_params()
   params["optimizer_params"]["ga"]["energy_tolerance"] = 0.05
   params["optimizer_params"]["ga"]["comparator_pair_cor_max"] = 0.5

Not this
--------

**TS uniqueness is a separate, tighter policy** (same energy-and-geometry rule,
``DEFAULT_TS_PAIR_COR_MAX`` = ``0.1`` Å instead of ``0.7`` Å):

- Pre-pair minima use GO uniqueness (``minima_energy_tolerance`` plus GO
  geometry cutoffs).
- Pair near-dupe gating and final unique-TS clustering share
  ``similarity_tolerance``, ``similarity_pair_cor_max``, and
  ``ts_energy_tolerance``.

Seed and template signature deduplication during initialization is a different
mechanism (rounded distance signatures), not this comparator.
