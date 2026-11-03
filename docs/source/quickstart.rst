Quick start
===========

Install the base package and run one search. Later sections cover a MACE
potential, a surface, a molecule, a slab, transition states, and several
compositions. Every ``run_*`` call needs ``system_type``.

.. _quickstart-try:

Try it
------

.. code-block:: bash

   pip install scgo

Python 3.12 or newer. This install is enough for the EMT examples on this
page. A machine-learning potential is a separate extra; see
:doc:`/installation`.

.. code-block:: python

   from scgo import run_go
   from scgo.param_presets import get_testing_params

   results = run_go(
       ["Pt"] * 4,
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster",
   )

   for energy, atoms in results:
       print(f"{energy:.3f} eV  {atoms.get_chemical_formula()}")

``results`` is a list of ``(energy, Atoms)`` pairs, lowest energy first. The
composition can be a symbol list (``["Pt"] * 4``) or a formula string
(``"Pt4"``). :func:`~scgo.param_presets.get_testing_params` uses ASE's EMT
potential and a small search (2 generations, population 5). In the log,
``calculator=EMT`` and ``niter=2`` are that preset. A line that names
``get_default_params()`` only means those defaults were the starting point.

SCGO picks the algorithm from how many atoms move. One or two atoms in vacuum
are relaxed. Three atoms and no molecule use basin hopping, in vacuum or on
a slab. Every other search uses a genetic algorithm.

The run writes ``Pt4_searches/`` in the current directory.
``run_YYYYMMDD_HHMMSS_ffffff/`` is that attempt, and
``final_unique_minima/`` holds the structures from the loop above. For a bare
gas cluster the folder name is the composition. A molecule or a slab is added
to the name (``Pt5_OH``, ``Pt5_graphite``). Directory layout, including
transition-state output, is in :doc:`/output_layout`.

.. _quickstart-mace:

A real potential
----------------

.. code-block:: bash

   pip install "scgo[mace]"

:func:`~scgo.param_presets.get_default_params` is the usual MACE preset. The
model is ``mace_matpes_0`` (MatPES r²SCAN) with DFT-D3(BJ). The first run
downloads that file. If you do not set ``device``, MACE uses CUDA, then
Apple MPS, then CPU.

With no overrides, a gas-phase ``Pt5`` search uses a population of 83 and 65
generations. It stops if the best energy is unchanged for 10 generations.
The two lines below make the search smaller. Delete them to use the automatic
sizes in :doc:`/search_effort`.

.. code-block:: python

   from scgo import run_go
   from scgo.param_presets import get_default_params

   params = get_default_params()
   params["optimizer_params"]["ga"]["niter"] = 2
   params["optimizer_params"]["ga"]["population_size"] = 5

   results = run_go(
       "Pt5",
       params=params,
       seed=42,
       system_type="gas_cluster",
   )

``params["n_jobs"] = -2`` uses every CPU but one for the CPU stages. The
default is a single worker. Other checkpoints and dispersion settings are
in :doc:`/parameters`.

.. _quickstart-surface:

On a surface
------------

Pass a slab as ``surface_config`` and set ``system_type="surface_cluster"``.
:func:`~scgo.make_graphite_surface_config` builds a graphite slab and names
it ``graphite``, so the output folder is ``Pt5_graphite_searches/``.

.. code-block:: python

   from scgo import run_go, make_graphite_surface_config
   from scgo.param_presets import get_testing_params

   results = run_go(
       "Pt5",
       params=get_testing_params(),
       seed=42,
       surface_config=make_graphite_surface_config(),
       system_type="surface_cluster",
   )

This call uses the EMT preset. For MACE, use ``get_default_params()`` as in
the previous section. Graphene, vacancies, dopants, and your own slab are in
:doc:`/surface_slab_guide`.

.. _quickstart-molecule:

With a molecule
---------------

An adsorbate is an ASE ``Atoms`` fragment, passed separately from the metal
composition. One fragment or a list of fragments both work. ``system_type``
is ``gas_cluster_adsorbate``.

.. code-block:: python

   from ase import Atoms
   from scgo import run_go
   from scgo.param_presets import get_testing_params

   oh = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])

   results = run_go(
       "Pt5",
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster_adsorbate",
       adsorbates=oh,
   )

Output goes to ``Pt5_OH_searches/``. Placement settings are on
:class:`~scgo.cluster_adsorbate.config.ClusterAdsorbateConfig`. See
:doc:`/api/cluster_adsorbate`. To put the same fragment on a slab, pass
``surface_config`` and set ``system_type="surface_cluster_adsorbate"``. See
:doc:`/surface_slab_guide`.

.. _quickstart-slab:

The slab itself
---------------

``surface`` moves the top layers of the slab. There is no cluster on top, so
the composition is an empty list. This helper removes one carbon and lets the
top layer move.

.. code-block:: python

   from scgo import run_go, make_defected_graphite_surface_config
   from scgo.param_presets import get_testing_params

   surface_config = make_defected_graphite_surface_config(
       n_vacancies=1,
       seed=42,
   )

   results = run_go(
       [],
       params=get_testing_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface",
   )

Output goes to ``defected_graphite_searches/``. The slab is a 5 by 5,
three-layer sheet with one carbon removed (149 atoms), so it takes longer
than ``Pt4``. Add an ``OH`` fragment and set
``system_type="surface_adsorbate"`` to move the top layer and the molecule
together. Other graphite slabs are in :doc:`/surface_slab_guide`.

.. _quickstart-ts:

Transition states
-----------------

A transition-state search reads minima from an earlier run and runs nudged
elastic band on a short list of pairs. ``max_pairs`` is how many bands run.
**Pair selection** in :doc:`/parameters` describes which pairs are kept.

Run this after the first example, so ``Pt4_searches/`` exists in the
working directory (or pass ``searches_dir``). EMT needs
``use_torchsim=False``.

.. code-block:: python

   from scgo import run_ts_search
   from scgo.param_presets import get_ts_search_params

   ts_params = get_ts_search_params(
       calculator="EMT",
       system_type="gas_cluster",
       seed=42,
   )
   ts_params["use_torchsim"] = False
   ts_params["max_pairs"] = 2

   results = run_ts_search(
       "Pt4",
       ts_params=ts_params,
       seed=42,
       system_type="gas_cluster",
   )

``results`` is a list of dicts, one per NEB. Files are written to
``Pt4_ts_results/``, next to ``Pt4_searches/``.

To optimize and then search in one call:

.. code-block:: python

   from scgo import run_go_ts
   from scgo.param_presets import get_testing_params, get_ts_search_params

   ts_params = get_ts_search_params(
       calculator="EMT",
       system_type="gas_cluster",
       seed=42,
   )
   ts_params["use_torchsim"] = False
   ts_params["max_pairs"] = 2

   summary = run_go_ts(
       "Pt4",
       go_params=get_testing_params(),
       ts_params=ts_params,
       seed=42,
       system_type="gas_cluster",
   )

``summary`` includes ``minima_by_formula`` and ``ts_results``. If you omit
``output_dir``, files go under ``scgo_runs/Pt4_emt/``, not in the
``Pt4_searches/`` folder from the first example. When ``seed`` is set in more
than one place, the call, ``go_params``, and ``ts_params`` must use the same
value.

.. _quickstart-campaign:

Several compositions
--------------------

:func:`~scgo.runner_api.build_one_element_compositions` and
:func:`~scgo.runner_api.build_two_element_compositions` are in
``scgo.runner_api``, not in the top-level ``scgo`` package.

.. code-block:: python

   from scgo import run_go_campaign
   from scgo.param_presets import get_testing_params
   from scgo.runner_api import build_one_element_compositions

   # Pt2, Pt3, Pt4
   compositions = build_one_element_compositions("Pt", min_atoms=2, max_atoms=4)

   results = run_go_campaign(
       compositions,
       params=get_testing_params(),
       seed=42,
       system_type="gas_cluster",
   )

``results`` is a dict. The keys are the folder names (``"Pt2"``, ``"Pt3"``,
``"Pt4"`` for this call). Each value is the same ``(energy, Atoms)`` list as
``run_go``. A composition that fails is logged and stored as an empty list.
The other compositions still run.

``run_ts_campaign`` and ``run_go_ts_campaign`` take the same composition
list. Signatures and the two-element builder are in :doc:`/api/runner_api`.

Presets
-------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Preset
     - Use for
   * - ``get_testing_params()``
     - Fast first run (EMT, small search)
   * - ``get_default_params()``
     - Full MACE search (MatPES r²SCAN + DFT-D3)
   * - ``get_torchsim_ga_params(...)``
     - Genetic algorithm with a TorchSim relaxer
   * - ``get_ts_search_params(...)``
     - Transition-state search

All settings are in :doc:`/parameters`. Preset functions are in
:doc:`/api/param_presets`. Scripts for all six system types are in
`examples/ <https://github.com/rlaplaza-lab/scgo/blob/main/examples/README.md>`_.
