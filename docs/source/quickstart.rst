Quick start
===========

Install the base package and run one search. Later sections are one snippet
each for a real potential, a surface, a molecule, a slab, transition states,
and a campaign. ``system_type`` is a required argument on every ``run_*`` call.

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
``get_default_params()`` is the merge base.

SCGO picks the algorithm from how many atoms move. One or two atoms in vacuum
are relaxed. Three atoms and no molecule use basin hopping, on a slab as well
as in vacuum. Every other search uses a genetic algorithm.

The run writes ``Pt4_searches/`` in the current directory. Inside it,
``run_YYYYMMDD_HHMMSS_ffffff/`` holds that attempt, and
``final_unique_minima/`` holds the structures you just printed. The folder
name is the composition for a bare gas cluster. A molecule or a slab appends
its own pieces (``Pt5_OH``, ``Pt5_graphite``). The full tree, including
transition-state output, is in :doc:`/output_layout`.

.. _quickstart-mace:

A real potential
----------------

.. code-block:: bash

   pip install "scgo[mace]"

:func:`~scgo.param_presets.get_default_params` is the production preset. The
calculator is MACE ``mace_matpes_0`` (MatPES r²SCAN) with DFT-D3(BJ). The
first run downloads that checkpoint. With ``device`` unset, MACE uses CUDA,
then Apple MPS, then CPU.

Left alone, a gas-phase ``Pt5`` search uses a population of 83 and 65
generations. It stops early if the best energy is unchanged for 10
generations. The two overrides below shrink the search. Remove them for the
automatic budget in :doc:`/search_effort`.

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

This call uses the EMT preset. For MACE, use the preset from the previous
section. Graphene, vacancies, dopants, and a slab you built yourself are in
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

Output goes to ``Pt5_OH_searches/``. Placement knobs live on
:class:`~scgo.cluster_adsorbate.config.ClusterAdsorbateConfig`; see
:doc:`/api/cluster_adsorbate`. The same fragment on a slab uses
``surface_config`` and ``system_type="surface_cluster_adsorbate"``, covered
in :doc:`/surface_slab_guide`.

.. _quickstart-slab:

The slab itself
---------------

``surface`` searches the top layers of the slab. There is no deposited
cluster, so the composition is an empty list. This helper punches one
vacancy and leaves the top layer free to move.

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

Output goes to ``defected_graphite_searches/``. The preset is a 5×5,
three-layer sheet with one carbon removed (149 atoms), so this is a heavier
EMT run than ``Pt4``. Add an ``OH`` fragment and set
``system_type="surface_adsorbate"`` to move the top layer and the molecule
together. Pristine, defected, and N-doped graphite presets are in
:doc:`/surface_slab_guide`.

.. _quickstart-ts:

Transition states
-----------------

A transition-state search reads minima you already wrote and runs nudged
elastic band on a short list of pairs. ``max_pairs`` is how many bands run.
Which pairs make that list is documented under **Pair selection** in
:doc:`/parameters`.

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

``results`` is a list of dicts, one per NEB. Files land in
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

``summary`` includes ``minima_by_formula`` and ``ts_results``. With
``output_dir`` omitted, files go under ``scgo_runs/Pt4_emt/``, separate from
the ``Pt4_searches/`` folder of the first example. ``seed`` must agree across
the call, ``go_params``, and ``ts_params`` when more than one of them sets it.

.. _quickstart-campaign:

Several compositions
--------------------

:func:`~scgo.runner_api.build_one_element_compositions` and
:func:`~scgo.runner_api.build_two_element_compositions` live in
``scgo.runner_api``, not the top-level ``scgo`` package.

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

``results`` is a dict keyed by the on-disk folder stem (``"Pt2"``,
``"Pt3"``, ``"Pt4"`` for this call). Each value is the same
``(energy, Atoms)`` list as ``run_go``. A composition that fails is logged,
stored as an empty list, and skipped. The rest of the campaign continues.

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

Every option is in :doc:`/parameters`. What each preset function changes is
in :doc:`/api/param_presets`. Working scripts for all six system types are
in the repository `examples/ <https://github.com/rlaplaza-lab/scgo/blob/main/examples/README.md>`_.
