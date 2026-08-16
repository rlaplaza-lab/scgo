Surface slabs for global optimization
======================================

This guide shows how to build graphene and graphite slabs and use them as
substrates for cluster global optimization, both pristine and with defects
(vacancies, dopants). It complements the API reference in :doc:`/api/surface`
and the recipes in :doc:`/quickstart`.

SCGO supports two slab roles for ``surface_cluster`` /
``surface_cluster_adsorbate`` (deposit and optimize a nanoparticle on top) and
for ``surface`` / ``surface_adsorbate`` (let the search move the top slab
layers, with or without an adsorbate fragment, and leave the lower layers
fixed).

.. note::

   All slabs use slab-style periodic boundary conditions: periodic in the two
   in-plane directions and open along the vacuum (surface-normal) axis. For the
   default orientation the surface normal is ``z`` (``pbc=(True, True, False)``).
   :class:`~scgo.surface.config.SurfaceSystemConfig` normalizes any slab you
   pass and raises if no periodic dimension is present.

Choosing a substrate
--------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Slab
     - Use when
   * - Graphite (pristine)
     - Multi-layer support; the deposited cluster sits on the top graphene
       sheet. Built with :func:`~scgo.make_graphite_surface_config`.
   * - Graphene (pristine, monolayer)
     - Single free-standing sheet. Built with
       :func:`~scgo.make_graphene_surface_config`.
   * - Defected graphite
     - Graphite with one or more top-layer carbon vacancies. Built with
       :func:`~scgo.make_defected_graphite_surface_config`.
   * - Monovacancy graphene
     - Single-layer graphene with one removed carbon atom; placement can target
       the vacancy. Built with :func:`~scgo.make_graphene_surface_config`
       (``monovacancy=True``).
   * - N-doped graphite
     - Graphite with top-layer carbons substituted by nitrogen. Built with
       :func:`~scgo.make_n_doped_graphite_surface_config`.

Pristine slabs (no defects)
---------------------------

The presets build the slab, configure deposition heights, and return a ready
:class:`~scgo.surface.config.SurfaceSystemConfig`.

.. code-block:: python

   from scgo import run_go, make_graphite_surface_config, make_graphene_surface_config
   from scgo.param_presets import get_default_params

   # Graphite: 3 layers, 4x4 in-plane repetition
   graphite_cfg = make_graphite_surface_config(slab_layers=3, slab_repeat_xy=4)

   # Graphene monolayer: 4x4 repetition, 18 Angstrom cell height
   graphene_cfg = make_graphene_surface_config(nx=4, ny=4, cell_height=18.0)

   results = run_go(
       "Pt5",
       params=get_default_params(),
       seed=42,
       surface_config=graphite_cfg,
       system_type="surface_cluster",
   )

Both configs set ``fix_all_slab_atoms=False`` with ``n_relax_top_slab_layers=1``
so the surface layer can relax with the deposited cluster. Raise the layer count
or pass ``fix_all_slab_atoms=True`` to freeze the whole slab.

Metal-carbon contacts (for example Pt on graphite or graphene) are often longer
than a default ``1.4×`` covalent-radius sum. Loosen only that contact via
``go_params["connectivity_factor"]`` (also used by the final structural check),
for example ``{"Pt": 1.4, "C": 1.4, "Pt-C": 1.8}``.

Defective slabs
---------------

Defects are recorded on the returned ``Atoms`` so the deposition pipeline can
see them. Two pieces of metadata are written to ``slab.info``:

* ``vacancy_cartesian_angstrom``: the Cartesian position (3-vector) of the
  removed atom (the first removed atom for multi-vacancy graphite).
* ``vacancy_removed_original_index_zero_based``: its index in the pristine slab.

Graphite with vacancies
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scgo import make_defected_graphite_surface_config

   cfg = make_defected_graphite_surface_config(
       slab_layers=3, slab_repeat_xy=3, n_vacancies=1, seed=42
   )
   assert "vacancy_cartesian_angstrom" in cfg.slab.info

The vacancies are removed from the **top layer** of the graphite stack. The
preset defaults ``defect_bias_probability=0.5`` (same as graphene monovacancy)
so cluster-on-slab searches bias a fraction of placements onto the vacancy;
pass ``defect_bias_probability=0.0`` for fully random placement. To use the
same slab directly as the GO target (no nanoparticle core), pass an empty
composition and ``system_type="surface"``. See :ref:`slab-as-target`.

Graphene monovacancy
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scgo import make_graphene_surface_config

   # Default defect_bias_probability=0.5 and name="graphene_monovacancy"
   cfg = make_graphene_surface_config(nx=4, ny=4, monovacancy=True)

   # Optional reconstruction seed: displace two nearest vacancy neighbours
   cfg = make_graphene_surface_config(
       nx=4, ny=4, monovacancy=True, reconstruct=True, reconstruction_shift=0.10
   )

The removed atom is the one closest to the cell center (see
:func:`~scgo.surface.presets.build_monovacancy_graphene_slab`). With
``reconstruct=True`` the two closest vacancy neighbours are shifted along their
bond axes, seeding the usual reconstruction around the vacancy.

N-doped graphite
~~~~~~~~~~~~~~~~

.. code-block:: python

   from scgo import make_n_doped_graphite_surface_config

   cfg = make_n_doped_graphite_surface_config(
       slab_layers=3, slab_repeat_xy=3, n_dopants=2, seed=42
   )

Defect-biased nanoparticle placement
------------------------------------

By default the deposited cluster is placed on a random slab atom (with a small
in-plane jitter). When a slab carries ``vacancy_cartesian_angstrom`` and you set
``defect_bias_probability > 0`` on the config, a fraction of placements land on
the vacancy, while the rest stay random. That keeps some defect coverage without
losing diversity.

.. code-block:: python

   from scgo import make_defected_graphite_surface_config, make_graphene_surface_config

   # Land on the vacancy every time
   cfg = make_graphene_surface_config(
       nx=4, ny=4, monovacancy=True, defect_bias_probability=1.0
   )

   # Mix of defect-targeted and random placements (default for monovacancy
   # graphene and for make_defected_graphite_surface_config)
   cfg = make_graphene_surface_config(nx=4, ny=4, monovacancy=True)  # 0.5

   cfg = make_defected_graphite_surface_config(
       slab_layers=3, slab_repeat_xy=3, n_vacancies=1, seed=42
   )  # defect_bias_probability=0.5 by default

The bias only changes the in-plane center of the cluster. The height above the
slab still follows the normal adsorption-height range, so the cluster stays in
bonding range of the vacancy's neighbours. Setting
``defect_bias_probability=0.0`` (the default for pristine slabs) leaves
placement fully random.

.. _slab-as-target:

Using a slab as the GO target
-----------------------------

For ``surface`` / ``surface_adsorbate`` the top slab layers (not a deposited
core) are the search variables. Pass an empty composition and set the slab to
relax its surface region:

.. code-block:: python

   from scgo import run_go, make_defected_graphite_surface_config
   from scgo.param_presets import get_default_params

   surface_config = make_defected_graphite_surface_config(
       slab_layers=3, slab_repeat_xy=3, n_vacancies=1, seed=42
   )

   results = run_go(
       [],                       # no nanoparticle core
       params=get_default_params(),
       seed=42,
       surface_config=surface_config,
       system_type="surface",
   )

Add an adsorbate (for example ``OH``) and switch to ``surface_adsorbate`` to
optimize the top layers plus the fragment. See :doc:`/quickstart` for the full
adsorbate recipe.

Tuning deposition
-----------------

Key options on :class:`~scgo.surface.config.SurfaceSystemConfig`:

* ``adsorption_height_min`` / ``adsorption_height_max``: vertical gap between
  slab top and cluster bottom. Graphene and n-doped graphite presets use
  0.5-1.5 Å; graphite and defected-graphite presets use 0.5-1.0 Å. That range
  keeps the cluster close to the surface (or to the vacancy neighbours). You
  can widen or override it per run. Accepted placements still pass a clash
  check (``blmin_ratio = 0.7``).
* ``max_placement_attempts``: retries per structure (default 1000 for the
  presets) before giving up on a sterically valid placement.
* ``defect_bias_probability``: fraction of placements targeting the vacancy
  (0.0-1.0; ignored when the slab has no recorded vacancy).
* ``fix_all_slab_atoms`` / ``n_relax_top_slab_layers``: how much of the slab
  relaxes during local optimization.

Low-level builders
------------------

For full control, build the ``Atoms`` directly and wrap them in a config:

.. code-block:: python

   from ase import Atoms
   from scgo.surface import SurfaceSystemConfig, build_graphene_slab
   from scgo.surface.presets import build_monovacancy_graphene_slab

   graphene_sheet = build_graphene_slab(nx=6, ny=6, a=2.46, cell_height=18.0)
   defected = build_monovacancy_graphene_slab(nx=6, ny=6, reconstruct=True)

   cfg = SurfaceSystemConfig(
       slab=defected,
       adsorption_height_min=0.5,
       adsorption_height_max=1.5,
       n_relax_top_slab_layers=1,
       defect_bias_probability=0.5,
   )

Available builders in :mod:`scgo.surface.presets`:
``build_graphite_slab``, ``build_defected_graphite_slab``,
``build_n_doped_graphite_slab``, ``build_graphene_slab``,
``build_monovacancy_graphene_slab``. The config wrappers
(``make_*_surface_config``) are the recommended entry point for GO runs.
