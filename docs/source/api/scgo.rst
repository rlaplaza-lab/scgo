Top-level API
==============

The most commonly referenced re-exports of the ``scgo`` package. These names
are imported directly from ``scgo`` (they live in ``scgo.__all__``). Lower-level
modules have their own pages under ``api/``.

Surface and adsorption
----------------------

.. autofunction:: scgo.adsorption_energy
.. autofunction:: scgo.is_true_minimum
.. autofunction:: scgo.perform_local_relaxation

Database
--------

.. autofunction:: scgo.setup_database
.. autofunction:: scgo.load_previous_run_results

Surface configuration builders
------------------------------

.. autofunction:: scgo.make_surface_config
.. autofunction:: scgo.make_graphite_surface_config
.. autofunction:: scgo.make_hopg_5x5_graphite_surface_config
.. autofunction:: scgo.make_graphene_surface_config
.. autofunction:: scgo.make_defected_graphite_surface_config
.. autofunction:: scgo.make_hopg_5x5_defected_graphite_surface_config
.. autofunction:: scgo.make_n_doped_graphite_surface_config

Parameter resolution
--------------------

.. autofunction:: scgo.get_ts_search_params
.. autofunction:: scgo.get_system_path_key

Logging
-------

.. autofunction:: scgo.configure_logging

GO + TS pipeline
----------------

.. autofunction:: scgo.run_go_ts

Version
-------

.. autodata:: scgo.__version__

:class:`~scgo.surface.config.SurfaceSystemConfig` is documented under
:doc:`/api/surface`.
