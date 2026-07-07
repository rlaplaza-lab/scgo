Database access
===============

HPC-oriented SQLite helpers for reading and writing GO/TS optimizer databases.

Use :func:`~scgo.load_previous_run_results` or
:class:`~scgo.database.manager.SCGODatabaseManager` for the common case of
reloading minima from completed ``run_*`` searches. For low-level access,
:func:`~scgo.database.connection.get_connection` opens a scoped
:class:`~ase_ga.data.DataConnection` with SCGO PRAGMA settings applied.

.. automodule:: scgo.database
   :members:
   :undoc-members:
   :show-inheritance:
