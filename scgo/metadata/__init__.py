"""SCGO metadata package.

Separate concerns (do not merge schemas):

- :mod:`scgo.metadata.atoms` — structure tags on ASE Atoms (``key_value_pairs``)
- :mod:`scgo.metadata.persist` — SQL updates to stored tags (final minima)
- :mod:`scgo.metadata.provenance` — output-JSON provenance header
- :mod:`scgo.metadata.run_dir` — ``run_*/metadata.json`` records and run IDs
- :mod:`scgo.metadata.db_stamp` — SQLite ``scgo_metadata`` identity stamp

Import from the submodules above; this package does not re-export symbols.
"""
