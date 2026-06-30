Runner API
==========

High-level API entry points for global optimization and transition state searches.

GA timing and profiling
-----------------------

Configure timing in ``params`` / ``go_params`` only (``optimizer_params['ga']`` or
``bh``):

* ``write_timing_json=True`` — write ``timing.json`` under each trial directory.
* ``detailed_timing=True`` — add ``per_generation`` rows (requires
  ``write_timing_json=True``).

For TS, set ``write_timing_json`` in ``ts_params`` when needed.

See :mod:`scgo.utils.timing_report` for the JSON layout.

Workflow functions
------------------

.. automodule:: scgo.runner_api
   :members:
     run_go,
     run_go_campaign,
     run_ts_search,
     run_ts_campaign,
     run_go_ts,
     run_go_ts_campaign,
     log_go_ts_summary,
     resolve_workflow_seed,
     parse_composition_arg,
     build_one_element_compositions,
     build_two_element_compositions,
   :undoc-members:
   :show-inheritance:
