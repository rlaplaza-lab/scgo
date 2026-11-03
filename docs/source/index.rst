SCGO Documentation
===================

.. raw:: html

   <div style="text-align: center; margin: 30px 0 20px 0;">
       <img src="_static/scgo_logo.svg" alt="SCGO" style="width: 200px;">
   </div>

**SCGO** searches for low-energy arrangements of atoms. A **cluster** is the
metal nanoparticle. An **adsorbate** is a molecule or fragment on that cluster
or on a surface. A **slab** is the periodic surface underneath. The search
uses basin hopping or a genetic algorithm. Transition states use nudged
elastic band. MACE, UMA, and UPET are optional and run through ASE.

Install the base package and run the first example in :doc:`/quickstart`.

What you can search
-------------------

Set ``system_type`` to one of these. The calls are in the quickstart.

- ``gas_cluster``: cluster in vacuum
- ``surface_cluster``: cluster on a slab
- ``gas_cluster_adsorbate``: cluster plus molecules in vacuum
- ``surface_cluster_adsorbate``: cluster plus molecules on a slab
- ``surface``: top layers of a bare slab
- ``surface_adsorbate``: top slab layers plus molecules, no cluster

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   surface_slab_guide
   uniqueness
   validation_and_constraints
   output_layout
   parameters
   search_effort
   benchmarks

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/runner_api
   api/runners
   api/scgo
   api/calculators
   api/database
   api/metadata
   api/exceptions
   api/initialization
   api/surface
   api/cluster_adsorbate
   api/param_presets
   api/validation_and_constraints_api
   api/system_types_guide
   api/system_types
   api/utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
