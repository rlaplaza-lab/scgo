SCGO Documentation
===================

.. raw:: html

   <div style="text-align: center; margin: 30px 0 20px 0;">
       <img src="_static/scgo_logo.svg" alt="SCGO" style="width: 200px;">
   </div>

SCGO: Simple Cluster Global Optimization
=========================================

A compact toolkit for global optimization of atomic clusters using ASE, with
Basin Hopping, Genetic Algorithms, NEB transition-state search, and MLIP support.

Key Features
------------

- **Basin Hopping and Genetic Algorithm** global optimization with automatic
  algorithm selection by cluster size
- **MLIP Support**: MACE and UMA (fairchem) for GPU-accelerated optimization
- **Surface Workflows**: slab-supported clusters and adsorbates; hull-site
  fragment placement and tag-aware GA operators
- **Transition State Search**: NEB-based TS search with automated pair selection
  and PBC-aware endpoint alignment
- **Flexible API**: high-level runners and low-level control for custom workflows
- **Reproducible initialization**: composition-canonical atom ordering for
  multi-element GA runs

See :doc:`/quickstart` to get started.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   parameters
   benchmarks

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/runner_api
   api/initialization
   api/surface
   api/cluster_adsorbate
   api/param_presets
   api/system_types

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
