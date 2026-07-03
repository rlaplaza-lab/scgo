"""Cluster initialization package.

Builds starting structures for global optimization and surface deposition.

Main entry points:

- ``create_initial_cluster`` and ``create_initial_cluster_batch``
- ``random_spherical`` and ``grow_from_seed``
- ``combine_and_grow``
- ``generate_template_structure``

All randomness flows through ``numpy.random.Generator`` arguments.
See the initialization chapter in the project documentation for modes,
ordering, and reproducibility.
"""

from __future__ import annotations

from .atomic_radii import get_covalent_radius, get_vdw_radius
from .geometry_helpers import (
    StructureDiagnostics,
    get_structure_diagnostics,
    is_cluster_connected,
    validate_cluster,
    validate_cluster_structure,
)
from .initializers import (
    compute_cell_side,
    create_initial_cluster,
    create_initial_cluster_batch,
)
from .random_spherical import (
    grow_from_seed,
    random_spherical,
)
from .seed_combiners import combine_and_grow, combine_seeds
from .templates import (
    generate_cube,
    generate_cuboctahedron,
    generate_decahedron,
    generate_icosahedron,
    generate_octahedron,
    generate_template_structure,
    generate_tetrahedron,
    generate_truncated_octahedron,
    get_nearest_magic_number,
    is_near_magic_number,
)

__all__ = [
    # Main functions
    "create_initial_cluster",
    "create_initial_cluster_batch",
    "random_spherical",
    "grow_from_seed",
    "combine_seeds",
    "combine_and_grow",
    "compute_cell_side",
    "is_cluster_connected",
    "validate_cluster",
    "validate_cluster_structure",
    # Diagnostics and utilities
    "StructureDiagnostics",
    "get_covalent_radius",
    "get_vdw_radius",
    "get_structure_diagnostics",
    # Template functions
    "generate_icosahedron",
    "generate_decahedron",
    "generate_octahedron",
    "generate_tetrahedron",
    "generate_cube",
    "generate_cuboctahedron",
    "generate_truncated_octahedron",
    "generate_template_structure",
    "get_nearest_magic_number",
    "is_near_magic_number",
]
