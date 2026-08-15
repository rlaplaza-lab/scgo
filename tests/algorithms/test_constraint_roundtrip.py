"""GA/TS constraint persistence: FixAtoms / FixBondLengths survive DB and JSON IO."""

import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms, FixBondLengths

from scgo.algorithms.ga_common import (
    extract_constraint_index_lists,
    reconstruct_constraints_from_index_lists,
)
from scgo.algorithms.geneticalgorithm_go_torchsim import (
    GAWriteContext,
    _write_relaxed_candidate,
)
from scgo.database import setup_database
from scgo.metadata.atoms import get_tag
from scgo.surface.config import SurfaceSystemConfig
from scgo.ts_search.transition_state_io import (
    load_minima_by_composition,
    save_transition_state_results,
)


def _make_combined() -> Atoms:
    slab_positions = []
    for layer in range(3):
        for j in range(2):
            slab_positions.append((j * 1.5, 0.0, float(layer)))
    slab = Atoms(
        symbols=["C"] * 6,
        positions=slab_positions,
        cell=[8, 8, 12],
        pbc=[True, True, False],
    )
    ads = Atoms(
        symbols=["C", "O"],
        positions=[(0.75, 0.75, 3.2), (0.75, 0.75, 4.33)],
    )
    return slab + ads


def _surface_config(combined: Atoms) -> SurfaceSystemConfig:
    return SurfaceSystemConfig(
        slab=combined[:6].copy(), fix_all_slab_atoms=False, n_relax_top_slab_layers=1
    )


def _write_context(combined: Atoms) -> GAWriteContext:
    return GAWriteContext(
        n_slab=6,
        n_frozen_prefix=6,
        composition=None,
        adsorbate_definition=None,
        system_type="surface",
        surface_mode=True,
        surface_config=_surface_config(combined),
        connectivity_factor=None,
        allow_cluster_fragmentation=False,
        allow_adsorbate_surface_detachment=False,
        enforce_adsorbate_subgraph_integrity=True,
        freeze_adsorbate_internal_geometry=False,
        adsorbate_fragment_templates=None,
    )


def test_extract_and_reconstruct_constraint_index_lists() -> None:
    atoms = Atoms(symbols=["C"] * 6, positions=np.zeros((6, 3)))
    atoms.set_constraint([FixAtoms(indices=[0, -1, -2]), FixBondLengths([(4, 5)])])
    lists = extract_constraint_index_lists(atoms)
    assert lists["fix_atoms_indices"] == [0, 4, 5]
    assert lists["fix_bond_lengths_pairs"] == [[4, 5]]

    fresh = Atoms(symbols=["C"] * 6, positions=np.zeros((6, 3)))
    changed = reconstruct_constraints_from_index_lists(
        fresh,
        fix_atoms_indices=lists["fix_atoms_indices"],
        fix_bond_lengths_pairs=lists["fix_bond_lengths_pairs"],
    )
    assert changed
    fa = [c for c in fresh.constraints if isinstance(c, FixAtoms)][0]
    assert sorted(fa.index) == [0, 4, 5]
    fb = [c for c in fresh.constraints if isinstance(c, FixBondLengths)][0]
    assert list(fb.pairs[0]) == [4, 5]

    atoms2 = Atoms(symbols=["C"] * 6, positions=np.zeros((6, 3)))
    atoms2.set_constraint(FixAtoms(indices=[1]))
    assert (
        reconstruct_constraints_from_index_lists(
            atoms2, fix_atoms_indices=[0, 1, 2], fix_bond_lengths_pairs=None
        )
        is False
    )
    assert sorted(
        [c for c in atoms2.constraints if isinstance(c, FixAtoms)][0].index
    ) == [1]


def test_ga_relaxed_row_preserves_constraints(tmp_path) -> None:
    base = tmp_path / "run_0"
    base.mkdir()

    combined = _make_combined()
    relaxed = combined.copy()
    relaxed.set_constraint([FixAtoms(indices=[0, 1, 2, 3]), FixBondLengths([(6, 7)])])
    original = combined.copy()
    original.positions[6] = (0.75, 0.75, 3.2)
    original.positions[7] = (0.75, 0.75, 4.33)
    original.info["confid"] = 1
    original.info.setdefault("data", {})

    da = setup_database(str(base), "ga_go.db", atoms_template=combined, run_id="0")

    _write_relaxed_candidate(
        da,
        original,
        relaxed,
        energy=-1.0,
        ctx=_write_context(combined),
        generation=0,
        run_id="0",
    )

    minima = load_minima_by_composition(str(tmp_path), prefer_final_unique=False)
    assert minima
    formula = next(iter(minima))
    assert minima[formula]
    _energy, atoms = minima[formula][0]

    assert get_tag(atoms, "fix_atoms_indices_json") == [0, 1, 2, 3]
    assert get_tag(atoms, "fix_bond_lengths_pairs_json") == [[6, 7]]

    fa = [c for c in atoms.constraints if isinstance(c, FixAtoms)]
    fb = [c for c in atoms.constraints if isinstance(c, FixBondLengths)]
    assert fa and sorted(fa[0].index) == [0, 1, 2, 3]
    assert fb and list(fb[0].pairs[0]) == [6, 7]

    atoms.set_constraint([])
    assert reconstruct_constraints_from_index_lists(
        atoms,
        fix_atoms_indices=get_tag(atoms, "fix_atoms_indices_json"),
        fix_bond_lengths_pairs=get_tag(atoms, "fix_bond_lengths_pairs_json"),
    )
    fa2 = [c for c in atoms.constraints if isinstance(c, FixAtoms)]
    fb2 = [c for c in atoms.constraints if isinstance(c, FixBondLengths)]
    assert fa2 and sorted(fa2[0].index) == [0, 1, 2, 3]
    assert fb2 and list(fb2[0].pairs[0]) == [6, 7]


def test_ts_save_persists_constraint_index_lists(tmp_path) -> None:
    combined = _make_combined()
    relaxed = combined.copy()
    relaxed.set_constraint([FixAtoms(indices=[0, 1, 2, 3]), FixBondLengths([(6, 7)])])

    ts_results = [
        {
            "pair_id": "0_1",
            "status": "success",
            "neb_converged": True,
            "n_images": 5,
            "spring_constant": 1.0,
            "reactant_energy": -2.0,
            "product_energy": -1.5,
            "ts_energy": -1.0,
            "barrier_height": 1.0,
            "reactant_structure": relaxed.copy(),
            "product_structure": relaxed.copy(),
            "transition_state": relaxed.copy(),
        }
    ]
    path = save_transition_state_results(
        ts_results, str(tmp_path), ["C"] * 8, run_id="0"
    )
    summary = json.loads(Path(path).read_text())
    result_json = summary["results"][0]
    assert result_json["reactant_fix_atoms_indices_json"] == [0, 1, 2, 3]
    assert result_json["reactant_fix_bond_lengths_pairs_json"] == [[6, 7]]
    assert "product_fix_atoms_indices_json" in result_json
    assert "transition_state_fix_atoms_indices_json" in result_json
