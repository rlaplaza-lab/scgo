"""Process-safety regression tests for :func:`is_true_minimum`.

The Hessian check used to run ``Vibrations(atoms, name="vib_check")`` in the
current working directory, so concurrent ``ProcessPoolExecutor`` workers shared
one cache namespace and could clobber each other's displacement files (wrong
verdicts) or leave files behind.
"""

from __future__ import annotations

import concurrent.futures
import glob
import os
import tempfile

from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import LBFGS

from scgo.utils.helpers import is_true_minimum, perform_local_relaxation

# (symbol, initial bond length, relax?) — relaxed dimers are true minima,
# the compressed unrelaxed one fails the force check.
CASES: list[tuple[str, float, bool]] = [
    ("Pt", 2.6, True),
    ("Ag", 2.8, True),
    ("Cu", 2.4, True),
    ("Au", 2.7, True),
    ("Pt", 1.2, False),
    ("Ag", 1.2, False),
]


def _build(case: tuple[str, float, bool]) -> Atoms:
    symbol, distance, relax = case
    atoms = Atoms(f"{symbol}2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
    if relax:
        perform_local_relaxation(atoms, EMT(), LBFGS, fmax=0.005, steps=100)
    return atoms


def _check_case(case: tuple[str, float, bool]) -> bool:
    """Top-level worker (must be picklable for ProcessPoolExecutor)."""
    atoms = _build(case)
    return is_true_minimum(
        atoms,
        calculator=EMT(),
        fmax_threshold=0.05,
        check_hessian=True,
        imag_freq_threshold=50.0,
    )


def _vib_leftovers() -> list[str]:
    patterns = [
        os.path.join(os.getcwd(), "vib_check*"),
        os.path.join(os.getcwd(), "vib*.json"),
        os.path.join(tempfile.gettempdir(), "scgo_vib_*"),
    ]
    found: list[str] = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))
    return found


def test_is_true_minimum_parallel_matches_serial_and_leaves_no_files(tmp_path):
    """Concurrent calls must not collide and must return the serial verdicts."""
    before = set(_vib_leftovers())

    serial = [_check_case(case) for case in CASES]

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        parallel = list(executor.map(_check_case, CASES))

    assert parallel == serial
    # Relaxed dimers are minima, the compressed unrelaxed ones are not.
    assert parallel == [case[2] for case in CASES]

    after = set(_vib_leftovers())
    assert after == before, f"leftover vibration cache files: {sorted(after - before)}"


def test_is_true_minimum_repeated_calls_are_independent():
    """Sequential calls reusing the same process must not share a vib cache."""
    first = _check_case(("Pt", 2.6, True))
    second = _check_case(("Pt", 1.2, False))
    third = _check_case(("Pt", 2.6, True))

    assert (first, second, third) == (True, False, True)
