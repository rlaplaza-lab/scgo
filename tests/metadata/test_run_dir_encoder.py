"""Regression tests for :class:`RunDirJSONEncoder` numpy support.

Params snapshots frequently carry numpy scalars (e.g. ``np.int64`` from a
numpy RNG or an array reduction). The encoder only handled ``type`` objects, so
``save_run_dir_record`` raised ``TypeError: Object of type int64 is not JSON
serializable``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from scgo.metadata.run_dir import (
    RunDirJSONEncoder,
    load_run_dir_record,
    save_run_dir_record,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.int64(3), 3),
        (np.int32(7), 7),
        (np.float64(0.05), 0.05),
        (np.bool_(True), True),
    ],
)
def test_encoder_serializes_numpy_scalars(value, expected):
    dumped = json.dumps({"param": value}, cls=RunDirJSONEncoder)
    assert json.loads(dumped) == {"param": expected}


def test_encoder_still_serializes_type_objects():
    from ase.optimize import LBFGS

    dumped = json.dumps({"optimizer": LBFGS}, cls=RunDirJSONEncoder)
    assert json.loads(dumped) == {"optimizer": "LBFGS"}


def test_save_run_dir_record_with_numpy_int_params(tmp_path):
    run_dir = str(tmp_path / "run_20240101_000000_000000")
    record = {
        "path_key": "Pt2",
        "composition": ["Pt", "Pt"],
        "params": {
            "n_atoms": np.int64(3),
            "fmax": np.float64(0.05),
            "seeds": [np.int64(1), np.int64(2)],
            "nested": {"n_trials": np.int32(4)},
        },
    }

    save_run_dir_record(run_dir, "run_20240101_000000_000000", record=record)

    metadata_file = os.path.join(run_dir, "metadata.json")
    assert os.path.exists(metadata_file)

    with open(metadata_file) as handle:
        payload = json.load(handle)

    assert payload["params"]["n_atoms"] == 3
    assert payload["params"]["fmax"] == pytest.approx(0.05)
    assert payload["params"]["seeds"] == [1, 2]
    assert payload["params"]["nested"]["n_trials"] == 4

    loaded = load_run_dir_record(run_dir)
    assert loaded is not None
    assert loaded.params["n_atoms"] == 3
