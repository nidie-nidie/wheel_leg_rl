from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from debug.sim2sim.root_cause_suite.trace_contract import (
    FieldSpec,
    load_verified_trace,
    normalize_availability,
    write_trace,
)


def test_trace_round_trip_and_hash_verification(tmp_path: Path) -> None:
    arrays = {"time_s": np.asarray([0.0, 0.005]), "value": np.asarray([[1.0], [2.0]])}
    fields = {
        "time_s": FieldSpec("s", "simulation", "post_step", "scalar"),
        "value": FieldSpec("rad", "canonical", "post_step", "joint"),
    }
    identity = write_trace(tmp_path / "trace", arrays, fields)
    loaded = load_verified_trace(tmp_path / "trace")
    assert identity.trace_sha256 == loaded.identity.trace_sha256
    assert np.array_equal(loaded.arrays["value"], arrays["value"])

    (tmp_path / "trace" / "trace.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        load_verified_trace(tmp_path / "trace")


def test_availability_never_treats_minus_one_as_false() -> None:
    unavailable = normalize_availability(-1, sentinel=-1)
    assert unavailable.available is False
    assert unavailable.value is None
    assert unavailable.reason == "sentinel"


def test_field_spec_rejects_missing_metadata() -> None:
    with pytest.raises(ValueError):
        FieldSpec("", "canonical", "post_step", "joint")

