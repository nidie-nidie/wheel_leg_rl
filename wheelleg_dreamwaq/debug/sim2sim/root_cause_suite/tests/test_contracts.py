from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from debug.sim2sim.root_cause_suite.contracts import (
    DECLARED_USD_INFINITY_SPECS,
    DESIGN_SHA256,
    DESIGN_VERSION,
    DeclaredInfinitySpec,
    EvidenceIntegrityError,
    StageName,
    StageStatus,
    canonical_json_bytes,
    declared_infinity_tag,
    encode_declared_usd_value,
    stable_hash,
    validate_declared_infinity_payload,
    validate_design_identity,
)


@dataclass(frozen=True)
class Payload:
    path: Path
    values: np.ndarray


def test_canonical_json_is_stable_for_paths_dataclasses_and_numpy() -> None:
    payload = Payload(Path("a/b"), np.asarray([1.0, 2.0], dtype=np.float32))
    first = canonical_json_bytes(payload)
    second = canonical_json_bytes(payload)
    assert first == second
    assert stable_hash(payload) == stable_hash(payload)
    assert b'"path":"a/b"' in first


@pytest.mark.parametrize(
    ("identity", "spec"), list(DECLARED_USD_INFINITY_SPECS.items())
)
def test_every_declared_usd_infinity_has_exact_field_sign_and_shape(
    identity: tuple[str, str], spec: DeclaredInfinitySpec
) -> None:
    prim_path, attribute = identity
    sign = 1.0 if spec.kind == "positive_infinity" else -1.0
    scalar = sign * float("inf")
    value = [scalar, scalar, scalar] if spec.expected_shape == (3,) else scalar
    encoded = encode_declared_usd_value(
        value, prim_path=prim_path, attribute=attribute
    )
    expected_tag = declared_infinity_tag(spec)
    expected = (
        [expected_tag, expected_tag, expected_tag]
        if spec.expected_shape
        else expected_tag
    )
    assert encoded == expected
    canonical_json_bytes(encoded)


def test_declared_usd_infinity_rejects_nan_wrong_sign_and_undeclared_fields() -> None:
    with pytest.raises(EvidenceIntegrityError, match="contains NaN"):
        encode_declared_usd_value(
            float("nan"),
            prim_path="/physicsScene",
            attribute="physxScene:maxBiasCoefficient",
        )
    with pytest.raises(EvidenceIntegrityError, match="sentinel drifted"):
        encode_declared_usd_value(
            -float("inf"),
            prim_path="/physicsScene",
            attribute="physxScene:maxBiasCoefficient",
        )
    with pytest.raises(EvidenceIntegrityError, match="Undeclared USD infinity"):
        encode_declared_usd_value(
            float("inf"),
            prim_path="/World/envs/env_0/Robot/base",
            attribute="physics:mass",
        )
    with pytest.raises(EvidenceIntegrityError, match="sentinel drifted"):
        encode_declared_usd_value(
            [-float("inf"), 0.0, -float("inf")],
            prim_path="/World/envs/env_0/Coupon",
            attribute="physics:centerOfMass",
        )
    with pytest.raises(ValueError, match="NaN or Inf"):
        canonical_json_bytes(float("inf"))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("mass", float("inf")),
        ("inertia", -float("inf")),
        ("stiffness", np.float32(np.inf)),
        ("damping", np.float64(-np.inf)),
        ("material", float("nan")),
        ("solver", float("inf")),
    ),
)
def test_payload_validator_rejects_nonfinite_physical_fields(
    field: str, value: float
) -> None:
    with pytest.raises(EvidenceIntegrityError, match="raw NaN/Inf"):
        validate_declared_infinity_payload(
            {field: value},
            declaration_for_path=lambda path: None,
            label="test payload",
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("schema_version", "WrongVersion"),
        ("kind", "negative_infinity"),
        ("semantics", "wrong"),
        ("source", "wrong"),
        ("attribute", "wrong"),
    ),
)
def test_payload_validator_rejects_malformed_declared_infinity_tags(
    field: str, replacement: str
) -> None:
    spec = DECLARED_USD_INFINITY_SPECS[
        ("/physicsScene", "physxScene:maxBiasCoefficient")
    ]
    tag = declared_infinity_tag(spec)
    tag[field] = replacement
    with pytest.raises(EvidenceIntegrityError, match="invalid infinity tag"):
        validate_declared_infinity_payload(
            {"value": tag},
            declaration_for_path=lambda path: spec if path == ("value",) else None,
            label="test payload",
        )


def test_payload_validator_rejects_extra_keys_and_illegal_tag_paths() -> None:
    spec = DECLARED_USD_INFINITY_SPECS[
        ("/physicsScene", "physxScene:maxBiasCoefficient")
    ]
    tag = declared_infinity_tag(spec)
    extra = {**tag, "extra": True}
    with pytest.raises(EvidenceIntegrityError, match="malformed infinity tag"):
        validate_declared_infinity_payload(
            {"value": extra},
            declaration_for_path=lambda path: spec if path == ("value",) else None,
            label="test payload",
        )
    with pytest.raises(EvidenceIntegrityError, match="undeclared path"):
        validate_declared_infinity_payload(
            {"mass": tag},
            declaration_for_path=lambda path: None,
            label="test payload",
        )


def test_payload_validator_rejects_partial_shaped_infinity_sentinel() -> None:
    spec = DECLARED_USD_INFINITY_SPECS[
        ("/World/envs/env_0/Coupon", "physics:centerOfMass")
    ]
    tag = declared_infinity_tag(spec)
    with pytest.raises(EvidenceIntegrityError, match="incomplete shaped infinity"):
        validate_declared_infinity_payload(
            {"value": [tag, 0.0, tag]},
            declaration_for_path=lambda path: (
                spec
                if len(path) == 2
                and path[0] == "value"
                and path[1] in (0, 1, 2)
                else None
            ),
            label="test payload",
        )


def test_frozen_design_identity() -> None:
    assert DESIGN_VERSION == "RootCauseSuiteCoreV1.17"
    assert DESIGN_SHA256 == "8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD"
    validate_design_identity()


def test_stage_enums_reject_unknown_values() -> None:
    assert StageName("P20_static_properties") is StageName.P20_STATIC_PROPERTIES
    assert StageStatus("complete") is StageStatus.COMPLETE
    with pytest.raises(ValueError):
        StageName("P20_dynamic_wrench")
    with pytest.raises(ValueError):
        StageStatus("mostly_complete")
