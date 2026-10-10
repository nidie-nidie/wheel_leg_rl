from __future__ import annotations

from debug.sim2sim.root_cause_suite.verdict import build_verdict


def test_invalid_gate_precedes_physics() -> None:
    verdict = build_verdict({"failed_gates": ["G00"], "candidates": {}})
    assert verdict["conclusion_status"] == "invalid"
    assert verdict["primary_classification"] == "INVALID_EVIDENCE_PIPELINE"


def test_multiple_primary_and_checkpoint_never_primary() -> None:
    verdict = build_verdict(
        {
            "failed_gates": [],
            "candidates": {
                "ACTUATOR_INTEGRATION_MISMATCH": "supported_primary",
                "CLOSED_CHAIN_CONSTRAINT_MISMATCH": "supported_primary",
            },
            "checkpoint_role": "amplifier",
        }
    )
    assert verdict["primary_classification"] == "MULTIPLE_PHYSICS_MISMATCHES"
    assert "CHECKPOINT_ROBUSTNESS_FAILURE" not in verdict["supported_primary"]


def test_no_primary_is_inconclusive() -> None:
    verdict = build_verdict(
        {
            "failed_gates": [],
            "candidates": {"CONTACT_OR_FRICTION_MISMATCH:normal": "supported_contributor"},
            "checkpoint_role": "not_distinguished",
        }
    )
    assert verdict["conclusion_status"] == "inconclusive"

