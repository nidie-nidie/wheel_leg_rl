from __future__ import annotations

from pathlib import Path

import mujoco

from debug.sim2sim.dreamwaq_debug_contract import TIMING_PROFILES
from debug.sim2sim.evaluate_debug_mujoco import evaluate_debug_mujoco
from debug.sim2sim.trace_schema import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPORT = PROJECT_ROOT / "artifacts/phase2_dreamwaq/training-suite-20261007-185711/run-01-evaluation/export"
FORMAL_MODEL = PROJECT_ROOT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"


def test_sync_timing_evaluator_is_debug_only_and_keeps_formal_model(tmp_path) -> None:
    formal_hash = sha256_file(FORMAL_MODEL)
    report = evaluate_debug_mujoco(
        actor_path=EXPORT / "actor.ts",
        manifest_path=EXPORT / "policy_manifest.json",
        output_directory=tmp_path / "debug" / "evaluation",
        timing=TIMING_PROFILES["isaac_sync_5ms"],
        expected_ticks=2,
        scenarios=(("nominal_stand", (0.0, 0.0, 0.20)),),
    )
    assert report["debug_only"] is True
    assert report["formal_ranking_eligible"] is False
    assert report["timing_profile"]["physics_dt_s"] == 0.005
    assert report["aggregate"]["scenario_count"] == 1
    assert sha256_file(FORMAL_MODEL) == formal_hash
    model = mujoco.MjModel.from_xml_path(report["debug_model"])
    assert model.opt.timestep == 0.005
