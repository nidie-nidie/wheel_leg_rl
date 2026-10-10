from __future__ import annotations

from pathlib import Path

import numpy as np

from debug.sim2sim.collect_dreamwaq_mujoco_trace import collect_dreamwaq_mujoco_trace
from debug.sim2sim.dreamwaq_debug_contract import TIMING_PROFILES
from debug.sim2sim.trace_schema import load_npz, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPORT = PROJECT_ROOT / "artifacts/phase2_dreamwaq/training-suite-20261007-185711/run-01-evaluation/export"
FORMAL_MODEL = PROJECT_ROOT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"


def test_dreamwaq_mujoco_trace_records_history_cenet_and_substeps(tmp_path) -> None:
    formal_hash = sha256_file(FORMAL_MODEL)
    output = tmp_path / "debug" / "trace"
    metadata = collect_dreamwaq_mujoco_trace(
        actor_path=EXPORT / "actor.ts",
        manifest_path=EXPORT / "policy_manifest.json",
        output_directory=output,
        timing=TIMING_PROFILES["isaac_sync_5ms"],
        control_ticks=2,
        mode="closed_loop",
    )
    control = load_npz(output / "control_trace.npz")
    substeps = load_npz(output / "substep_trace.npz")
    assert control["actor_obs_policy_pre_step"].shape == (2, 125)
    assert control["actor_obs_current_pre_step"].shape == (2, 25)
    assert control["cenet_estimated_velocity"].shape == (2, 3)
    assert control["cenet_context_mu"].shape == (2, 16)
    assert "pd_torque_effort_clipped_canonical_mean" in control
    assert "pd_torque_effort_clipped_canonical_canonical_mean" not in control
    assert substeps["control_tick"].shape == (8,)
    assert metadata["timing_profile"]["physics_dt_s"] == 0.005
    assert sha256_file(FORMAL_MODEL) == formal_hash


def test_action_replay_uses_exact_supplied_rows(tmp_path) -> None:
    output = tmp_path / "debug" / "replay"
    sequence = np.asarray([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]], dtype=np.float32)
    collect_dreamwaq_mujoco_trace(
        actor_path=EXPORT / "actor.ts",
        manifest_path=EXPORT / "policy_manifest.json",
        output_directory=output,
        timing=TIMING_PROFILES["formal_1ms"],
        control_ticks=1,
        mode="action_replay",
        action_sequence=sequence,
    )
    control = load_npz(output / "control_trace.npz")
    np.testing.assert_allclose(control["action_clipped"], sequence, rtol=0.0, atol=1.0e-8)
