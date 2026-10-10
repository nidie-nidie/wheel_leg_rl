from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from debug.sim2sim.dreamwaq_debug_contract import (
    DebugPolicyAdapter,
    TIMING_PROFILES,
    contract_for_timing,
    write_debug_timing_model_copy,
)
from debug.sim2sim.trace_schema import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DREAMWAQ_EXPORT = (
    PROJECT_ROOT
    / "artifacts/phase2_dreamwaq/training-suite-20261007-185711/run-01-evaluation/export"
)
MODEL_MANIFEST = PROJECT_ROOT / "sim2sim/mujoco/model_manifest.json"
FORMAL_MODEL = PROJECT_ROOT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"


def _adapter(*, load_mujoco_contract: bool = False) -> DebugPolicyAdapter:
    return DebugPolicyAdapter(
        actor_path=DREAMWAQ_EXPORT / "actor.ts",
        manifest_path=DREAMWAQ_EXPORT / "policy_manifest.json",
        model_manifest_path=MODEL_MANIFEST,
        load_mujoco_contract=load_mujoco_contract,
    )


@dataclass(frozen=True)
class _TimingContract:
    physics_dt_s: float = 0.001
    physics_steps_per_action: int = 20
    control_dt_s: float = 0.020


def test_dreamwaq_history_initialization_and_advance() -> None:
    adapter = _adapter()
    assert adapter.contract is None
    current = np.arange(25, dtype=np.float32)
    history = adapter.initialize_policy_input(current)
    assert history.shape == (125,)
    np.testing.assert_array_equal(history.reshape(5, 25), np.tile(current, (5, 1)))

    following = current + np.float32(100.0)
    advanced = adapter.advance_policy_input(history, following)
    np.testing.assert_array_equal(advanced.reshape(5, 25)[:-1], history.reshape(5, 25)[1:])
    np.testing.assert_array_equal(advanced.reshape(5, 25)[-1], following)


def test_dreamwaq_encoder_inspection_matches_frozen_actor() -> None:
    adapter = _adapter()
    golden = torch.load(DREAMWAQ_EXPORT / "golden_vectors.pt", map_location="cpu", weights_only=False)
    history = golden["history"][0].numpy().astype(np.float32, copy=True)
    output = adapter.infer(history)
    with torch.inference_mode():
        expected = adapter.actor(torch.from_numpy(history).unsqueeze(0)).numpy()[0]
        encoded = adapter.actor.encoder(torch.from_numpy(history).unsqueeze(0)).numpy()[0]
    np.testing.assert_allclose(output.raw_action, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(output.estimated_velocity, encoded[:3], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(output.context_mu, encoded[3:19], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(output.context_logvar, encoded[19:35], rtol=0.0, atol=0.0)


def test_debug_timing_profiles_keep_control_period_and_refresh_schedule() -> None:
    assert TIMING_PROFILES["formal_1ms"].physics_steps_per_action == 20
    assert TIMING_PROFILES["isaac_sync_5ms"].physics_steps_per_action == 4
    hold = TIMING_PROFILES["hold_5ms"]
    assert [index for index in range(20) if hold.refreshes_at(index)] == [0, 5, 10, 15]
    contract = contract_for_timing(_TimingContract(), TIMING_PROFILES["isaac_sync_5ms"])
    assert contract.physics_dt_s == 0.005
    assert contract.physics_steps_per_action == 4
    assert contract.control_dt_s == 0.020


def test_debug_timing_model_copy_does_not_change_formal_model(tmp_path) -> None:
    source_hash = sha256_file(FORMAL_MODEL)
    output = tmp_path / "debug" / "wheelleg_sync.xml"
    write_debug_timing_model_copy(FORMAL_MODEL, output, TIMING_PROFILES["isaac_sync_5ms"])
    assert sha256_file(FORMAL_MODEL) == source_hash
    tree = ET.parse(output)
    assert tree.getroot().find("option").get("timestep") == "0.0050000000000000001"
    assert Path(tree.getroot().find("compiler").get("meshdir")).is_absolute()
    assert output.with_suffix(".json").is_file()
