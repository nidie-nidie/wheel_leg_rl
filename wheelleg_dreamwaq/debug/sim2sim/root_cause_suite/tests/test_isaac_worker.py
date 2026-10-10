from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from debug.sim2sim.root_cause_suite.contracts import EvidenceIntegrityError, stable_hash
from debug.sim2sim.root_cause_suite.isaac_worker import (
    DIRECT_EFFORT_SCENARIOS,
    ISAAC_ROBOT_VARIANTS,
    RESET_UNAVAILABLE_SCHEMA_VERSION,
    _capture_authoritative_identity_array,
    _isaac_robot_transform_semantics,
    _merge_serial_repetition_arrays,
    _normalized_serial_physics_time_s,
    _phase_hash_payload,
    _reset_phase_identity_payload,
    _repeatability_profile_state,
    _repeatability_returned_policy_state,
    _robot_profiles,
    _robot_probe_fields,
    _serial_robot_profiles,
)
from debug.sim2sim.root_cause_suite.variant_builders import ROBOT_HINGES


WORKER = Path(__file__).resolve().parents[1] / "isaac_worker.py"


def test_worker_installs_guard_before_third_party_imports() -> None:
    source = WORKER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    top_level_imports.update(
        node.module.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module and not node.module.startswith(".")
    )
    assert not ({"numpy", "torch", "isaaclab", "isaacsim", "carb", "pxr"} & top_level_imports)
    install = source.index("stack.enter_context(guard)")
    assert install < source.index("from isaaclab.app import AppLauncher")


def test_worker_requires_explicit_run_and_kit_roots() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert 'parser.add_argument("--run-root", type=Path, required=True)' in source
    assert 'parser.add_argument("--kit-root", type=Path, required=True)' in source
    assert '"replay-source"' in source
    assert '"replay"' in source


def test_worker_closes_simulation_app_inside_write_guard() -> None:
    source = WORKER.read_text(encoding="utf-8")
    close = source.index("simulation_app.close()")
    ledger = source.index('"ledger": list(guard.ledger)', close)
    context_end = source.index('(output / "identity.json").write_bytes', ledger)
    assert close < ledger < context_end


def test_worker_disables_fast_shutdown_for_final_audit_evidence() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert 'fast_shutdown = args.command != "identity"' in source
    finalize = source.index("finalize_bootstrap(0)")
    close = source.index("simulation_app.close(skip_cleanup=True)")
    assert finalize < close
    assert 'prelaunch.get("fast_shutdown") is not fast_shutdown' in source


def test_robot_probe_variants_cover_core_configuration_families() -> None:
    assert set(ISAAC_ROBOT_VARIANTS) == {
        "p10_a", "p10_b", "p10_c",
        "p30_open_direct", "p30_open_target",
        "p30_closed_direct", "p30_closed_target",
        "p40_on", "p40_off", "p60_b", "p60_c",
    }
    assert ISAAC_ROBOT_VARIANTS["p30_open_direct"] == {
        "ground_enabled": False,
        "gravity_enabled": False,
        "closure_enabled": False,
        "drive_enabled": False,
        "fixed_base": True,
    }
    assert "p30_open_direct" in DIRECT_EFFORT_SCENARIOS
    assert "p30_open_target" not in DIRECT_EFFORT_SCENARIOS


def test_robot_transform_semantics_are_present_for_target_and_direct_variants() -> None:
    identity = SimpleNamespace(
        compiled={
            "robot_hinges": {
                name: {"stiffness": 1.0, "damping": 2.0}
                for name in ROBOT_HINGES
            },
            "closure_constraints": {
                "loop": {"enabled": True},
            },
        }
    )
    target = _isaac_robot_transform_semantics(
        scenario="p30_closed_target", identity=identity
    )
    direct = _isaac_robot_transform_semantics(
        scenario="p30_open_direct", identity=identity
    )
    assert target["actuation"]["drive_mode"] == "formal_target_drive"
    assert target["closure"] == {
        "mode": "enabled",
        "constraints": {"loop": {"enabled": True}},
    }
    assert direct["actuation"]["drive_mode"] == "direct_effort_bypass"
    assert direct["closure"]["mode"] == "disabled"


def test_p30_profiles_are_matched_at_point_three_nm() -> None:
    direct = _robot_profiles("p30_open_direct", repetitions=3, amplitude=None)
    target = _robot_profiles("p30_open_target", repetitions=3, amplitude=None)
    assert len(direct) == len(target) == 54
    for direct_row, target_row in zip(direct, target, strict=True):
        assert direct_row["channel"] == target_row["channel"]
        assert direct_row["sign"] == target_row["sign"]
        assert direct_row["repetition"] == target_row["repetition"]
        assert direct_row["canonical_torque_equivalent"] == target_row["canonical_torque_equivalent"]
    leg_positive = next(row for row in target if row["channel"] == 0 and row["sign"] == 1)
    wheel_positive = next(row for row in target if row["channel"] == 4 and row["sign"] == 1)
    assert leg_positive["actual_input"][0] == 0.3 / (120.0 * 0.35)
    assert wheel_positive["actual_input"][4] == 0.3 / (0.6 * 25.0)


def test_serial_robot_profiles_use_one_environment_per_excitation() -> None:
    profiles = _robot_profiles("p30_closed_target", repetitions=3, amplitude=None)
    serial = _serial_robot_profiles(profiles, repetitions=3)
    assert len(profiles) == 54
    assert len(serial) == 18
    assert all("repetition" not in row for row in serial)
    assert serial[0]["channel"] == 0
    assert serial[0]["sign"] == 1
    assert serial[-1]["channel"] == 5
    assert serial[-1]["sign"] == -1


def test_serial_robot_profiles_reject_noncontiguous_repetitions() -> None:
    profiles = _robot_profiles("p10_a", repetitions=3, amplitude=None)
    profiles[1]["repetition"] = 2
    with pytest.raises(EvidenceIntegrityError, match="exact repetitions"):
        _serial_robot_profiles(profiles, repetitions=3)


def test_merge_serial_repetition_arrays_restores_semantics_major_order() -> None:
    repetitions = []
    for repetition in range(3):
        repetitions.append(
            {
                "scalar": np.asarray(
                    [[10 + repetition, 20 + repetition]], dtype=np.float32
                ),
                "vector": np.asarray(
                    [[[10 + repetition, 100], [20 + repetition, 200]]],
                    dtype=np.float32,
                ),
            }
        )
    merged = _merge_serial_repetition_arrays(repetitions)
    assert merged["scalar"].tolist() == [[10, 11, 12, 20, 21, 22]]
    assert merged["vector"].shape == (1, 6, 2)
    assert merged["vector"][0, :, 0].tolist() == [10, 11, 12, 20, 21, 22]


def test_sphere_probe_is_dispatched_inside_guarded_worker() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert 'elif args.command == "sphere-probe":' in source
    assert "sphere-probe requires --sphere-mode and --friction" in source


def test_instrumentation_probe_covers_all_frozen_modes() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert '"instrumentation-probe"' in source
    for mode in ("formal", "debug", "contact", "system_observer"):
        assert f'"{mode}"' in source
    assert "RootCauseIsaacProbeEnv.capture_system_state(self)" in source
    assert 'env.configure_debug_joint_order(list(model_manifest["joint_order"]))' in source


def test_replay_source_and_fresh_replay_are_separate_worker_modes() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert 'args.command in {"replay-source", "replay"}' in source
    assert 'source_mode=args.command == "replay-source"' in source
    assert "Fresh Isaac replay equivalence failed" in source


def test_p60_probe_explicitly_replicates_frozen_reset_row_zero() -> None:
    worker = WORKER.read_text(encoding="utf-8")
    probe_env = WORKER.with_name("isaac_probe_env.py").read_text(encoding="utf-8")
    assert (
        "reset_artifact is not None and len(serial_profiles) != 1"
        in worker
    )
    assert '"suite_frozen_row0_replicated"' in probe_env
    assert '"selected_environment_rows": selected_rows' in probe_env


def test_robot_probe_repeats_with_fresh_reset_on_stable_environment_rows() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert "serial_profiles = _serial_robot_profiles" in source
    assert "num_envs=len(serial_profiles)" in source
    assert "env.set_randomization_rng_state(reset_rng_state)" in source
    assert "for repetition in range(repetitions):" in source
    assert "physics_time_origin_s = float(env._sim_step_counter * env.physics_dt)" in source
    assert "relative_physics_time_s" in source
    assert "expected_physics_time_s" in source
    assert "Isaac serial robot probe physics clock drifted" in source
    assert "arrays = _merge_serial_repetition_arrays(repetition_arrays)" in source


def test_serial_clock_normalization_merges_three_absolute_origins() -> None:
    physics_dt_s = 0.005
    decimation = 4
    origins = (3.125, 17.75, 101.0)
    repetitions = []
    for origin in origins:
        schedule = [0.0]
        for tick in range(2):
            for substep_index in range(decimation):
                expected = (tick * decimation + substep_index + 1) * physics_dt_s
                schedule.append(
                    _normalized_serial_physics_time_s(
                        absolute_time_s=origin + expected,
                        origin_s=origin,
                        tick=tick,
                        substep_index=substep_index,
                        decimation=decimation,
                        physics_dt_s=physics_dt_s,
                    )
                )
        repetitions.append(
            {"time_s": np.asarray(schedule, dtype=np.float64)[:, None]}
        )

    merged = _merge_serial_repetition_arrays(repetitions)["time_s"]
    expected = np.asarray(
        [0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040],
        dtype=np.float64,
    )
    assert merged.shape == (9, 3)
    for repetition in range(3):
        np.testing.assert_array_equal(merged[:, repetition], expected)
    np.testing.assert_allclose(np.diff(merged[:, 0]), physics_dt_s, rtol=0.0, atol=1.0e-15)
    assert merged[0, 0] == 0.0
    assert merged[4, 0] == 0.02


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -float("inf")))
@pytest.mark.parametrize("field", ("absolute_time_s", "origin_s", "physics_dt_s"))
def test_serial_clock_normalization_rejects_nonfinite_authoritative_time(
    field: str, bad: float
) -> None:
    arguments = {
        "absolute_time_s": 3.13,
        "origin_s": 3.125,
        "tick": 0,
        "substep_index": 0,
        "decimation": 4,
        "physics_dt_s": 0.005,
    }
    arguments[field] = bad
    with pytest.raises(EvidenceIntegrityError, match="finite"):
        _normalized_serial_physics_time_s(**arguments)


def test_serial_clock_normalization_rejects_finite_drift() -> None:
    with pytest.raises(RuntimeError, match="physics clock drifted"):
        _normalized_serial_physics_time_s(
            absolute_time_s=3.131,
            origin_s=3.125,
            tick=0,
            substep_index=0,
            decimation=4,
            physics_dt_s=0.005,
        )


def test_impact_probe_rejects_pre_teleport_contact_sensor_sample() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert 'if mode == "impact" and time_s == 0.0:' in source
    assert "normal_force = np.zeros_like(normal_force)" in source


def test_robot_probe_field_builder_covers_every_array() -> None:
    names = (
        "time_s",
        "profile_channel",
        "profile_sign",
        "profile_repetition",
        "controlled_position_canonical",
        "controlled_velocity_canonical",
        "all_hinge_position",
        "all_hinge_velocity",
        "base_com_position_diag",
        "base_linear_velocity_control",
        "base_angular_velocity_control",
        "system_com_position_control",
        "linear_momentum_control",
        "angular_momentum_com_control",
        "kinetic_energy_j",
        "closure_residual_m",
        "commanded_input_canonical",
        "canonical_torque_equivalent",
        "host_applied_torque_canonical",
        "effort_limit_event",
        "joint_velocity_limit_exceeded",
        "terminated",
        "truncated",
    )
    fields = _robot_probe_fields({name: object() for name in names})
    assert set(fields) == set(names)
    assert fields["base_angular_velocity_control"].frame == "control_body"


def _reset_state(*, profiles: int = 1, contact_enabled: bool = False) -> dict:
    prefix = (profiles,)
    active = (
        np.tile(np.asarray([[0, 1]], dtype=np.int8), (profiles, 1))
        if contact_enabled
        else np.full((*prefix, 2), -1, dtype=np.int8)
    )
    normal = (
        np.zeros((*prefix, 2), dtype=np.float32)
        if contact_enabled
        else np.full((*prefix, 2), np.nan, dtype=np.float32)
    )
    normal_world = (
        np.zeros((*prefix, 2, 3), dtype=np.float32)
        if contact_enabled
        else np.full((*prefix, 2, 3), np.nan, dtype=np.float32)
    )
    return {
        "active_joint_position_canonical": np.zeros((*prefix, 6), dtype=np.float32),
        "base_com_position_engine_world": np.zeros((*prefix, 3), dtype=np.float32),
        "base_orientation_control_wxyz": np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (profiles, 1),
        ),
        "wheel_contact_active": active,
        "wheel_normal_force_n": normal.copy(),
        "wheel_normal_impulse_ns": normal.copy(),
        "wheel_normal_force_world": normal_world,
        "wheel_friction_force_world": np.full(
            (*prefix, 2, 3), np.nan, dtype=np.float32
        ),
        "unexpected_contact": np.full(prefix, -1, dtype=np.int8),
    }


def test_reset_hash_accepts_only_exact_disabled_contact_sentinels() -> None:
    payload = _phase_hash_payload(_reset_state())
    active = payload["wheel_contact_active"]
    assert active == {
        "schema_version": RESET_UNAVAILABLE_SCHEMA_VERSION,
        "availability": "unavailable",
        "reason": "debug_contact_observer_disabled",
        "sentinel": "all_negative_one",
        "shape": [2],
        "dtype": "|i1",
    }
    for name in (
        "wheel_normal_force_n",
        "wheel_normal_impulse_ns",
        "wheel_normal_force_world",
    ):
        assert payload[name]["reason"] == "debug_contact_observer_disabled"
        assert payload[name]["sentinel"] == "all_nan"
        assert payload[name]["dtype"] == "<f4"
    assert payload["wheel_friction_force_world"]["reason"] == (
        "isaaclab_friction_force_unavailable"
    )
    assert payload["unexpected_contact"]["sentinel"] == "all_negative_one"


def test_reset_hash_preserves_available_contact_values() -> None:
    payload = _phase_hash_payload(_reset_state(contact_enabled=True))
    assert payload["wheel_contact_active"] == [0, 1]
    assert payload["wheel_normal_force_n"] == [0.0, 0.0]
    assert payload["wheel_normal_force_world"] == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert payload["wheel_friction_force_world"]["availability"] == "unavailable"


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -float("inf")))
@pytest.mark.parametrize(
    "field", ("active_joint_position_canonical", "base_orientation_control_wxyz")
)
def test_reset_hash_rejects_nonfinite_authoritative_physical_fields(
    bad: float, field: str
) -> None:
    state = _reset_state()
    state[field][0, 0] = bad
    with pytest.raises(EvidenceIntegrityError, match="Authoritative reset field"):
        _phase_hash_payload(state)


def test_reset_hash_rejects_mixed_or_malformed_contact_sentinels() -> None:
    mixed_active = _reset_state()
    mixed_active["wheel_contact_active"][0, 1] = 0
    with pytest.raises(EvidenceIntegrityError, match="mixed disabled-observer"):
        _phase_hash_payload(mixed_active)

    mixed = _reset_state()
    mixed["wheel_normal_force_n"][0, 0] = 0.0
    with pytest.raises(EvidenceIntegrityError, match="not exact all-NaN"):
        _phase_hash_payload(mixed)

    wrong_dtype = _reset_state()
    wrong_dtype["wheel_contact_active"] = wrong_dtype[
        "wheel_contact_active"
    ].astype(np.int64)
    with pytest.raises(EvidenceIntegrityError, match="exact int8 dtype"):
        _phase_hash_payload(wrong_dtype)

    wrong_shape = _reset_state()
    wrong_shape["wheel_normal_force_world"] = np.full(
        (1, 2), np.nan, dtype=np.float32
    )
    with pytest.raises(EvidenceIntegrityError, match="shape drifted"):
        _phase_hash_payload(wrong_shape)


def test_repeatability_profile_uses_only_frozen_phase_identity_fields() -> None:
    profiles = 3
    state = {
        "base_orientation_control_wxyz": np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (profiles, 1),
        ),
        "base_linear_velocity_control": np.zeros((profiles, 3), dtype=np.float32),
        "base_angular_velocity_control": np.zeros((profiles, 3), dtype=np.float32),
        "all_hinge_position_named": np.zeros((profiles, 26), dtype=np.float32),
        "all_hinge_velocity_named": np.zeros((profiles, 26), dtype=np.float32),
        "loop_closure_error": np.zeros((profiles, 2), dtype=np.float32),
        "wheel_contact_active": np.full((profiles, 2), -1, dtype=np.int8),
        "virtual_leg_length": np.ones((profiles, 2), dtype=np.float32),
    }
    root_position = np.zeros((profiles, 3), dtype=np.float32)
    command = np.asarray([0.2, -0.1, 0.19], dtype=np.float32)
    pre = _repeatability_profile_state(
        state,
        1,
        phase="pre_forward",
        root_com_position_control=root_position,
        command=command,
    )
    post = _repeatability_profile_state(
        state,
        1,
        phase="post_forward",
        root_com_position_control=root_position,
    )
    assert set(pre) == {
        "root_com_position_control",
        "root_orientation_control_wxyz",
        "root_linear_velocity_control",
        "root_angular_velocity_control",
        "all_hinge_position_named",
        "all_hinge_velocity_named",
        "command",
    }
    assert set(post) == {
        "root_com_position_control",
        "root_orientation_control_wxyz",
        "root_linear_velocity_control",
        "root_angular_velocity_control",
        "all_hinge_position_named",
        "all_hinge_velocity_named",
        "closure_residual_m",
    }
    assert pre["command"]["values"] == pytest.approx(command.tolist())
    for field in pre.values():
        assert field["dtype"] == "<f4"
        assert field["shape"] == list(np.asarray(field["values"]).shape)
    assert "wheel_contact_active" not in pre
    assert "virtual_leg_length" not in post


def _robot_repeatability_state() -> tuple[dict, np.ndarray, np.ndarray]:
    state = {
        "base_orientation_control_wxyz": np.asarray(
            [[1.0, 0.0, 0.0, 0.0]], dtype=np.float32
        ),
        "base_linear_velocity_control": np.zeros((1, 3), dtype=np.float32),
        "base_angular_velocity_control": np.zeros((1, 3), dtype=np.float32),
        "all_hinge_position_named": np.zeros((1, 26), dtype=np.float32),
        "all_hinge_velocity_named": np.zeros((1, 26), dtype=np.float32),
        "loop_closure_error": np.zeros((1, 2), dtype=np.float32),
    }
    root_position = np.zeros((1, 3), dtype=np.float32)
    command = np.zeros(3, dtype=np.float32)
    return state, root_position, command


@pytest.mark.parametrize(
    ("field", "bad_shape", "phase"),
    (
        ("root_com_position_control", (1, 2), "pre_forward"),
        ("base_orientation_control_wxyz", (1, 5), "pre_forward"),
        ("base_linear_velocity_control", (1, 2), "pre_forward"),
        ("base_angular_velocity_control", (1, 4), "pre_forward"),
        ("all_hinge_position_named", (1, 25), "pre_forward"),
        ("all_hinge_velocity_named", (1, 27), "pre_forward"),
        ("loop_closure_error", (1, 3), "post_forward"),
        ("command", (2,), "pre_forward"),
        ("base_orientation_control_wxyz", (1, 1, 4), "pre_forward"),
    ),
)
def test_repeatability_profile_rejects_every_authoritative_shape_drift(
    field: str, bad_shape: tuple[int, ...], phase: str
) -> None:
    state, root_position, command = _robot_repeatability_state()
    if field == "root_com_position_control":
        root_position = np.zeros(bad_shape, dtype=np.float32)
    elif field == "command":
        command = np.zeros(bad_shape, dtype=np.float32)
    else:
        state[field] = np.zeros(bad_shape, dtype=np.float32)
    with pytest.raises(EvidenceIntegrityError, match="exact shape"):
        _repeatability_profile_state(
            state,
            0,
            phase=phase,
            root_com_position_control=root_position,
            command=command if phase == "pre_forward" else None,
        )


@pytest.mark.parametrize(
    "dtype",
    (np.float64, np.int32, np.bool_, np.dtype(">f4")),
)
@pytest.mark.parametrize("field", ("root_com_position_control", "command"))
def test_repeatability_profile_rejects_noncanonical_dtype_and_endianness(
    dtype: np.dtype | type, field: str
) -> None:
    state, root_position, command = _robot_repeatability_state()
    if field == "root_com_position_control":
        root_position = np.zeros((1, 3), dtype=dtype)
    else:
        command = np.zeros(3, dtype=dtype)
    with pytest.raises(EvidenceIntegrityError, match="exact little-endian float32 dtype"):
        _repeatability_profile_state(
            state,
            0,
            phase="pre_forward",
            root_com_position_control=root_position,
            command=command,
        )


def test_repeatability_profile_hash_encodes_dtype_shape_and_values() -> None:
    state, root_position, command = _robot_repeatability_state()
    payload = _repeatability_profile_state(
        state,
        0,
        phase="pre_forward",
        root_com_position_control=root_position,
        command=command,
    )
    assert payload["root_com_position_control"] == {
        "dtype": "<f4",
        "shape": [3],
        "values": [0.0, 0.0, 0.0],
    }


def test_reset_identity_changes_only_pre_forward_phase_hash() -> None:
    physical_state = {"closure_residual_m": {"dtype": "<f4", "shape": [2], "values": [0.0, 0.0]}}
    changed_physical_state = {
        "closure_residual_m": {"dtype": "<f4", "shape": [2], "values": [1.0e-6, 0.0]}
    }
    reset_a = {"seed": 0, "rng_state_before_reset_hash": "A", "reset_artifact_sha256": None}
    reset_b = {"seed": 0, "rng_state_before_reset_hash": "B", "reset_artifact_sha256": None}

    pre_a = _reset_phase_identity_payload(
        phase="pre_forward", state_name="state", state=physical_state, reset_identity=reset_a
    )
    pre_b = _reset_phase_identity_payload(
        phase="pre_forward", state_name="state", state=physical_state, reset_identity=reset_b
    )
    post_a = _reset_phase_identity_payload(
        phase="post_forward", state_name="state", state=physical_state
    )
    post_b = _reset_phase_identity_payload(
        phase="post_forward", state_name="state", state=physical_state
    )
    post_physical_change = _reset_phase_identity_payload(
        phase="post_forward",
        state_name="state",
        state=changed_physical_state,
    )

    assert stable_hash(pre_a) != stable_hash(pre_b)
    assert stable_hash(post_a) == stable_hash(post_b)
    assert stable_hash(post_a) != stable_hash(post_physical_change)
    assert not (set(reset_a) & set(post_a))
    with pytest.raises(EvidenceIntegrityError, match="must not include"):
        _reset_phase_identity_payload(
            phase="post_forward",
            state_name="state",
            state=physical_state,
            reset_identity=reset_a,
        )


def test_returned_policy_identity_enforces_and_encodes_frozen_layout() -> None:
    current = np.zeros(25, dtype=np.float32)
    previous_action = np.zeros(6, dtype=np.float32)
    payload = _repeatability_returned_policy_state(current, previous_action)
    assert payload["actor_obs_policy_returned"]["shape"] == [25]
    assert payload["previous_action"]["shape"] == [6]
    assert payload["initialized_history"]["shape"] == [5, 25]
    assert all(field["dtype"] == "<f4" for field in payload.values())

    with pytest.raises(EvidenceIntegrityError, match="actor_obs_policy_returned.*exact shape"):
        _repeatability_returned_policy_state(
            np.zeros(24, dtype=np.float32), previous_action
        )
    with pytest.raises(EvidenceIntegrityError, match="previous_action.*exact little-endian"):
        _repeatability_returned_policy_state(
            current, np.zeros(6, dtype=np.float64)
        )


def test_robot_probe_capture_preserves_dtype_until_identity_validation() -> None:
    state, root_position, command = _robot_repeatability_state()
    raw_returned = np.zeros((1, 25), dtype=np.float64)
    raw_previous_action = np.zeros((1, 6), dtype=">f4")
    raw_root = np.zeros((1, 3), dtype=np.float64)
    raw_command = np.zeros((1, 3), dtype=np.float64)

    returned = _capture_authoritative_identity_array(
        "returned_observation", raw_returned
    )
    previous_action = _capture_authoritative_identity_array(
        "previous_action", raw_previous_action
    )
    captured_root = _capture_authoritative_identity_array(
        "pre_root_position_control", raw_root
    )
    captured_command = _capture_authoritative_identity_array("commands", raw_command)

    assert returned.dtype.str == "<f8"
    assert previous_action.dtype.str == ">f4"
    assert captured_root.dtype.str == "<f8"
    assert captured_command.dtype.str == "<f8"
    with pytest.raises(EvidenceIntegrityError, match="actor_obs_policy_returned.*<f4"):
        _repeatability_returned_policy_state(returned[0], np.zeros(6, dtype=np.float32))
    with pytest.raises(EvidenceIntegrityError, match="previous_action.*<f4"):
        _repeatability_returned_policy_state(
            np.zeros(25, dtype=np.float32), previous_action[0]
        )
    with pytest.raises(EvidenceIntegrityError, match="root_com_position_control.*<f4"):
        _repeatability_profile_state(
            state,
            0,
            phase="pre_forward",
            root_com_position_control=captured_root,
            command=command,
        )
    with pytest.raises(EvidenceIntegrityError, match="command.*<f4"):
        _repeatability_profile_state(
            state,
            0,
            phase="pre_forward",
            root_com_position_control=root_position,
            command=captured_command[0],
        )


def test_repeatability_profile_rejects_nonfinite_authoritative_state() -> None:
    state = {
        "base_orientation_control_wxyz": np.asarray(
            [[1.0, 0.0, 0.0, 0.0]], dtype=np.float32
        ),
        "base_linear_velocity_control": np.zeros((1, 3), dtype=np.float32),
        "base_angular_velocity_control": np.zeros((1, 3), dtype=np.float32),
        "all_hinge_position_named": np.zeros((1, 26), dtype=np.float32),
        "all_hinge_velocity_named": np.zeros((1, 26), dtype=np.float32),
        "loop_closure_error": np.zeros((1, 2), dtype=np.float32),
    }
    state["all_hinge_position_named"][0, 0] = np.nan
    with pytest.raises(EvidenceIntegrityError, match="all_hinge_position_named"):
        _repeatability_profile_state(
            state,
            0,
            phase="post_forward",
            root_com_position_control=np.zeros((1, 3), dtype=np.float32),
        )
