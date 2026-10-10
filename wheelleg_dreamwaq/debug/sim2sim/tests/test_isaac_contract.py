from __future__ import annotations

from pathlib import Path


DEBUG_ENV = Path(__file__).resolve().parents[1] / "isaac_debug_env.py"


def test_debug_step_preserves_required_phase_order() -> None:
    source = DEBUG_ENV.read_text(encoding="utf-8")
    step_start = source.index("    def step(")
    ordered_tokens = (
        "self._pre_physics_step(action)",
        "self._sim_step_counter += 1",
        "self._apply_action()",
        "pre_state = self._debug_capture_direct_state()",
        "self.scene.write_data_to_sim()",
        "host_torque_native = self.robot.data.applied_torque",
        "self.sim.step(render=False)",
        "self.scene.update(dt=self.physics_dt)",
        "post_state = self._debug_capture_direct_state()",
        "self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()",
        "self._debug_terminal_state = self._debug_clone_state(post_state)",
        "self._reset_idx(reset_env_ids)",
        "self.obs_buf = self._get_observations()",
    )
    positions = [source.index(token, step_start) for token in ordered_tokens]
    assert positions == sorted(positions)


def test_debug_observer_does_not_call_mutating_capture_state() -> None:
    source = DEBUG_ENV.read_text(encoding="utf-8")
    start = source.index("def _debug_capture_direct_state")
    end = source.index("\n    def ", start + 10)
    assert "self._capture_state(" not in source[start:end]


def test_nested_spawn_replacement_is_explicit() -> None:
    source = DEBUG_ENV.read_text(encoding="utf-8")
    assert "spawn=WHEELLEG_CFG.spawn.replace(" in source
    assert "activate_contact_sensors=enable_contact_sensors" in source


def test_contact_sensor_paths_are_explicit_runtime_regexes() -> None:
    source = DEBUG_ENV.read_text(encoding="utf-8")

    assert 'prim_path="/World/envs/env_.*/Robot/jwheel_left"' in source
    assert 'prim_path="/World/envs/env_.*/Robot/jwheel_right"' in source
    assert 'prim_path="{ENV_REGEX_NS}/Robot/jwheel_left"' not in source
    assert 'prim_path="{ENV_REGEX_NS}/Robot/jwheel_right"' not in source


def test_gpu_contact_observer_uses_unfiltered_wheel_net_force() -> None:
    source = DEBUG_ENV.read_text(encoding="utf-8")

    assert '"filter_prim_paths_expr"' not in source
    assert '"track_friction_forces": False' in source
    assert "data.net_forces_w" in source
    assert "normal_force = torch.linalg.vector_norm(normal_world, dim=-1)" in source
    assert "friction_world = torch.full_like(normal_world, torch.nan)" in source


def test_external_pitch_torque_is_written_before_physics_step() -> None:
    source = DEBUG_ENV.read_text(encoding="utf-8")
    step_start = source.index("    def step(")
    write_torque = source.index("self._debug_write_pitch_torque()", step_start)
    write_scene = source.index("self.scene.write_data_to_sim()", step_start)

    assert write_torque < write_scene
