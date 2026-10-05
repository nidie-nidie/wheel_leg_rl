#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB = Path(os.environ.get("ISAACLAB_PATH", "/home/cb/IsaacLab"))
TRAIN_SCRIPT = ISAACLAB / "scripts/reinforcement_learning/rsl_rl/train.py"
RSL_RL_SCRIPT_DIR = TRAIN_SCRIPT.parent
RSL_RL_PACKAGE_CANDIDATES = (
    os.environ.get("GOGO_RSL_RL_PATH"),
    "/home/changba01/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages",
    "/home/cb/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages",
)


def _prepend_existing_path(path: str | Path | None) -> None:
    if not path:
        return
    candidate = Path(path)
    if not (candidate / "rsl_rl" / "__init__.py").is_file():
        return
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RSL_RL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))
for candidate in RSL_RL_PACKAGE_CANDIDATES:
    _prepend_existing_path(candidate)


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) not in {"0", "false", "False", "no", "No"}


def _env_float_pair(name: str, default: tuple[float, float]) -> tuple[float, float]:
    value = os.environ.get(name)
    if not value:
        return default
    parts = [part.strip() for part in value.replace(":", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must contain two comma-separated floats, got {value!r}")
    return float(parts[0]), float(parts[1])


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def _uses_external_a1_latency_model(task: str | None) -> bool:
    task_name = str(task or "")
    return task_name.startswith("Gogo-A1-Flat") and not task_name.startswith(
        "Gogo-A1-PACE-"
    )


def _gogo_configure_a1_sensor_noise_and_friction(env_cfg, task: str | None) -> None:
    """Apply per-run A1 observation noise and light contact-friction randomization."""
    if not str(task or "").startswith("Gogo-A1-Flat"):
        return
    if not _env_flag("GOGO_A1_SENSOR_NOISE_AND_FRICTION", "0"):
        return

    from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

    gravity_noise = max(0.0, _env_float("GOGO_A1_PROJECTED_GRAVITY_NOISE", 0.03))
    gyro_noise = max(0.0, _env_float("GOGO_A1_GYRO_NOISE", 0.10))
    joint_pos_noise = max(0.0, _env_float("GOGO_A1_JOINT_POS_NOISE", 0.006))
    joint_vel_noise = max(0.0, _env_float("GOGO_A1_JOINT_VEL_NOISE", 0.35))

    def _set_noise(group, term_name: str, half_range: float) -> None:
        term = getattr(group, term_name, None)
        if term is not None:
            term.noise = Unoise(n_min=-half_range, n_max=half_range)

    for group_name in ("policy", "critic"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is None:
            continue
        group.enable_corruption = True
        _set_noise(group, "projected_gravity", gravity_noise)
        _set_noise(group, "base_ang_vel", gyro_noise)
        _set_noise(group, "joint_pos", joint_pos_noise)
        _set_noise(group, "joint_vel", joint_vel_noise)

    events = getattr(env_cfg, "events", None)
    physics_material = getattr(events, "physics_material", None) if events is not None else None
    static_friction = _env_float_pair("GOGO_A1_STATIC_FRICTION_RANGE", (0.75, 0.90))
    dynamic_friction = _env_float_pair("GOGO_A1_DYNAMIC_FRICTION_RANGE", (0.55, 0.70))
    restitution = _env_float_pair("GOGO_A1_RESTITUTION_RANGE", (0.0, 0.0))
    if physics_material is not None:
        physics_material.params["static_friction_range"] = static_friction
        physics_material.params["dynamic_friction_range"] = dynamic_friction
        physics_material.params["restitution_range"] = restitution
        physics_material.params["num_buckets"] = int(os.environ.get("GOGO_A1_FRICTION_BUCKETS", "64"))

    print(
        "[INFO] A1 sensor noise/friction randomization: "
        f"gravity=+/-{gravity_noise:.3f}, gyro=+/-{gyro_noise:.3f} rad/s, "
        f"joint_pos=+/-{joint_pos_noise:.4f} rad, joint_vel=+/-{joint_vel_noise:.3f} rad/s, "
        f"static_friction={static_friction[0]:.2f}..{static_friction[1]:.2f}, "
        f"dynamic_friction={dynamic_friction[0]:.2f}..{dynamic_friction[1]:.2f}",
        flush=True,
    )


def _gogo_install_a1_motor_latency_model(env) -> None:
    """Patch A1 joint-motor tasks with small physics-step command and sensor latency."""
    if not _env_flag("GOGO_A1_LATENCY_MODEL", "1"):
        print("[INFO] A1 motor latency model disabled by GOGO_A1_LATENCY_MODEL=0", flush=True)
        return

    import types

    import torch

    unwrapped = env.unwrapped
    action_manager = unwrapped.action_manager
    obs_manager = unwrapped.observation_manager
    try:
        term = action_manager.get_term("joint_pos")
    except Exception:
        active_terms = list(getattr(action_manager, "active_terms", []))
        if len(active_terms) != 1:
            raise RuntimeError(f"expected A1 joint_pos action term, got action terms {active_terms}")
        term = action_manager.get_term(active_terms[0])

    if getattr(term, "_gogo_a1_latency_model_installed", False):
        return

    physics_dt = float(getattr(unwrapped, "physics_dt", getattr(unwrapped.cfg.sim, "dt", 0.005)))
    sim_dt_ms = physics_dt * 1000.0
    decimation = int(getattr(unwrapped.cfg, "decimation", 1))
    max_command_delay_ms = max(0.0, float(os.environ.get("GOGO_A1_MAX_COMMAND_DELAY_MS", "5.0")))
    max_sensor_delay_ms = max(0.0, float(os.environ.get("GOGO_A1_MAX_SENSOR_DELAY_MS", "5.0")))
    max_command_delay_steps = min(max(0, int(round(max_command_delay_ms / sim_dt_ms))), max(0, decimation - 1))
    max_sensor_delay_steps = max(0, int(round(max_sensor_delay_ms / sim_dt_ms)))

    original_process_actions = term.process_actions
    original_apply_actions = term.apply_actions
    original_reset = term.reset
    original_compute_group = obs_manager.compute_group
    original_scene_update = unwrapped.scene.update

    def _sample_lags(num_envs: int, max_delay: int, device) -> torch.Tensor:
        if max_delay <= 0:
            return torch.zeros(num_envs, dtype=torch.long, device=device)
        return torch.randint(0, max_delay + 1, (num_envs,), dtype=torch.long, device=device)

    def _ensure_command_state(self) -> None:
        target = self.processed_actions
        if getattr(self, "_gogo_a1_current_target", None) is not None and self._gogo_a1_current_target.shape == target.shape:
            return
        current_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._gogo_a1_previous_target = current_pos.clone()
        self._gogo_a1_current_target = current_pos.clone()
        self._gogo_a1_command_lag = _sample_lags(target.shape[0], max_command_delay_steps, target.device)
        self._gogo_a1_apply_substep = 0

    def _process_actions_with_latency(self, actions):
        _ensure_command_state(self)
        self._gogo_a1_previous_target[:] = self._gogo_a1_current_target
        original_process_actions(actions)
        self._gogo_a1_current_target[:] = self.processed_actions
        self._gogo_a1_apply_substep = 0
        self._gogo_a1_command_lag[:] = _sample_lags(
            self._gogo_a1_command_lag.shape[0], max_command_delay_steps, self._gogo_a1_command_lag.device
        )

    def _apply_actions_with_latency(self):
        if max_command_delay_steps <= 0:
            original_apply_actions()
            return
        _ensure_command_state(self)
        lag_mask = self._gogo_a1_command_lag > int(self._gogo_a1_apply_substep)
        target = torch.where(lag_mask.unsqueeze(1), self._gogo_a1_previous_target, self._gogo_a1_current_target)
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)
        self._gogo_a1_apply_substep += 1

    def _reset_action_latency(self, env_ids=None):
        original_reset(env_ids)
        if getattr(self, "_gogo_a1_current_target", None) is None:
            return
        ids = slice(None) if env_ids is None else env_ids
        current_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._gogo_a1_previous_target[ids] = current_pos[ids]
        self._gogo_a1_current_target[ids] = current_pos[ids]
        if max_command_delay_steps > 0:
            num_ids = current_pos.shape[0] if isinstance(ids, slice) else len(env_ids)
            self._gogo_a1_command_lag[ids] = _sample_lags(num_ids, max_command_delay_steps, current_pos.device)

    delayed_obs_terms = {"projected_gravity", "base_ang_vel", "joint_pos", "joint_vel"}

    def _ensure_obs_state() -> None:
        if hasattr(obs_manager, "_gogo_a1_obs_delay_buffers"):
            return
        obs_manager._gogo_a1_obs_delay_buffers = {}
        obs_manager._gogo_a1_obs_lags = {}

    def _source_obs(term_cfg):
        source = term_cfg.func(obs_manager._env, **term_cfg.params).clone()
        if term_cfg.modifiers is not None:
            for modifier in term_cfg.modifiers:
                source = modifier.func(source, **modifier.params)
        return source

    def _ensure_obs_term(group_name: str, term_name: str, source: torch.Tensor) -> None:
        _ensure_obs_state()
        obs_manager._gogo_a1_obs_delay_buffers.setdefault(group_name, {})
        obs_manager._gogo_a1_obs_lags.setdefault(group_name, {})
        term_buffers = obs_manager._gogo_a1_obs_delay_buffers[group_name]
        if term_name in term_buffers and term_buffers[term_name].shape[0] == source.shape[0]:
            return
        term_buffers[term_name] = source.unsqueeze(1).repeat(1, max_sensor_delay_steps + 1, 1).clone()
        obs_manager._gogo_a1_obs_lags[group_name][term_name] = _sample_lags(
            source.shape[0], max_sensor_delay_steps, source.device
        )

    def _capture_obs_latency_state() -> None:
        if max_sensor_delay_steps <= 0:
            return
        for group_name, term_names in obs_manager._group_obs_term_names.items():
            for term_name, term_cfg in zip(term_names, obs_manager._group_obs_term_cfgs[group_name]):
                if term_name not in delayed_obs_terms:
                    continue
                source = _source_obs(term_cfg)
                _ensure_obs_term(group_name, term_name, source)
                buffer = obs_manager._gogo_a1_obs_delay_buffers[group_name][term_name]
                buffer[:] = torch.roll(buffer, shifts=1, dims=1)
                buffer[:, 0, :] = source

    def _reset_obs_latency_state(group_name: str | None = None, env_ids=None) -> None:
        if max_sensor_delay_steps <= 0 or getattr(obs_manager, "_gogo_a1_obs_delay_buffers", None) is None:
            return
        groups = [group_name] if group_name else list(obs_manager._gogo_a1_obs_delay_buffers.keys())
        ids = slice(None) if env_ids is None else env_ids
        for group in groups:
            term_cfgs = dict(zip(obs_manager._group_obs_term_names[group], obs_manager._group_obs_term_cfgs[group]))
            for term_name, buffer in obs_manager._gogo_a1_obs_delay_buffers.get(group, {}).items():
                source = _source_obs(term_cfgs[term_name])
                buffer[ids] = source[ids].unsqueeze(1).repeat(1, max_sensor_delay_steps + 1, 1)
                if max_sensor_delay_steps > 0:
                    num_ids = source.shape[0] if isinstance(ids, slice) else len(env_ids)
                    obs_manager._gogo_a1_obs_lags[group][term_name][ids] = _sample_lags(
                        num_ids, max_sensor_delay_steps, source.device
                    )

    def _scene_update_with_obs_latency(scene_self, dt: float) -> None:
        original_scene_update(dt)
        _capture_obs_latency_state()

    def _compute_group_with_obs_latency(self, group_name: str, update_history: bool = False):
        if max_sensor_delay_steps <= 0 or group_name not in self._group_obs_term_names:
            return original_compute_group(group_name, update_history=update_history)

        group_term_names = self._group_obs_term_names[group_name]
        group_obs = dict.fromkeys(group_term_names, None)
        obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])

        for term_name, term_cfg in obs_terms:
            obs = _source_obs(term_cfg)

            if term_name in delayed_obs_terms:
                _ensure_obs_term(group_name, term_name, obs)
                env_ids = torch.arange(obs.shape[0], device=obs.device)
                obs = self._gogo_a1_obs_delay_buffers[group_name][term_name][
                    env_ids, self._gogo_a1_obs_lags[group_name][term_name], :
                ]

            if isinstance(term_cfg.noise, torch.nn.Module):
                obs = term_cfg.noise(obs)
            else:
                from isaaclab.utils import noise

                if isinstance(term_cfg.noise, noise.NoiseCfg):
                    obs = term_cfg.noise.func(obs, term_cfg.noise)
                elif isinstance(term_cfg.noise, noise.NoiseModelCfg) and term_cfg.noise.func is not None:
                    obs = term_cfg.noise.func(obs)
            if term_cfg.clip:
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
            if term_cfg.scale is not None:
                obs = obs.mul_(term_cfg.scale)
            if term_cfg.history_length > 0:
                circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
                if update_history:
                    circular_buffer.append(obs)
                elif circular_buffer._buffer is None:
                    from isaaclab.utils.buffers import CircularBuffer

                    circular_buffer = CircularBuffer(
                        max_len=circular_buffer.max_length,
                        batch_size=circular_buffer.batch_size,
                        device=circular_buffer.device,
                    )
                    circular_buffer.append(obs)

                if term_cfg.flatten_history_dim:
                    group_obs[term_name] = circular_buffer.buffer.reshape(self._env.num_envs, -1)
                else:
                    group_obs[term_name] = circular_buffer.buffer
            else:
                group_obs[term_name] = obs

        if self._group_obs_concatenate[group_name]:
            return torch.cat(list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name])
        return group_obs

    original_env_reset_idx = unwrapped._reset_idx

    def _reset_idx_with_latency(env_self, env_ids):
        result = original_env_reset_idx(env_ids)
        _reset_obs_latency_state(env_ids=env_ids)
        return result

    term.process_actions = types.MethodType(_process_actions_with_latency, term)
    term.apply_actions = types.MethodType(_apply_actions_with_latency, term)
    term.reset = types.MethodType(_reset_action_latency, term)
    term._gogo_a1_latency_model_installed = True
    obs_manager.compute_group = types.MethodType(_compute_group_with_obs_latency, obs_manager)
    unwrapped.scene.update = types.MethodType(_scene_update_with_obs_latency, unwrapped.scene)
    unwrapped._reset_idx = types.MethodType(_reset_idx_with_latency, unwrapped)
    print(
        "[INFO] Installed A1 motor latency model: "
        f"command_delay=0..{max_command_delay_steps} sim steps (~0..{max_command_delay_steps * sim_dt_ms:.1f} ms), "
        f"sensor_delay=0..{max_sensor_delay_steps} sim steps (~0..{max_sensor_delay_steps * sim_dt_ms:.1f} ms)",
        flush=True,
    )


def _gogo_install_dog_servo_dc_action_model(env) -> None:
    """Patch dog joint-position targets to mimic light DC/servo non-idealities during training."""
    if not _env_flag("GOGO_DOG_DC_ACTION_MODEL", "1"):
        print("[INFO] Dog servo DC target model disabled by GOGO_DOG_DC_ACTION_MODEL=0", flush=True)
        return

    import types

    import torch

    unwrapped = env.unwrapped
    action_manager = unwrapped.action_manager
    try:
        term = action_manager.get_term("joint_pos")
    except Exception:
        active_terms = list(getattr(action_manager, "active_terms", []))
        if len(active_terms) != 1:
            raise RuntimeError(f"expected dog servo joint_pos action term, got action terms {active_terms}")
        term = action_manager.get_term(active_terms[0])

    if getattr(term, "_gogo_dc_action_model_installed", False):
        return

    max_delay = max(0, int(os.environ.get("GOGO_DOG_DC_MAX_DELAY", "1")))
    deadband = max(0.0, float(os.environ.get("GOGO_DOG_DC_DEADBAND", "0.004")))
    max_target_step = max(1.0e-6, float(os.environ.get("GOGO_DOG_DC_MAX_TARGET_STEP", "0.055")))
    no_load_speed = max(1.0e-6, float(os.environ.get("GOGO_DOG_DC_NO_LOAD_SPEED", "7.0")))
    min_speed_scale = min(1.0, max(0.05, float(os.environ.get("GOGO_DOG_DC_MIN_SPEED_SCALE", "0.45"))))
    voltage_range = _env_float_pair("GOGO_DOG_DC_VOLTAGE_SCALE_RANGE", (0.92, 1.00))
    voltage_low, voltage_high = min(voltage_range), max(voltage_range)

    original_reset = term.reset

    def _ensure_state(self):
        target = self.processed_actions
        if getattr(self, "_gogo_dc_target", None) is not None and self._gogo_dc_target.shape == target.shape:
            return
        current_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._gogo_dc_target = current_pos.clone()
        self._gogo_dc_delay_buffer = target.unsqueeze(1).repeat(1, max_delay + 1, 1).clone()
        if max_delay > 0:
            self._gogo_dc_lag = torch.randint(
                0,
                max_delay + 1,
                (target.shape[0],),
                dtype=torch.long,
                device=target.device,
            )
        else:
            self._gogo_dc_lag = torch.zeros(target.shape[0], dtype=torch.long, device=target.device)
        self._gogo_dc_voltage = voltage_low + (voltage_high - voltage_low) * torch.rand(
            target.shape[0],
            device=target.device,
        )

    def _reset_state(self, env_ids=None):
        original_reset(env_ids)
        if getattr(self, "_gogo_dc_target", None) is None:
            return
        ids = slice(None) if env_ids is None else env_ids
        current_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._gogo_dc_target[ids] = current_pos[ids]
        self._gogo_dc_delay_buffer[ids] = current_pos[ids].unsqueeze(1).repeat(1, max_delay + 1, 1)
        if max_delay > 0:
            num_ids = current_pos.shape[0] if isinstance(ids, slice) else len(env_ids)
            self._gogo_dc_lag[ids] = torch.randint(
                0,
                max_delay + 1,
                (num_ids,),
                dtype=torch.long,
                device=current_pos.device,
            )
        self._gogo_dc_voltage[ids] = voltage_low + (voltage_high - voltage_low) * torch.rand(
            current_pos[ids].shape[0],
            device=current_pos.device,
        )

    def _apply_actions_with_dc_target_model(self):
        _ensure_state(self)
        target = self.processed_actions
        if max_delay > 0:
            self._gogo_dc_delay_buffer = torch.roll(self._gogo_dc_delay_buffer, shifts=1, dims=1)
            self._gogo_dc_delay_buffer[:, 0, :] = target
            env_ids = torch.arange(target.shape[0], device=target.device)
            delayed_target = self._gogo_dc_delay_buffer[env_ids, self._gogo_dc_lag, :]
        else:
            delayed_target = target

        current_vel = self._asset.data.joint_vel[:, self._joint_ids]
        speed_scale = torch.clamp(1.0 - torch.abs(current_vel) / no_load_speed, min=min_speed_scale, max=1.0)
        allowed_step = max_target_step * speed_scale * self._gogo_dc_voltage.unsqueeze(1)
        target_delta = delayed_target - self._gogo_dc_target
        if deadband > 0.0:
            target_delta = torch.where(torch.abs(target_delta) < deadband, torch.zeros_like(target_delta), target_delta)
        self._gogo_dc_target = self._gogo_dc_target + torch.clamp(target_delta, -allowed_step, allowed_step)
        self._asset.set_joint_position_target(self._gogo_dc_target, joint_ids=self._joint_ids)

    term.reset = types.MethodType(_reset_state, term)
    term.apply_actions = types.MethodType(_apply_actions_with_dc_target_model, term)
    term._gogo_dc_action_model_installed = True
    print(
        "[INFO] Installed dog servo PD target DC model: "
        f"delay=0..{max_delay} steps, deadband={deadband:.4f} rad, "
        f"max_step={max_target_step:.4f} rad/apply, no_load_speed={no_load_speed:.2f} rad/s, "
        f"min_speed_scale={min_speed_scale:.2f}, voltage_scale={voltage_low:.2f}..{voltage_high:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    source = source.replace(
        "installed_version = metadata.version(\"rsl-rl-lib\")\n",
        "try:\n"
        "    installed_version = metadata.version(\"rsl-rl-lib\")\n"
        "except metadata.PackageNotFoundError:\n"
        "    installed_version = RSL_RL_VERSION\n",
        1,
    )
    source = source.replace(
        "# PLACEHOLDER: Extension template (do not remove this comment)",
        "import isaaclab_ext.a1_locomotion  # noqa: F401\n"
        "# PLACEHOLDER: Extension template (do not remove this comment)",
        1,
    )
    source = source.replace(
        "    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)\n",
        "    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)\n"
        "    experiment_name_override = os.environ.get(\"GOGO_EXPERIMENT_NAME\")\n"
        "    if experiment_name_override:\n"
        "        agent_cfg.experiment_name = experiment_name_override\n"
        "    run_name_override = os.environ.get(\"GOGO_RUN_NAME\")\n"
        "    if run_name_override:\n"
        "        agent_cfg.run_name = run_name_override\n",
        1,
    )
    source = source.replace(
        "    agent_cfg.max_iterations = (\n"
        "        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations\n"
        "    )\n",
        "    agent_cfg.max_iterations = (\n"
        "        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations\n"
        "    )\n"
        "    lr_override = os.environ.get(\"GOGO_LR\")\n"
        "    if lr_override:\n"
        "        agent_cfg.algorithm.learning_rate = float(lr_override)\n"
        "    entropy_override = os.environ.get(\"GOGO_ENTROPY_COEF\")\n"
        "    if entropy_override:\n"
        "        agent_cfg.algorithm.entropy_coef = float(entropy_override)\n"
        "    desired_kl_override = os.environ.get(\"GOGO_DESIRED_KL\")\n"
        "    if desired_kl_override:\n"
        "        agent_cfg.algorithm.desired_kl = float(desired_kl_override)\n"
        "    cenet_estimation_override = os.environ.get(\"GOGO_CENET_ESTIMATION_COEF\")\n"
        "    if cenet_estimation_override and hasattr(agent_cfg.algorithm, \"cenet_estimation_coef\"):\n"
        "        agent_cfg.algorithm.cenet_estimation_coef = float(cenet_estimation_override)\n"
        "    cenet_reconstruction_override = os.environ.get(\"GOGO_CENET_RECONSTRUCTION_COEF\")\n"
        "    if cenet_reconstruction_override and hasattr(agent_cfg.algorithm, \"cenet_reconstruction_coef\"):\n"
        "        agent_cfg.algorithm.cenet_reconstruction_coef = float(cenet_reconstruction_override)\n"
        "    cenet_kl_override = os.environ.get(\"GOGO_CENET_KL_BETA\")\n"
        "    if cenet_kl_override and hasattr(agent_cfg.algorithm, \"cenet_kl_beta\"):\n"
        "        agent_cfg.algorithm.cenet_kl_beta = float(cenet_kl_override)\n"
        "    cenet_reconstruction_target_override = os.environ.get(\"GOGO_CENET_RECONSTRUCTION_TARGET\")\n"
        "    if cenet_reconstruction_target_override and hasattr(agent_cfg.algorithm, \"cenet_reconstruction_target\"):\n"
        "        agent_cfg.algorithm.cenet_reconstruction_target = cenet_reconstruction_target_override\n"
        "    adaboot_override = os.environ.get(\"GOGO_ADABOOT\")\n"
        "    if adaboot_override is not None and hasattr(agent_cfg.algorithm, \"adaboot\"):\n"
        "        agent_cfg.algorithm.adaboot = adaboot_override not in {\"0\", \"false\", \"False\", \"no\", \"No\"}\n"
        "    save_interval_override = os.environ.get(\"GOGO_SAVE_INTERVAL\")\n"
        "    if save_interval_override:\n"
        "        agent_cfg.save_interval = int(save_interval_override)\n",
        1,
    )
    source = source.replace(
        "    # set the environment seed\n",
        "    _gogo_configure_a1_sensor_noise_and_friction(env_cfg, args_cli.task)\n"
        "    # set the environment seed\n",
        1,
    )
    source = source.replace(
        "    if agent_cfg.resume or agent_cfg.algorithm.class_name == \"Distillation\":\n"
        "        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)\n",
        "    if agent_cfg.resume or agent_cfg.algorithm.class_name == \"Distillation\":\n"
        "        abs_checkpoint = os.environ.get(\"GOGO_RESUME_PATH\")\n"
        "        if not abs_checkpoint and agent_cfg.load_checkpoint and os.path.isabs(str(agent_cfg.load_checkpoint)):\n"
        "            abs_checkpoint = str(agent_cfg.load_checkpoint)\n"
        "        if abs_checkpoint:\n"
        "            resume_path = abs_checkpoint\n"
        "        else:\n"
        "            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)\n",
        1,
    )
    source = source.replace(
        "    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=\"rgb_array\" if args_cli.video else None)\n",
        "    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=\"rgb_array\" if args_cli.video else None)\n"
        "    if _uses_external_a1_latency_model(args_cli.task):\n"
        "        robot = env.unwrapped.scene[\"robot\"]\n"
        "        actuator = robot.actuators.get(\"base_legs\")\n"
        "        actuator_name = actuator.__class__.__name__ if actuator is not None else \"<missing>\"\n"
        "        if actuator_name != \"DCMotor\":\n"
        "            raise RuntimeError(f\"A1 flat latency task must use DCMotor base_legs, got {actuator_name}\")\n"
        "        _gogo_install_a1_motor_latency_model(env)\n"
        "    if str(args_cli.task or \"\").startswith(\"Gogo-Dog-Servo\"):\n"
        "        robot_cfg = env_cfg.scene.robot\n"
        "        asset_path = str(getattr(robot_cfg.spawn, \"asset_path\", \"\"))\n"
        "        if not asset_path.endswith(\"assets/dog/dog.xml\"):\n"
        "            raise RuntimeError(f\"dog servo task must use assets/dog/dog.xml, got {asset_path}\")\n"
        "        spawn_cfg = robot_cfg.spawn\n"
        "        if getattr(spawn_cfg, \"self_collision\", None) is not False:\n"
        "            raise RuntimeError(\"dog servo task must import with self_collision=False\")\n"
        "        articulation_props = getattr(spawn_cfg, \"articulation_props\", None)\n"
        "        if getattr(articulation_props, \"enabled_self_collisions\", None) is not False:\n"
        "            raise RuntimeError(\"dog servo task must set enabled_self_collisions=False\")\n"
        "        if tuple(robot_cfg.init_state.pos) != (0.0, 0.0, 0.138):\n"
        "            raise RuntimeError(f\"dog servo task must start at pos z=0.138, got {robot_cfg.init_state.pos}\")\n"
        "        robot = env.unwrapped.scene[\"robot\"]\n"
        "        actuator = robot.actuators.get(\"servos\")\n"
        "        actuator_name = actuator.__class__.__name__ if actuator is not None else \"<missing>\"\n"
        "        if actuator_name != \"ImplicitActuator\":\n"
        "            raise RuntimeError(f\"dog servo task must use ImplicitActuator, got {actuator_name}\")\n"
        "        _gogo_install_dog_servo_dc_action_model(env)\n"
        "        print(\"[INFO] Verified dog servo runtime model: assets/dog/dog.xml + ImplicitActuator PD\", flush=True)\n",
        1,
    )
    source = source.replace(
        "        runner.load(resume_path)\n",
        "        runner.load(resume_path, load_optimizer=os.environ.get(\"GOGO_LOAD_OPTIMIZER\", \"1\") != \"0\")\n"
        "        action_std_set = os.environ.get(\"GOGO_ACTION_STD_SET\")\n"
        "        if action_std_set:\n"
        "            std_value = float(action_std_set)\n"
        "            policy = runner.alg.policy\n"
        "            if hasattr(policy, \"std\"):\n"
        "                policy.std.data.fill_(std_value)\n"
        "            elif hasattr(policy, \"log_std\"):\n"
        "                policy.log_std.data.fill_(__import__(\"math\").log(std_value))\n"
        "        action_std_max = os.environ.get(\"GOGO_ACTION_STD_MAX\")\n"
        "        if action_std_max:\n"
        "            max_std = float(action_std_max)\n"
        "            policy = runner.alg.policy\n"
        "            if hasattr(policy, \"std\"):\n"
        "                policy.std.data.clamp_(max=max_std)\n"
        "            elif hasattr(policy, \"log_std\"):\n"
        "                policy.log_std.data.clamp_(max=__import__(\"math\").log(max_std))\n",
        1,
    )
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(TRAIN_SCRIPT),
        "__package__": None,
        "_gogo_configure_a1_sensor_noise_and_friction": _gogo_configure_a1_sensor_noise_and_friction,
        "_uses_external_a1_latency_model": _uses_external_a1_latency_model,
        "_gogo_install_a1_motor_latency_model": _gogo_install_a1_motor_latency_model,
        "_gogo_install_dog_servo_dc_action_model": _gogo_install_dog_servo_dc_action_model,
    }
    exec(compile(source, str(TRAIN_SCRIPT), "exec"), globals_dict)
