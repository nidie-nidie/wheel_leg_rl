"""Real PhysX control-step and command/history boundary probe (no policy training)."""
from __future__ import annotations
import argparse
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import torch
import tensordict  # Import before Kit on Windows.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

from wheelleg_dreamwaq.algorithms.dreamwaq.history_wrapper import DreamWaQHistoryVecEnvWrapper
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import command_practice_factor
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import compute_reward
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import apply_task_profile


def main():
    cfg = WheelLegFlatEnvCfg()
    cfg.seed = 20261011
    cfg.scene.num_envs = 2
    cfg.sim.device = args.device
    apply_task_profile(cfg, "stop_reverse_v1")
    env = WheelLegFlatEnv(cfg)
    wrapper = DreamWaQHistoryVecEnvWrapper(env)
    checks = []
    def check(name, condition, **details):
        checks.append({"name": name, "passed": bool(condition), **details})
    def close(name, actual, expected, tolerance=1.e-6):
        error = float((actual - expected).abs().max())
        check(name, error <= tolerance, max_abs_error=error)
    def clocks():
        return (float(env.sim.current_time), int(env._sim_step_counter), int(env.common_step_counter))
    def command_rng():
        return env.get_randomization_rng_state()["streams"]["command_rng"]["state"].clone()
    def seed_phase(tick):
        # Seed the independent diagnostic clock close to a boundary: an untrained
        # zero-action robot need not survive two seconds to test command timing.
        wrapper.reset()
        initial = torch.tensor([[.5, -.6, .2], [-.4, .3, .23]], device=env.device)
        env._command_practice.reset(torch.arange(2, device=env.device), initial)
        env._command_practice.steps.fill_(tick)
        env._commands.copy_(env._command_practice.current_commands())
        env._current_state().command.copy_(env._commands)
        env._command_practice_last_tick = env.common_step_counter
        base = wrapper._validate_base_observation(tensordict.TensorDict(
            env._get_observations(), batch_size=[2], device=env.device))
        wrapper._history = base["policy"].unsqueeze(1).repeat(1, 5, 1)
        wrapper._cached_observations = wrapper._build_snapshot(base)
        return initial
    try:
        obs = wrapper.get_observations()
        check("reset/phase_zero", torch.equal(env._command_practice.steps, torch.zeros(2, device=env.device, dtype=torch.long)))
        close("reset/repeated_history", obs["policy_history"].view(2, 5, 25),
              obs["policy"].unsqueeze(1).repeat(1, 5, 1))
        original_reward = env._get_rewards
        captured = {}
        def reward_call():
            captured["old_command"] = env._current_state().command.clone()
            captured["expected_reward"], _ = compute_reward(env._current_state(), cfg.reward_weights,
                control_dt=env.step_dt, terminated=env.reset_terminated)
            captured["expected_error"] = (captured["old_command"][:, 0] -
                env._current_state().root_com_linear_velocity[:, 0]).abs().mean().clone()
            return original_reward()
        env._get_rewards = reward_call
        for tick in [98, 99, 148, 149, 248, 249, 298, 299, 398, 399, 499, 500]:
            initial = seed_phase(tick)
            old_commands = env._commands.clone()
            previous = wrapper.get_observations()["policy_history"].view(2, 5, 25).clone()
            before_clock, before_rng = clocks(), command_rng()
            obs, reward, dones, extras = wrapper.step(torch.zeros(2, 6, device=env.device))
            check(f"{tick}/alive", not bool(dones.any()))
            check(f"{tick}/independent_ticks", env._command_practice.steps.tolist() == [tick + 1] * 2)
            expected = initial.clone()
            expected[:, :2] *= command_practice_factor(torch.tensor([tick + 1], device=env.device)).view(1, 1)
            close(f"{tick}/old_reward", reward, captured["expected_reward"])
            close(f"{tick}/old_tracking_log", extras["log"]["Tracking/vx_abs_error"], captured["expected_error"])
            close(f"{tick}/old_command_log", extras["log"]["Command/vx"], old_commands[:, 0].mean())
            close(f"{tick}/current_command", env._commands, expected)
            close(f"{tick}/cached_command", env._current_state().command, expected)
            for group in ["policy", "critic"]:
                close(f"{tick}/{group}_command", obs[group][:, 6:9], cfg.normalization.normalize_command(expected))
            frames = obs["policy_history"].view(2, 5, 25)
            close(f"{tick}/new_history", frames[:, -1], obs["policy"])
            close(f"{tick}/past_history", frames[:, :4], previous[:, 1:])
            check(f"{tick}/physics_steps", clocks()[1] - before_clock[1] == cfg.decimation)
            check(f"{tick}/physical_dt", abs(clocks()[0] - before_clock[0] - .02) < 1.e-8)
            check(f"{tick}/command_rng", torch.equal(command_rng(), before_rng))
            before_clock = clocks()
            for _ in range(2):
                wrapper.get_observations()
                env._get_observations()
                env._get_rewards()
            check(f"{tick}/extra_reads_clock", clocks() == before_clock)
            check(f"{tick}/extra_reads_ticks", env._command_practice.steps.tolist() == [tick + 1] * 2)
            check(f"{tick}/extra_reads_rng", torch.equal(command_rng(), before_rng))
            close(f"{tick}/extra_reads_history", wrapper.get_observations()["policy_history"], obs["policy_history"])
        seed_phase(149)
        old_initial = env._command_practice.initial_commands[1].clone()
        before_clock = clocks()
        env._reset_idx(torch.tensor([0], device=env.device))
        check("partial/no_physics", clocks() == before_clock)
        check("partial/selected_clock", env._command_practice.steps.tolist() == [0, 149])
        close("partial/unselected_initial", env._command_practice.initial_commands[1], old_initial)
        # Partial reset clones the state: next boundary must also sync that clone.
        obs, _, dones, _ = wrapper.step(torch.zeros(2, 6, device=env.device))
        check("partial/next_alive", not bool(dones.any()))
        check("partial/next_ticks", env._command_practice.steps.tolist() == [1, 150])
        close("partial/next_cached_command", env._current_state().command, env._commands)
        close("partial/next_critic_command", obs["critic"][:, 6:9],
              cfg.normalization.normalize_command(env._commands))
        seed_phase(99)
        env.episode_length_buf[0] = env.max_episode_length - 2
        obs, _, dones, extras = wrapper.step(torch.zeros(2, 6, device=env.device))
        check("timeout/selected_only", bool(dones[0]) and not bool(dones[1]))
        check("timeout/truncated", bool(extras["time_outs"][0]))
        check("timeout/real_clock", env._command_practice.steps.tolist() == [0, 100])
        close("timeout/repeated_reset_history", obs["policy_history"].view(2, 5, 25)[0],
              obs["policy"][0].unsqueeze(0).repeat(5, 1))
        close("timeout/other_command", env._commands[1, :2], torch.zeros(2, device=env.device))
    except Exception:
        check("probe_exception", False, traceback=traceback.format_exc())
    finally:
        report = {"schema_version": "IsaacCommandPracticeProbeV1", "checks": checks,
                  "passed": bool(checks) and all(item["passed"] for item in checks),
                  "physics_schema": "PhysicsV5", "control_dt_s": env.step_dt}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        env.close()
    print(json.dumps({"passed": report["passed"], "failed": [item for item in checks if not item["passed"]]}))
    return 0 if report["passed"] else 1


try:
    exit_code = main()
finally:
    app.close(skip_cleanup=True)
raise SystemExit(exit_code)
