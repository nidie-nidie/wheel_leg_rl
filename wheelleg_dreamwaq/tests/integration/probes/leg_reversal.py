"""Real PhysX control-tick, repeated-read and partial-reset probe."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import traceback

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import torch
import tensordict
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from wheelleg_dreamwaq.algorithms.dreamwaq.history_wrapper import DreamWaQHistoryVecEnvWrapper
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.control import compute_action_targets
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import compute_reward
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import apply_task_profile


def main():
    cfg = WheelLegFlatEnvCfg()
    cfg.seed = 20261011
    cfg.scene.num_envs = 2
    cfg.sim.device = args.device
    apply_task_profile(cfg, "stop_reverse_reversal12_v1")
    env = WheelLegFlatEnv(cfg)
    wrapper = DreamWaQHistoryVecEnvWrapper(env)
    controller = env._leg_reversal_reward
    original_update = controller.counter.update
    calls = []
    def update(targets):
        calls.append(targets.clone())
        return original_update(targets)
    controller.counter.update = update
    original_reward = env._get_rewards
    captured = {}
    def reward_read():
        expected, _ = compute_reward(env._current_state(), cfg.reward_weights,
            control_dt=env.step_dt, terminated=env.reset_terminated)
        captured["expected"] = expected + controller.cached_reward
        return original_reward()
    env._get_rewards = reward_read
    checks = []
    def check(name, value):
        checks.append({"name": name, "passed": bool(value)})
    def clocks():
        return (float(env.sim.current_time), int(env._sim_step_counter), int(env.common_step_counter))
    try:
        for tick in range(72):
            before = clocks()
            actions = torch.zeros(2, 6, device=env.device)
            actions[:, 0] = .15 if tick % 2 == 0 else -.15
            expected_targets = compute_action_targets(actions, env._q_reference, cfg.control)[1]
            obs, reward, dones, _ = wrapper.step(actions)
            check(f"{tick}/once_per_control", len(calls) == tick + 1)
            check(f"{tick}/unaltered_target", torch.equal(calls[-1], expected_targets))
            check(f"{tick}/reward_matches", torch.allclose(reward, captured["expected"], atol=1e-7))
            check(f"{tick}/physics_steps", clocks()[1] - before[1] == cfg.decimation)
            check(f"{tick}/physical_dt", abs(clocks()[0] - before[0] - .02) < 1e-8)
            stable_clock = clocks()
            stable_history = wrapper.get_observations()["policy_history"].clone()
            stable_count = controller.counter.count.clone()
            step_in_rollout = controller.step_in_rollout
            for _ in range(2):
                wrapper.get_observations()
                env._get_observations()
                env._get_rewards()
            check(f"{tick}/read_only_counter", len(calls) == tick + 1 and
                  torch.equal(controller.counter.count, stable_count) and
                  controller.step_in_rollout == step_in_rollout)
            check(f"{tick}/read_only_clock", clocks() == stable_clock)
            check(f"{tick}/read_only_history", torch.equal(
                wrapper.get_observations()["policy_history"], stable_history))
        check("rollout/global_not_reset_by_episodes", controller.completed_rollouts == 3)
        saved_gain = controller.state_dict()
        snapshots = {name: getattr(controller.counter, name).clone()
                     for name in ("age", "direction", "count", "rate", "extreme", "events")}
        stable_clock = clocks()
        env._reset_idx(torch.tensor([0], device=env.device))
        check("partial/no_physics", clocks() == stable_clock)
        check("partial/zero_selected", controller.counter.age[0] == 0 and controller.counter.count[0].sum() == 0)
        check("partial/anchor", torch.equal(controller.counter.extreme[0], env._leg_position_targets[0]))
        for name in ("age", "direction", "count", "rate", "extreme"):
            check(f"partial/other_{name}", torch.equal(getattr(controller.counter, name)[1], snapshots[name][1]))
        check("partial/other_window", torch.equal(controller.counter.events[:, 1], snapshots["events"][:, 1]))
        check("partial/global_gain_unchanged", controller.state_dict() == saved_gain)
        report = {"passed": all(c["passed"] for c in checks), "checks": checks,
                  "counted_control_steps": len(calls), "rollouts": controller.completed_rollouts}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        assert report["passed"], [c for c in checks if not c["passed"]]
    finally:
        wrapper.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        os._exit(1)
    app.close(skip_cleanup=True)
