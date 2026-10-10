"""Real PhysX reset regression; no policy training or optimizer is created."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import torch
import tensordict  # Import before Kit on Windows.
from isaaclab.app import AppLauncher

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--kind", choices=("formal", "randomized", "debug"), default="formal")
parser.add_argument("--reset-cache", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

from wheelleg_dreamwaq.algorithms.dreamwaq.history_wrapper import DreamWaQHistoryVecEnvWrapper
from wheelleg_dreamwaq.schemas.frames import (
    quat_rotate_inverse_wxyz, quat_rotate_wxyz, transform_usd_vector_to_control,
)
from wheelleg_dreamwaq.schemas.randomization import FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import CommandRanges
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def cpu(value):
    return value.detach().cpu().clone()


def clocks(env):
    return [float(env.sim.current_time), int(env._sim_step_counter),
            int(env.common_step_counter), float(env.robot.data._sim_timestamp)]


def rng(env):
    return copy.deepcopy(env.get_randomization_rng_state())


def same_rng(left, right):
    return all(torch.equal(left["streams"][key]["state"], right["streams"][key]["state"])
               for key in left["streams"])


def direct(env):
    view = env.robot.root_physx_view
    pose = view.get_root_transforms().clone()
    vel = view.get_root_velocities().clone()
    quat = pose[:, 3:7][:, [3, 0, 1, 2]]
    offset = view.get_coms().to(env.device).clone()[:, 0, :3]
    com = pose[:, :3] + quat_rotate_wxyz(quat, offset)
    height = (com[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)
    angular = transform_usd_vector_to_control(quat_rotate_inverse_wxyz(quat, vel[:, 3:6]))
    return {"pose_xyzw": pose, "velocity_world": vel, "local_com": offset,
            "com": com, "height": height, "angular": angular}


def main():
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = WheelLegFlatEnvCfg()
    cfg.seed = 20261009
    cfg.scene.num_envs = 2
    cfg.sim.device = args.device
    cfg.commands = CommandRanges(vx=(0., 0.), yaw_rate=(0., 0.), base_height=(.20, .20))
    if args.kind == "randomized":
        cfg.randomization = FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
    cache = None if args.reset_cache is None else torch.load(args.reset_cache, map_location="cpu", weights_only=False)
    if args.kind == "debug":
        from debug.sim2sim.isaac_debug_env import WheelLegSim2SimDebugEnv
        env = WheelLegSim2SimDebugEnv(cfg, closed_chain_reset_cache=cache)
        env.configure_debug_joint_order(list(env.robot.joint_names))
    else:
        env = WheelLegFlatEnv(cfg, closed_chain_reset_cache=cache)
    torch.save(env.closed_chain_reset_cache_artifact, args.output / "reset-cache.pt")
    checks, evidence = [], []
    counts = {"observations": 0, "noise": 0}
    noises = []
    original_obs = env._get_observations
    original_noise = env._randomization.sample_actor_noise

    def observation_call():
        counts["observations"] += 1
        return original_obs()

    def noise_call():
        counts["noise"] += 1
        result = original_noise()
        noises.append(copy.deepcopy(result))
        return result

    env._get_observations = observation_call
    env._randomization.sample_actor_noise = noise_call

    def check(name, value, **details):
        checks.append({"name": name, "passed": bool(value), **details})

    def close(name, value, expected, tolerance=1e-6):
        error = float((value - expected).abs().max().item())
        check(name, error <= tolerance, max_abs_error=error, tolerance=tolerance)

    def capture(name, observations, rows=None):
        physical = direct(env)
        selected = slice(None) if rows is None else rows
        angular_obs = cfg.normalization.normalize_angular_velocity(physical["angular"])
        close(name + "/critic_angular", observations["critic"][selected, :3], angular_obs[selected])
        close(name + "/raw_com", env._state.root_com_pos_w[selected], physical["com"][selected])
        close(name + "/raw_height", env._state.base_height[selected], physical["height"][selected])
        close(name + "/critic_height", observations["critic"][selected, 28:29],
              cfg.normalization.normalize_base_height(physical["height"])[selected], 1e-5)
        noise = noises[-1]
        noisy_angular = physical["angular"] if noise is None else physical["angular"] + noise.angular_velocity
        close(name + "/actor_angular", observations["policy"][selected, :3],
              cfg.normalization.normalize_angular_velocity(noisy_angular)[selected])
        check(name + "/zero_action", torch.equal(observations["policy"][selected, 19:25],
                                                  torch.zeros_like(observations["policy"][selected, 19:25])))
        if "policy_history" in observations.keys():
            frames = observations["policy_history"].view(2, 5, 25)
            check(name + "/history", torch.equal(frames[selected],
                  observations["policy"][selected].unsqueeze(1).repeat(1, 5, 1)))
        evidence.append({"name": name, "direct": {key: cpu(value) for key, value in physical.items()},
                         "state": {key: cpu(value) for key, value in vars(env._state).items()},
                         "observations": {key: cpu(value) for key, value in observations.items()},
                         "counts": counts.copy(), "clock": clocks(env), "rng": rng(env),
                         "noise": copy.deepcopy(noise)})
        if args.kind == "debug" and rows is None:
            snapshot = copy.deepcopy(env._debug_reset_snapshot)
            evidence[-1]["debug_reset_phases"] = snapshot
            for phase, observation_key in (
                ("reset_written_pre_forward", "reset_written_cached_actor_obs_policy"),
                ("reset_forwarded_post_forward", "reset_returned_actor_obs_policy"),
            ):
                angular = snapshot[phase]["base_angular_velocity_control"]
                close(name + "/" + phase + "/angular", snapshot[observation_key][:, :3],
                      cfg.normalization.normalize_angular_velocity(angular))
                if phase == "reset_forwarded_post_forward":
                    close(name + "/" + phase + "/com", env._state.root_com_pos_w,
                          snapshot[phase]["base_com_position_engine_world"])
            check(name + "/debug_returned_first", torch.equal(
                snapshot["reset_returned_actor_obs_policy"], observations["policy"]))

    try:
        initial_clock = clocks(env)
        wrapper = DreamWaQHistoryVecEnvWrapper(env)
        expected_count = 3 if args.kind == "debug" else 2
        check("constructor/count", counts == {"observations": expected_count, "noise": expected_count})
        check("constructor/clock", clocks(env) == initial_clock)
        capture("constructor", wrapper.get_observations())
        check("cache_origin", env._closed_chain_cache_origin == ("generated" if cache is None else "loaded"))
        for index in range(4):
            velocity = env.robot.root_physx_view.get_root_velocities().clone()
            velocity[:, 3:] = velocity.new_tensor((.4, -.6, .8))
            env.robot.write_root_velocity_to_sim(velocity)
            _, _, dones, _ = wrapper.step(torch.zeros((2, 6), device=env.device))
            check(f"repeat{index}/preparation_alive", not bool(dones.any()))
            check(f"repeat{index}/nonzero_terminal", float(direct(env)["angular"].abs().max()) > .01)
            before_clock, before_count = clocks(env), counts.copy()
            if args.kind != "debug":
                env.begin_resume_sequence_capture()
            obs, _ = wrapper.reset()
            if args.kind != "debug":
                trace = env.get_resume_sequence_trace_tensors()
                check(f"repeat{index}/resume_first", torch.equal(
                    trace["first_policy_observation"], cpu(obs["policy"])))
            check(f"repeat{index}/clock", clocks(env) == before_clock)
            check(f"repeat{index}/count", all(counts[key]-before_count[key] == expected_count for key in counts))
            capture(f"repeat{index}", obs)
            before_rng, before_count = rng(env), counts.copy()
            one, two = wrapper.get_observations(), wrapper.get_observations()
            check(f"repeat{index}/cached_rng", same_rng(before_rng, rng(env)))
            check(f"repeat{index}/cached_count", counts == before_count)
            check(f"repeat{index}/cached_tensors", all(torch.equal(one[key], two[key]) for key in one.keys()))
        # Restore the actual seven-stream state; repeated reset should reproduce
        # commands, new root velocity, noisy policy and history bit for bit.
        saved_rng = rng(env)
        first, _ = wrapper.reset()
        env.set_randomization_rng_state(saved_rng)
        second, _ = wrapper.reset()
        check("reset_rng_replay", all(torch.equal(first[key], second[key]) for key in first.keys()))
        # Empty reset must not change clocks, caches, state or RNG.
        before_clock, before_rng = clocks(env), rng(env)
        cache_times = (env.robot.data._root_link_vel_w.timestamp, env.robot.data._root_com_pose_w.timestamp)
        empty_before = {key: value.clone() for key, value in vars(env._state).items()}
        env._reset_idx(torch.empty(0, dtype=torch.long, device=env.device))
        check("empty/unchanged", clocks(env) == before_clock and same_rng(before_rng, rng(env))
              and cache_times == (env.robot.data._root_link_vel_w.timestamp, env.robot.data._root_com_pose_w.timestamp)
              and all(torch.equal(value, getattr(env._state, key)) for key, value in empty_before.items()))
        # Real partial reset, with non-reset snapshot checked separately.
        wrapper.step(torch.zeros((2, 6), device=env.device))
        old_state = {key: value.clone() for key, value in vars(env._state).items()}
        before = clocks(env)
        env._reset_idx(torch.tensor([0], device=env.device))
        obs = env._get_observations()
        check("partial/clock", before == clocks(env))
        check("partial/other_snapshot", all(torch.equal(value[1], getattr(env._state, key)[1])
                                             for key, value in old_state.items()))
        capture("partial", obs, torch.tensor([0], device=env.device))
        wrapper.reset()
        # Genuine timeout from the frozen compute_dones, without changing limits.
        env.episode_length_buf[0] = env.max_episode_length - 2
        before = clocks(env)
        previous = wrapper.get_observations()["policy_history"].view(2, 5, 25).clone()
        obs, _, dones, extras = wrapper.step(torch.zeros((2, 6), device=env.device))
        check("timeout/dones", bool(dones[0]) and not bool(dones[1]))
        check("timeout/truncated", bool(extras["time_outs"][0]))
        check("timeout/physics_steps", clocks(env)[1] - before[1] == cfg.decimation)
        check("timeout/time", abs(clocks(env)[0] - before[0] - env.step_dt) < 1e-8)
        capture("timeout", obs, torch.tensor([0], device=env.device))
        frames = obs["policy_history"].view(2, 5, 25)
        check("timeout/other_history", torch.equal(frames[1, :4], previous[1, 1:]))
        # Genuine tilt termination: prepare pose; compute_dones remains untouched.
        wrapper.reset()
        pose = env.robot.data.root_link_pose_w.clone()
        angle = pose.new_tensor(1.2)
        pose[0, 3:7] = torch.stack((torch.cos(angle/2), torch.sin(angle/2), angle*0, angle*0))
        env.robot.write_root_pose_to_sim(pose)
        # The frozen failure rule only applies after its ten-step grace period.
        env.episode_length_buf[0] = cfg.termination.grace_steps + 1
        before = clocks(env)
        obs, _, dones, _ = wrapper.step(torch.zeros((2, 6), device=env.device))
        check("failure/terminated", bool(env.reset_terminated[0]) and not bool(dones[1]))
        check("failure/physics_steps", clocks(env)[1]-before[1] == cfg.decimation)
        capture("failure", obs, torch.tensor([0], device=env.device))
    except Exception:
        check("probe_exception", False, traceback=traceback.format_exc())
    finally:
        sources = [Path(__file__), PROJECT / "source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py",
                   PROJECT / "source/wheelleg_dreamwaq/wheelleg_dreamwaq/algorithms/dreamwaq/history_wrapper.py",
                   PROJECT / "debug/sim2sim/isaac_debug_env.py"]
        for name in ("articulation.py", "articulation_data.py"):
            sources.append(PROJECT / "dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/assets/articulation" / name)
        sources.extend([
            PROJECT / "dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/envs/direct_rl_env.py",
            PROJECT / "dependencies/IsaacLab-v2.3.2/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py",
        ])
        torch.save(evidence, args.output / "evidence.pt")
        report = {"schema": "ResetObservationCoherenceProbeV1", "kind": args.kind,
                  "actual_class": f"{type(env).__module__}.{type(env).__name__}",
                  "seed": cfg.seed, "num_envs": 2, "profile": cfg.randomization.name,
                  "argv": sys.argv, "config": repr(cfg.to_dict()),
                  "sources": {str(path): digest(path) for path in sources},
                  "evidence_sha256": digest(args.output / "evidence.pt"),
                  "checks": checks, "passed": bool(checks) and all(item["passed"] for item in checks)}
        (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        env.close()
    print(json.dumps({"passed": report["passed"], "failed": [item for item in checks if not item["passed"]]}))
    return 0 if report["passed"] else 1


try:
    exit_code = main()
finally:
    app.close()
raise SystemExit(exit_code)
