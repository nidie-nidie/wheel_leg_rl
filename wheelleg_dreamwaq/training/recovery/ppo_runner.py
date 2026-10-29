from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Callable
from time import perf_counter

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from .validation import all_finite


class RecoveryPPORunner:
    def __init__(self, env, resolved_config: dict):
        self.env = env
        self.resolved_config = deepcopy(resolved_config)
        self.device = str(resolved_config["environment"]["device"])
        self.config = deepcopy(resolved_config["ppo"])
        if not env.cfg.is_finite_horizon or env.clip_actions is not None:
            raise ValueError("起身 PPO 要求有限回合且不裁剪原始动作")
        self.num_steps_per_env = int(self.config["num_steps_per_env"])
        if self.num_steps_per_env != 24:
            raise ValueError("起身 rollout 必须为 24 步")
        self.completed_updates = 0
        self.guided = resolved_config["schema_version"] == "RecoveryResolvedConfigV3"
        observation = env.get_observations().to(self.device)
        actor_dim, critic_dim = (111, 35) if self.guided else (110, 34)
        if observation["policy"].shape != (env.num_envs, actor_dim):
            raise ValueError("起身 Actor 输入与训练契约不一致")
        if observation["critic"].shape != (env.num_envs, critic_dim):
            raise ValueError("起身 Critic 输入与训练契约不一致")
        policy_config = deepcopy(self.config["policy"])
        if policy_config.pop("class_name", "ActorCritic") != "ActorCritic":
            raise ValueError("起身训练只接受普通 ActorCritic")
        if policy_config["noise_std_type"] != "log":
            raise ValueError("起身高斯策略必须使用 log 标准差")
        for key in ("actor_obs_normalization", "critic_obs_normalization"):
            if policy_config[key]:
                raise ValueError("起身观测已经按共享物理契约归一化")
        self.policy = ActorCritic(
            observation, {"policy": ["policy"], "critic": ["critic"]}, 6, **policy_config
        ).to(self.device)
        algorithm_config = deepcopy(self.config["algorithm"])
        if algorithm_config.pop("class_name", "PPO") != "PPO":
            raise ValueError("起身训练只接受普通 PPO")
        if algorithm_config["gamma"] != resolved_config["contract"]["gamma"]:
            raise ValueError("PPO 折扣必须与势函数奖励相同")
        batch_size = env.num_envs * self.num_steps_per_env
        if batch_size % int(algorithm_config["num_mini_batches"]):
            raise ValueError("rollout 大小必须能被 mini batch 数整除")
        if self.guided:
            from .guidance_ppo import GuidancePPO

            self.algorithm = GuidancePPO(self.policy, resolved_config["guidance"], device=self.device, **algorithm_config)
        else:
            self.algorithm = PPO(self.policy, device=self.device, **algorithm_config)
        self.algorithm.init_storage("rl", env.num_envs, self.num_steps_per_env, observation, [6])
        self._observation = observation
        self.update_records: list[dict] = []
        self._episode_reward = torch.zeros(env.num_envs, device=self.device)
        self._episode_steps = torch.zeros(env.num_envs, device=self.device, dtype=torch.int64)
        self._tensorboard = None
        if self.guided:
            from torch.utils.tensorboard import SummaryWriter

            self._tensorboard = SummaryWriter(str(Path(resolved_config["environment"]["run_dir"]) / "tensorboard"))

    def close(self):
        if self._tensorboard is not None:
            self._tensorboard.close()
            self._tensorboard = None

    @property
    def alg(self):
        return self.algorithm

    def learn(self, num_learning_iterations: int, *, checkpoint_writer: Callable | None = None, profiler=None) -> list[dict]:
        if num_learning_iterations <= 0:
            raise ValueError("PPO 更新次数必须为正整数")
        self.policy.train()
        run_dir = Path(self.resolved_config["environment"]["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        records = []
        direct_env = self.env.unwrapped
        performance = direct_env._performance
        for _ in range(num_learning_iterations):
            update_started = perf_counter()
            performance.begin_update()
            reward_sum = torch.zeros((), device=self.device, dtype=torch.float64)
            terminal_count = torch.zeros((), device=self.device, dtype=torch.int64)
            completed_reward = torch.zeros((), device=self.device)
            completed_length = torch.zeros((), device=self.device)
            guidance_cost = torch.zeros((), device=self.device)
            force_peak = torch.zeros((), device=self.device)
            reward_terms = {}
            initial_cost_estimate = initial_alpha = None
            if self.guided:
                initial_alpha = float(direct_env._guidance.alpha[0])
                initial_cost_estimate = self.algorithm.estimate_initial_cost(
                    direct_env._guidance_initial_critic, initial_alpha)
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    with performance.measure("policy_inference"):
                        actions = self.algorithm.act(self._observation)
                    if not torch.isfinite(actions).all():
                        raise FloatingPointError("PPO 采样动作包含非有限数值")
                    with performance.measure("environment_step"):
                        observation, rewards, dones, extras = self.env.step(actions)
                    if "time_outs" in extras:
                        raise RuntimeError("起身真实终止不得添加 timeout bootstrap")
                    observation = observation.to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    direct_env.check_workers()
                    if not all_finite(rewards, *observation.values()):
                        raise FloatingPointError("非有限转移不得进入 PPO storage")
                    self.algorithm.process_env_step(observation, rewards, dones, extras)
                    self._observation = observation
                    reward_sum += rewards.sum()
                    terminal_count += dones.sum()
                    self._episode_reward.add_(rewards)
                    self._episode_steps.add_(1)
                    terminal_mask = dones.bool()
                    completed_reward.add_(torch.where(terminal_mask, self._episode_reward, 0).sum())
                    completed_length.add_(torch.where(terminal_mask, self._episode_steps, 0).sum())
                    self._episode_reward[terminal_mask] = 0
                    self._episode_steps[terminal_mask] = 0
                    if self.guided:
                        guidance_cost.add_(extras["guidance"]["cost_force_n"].sum())
                        force_peak.copy_(torch.maximum(force_peak, extras["guidance"]["peak_force_n"].max()))
                        for key, value in direct_env._reward_terms.items():
                            if key not in reward_terms:
                                reward_terms[key] = torch.zeros((), device=self.device)
                            reward_terms[key].add_(value.sum())
                # 24 步采集边界保留正常价值估计；400 步任务终态已经由 dones 消除。
                with performance.measure("returns"):
                    self.algorithm.compute_returns(self._observation)
            with performance.measure("ppo_update"):
                losses = self.algorithm.update()
            if any(not math.isfinite(float(value)) for value in losses.values()):
                raise FloatingPointError("PPO 更新产生非有限损失")
            if not all_finite(*self.policy.parameters()):
                raise FloatingPointError("PPO 更新产生非有限网络参数")
            self.completed_updates += 1
            with performance.measure("log_rollout_submit"):
                direct_env.rollout_barrier()
            save_due = self.completed_updates % int(self.config["save_interval"]) == 0
            final_update = len(records) + 1 == num_learning_iterations
            if checkpoint_writer is not None and (save_due or final_update):
                with performance.measure("checkpoint"):
                    checkpoint_writer(self)
            phase = performance.snapshot()
            elapsed = perf_counter() - update_started
            record = {
                "completed_updates": self.completed_updates,
                "mean_step_reward": float(reward_sum) / (self.env.num_envs * self.num_steps_per_env),
                "terminal_count": int(terminal_count),
                "learning_rate": float(self.algorithm.learning_rate),
                "losses": {key: float(value) for key, value in losses.items()},
                "elapsed_seconds": elapsed,
                "environment_steps_per_second": self.env.num_envs * self.num_steps_per_env / elapsed,
                "performance": phase,
                "logging": dict(direct_env._metrics_stream.statistics),
                "timing_scope": "includes update, rollout submission, due checkpoint and end-of-update GPU completion; final drain reported separately",
            }
            if self.guided:
                count = int(terminal_count)
                record["mean_episode_reward"] = None if count == 0 else float(completed_reward) / count
                record["mean_episode_length"] = None if count == 0 else float(completed_length) / count
                record["guidance"] = {**direct_env._guidance.statistics(),
                    "mean_force_cost_n": float(guidance_cost) / (self.env.num_envs * self.num_steps_per_env),
                    "peak_force_n": float(force_peak),
                    "old_initial_cost_estimate_n": float(initial_cost_estimate),
                    "old_initial_cost_estimate_alpha": initial_alpha,
                    "old_initial_cost_estimate_source": "pre_update_cost_critic_on_last_reset_states_at_rollout_start_alpha",
                    "cost_surrogate_estimate_n": float(initial_cost_estimate) + losses["guidance_cost_surrogate_delta_n"],
                    "cost_surrogate_scope": "initial_cost_estimate_plus_intermediate_minibatch_mean_delta"}
                record["reward_terms"] = {key: float(value) / (self.env.num_envs * self.num_steps_per_env)
                                          for key, value in reward_terms.items()}
                if direct_env._guidance.global_control_steps != self.completed_updates * self.num_steps_per_env:
                    raise RuntimeError("正式采集进度与辅助控制步不同")
                self._write_tensorboard(record)
            records.append(record)
            self.update_records.append(record)
            with (run_dir / "updates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            print("RECOVERY_PPO_UPDATE " + json.dumps(record, ensure_ascii=False), flush=True)
            if profiler is not None:
                profiler.step()
        return records

    def state_dict(self) -> dict:
        state = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.algorithm.optimizer.state_dict(),
            "completed_updates": self.completed_updates,
            "learning_rate": float(self.algorithm.learning_rate),
            "environment_state": self.env.unwrapped.environment_state_dict(),
        }
        if self.guided:
            guidance = self.env.unwrapped._guidance
            state["guidance"] = {"schema_version": "RecoveryGuidanceLearnerStateV1",
                **self.algorithm.guidance_state_dict(), "global_control_steps": guidance.global_control_steps,
                "alpha": float(guidance.alpha[0]), "guidance_config": deepcopy(self.resolved_config["guidance"]),
                "training_contract": deepcopy(self.resolved_config["training_contract"])}
        return state

    def load_state_dict(self, payload: dict) -> None:
        if int(payload["completed_updates"]) <= 0:
            raise ValueError("resume checkpoint 必须包含已经完成的 PPO 更新")
        self.policy.load_state_dict(payload["model_state_dict"], strict=True)
        self.algorithm.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.completed_updates = int(payload["completed_updates"])
        if self.guided:
            guidance = payload["guidance"]
            if (guidance["guidance_config"] != self.resolved_config["guidance"]
                    or guidance["training_contract"] != self.resolved_config["training_contract"]
                    or guidance["global_control_steps"] != self.completed_updates * self.num_steps_per_env):
                raise ValueError("辅助恢复配置、契约或更新进度不一致")
            self.algorithm.load_guidance_state_dict(guidance)
            self.env.unwrapped._guidance.restore_progress(guidance["global_control_steps"])
        elif "guidance" in payload:
            raise ValueError("普通 PPO 不能加载辅助训练模型")
        self.algorithm.learning_rate = float(payload["learning_rate"])
        for group in self.algorithm.optimizer.param_groups:
            group["lr"] = self.algorithm.learning_rate

    def refresh_observation_after_resume(self) -> None:
        self._observation = self.env.get_observations().to(self.device)

    def _write_tensorboard(self, record):
        step = self.completed_updates
        for key in ("mean_step_reward", "mean_episode_reward", "mean_episode_length", "elapsed_seconds"):
            if record[key] is not None:
                self._tensorboard.add_scalar(f"Train/{key}", record[key], step)
        for section in ("losses", "guidance", "reward_terms"):
            for key, value in record[section].items():
                if isinstance(value, (int, float)):
                    self._tensorboard.add_scalar(f"{section}/{key}", value, step)
        self._tensorboard.flush()


def as_tensordict(observations: dict[str, torch.Tensor], num_envs: int) -> TensorDict:
    return TensorDict(observations, batch_size=[num_envs])
