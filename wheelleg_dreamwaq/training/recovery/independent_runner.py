from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from .bounded_actor_critic import BoundedActorCritic, DeterministicStandingActor
from .checkpoint import restore_global_rng_state
from .compact_metrics import CompactStandingMetrics
from .contracts import file_sha256, stable_hash
from .independent_checkpoint import load_independent_checkpoint, save_independent_checkpoint, validate_identity
from .independent_ppo import IndependentPPO


class SelectedFailureTrajectories:
    def __init__(self, num_envs, identity, quota_bytes=512 * 1024 * 1024):
        self.selected_count = min(12, int(num_envs))
        self.maximum_steps = 500
        self.quota_bytes = int(quota_bytes)
        self.identity_sha256 = stable_hash(identity)
        self.lengths = torch.zeros(self.selected_count, dtype=torch.long)
        self.episode_numbers = torch.zeros(self.selected_count, dtype=torch.long)
        self.component_names = None
        self.fields = {"policy": torch.zeros(self.selected_count, 500, 110),
                       "critic": torch.zeros(self.selected_count, 500, 41),
                       "latent": torch.zeros(self.selected_count, 500, 6),
                       "action": torch.zeros(self.selected_count, 500, 6),
                       "reward": torch.zeros(self.selected_count, 500),
                       "terminated": torch.zeros(self.selected_count, 500, dtype=torch.bool),
                       "truncated": torch.zeros(self.selected_count, 500, dtype=torch.bool),
                       "global_control_step": torch.zeros(self.selected_count, 500, dtype=torch.long)}
        self._validate_memory_bound()

    def _validate_memory_bound(self):
        self.tensor_bytes = sum(value.numel() * value.element_size() for value in self.fields.values())
        self.tensor_bytes += self.lengths.numel() * self.lengths.element_size() * 2
        # JSON 临时对象按每个数值 64 字节预留，并计算 checkpoint 复制占用。
        self.memory_upper_bound_bytes = 3 * self.tensor_bytes + 64 * sum(value.numel() for value in self.fields.values())
        self.memory_upper_bound_bytes += self.selected_count * 65536
        if self.memory_upper_bound_bytes > self.quota_bytes:
            raise ValueError("选定失败轨迹内存上界超过明确配额")

    def append(self, observation, sampled, reward, terminated, truncated, extras, global_control_step, identity):
        if stable_hash(identity) != self.identity_sha256:
            raise ValueError("选定失败轨迹身份改变")
        names = tuple(sorted(extras["reward_components"]))
        if len(names) > 256 or any(not isinstance(name, str) or len(name) > 128 for name in names):
            raise ValueError("诊断奖励字段数量或名称超出已限定范围")
        if self.component_names is None:
            self.component_names = names
            self.fields["reward_components"] = torch.zeros(self.selected_count, 500, len(names))
            self._validate_memory_bound()
        elif names != self.component_names:
            raise ValueError("同一训练的诊断奖励分量发生变化")
        if torch.any(self.lengths >= self.maximum_steps):
            raise RuntimeError("选定回合超过 500 步仍未收到真实终态")
        count = self.selected_count
        components = torch.stack([extras["reward_components"][name][:count] for name in names], -1)
        values = {"policy": observation["policy"][:count], "critic": observation["critic"][:count],
                  "latent": sampled.latent[:count], "action": sampled.action[:count],
                  "reward": reward[:count], "terminated": terminated[:count].reshape(count),
                  "truncated": truncated[:count].reshape(count), "reward_components": components}
        for name, value in values.items():
            source = value.detach().cpu()
            if source.dtype != self.fields[name].dtype or not torch.isfinite(source).all():
                raise ValueError(f"选定失败轨迹字段类型或有限性无效: {name}")
            self.fields[name][torch.arange(count), self.lengths] = source
        self.fields["global_control_step"][torch.arange(count), self.lengths] = global_control_step
        self.lengths += 1
        episode_map = {int(episode["env_id"]): episode for episode in extras["episodes"]}
        if len(episode_map) != len(extras["episodes"]):
            raise ValueError("终态回合 env_id 重复")
        done = (terminated[:count].reshape(count).bool() | truncated[:count].reshape(count).bool()).cpu()
        records = []
        for env_id in done.nonzero(as_tuple=False).flatten().tolist():
            if env_id not in episode_map:
                raise ValueError("选定环境真实终态缺少回合 metadata")
            episode = deepcopy(episode_map[env_id])
            length = int(self.lengths[env_id])
            if int(episode["length"]) != length:
                raise ValueError("选定回合长度与实际诊断步骤不一致")
            if len(json.dumps(episode, ensure_ascii=False, allow_nan=False).encode("utf-8")) > 65536:
                raise ValueError("终态 metadata 超出内存预留")
            if not bool(episode["success"]):
                terminal = extras["terminal_observations"]
                terminal_ids = terminal["env_ids"].detach().cpu().tolist()
                if len(terminal_ids) != len(set(terminal_ids)) or env_id not in terminal_ids:
                    raise ValueError("失败回合缺少唯一真实 terminal observation")
                index = terminal_ids.index(env_id)
                end = {name: terminal[name][index].detach().cpu() for name in ("policy", "critic")}
                if end["policy"].shape != (110,) or end["critic"].shape != (41,) or any(not torch.isfinite(v).all() for v in end.values()):
                    raise ValueError("失败回合 terminal observation 无效")
                records.append({"schema_version": "IndependentStandingSelectedFailureTrajectoryV1",
                    "identity": deepcopy(identity), "identity_sha256": self.identity_sha256,
                    "selection": {"env_id": env_id, "selected_env_count": count,
                                  "episode_number": int(self.episode_numbers[env_id]), "maximum_steps": 500},
                    "episode": episode, "reward_component_names": list(names),
                    "transitions": {name: value[env_id, :length].tolist() for name, value in self.fields.items()},
                    "terminal_observation": {name: value.tolist() for name, value in end.items()}})
            for value in self.fields.values():
                value[env_id].zero_()
            self.lengths[env_id] = 0
            self.episode_numbers[env_id] += 1
        if any(env_id < count and not done[env_id] for env_id in episode_map):
            raise ValueError("选定环境回合 metadata 与真实 done 不一致")
        return records

    @staticmethod
    def _state_hash(state):
        fields = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                         "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest().upper()}
                  for name, value in state["fields"].items()}
        metadata = {key: value for key, value in state.items() if key not in ("fields", "state_sha256")}
        metadata["lengths"] = state["lengths"].tolist()
        metadata["episode_numbers"] = state["episode_numbers"].tolist()
        return stable_hash({"metadata": metadata, "fields": fields})

    def state_dict(self):
        state = {"schema_version": "IndependentStandingSelectedTrajectoryBufferV1",
                 "identity_sha256": self.identity_sha256, "selected_count": self.selected_count,
                 "maximum_steps": self.maximum_steps, "quota_bytes": self.quota_bytes,
                 "component_names": self.component_names, "lengths": self.lengths.clone(),
                 "episode_numbers": self.episode_numbers.clone(),
                 "fields": {name: value.clone() for name, value in self.fields.items()}}
        state["state_sha256"] = self._state_hash(state)
        return state

    def load_state_dict(self, state):
        if (state["schema_version"] != "IndependentStandingSelectedTrajectoryBufferV1"
                or state["identity_sha256"] != self.identity_sha256 or state["selected_count"] != self.selected_count
                or state["maximum_steps"] != self.maximum_steps or state["quota_bytes"] != self.quota_bytes
                or state["state_sha256"] != self._state_hash(state)):
            raise ValueError("选定失败轨迹 checkpoint 身份或内容哈希无效")
        names = state["component_names"]
        if names is not None:
            if len(names) > 256 or tuple(sorted(names)) != tuple(names) or len(set(names)) != len(names):
                raise ValueError("诊断奖励分量恢复字段无效")
            self.component_names = tuple(names)
            self.fields["reward_components"] = torch.zeros(self.selected_count, 500, len(names))
        if set(state["fields"]) != set(self.fields):
            raise ValueError("诊断轨迹恢复字段不完整")
        for name, target in self.fields.items():
            source = state["fields"][name]
            if source.shape != target.shape or source.dtype != target.dtype or not torch.isfinite(source).all():
                raise ValueError("诊断轨迹恢复张量形状、类型或有限性无效")
            target.copy_(source.cpu())
        for name in ("lengths", "episode_numbers"):
            source, target = state[name], getattr(self, name)
            if source.shape != target.shape or source.dtype != target.dtype or torch.any(source < 0):
                raise ValueError("诊断轨迹恢复计数无效")
            target.copy_(source.cpu())
        if torch.any(self.lengths >= self.maximum_steps):
            raise ValueError("诊断轨迹 checkpoint 包含未处理终态")
        self._validate_memory_bound()


class IndependentRunner:
    def __init__(self, env, config, identity, reference_q_median):
        if config["schema_version"] != "RecoveryIndependentStandV1":
            raise ValueError("独立站立 runner 配置版本不一致")
        validate_identity(identity)
        if identity["config_sha256"] != stable_hash(config) or identity["contract_sha256"] != stable_hash(config["contract"]):
            raise ValueError("独立站立 runner 配置或契约与身份哈希不一致")
        self.env = env
        self.config = deepcopy(config)
        self.identity = deepcopy(identity)
        self.device = str(config["environment"]["device"])
        self.run_dir = Path(config["environment"]["run_dir"])
        self.policy = BoundedActorCritic(reference_q_median).to(self.device)
        self.algorithm = IndependentPPO(self.policy, config["ppo"])
        self.completed_updates = 0
        self.num_steps_per_env = int(config["ppo"]["num_steps_per_env"])
        if self.num_steps_per_env != 24:
            raise ValueError("独立站立 rollout 必须包含 24 个控制步骤")
        self._observation = self._validate_observation(env.get_observations())
        self.metrics = None
        self._resume_logging = None
        self._tensorboard = None
        self.closed = False
        self.update_records = []
        self.performance = {"measured_update_count": 0, "environment_steps": 0,
                            "rollout_seconds": 0.0, "optimization_seconds": 0.0,
                            "measured_seconds": 0.0}
        self._diagnostics = SelectedFailureTrajectories(env.num_envs, identity,
            config["assessment"]["trajectory_quota_bytes"])

    @property
    def alg(self):
        return self.algorithm

    def _validate_observation(self, observation):
        if set(observation) != {"policy", "critic"}:
            raise ValueError("独立站立观测必须包含 policy 与 critic")
        result = {key: value.to(self.device) for key, value in observation.items()}
        for key, dimension in (("policy", 110), ("critic", 41)):
            if result[key].shape != (self.env.num_envs, dimension) or not torch.isfinite(result[key]).all():
                raise ValueError(f"独立站立 {key} 观测形状或有限性无效")
        return result

    def _open_metrics(self):
        if self.closed:
            raise RuntimeError("runner 已经结束")
        if self.metrics is None:
            from torch.utils.tensorboard import SummaryWriter

            self.metrics = CompactStandingMetrics(self.run_dir, self._resume_logging)
            self._tensorboard = SummaryWriter(str(self.run_dir / "tensorboard"))

    def learn(self, num_updates, *, start_update=None):
        if num_updates <= 0 or (start_update is not None and start_update != self.completed_updates):
            raise ValueError("PPO 更新预算或起始更新无效")
        if self.completed_updates + num_updates > int(self.config["ppo"]["max_iterations"]):
            raise ValueError("PPO 更新超出配置预算")
        self._open_metrics()
        self.policy.train()
        records = []
        initial_completed = self.completed_updates
        for update in range(initial_completed + 1, initial_completed + num_updates + 1):
            started = perf_counter()
            transitions = []
            episodes = []
            component_sums = {}
            action_min = torch.full((6,), torch.inf, device=self.device)
            action_max = torch.full((6,), -torch.inf, device=self.device)
            reward_sum = 0.0
            action_saturated = 0
            effort_saturation_sum = 0.0
            effort_saturation_count = 0
            for rollout_step in range(self.num_steps_per_env):
                observation = self._observation
                with torch.no_grad():
                    sampled = self.policy.sample(observation["policy"], observation["critic"])
                    next_observation, reward, terminated, truncated, extras = self.env.step(sampled.action)
                    reward = reward.to(self.device).reshape(self.env.num_envs)
                    done = terminated.to(self.device).bool().reshape(self.env.num_envs) | truncated.to(self.device).bool().reshape(self.env.num_envs)
                    if not torch.isfinite(reward).all() or not {"episodes", "reward_components"}.issubset(extras):
                        raise ValueError("环境奖励非有限或回合/奖励分量缺失")
                    transitions.append({"policy": observation["policy"].clone(), "critic": observation["critic"].clone(),
                                        "latent": sampled.latent.clone(), "action": sampled.action.clone(),
                                        "normal_log_prob": sampled.normal_log_prob.clone(), "mean": sampled.mean.clone(),
                                        "std": sampled.std.clone(), "value": sampled.value.clone(),
                                        "reward": reward.clone(), "done": done.clone()})
                    self._observation = self._validate_observation(next_observation)
                    episodes.extend(deepcopy(extras["episodes"]))
                    for key, value in extras["reward_components"].items():
                        if not isinstance(value, torch.Tensor) or value.shape != reward.shape or not torch.isfinite(value).all():
                            raise ValueError(f"奖励分量字段无效: {key}")
                        component_sums[key] = component_sums.get(key, 0.0) + float(value.sum())
                    diagnostic_records = self._diagnostics.append(observation, sampled, reward,
                        terminated.to(self.device).bool(), truncated.to(self.device).bool(), extras,
                        (update - 1) * self.num_steps_per_env + rollout_step + 1, self.identity)
                    for diagnostic in diagnostic_records:
                        self.metrics.append_diagnostic(diagnostic)
                    reward_sum += float(reward.sum())
                    action_saturated += int((sampled.action.abs() >= 0.99).sum())
                    if "effort_saturation_fraction" in extras:
                        saturation = extras["effort_saturation_fraction"]
                        if not isinstance(saturation, torch.Tensor) or saturation.shape != reward.shape or not torch.isfinite(saturation).all():
                            raise ValueError("真实执行器饱和统计形状或有限性无效")
                        effort_saturation_sum += float(saturation.sum())
                        effort_saturation_count += self.env.num_envs
                    action_min = torch.minimum(action_min, sampled.action.amin(0))
                    action_max = torch.maximum(action_max, sampled.action.amax(0))
            rollout_finished = perf_counter()
            storage = {key: torch.stack([transition[key] for transition in transitions]) for key in transitions[0]}
            with torch.no_grad():
                last_value = self.policy.value(self._observation["critic"])
                returns, advantages = self.algorithm.returns(storage.pop("reward"), storage.pop("done"), storage["value"], last_value)
            storage.update(returns=returns, advantages=advantages)
            losses = self.algorithm.update(storage)
            finished = perf_counter()
            self.completed_updates = update
            steps = self.env.num_envs * self.num_steps_per_env
            self.performance["measured_update_count"] += 1
            self.performance["environment_steps"] += steps
            self.performance["rollout_seconds"] += rollout_finished - started
            self.performance["optimization_seconds"] += finished - rollout_finished
            self.performance["measured_seconds"] += finished - started
            groups = {}
            for episode in episodes:
                source = str(episode["source"])
                groups.setdefault(source, []).append(episode)
            group_statistics = {source: self._episode_statistics(items) for source, items in groups.items()}
            record = {"update": update, "global_control_steps": update * 24, "environment_steps": update * steps,
                      "mean_step_reward": reward_sum / steps, **self._episode_statistics(episodes),
                      "episode_groups": group_statistics,
                      "reward_components": {key: value / steps for key, value in component_sums.items()},
                      "losses": losses, "action_min": action_min.cpu().tolist(), "action_max": action_max.cpu().tolist(),
                      "action_saturation_fraction": action_saturated / (steps * 6),
                      "effort_saturation_fraction": effort_saturation_sum / effort_saturation_count if effort_saturation_count else None,
                      "leg_target_min_rad": action_min[:4].cpu().tolist(), "leg_target_max_rad": action_max[:4].cpu().tolist(),
                      "wheel_target_min_rad_s": (30.0 * action_min[4:]).cpu().tolist(),
                      "wheel_target_max_rad_s": (30.0 * action_max[4:]).cpu().tolist(),
                      "guidance_alpha": 0.0, "physical_failure_counts": self.env.physical_failure_counts(),
                      "selected_failure_trajectory_envs": self._diagnostics.selected_count,
                      "diagnostic_tensor_bytes": self._diagnostics.tensor_bytes,
                      "diagnostic_memory_upper_bound_bytes": self._diagnostics.memory_upper_bound_bytes,
                      "rollout_seconds": rollout_finished - started, "optimization_seconds": finished - rollout_finished,
                      "performance": deepcopy(self.performance)}
            self.metrics.append_episodes(episodes, update)
            self.metrics.append_update(record)
            record["logging_barrier"] = self.metrics.snapshot()
            self._write_tensorboard(record)
            self.update_records.append(record)
            records.append(record)
            if update % int(self.config["ppo"]["save_interval"]) == 0 or update == int(self.config["ppo"]["max_iterations"]):
                self.save(self.run_dir / f"model_{update}.pt", update)
        return records

    @staticmethod
    def _episode_statistics(episodes):
        count = len(episodes)
        return {"episode_count": count,
                "mean_episode_reward": sum(float(item["reward"]) for item in episodes) / count if count else None,
                "mean_episode_length": sum(int(item["length"]) for item in episodes) / count if count else None,
                "success_rate": sum(bool(item["success"]) for item in episodes) / count if count else None}

    def _write_tensorboard(self, record):
        update = record["update"]
        self._tensorboard.add_scalar("Train/mean_step_reward", record["mean_step_reward"], update)
        self._tensorboard.add_scalar("Train/episode_count", record["episode_count"], update)
        for key in ("mean_episode_reward", "mean_episode_length", "success_rate"):
            if record[key] is not None:
                self._tensorboard.add_scalar("Train/" + key, record[key], update)
        for source, values in record["episode_groups"].items():
            for key, value in values.items():
                if value is not None:
                    self._tensorboard.add_scalar(f"Episodes/{source}/{key}", value, update)
        for key, value in record["reward_components"].items():
            self._tensorboard.add_scalar("Rewards/" + key, value, update)
        for key, value in record["losses"].items():
            if isinstance(value, (int, float)):
                self._tensorboard.add_scalar("losses/" + key, value, update)
        for index, value in enumerate(record["losses"]["latent_std"]):
            self._tensorboard.add_scalar(f"Policy/latent_std_{index}", value, update)
        self._tensorboard.flush()

    def save(self, path, update=None):
        self._open_metrics()
        if update is not None and update != self.completed_updates:
            raise ValueError("checkpoint 更新必须等于真实完成更新")
        payload = {"completed_updates": self.completed_updates, "policy": self.policy.state_dict(),
                   "algorithm": self.algorithm.state_dict(), "environment_state": self.env.environment_state(),
                   "observation": {key: value.clone() for key, value in self._observation.items()},
                   "performance": deepcopy(self.performance), "logging_barrier": self.metrics.snapshot(),
                   "selected_failure_trajectories": self._diagnostics.state_dict(),
                   "action_mapping": {"type": "tanh_latent_normal", "leg_scale_rad": 1.0, "wheel_scale_rad_s": 30.0},
                   "observation_dimensions": {"policy": 110, "critic": 41}}
        return save_independent_checkpoint(path, payload, self.identity)

    def load(self, path):
        if self.metrics is not None or self.completed_updates:
            raise RuntimeError("checkpoint 必须在本 runner 开始更新前恢复")
        payload, rng = load_independent_checkpoint(path, self.identity, restore_rng=False, include_rng=True)
        expected = {"type": "tanh_latent_normal", "leg_scale_rad": 1.0, "wheel_scale_rad_s": 30.0}
        if payload["action_mapping"] != expected or payload["observation_dimensions"] != {"policy": 110, "critic": 41}:
            raise ValueError("checkpoint 动作或观测契约不一致")
        self.policy.load_state_dict(payload["policy"], strict=True)
        self.algorithm.load_state_dict(payload["algorithm"])
        self.env.restore_environment_state(payload["environment_state"])
        self._observation = self._validate_observation(payload["observation"])
        actual = self._validate_observation(self.env.get_observations())
        if any(not torch.equal(actual[key], self._observation[key]) for key in actual):
            raise ValueError("恢复后物理环境观测与 checkpoint 不一致")
        self.completed_updates = int(payload["completed_updates"])
        self.performance = deepcopy(payload["performance"])
        self._resume_logging = deepcopy(payload["logging_barrier"])
        self._diagnostics.load_state_dict(payload["selected_failure_trajectories"])
        if self._resume_logging["last_update"] != self.completed_updates:
            raise ValueError("checkpoint 更新与日志确认不一致")
        restore_global_rng_state(rng)
        return payload

    def export_actor(self, path):
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        actor = DeterministicStandingActor(deepcopy(self.policy.actor).cpu().eval())
        scripted = torch.jit.script(actor)
        scripted.save(str(destination))
        observation = self._observation["policy"].detach().cpu()
        with torch.no_grad():
            original = self.policy.deterministic(observation.to(self.device)).cpu()
            exported = scripted(observation)
        if not torch.allclose(original, exported, atol=2e-6, rtol=2e-6):
            raise ValueError("CPU 导出 Actor 与训练确定性动作不一致")
        metadata = {"schema_version": "IndependentStandingActorExportV1", "actor_file": destination.name,
                    "actor_sha256": file_sha256(destination), "identity": deepcopy(self.identity),
                    "identity_sha256": stable_hash(self.identity), "completed_updates": self.completed_updates,
                    "input_dimension": 110, "output_dimension": 6,
                    "action_mapping": {"type": "tanh_latent_normal", "leg_scale_rad": 1.0, "wheel_scale_rad_s": 30.0},
                    "verification_input_count": int(observation.shape[0]),
                    "maximum_action_error": float((original - exported).abs().max())}
        with Path(str(destination) + ".json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        return metadata

    def close(self, reason="training_run_end"):
        if self.closed:
            return None
        self._open_metrics()
        self._tensorboard.close()
        state = self.metrics.close(reason)
        self.closed = True
        return state
