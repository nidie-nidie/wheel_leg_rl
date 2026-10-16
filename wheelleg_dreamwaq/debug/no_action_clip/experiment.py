from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch
from rsl_rl.runners import OnPolicyRunner
from torch.distributions import Normal

from wheelleg_dreamwaq.algorithms.dreamwaq.actor_critic import (
    DreamWaQActorCritic, DreamWaQBatchOutput, _require_finite,
)
from wheelleg_dreamwaq.algorithms.dreamwaq.ppo import DreamWaQPPO
from wheelleg_dreamwaq.schemas.action import wheel_targets_control_to_usd
from wheelleg_dreamwaq.schemas.normalization import NormalizationV2
from wheelleg_dreamwaq.training.checkpoint import (
    capture_rng_state, expose_episode_info_key_union,
)


from debug.no_action_clip.contract import (
    ACTION_CONTRACT, BASELINE_EXPORT, EXPERIMENT_ID, EXPERIMENT_ROOT, GEOMETRY_GUARD_CONTRACT,
    PROJECT_ROOT, sha256, verify_formal_files, write_json,
)
from debug.no_action_clip.rewards import reward_manifest_fields, validate_reward_manifest


@dataclass(frozen=True)
class RawActionNormalization(NormalizationV2):
    schema_version: ClassVar[str] = "DebugRawActionNormalizationV1"

    @staticmethod
    def normalize_previous_action(values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != 6:
            raise ValueError("上一动作必须有六个分量")
        _require_finite("previous raw action", values)
        return values.clone()


def raw_action_targets(actions, q_reference, limits):
    if actions.ndim != 2 or actions.shape[-1] != 6:
        raise ValueError("实验动作必须为 [N,6]")
    _require_finite("raw runtime action", actions)
    nominal = q_reference.to(dtype=actions.dtype, device=actions.device)
    leg_targets = torch.clamp(
        nominal + limits.leg_action_scale * actions[:, :4],
        actions.new_tensor(limits.leg_target_lower),
        actions.new_tensor(limits.leg_target_upper),
    )
    wheel_targets = wheel_targets_control_to_usd(limits.wheel_action_scale * actions[:, 4:6])
    _require_finite("leg target", leg_targets)
    _require_finite("wheel target", wheel_targets)
    return actions.clone(), leg_targets, wheel_targets


class RawActionActorCritic(DreamWaQActorCritic):
    def batch_forward(self, obs, actions) -> DreamWaQBatchOutput:
        encoding = self._encoding(obs)
        mean = self._effective_mean(obs[self.current_obs_group], encoding)
        sigma = self._effective_std(mean)
        distribution = Normal(mean, sigma)
        context_z = self.cenet.sample_context(encoding.context_mu, encoding.context_logvar)
        predicted = self.cenet.decode(context_z, encoding.estimated_velocity, actions)
        output = DreamWaQBatchOutput(
            mean, sigma, distribution.log_prob(actions).sum(dim=-1),
            distribution.entropy().sum(dim=-1), self.critic(obs["critic"]),
            encoding.estimated_velocity, encoding.context_mu, encoding.context_logvar, predicted,
        )
        for name, value in zip(DreamWaQBatchOutput._fields, output, strict=True):
            _require_finite(name, value)
        return output


def verify_saved_inputs(run_dir: Path, manifest: dict) -> None:
    for relative_path, field in (("params/env.yaml", "environment_config_sha256"),
                                 ("params/agent.yaml", "agent_config_sha256"),
                                 ("closed_chain_reset_cache.pt", "reset_cache_sha256")):
        if sha256(run_dir / relative_path) != manifest[field]:
            raise ValueError(f"已保存的实验配置或缓存指纹不符：{relative_path}")


def load_experiment_checkpoint(path: Path):
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_reward_manifest(manifest)
    if (manifest["action_contract"] != ACTION_CONTRACT
            or manifest.get("geometry_guard") != GEOMETRY_GUARD_CONTRACT):
        raise ValueError("实验 manifest 身份或动作契约不符")
    verify_saved_inputs(path.parent, manifest)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    infos = payload["infos"]
    if (infos["experiment_id"] != manifest["experiment_id"]
            or infos.get("reward_contract") != manifest.get("reward_contract")
            or infos.get("geometry_guard") != GEOMETRY_GUARD_CONTRACT
            or infos["manifest_sha256"] != sha256(manifest_path)
            or infos["completed_iterations"] != int(payload["iter"]) + 1
            or infos["seed"] != manifest["seed"]):
        raise ValueError("实验 checkpoint 身份或迭代绑定不符")
    if infos["estimator_monitor_state"]["completed_rollouts"] != infos["completed_iterations"]:
        raise ValueError("实验 checkpoint 的 CENet rollout 计数不符")
    for name, value in payload["model_state_dict"].items():
        _require_finite(f"saved parameter {name}", value)
    for parameter_state in payload["optimizer_state_dict"]["state"].values():
        for name, value in parameter_state.items():
            if isinstance(value, torch.Tensor):
                _require_finite(f"saved optimizer {name}", value)
    return payload, manifest


class RawActionRunner(OnPolicyRunner):
    def __init__(self, env, config, run_dir: Path):
        self.run_dir = run_dir
        self.manifest_path = run_dir / "manifest.json"
        self.manifest_sha256 = sha256(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        validate_reward_manifest(self.manifest)
        for name, value in reward_manifest_fields(env.unwrapped.cfg.reward_weights).items():
            if self.manifest.get(name) != value:
                raise ValueError(f"环境实际奖励配置与 manifest 不符：{name}")
        if config["class_name"] != self.manifest["experiment_id"] or config["clip_actions"] is not None:
            raise ValueError("实验 runner 标识或动作裁剪设置错误")
        self.requested_iterations = config["max_iterations"]
        super().__init__(env, copy.deepcopy(config), log_dir=str(run_dir), device=config["device"])

    def _construct_algorithm(self, obs):
        policy_cfg = copy.deepcopy(self.policy_cfg)
        algorithm_cfg = copy.deepcopy(self.alg_cfg)
        if policy_cfg.pop("class_name") != "RawActionActorCritic":
            raise ValueError("实验策略类错误")
        if algorithm_cfg.pop("class_name") != "DreamWaQPPO":
            raise ValueError("实验 PPO 类错误")
        policy = RawActionActorCritic(obs, self.cfg["obs_groups"], self.env.num_actions, **policy_cfg).to(self.device)
        algorithm = DreamWaQPPO(policy, device=self.device, multi_gpu_cfg=self.multi_gpu_cfg, **algorithm_cfg)
        algorithm.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return algorithm

    def _verify_inputs(self):
        if sha256(self.manifest_path) != self.manifest_sha256:
            raise RuntimeError("训练 manifest 在运行期间发生变化")
        expected = self.manifest["formal_files_before_sha256"]
        for path in (EXPERIMENT_ROOT / "formal_files_before.json", self.run_dir / "formal_files_before.json"):
            if sha256(path) != expected:
                raise RuntimeError("正式文件起始指纹清单发生变化")
        verify_formal_files()
        verify_saved_inputs(self.run_dir, self.manifest)
        for name, expected_hash in self.manifest["debug_file_sha256"].items():
            if sha256(EXPERIMENT_ROOT / name) != expected_hash:
                raise RuntimeError(f"实验源文件发生变化：{name}")

    def save(self, path, infos=None):
        self._verify_inputs()
        completed = self.current_learning_iteration + 1
        monitor = self.alg.estimator_monitor_state.to_dict()
        if monitor["completed_rollouts"] != completed:
            raise RuntimeError("保存时 CENet rollout 计数与迭代不符")
        metadata = {
            "experiment_id": self.manifest["experiment_id"],
            "geometry_guard": GEOMETRY_GUARD_CONTRACT,
            "manifest_sha256": self.manifest_sha256,
            "completed_iterations": completed,
            "seed": self.cfg["seed"],
            "rng_state": capture_rng_state(),
            "environment_rng_state": self.env.unwrapped.get_randomization_rng_state(),
            "estimator_monitor_state": monitor,
            "algorithm_learning_rate": float(self.alg.learning_rate),
        }
        if "reward_contract" in self.manifest:
            metadata["reward_contract"] = self.manifest["reward_contract"]
        destination = Path(path)
        temporary = destination.with_suffix(".pending")
        super().save(str(temporary), metadata)
        os.replace(temporary, destination)
        load_experiment_checkpoint(destination)

    def load(self, *args, **kwargs):
        raise RuntimeError("本实验仅支持 fresh；禁止默认 RSL-RL load/resume")

    def log(self, locs, width=80, pad=35):
        self._verify_inputs()
        patched = dict(locs)
        patched["ep_infos"] = expose_episode_info_key_union(locs.get("ep_infos", []), device=self.device)
        super().log(patched, width, pad)
        completed = int(locs["it"]) + 1
        write_json(self.run_dir / "progress.json", {
            "experiment_id": self.manifest["experiment_id"], "completed_iterations": completed,
            "requested_iterations": self.requested_iterations,
            "state": "training", "losses": locs["loss_dict"],
            "total_timesteps": self.tot_timesteps, "training_seconds": self.tot_time,
        })
