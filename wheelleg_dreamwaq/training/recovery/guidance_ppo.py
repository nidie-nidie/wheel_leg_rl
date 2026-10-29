from __future__ import annotations

from copy import deepcopy
import math

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict
from torch import nn

from .guidance_networks import GuidanceCostCritic
from .guidance_storage import GuidanceRolloutStorage


def _require_tensor(condition: torch.Tensor, message: str) -> None:
    torch._assert_async(condition, message)


class GuidancePPO(PPO):
    def __init__(self, policy, guidance_config: dict, *, device="cpu", **algorithm_kwargs):
        if policy.is_recurrent:
            raise ValueError("辅助引导要求普通前馈 ActorCritic")
        for key in ("rnd_cfg", "symmetry_cfg", "multi_gpu_cfg"):
            if algorithm_kwargs.get(key) is not None:
                raise ValueError(f"辅助引导不支持 {key}")
        if algorithm_kwargs.get("normalize_advantage_per_mini_batch", False):
            raise ValueError("辅助优势只允许整批标准化一次")
        self.guidance_config = deepcopy(guidance_config)
        for key in ("beta_force", "cost_scale_n", "cost_learning_rate", "cost_max_grad_norm"):
            value = float(guidance_config[key])
            if not math.isfinite(value) or (value < 0 if key == "beta_force" else value <= 0):
                raise ValueError(f"辅助学习参数无效: {key}")
        if guidance_config["torque_enabled"] or guidance_config["beta_torque"] != 0 or guidance_config["max_torque_nm"] != 0:
            raise ValueError("当前辅助引导只支持关闭力矩通路")
        super().__init__(policy, device=device, **algorithm_kwargs)
        if self.gamma != guidance_config["gamma"] or self.lam != guidance_config["gae_lambda"]:
            raise ValueError("辅助与任务 GAE 参数必须一致")
        self.beta_force = float(guidance_config["beta_force"])
        self.cost_scale_n = float(guidance_config["cost_scale_n"])
        self.cost_max_grad_norm = float(guidance_config["cost_max_grad_norm"])
        self.end_control_steps = int(guidance_config["end_updates"]) * int(guidance_config["rollout_steps"])
        if self.end_control_steps <= 0:
            raise ValueError("辅助撤除控制步数必须为正整数")
        self.cost_critic = GuidanceCostCritic().to(self.device)
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=float(guidance_config["cost_learning_rate"]))
        self._pending_cost_value = None
        self._pending_alpha = None
        self._last_control_steps = None
        self.last_guidance_statistics: dict[str, float] = {}
        self._assistance_withdrawn = not bool(guidance_config["force_enabled"])

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        if obs["policy"].shape != (num_envs, 111) or obs["critic"].shape != (num_envs, 35):
            raise ValueError("辅助 Actor/Critic 输入必须为 111/35 维")
        if num_transitions_per_env != int(self.guidance_config["rollout_steps"]):
            raise ValueError("辅助 rollout 长度与撤除配置不一致")
        self.storage = GuidanceRolloutStorage(training_type, num_envs, num_transitions_per_env, obs, actions_shape, self.device)

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self._pending_cost_value is not None:
            raise RuntimeError("上一次辅助 transition 尚未提交")
        alpha = obs["critic"][:, -1:]
        _require_tensor(torch.isfinite(alpha).all() & (alpha >= 0).all() & (alpha <= 1).all(), "辅助观测 alpha 无效")
        _require_tensor((alpha.squeeze(-1) == obs["policy"][:, -1]).all(), "Actor/Critic 辅助进度不同")
        actions = super().act(obs)
        if self._assistance_withdrawn:
            self._pending_cost_value = torch.zeros((obs.batch_size[0], 1), device=self.device)
        else:
            self._pending_cost_value = self.cost_critic(obs["critic"]).detach()
        _require_tensor(torch.isfinite(self._pending_cost_value).all(), "辅助成本价值包含非有限数值")
        self._pending_alpha = alpha.detach().clone()
        return actions

    def process_env_step(self, obs, rewards, dones, extras):
        if self._pending_cost_value is None:
            raise RuntimeError("辅助 transition 缺少对应 act 调用")
        if "time_outs" in extras:
            raise ValueError("起身有限回合终止不允许 timeout bootstrap")
        guidance = extras["guidance"]
        if guidance["schema_version"] != "RecoveryGuidanceTransitionV1":
            raise ValueError("辅助 transition schema 不匹配")
        num_envs = self.storage.num_envs
        fields = {}
        for key in ("cost_force_n", "alpha_t", "alpha_next", "max_force_n"):
            value = guidance[key]
            if not isinstance(value, torch.Tensor) or value.shape != (num_envs,) or value.device != self.storage.cost_values.device:
                raise ValueError(f"辅助 transition 字段形状或设备不匹配: {key}")
            fields[key] = value.detach().reshape(num_envs, 1)
        cost, alpha, next_alpha, maximum = (fields[name] for name in ("cost_force_n", "alpha_t", "alpha_next", "max_force_n"))
        k = guidance["global_control_steps"]
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("辅助 transition 计数必须为已完成的控制步数")
        if self._last_control_steps is not None and k != self._last_control_steps + 1:
            raise ValueError("辅助 transition 控制步计数不连续")
        expected_alpha = max(1.0 - (k - 1) / self.end_control_steps, 0.0) if self.guidance_config["force_enabled"] else 0.0
        expected_next_alpha = max(1.0 - k / self.end_control_steps, 0.0) if self.guidance_config["force_enabled"] else 0.0
        conditions = torch.stack((
            torch.isfinite(torch.cat(tuple(fields.values()), dim=1)).all(),
            (cost >= 0).all(), (maximum >= 0).all(), (cost <= maximum * alpha + 1e-5).all(),
            (alpha == self._pending_alpha).all(),
            (torch.abs(alpha - expected_alpha) <= 1e-7).all(),
            (torch.abs(next_alpha - expected_next_alpha) <= 1e-7).all(),
            (next_alpha.squeeze(-1) == obs["critic"][:, -1]).all(),
            (next_alpha.squeeze(-1) == obs["policy"][:, -1]).all(),
            torch.where(alpha == 0, cost == 0, torch.ones_like(cost, dtype=torch.bool)).all(),
        ))
        _require_tensor(conditions.all(), "辅助 transition 数值、进度或施力上限无效")
        self.policy.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.storage.add_guidance_transition(
            self.transition, cost_force_n=cost, cost_values=self._pending_cost_value,
            alpha_t=alpha, alpha_next=next_alpha,
        )
        self.transition.clear()
        self.policy.reset(dones)
        self._pending_cost_value = self._pending_alpha = None
        self._last_control_steps = k
        self._assistance_withdrawn = not self.guidance_config["force_enabled"] or k >= self.end_control_steps

    @torch.no_grad()
    def compute_returns(self, obs):
        if self._pending_cost_value is not None:
            raise RuntimeError("辅助 transition 尚未提交")
        last_cost = torch.zeros((obs.batch_size[0], 1), device=self.device) if self._assistance_withdrawn else self.cost_critic(obs["critic"]).detach()
        self.storage.compute_guidance_returns(
            self.policy.evaluate(obs).detach(), last_cost,
            self.gamma, self.lam, self.cost_scale_n, self.beta_force,
        )

    def _adapt_learning_rate(self, old_mu, old_sigma, mu, sigma):
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        with torch.inference_mode():
            kl_mean = torch.sum(
                torch.log(sigma / old_sigma + 1e-5)
                + (old_sigma.square() + (old_mu - mu).square()) / (2 * sigma.square()) - 0.5,
                dim=-1,
            ).mean()
            _require_tensor(torch.isfinite(kl_mean), "PPO KL 包含非有限数值")
            if kl_mean > self.desired_kl * 2:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif 0 < kl_mean < self.desired_kl / 2:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def update(self) -> dict[str, float]:
        sums = torch.zeros(5, device=self.device)
        has_assisted_samples = bool((self.storage.alpha_t > 0).any())
        diagnostics = torch.stack((
            self.storage.cost_force_n.mean(), self.storage.cost_force_n.max(),
            self.storage.alpha_t.mean(), self.storage.alpha_next[-1].mean(),
            self.storage.reward_advantages.mean(), self.storage.reward_advantages.std(unbiased=False),
            self.storage.cost_advantages_physical.mean(), self.storage.cost_advantages_physical.std(unbiased=False),
            self.storage.penalty_advantages.mean(), self.storage.penalty_advantages.std(unbiased=False),
            self.storage.mixed_advantages.mean(), self.storage.mixed_advantages.std(unbiased=False),
        )).detach()
        num_updates = 0
        for obs, batch in self.storage.guidance_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            self.policy.act(obs)
            log_prob = self.policy.get_actions_log_prob(batch["actions"])
            value = self.policy.evaluate(obs)
            entropy = self.policy.entropy.mean()
            self._adapt_learning_rate(batch["mu"], batch["sigma"], self.policy.action_mean, self.policy.action_std)
            ratio = torch.exp(log_prob - batch["actions_log_prob"].squeeze(-1))
            advantage = batch["advantages"].squeeze(-1)
            surrogate_loss = torch.maximum(
                -advantage * ratio,
                -advantage * ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
            ).mean()
            value_loss = (value - batch["returns"]).square()
            if self.use_clipped_value_loss:
                clipped = batch["values"] + (value - batch["values"]).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum(value_loss, (clipped - batch["returns"]).square())
            value_loss = value_loss.mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            if has_assisted_samples:
                cost_value = self.cost_critic(obs["critic"])
                cost_loss = (cost_value - batch["cost_returns"]).square().mean()
            else:
                cost_loss = torch.zeros((), device=self.device)
            _require_tensor(torch.isfinite(torch.stack((loss, cost_loss, ratio.mean()))).all(), "辅助 PPO 损失或比率非有限")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            if has_assisted_samples:
                self.cost_optimizer.zero_grad(set_to_none=True)
                cost_loss.backward()
                nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.cost_max_grad_norm, error_if_nonfinite=True)
                self.cost_optimizer.step()
            with torch.no_grad():
                cost_surrogate_delta = (ratio.detach().unsqueeze(-1) * batch["cost_advantages_physical"]).mean() / (1 - self.gamma)
                sums += torch.stack((value_loss, surrogate_loss, entropy, cost_loss, cost_surrogate_delta)).detach()
            num_updates += 1
        expected_updates = self.num_learning_epochs * self.num_mini_batches
        if num_updates != expected_updates:
            raise RuntimeError("辅助 PPO mini batch 更新次数不完整")
        names = ("value_function", "surrogate", "entropy", "guidance_cost_value_function", "guidance_cost_surrogate_delta_n")
        statistic_names = (
            "guidance_force_mean_n", "guidance_force_max_n", "guidance_alpha_mean", "guidance_alpha_next",
            "guidance_reward_advantage_mean", "guidance_reward_advantage_std",
            "guidance_cost_advantage_mean_n", "guidance_cost_advantage_std_n",
            "guidance_penalty_advantage_mean", "guidance_penalty_advantage_std",
            "guidance_mixed_advantage_mean", "guidance_mixed_advantage_std",
        )
        packed = torch.cat((sums / num_updates, diagnostics))
        _require_tensor(torch.isfinite(packed).all(), "辅助 PPO 统计包含非有限数值")
        values = packed.cpu().tolist()
        result = dict(zip(names + statistic_names, values, strict=True))
        self.last_guidance_statistics = {name: result[name] for name in statistic_names}
        self.storage.clear()
        return result

    @torch.no_grad()
    def estimate_initial_cost(self, initial_critic_observation: torch.Tensor, alpha: float) -> torch.Tensor:
        if initial_critic_observation.ndim != 2 or initial_critic_observation.shape[-1] != 35:
            raise ValueError("初态成本估计要求 35 维 critic 观测")
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("初态成本估计 alpha 无效")
        initial = initial_critic_observation.to(self.device).clone()
        initial[:, -1] = alpha
        estimate = self.cost_critic(initial).mean() * self.cost_scale_n
        _require_tensor(torch.isfinite(estimate), "初态辅助成本估计包含非有限数值")
        return estimate

    def guidance_state_dict(self) -> dict:
        return {"cost_critic_state_dict": self.cost_critic.state_dict(),
                "cost_optimizer_state_dict": self.cost_optimizer.state_dict()}

    def load_guidance_state_dict(self, payload: dict) -> None:
        self.cost_critic.load_state_dict(payload["cost_critic_state_dict"], strict=True)
        self.cost_optimizer.load_state_dict(payload["cost_optimizer_state_dict"])
        _require_tensor(torch.stack([torch.isfinite(parameter).all() for parameter in self.cost_critic.parameters()]).all(),
                        "恢复的辅助成本网络包含非有限参数")
        self._pending_cost_value = self._pending_alpha = None
        self._last_control_steps = None
        self._assistance_withdrawn = not bool(self.guidance_config["force_enabled"])
