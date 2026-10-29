from __future__ import annotations

import math

import torch

from .bounded_actor_critic import latent_normal_kl


class IndependentPPO:
    def __init__(self, policy, config: dict):
        self.policy = policy
        self.config = dict(config)
        self.learning_rate = float(config["learning_rate"])
        if not 1e-5 <= self.learning_rate <= 3e-4:
            raise ValueError("独立站立 PPO 学习率超出契约范围")
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.learning_rate)

    def returns(self, rewards, dones, values, last_values):
        if rewards.shape != dones.shape or values.shape != rewards.shape:
            raise ValueError("GAE rollout 形状不一致")
        advantages = torch.empty_like(rewards)
        gae = torch.zeros_like(last_values)
        next_value = last_values
        for index in reversed(range(rewards.shape[0])):
            continuing = (~dones[index]).to(rewards.dtype)
            delta = rewards[index] + float(self.config["gamma"]) * next_value * continuing - values[index]
            gae = delta + float(self.config["gamma"]) * float(self.config["lam"]) * continuing * gae
            advantages[index] = gae
            next_value = values[index]
        return advantages + values, advantages

    def update(self, batch: dict[str, torch.Tensor]):
        required = {"policy", "critic", "latent", "action", "normal_log_prob", "mean", "std",
                    "value", "returns", "advantages"}
        if set(batch) != required or any(not torch.isfinite(value).all() for value in batch.values()):
            raise ValueError("PPO rollout 字段不完整或数值非有限")
        if not torch.equal(batch["action"], torch.tanh(batch["latent"])):
            raise ValueError("storage 的执行动作和潜在动作不一致")
        size = batch["value"].numel()
        minibatches = int(self.config["num_mini_batches"])
        if size < minibatches or size % minibatches:
            raise ValueError("PPO rollout 必须能被 mini batch 数整除")
        flat = {key: value.reshape(size, *value.shape[2:]) for key, value in batch.items()}
        advantage = flat["advantages"]
        flat["advantages"] = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
        totals = {key: 0.0 for key in ("policy_loss", "value_loss", "entropy", "kl", "gradient_norm", "clip_fraction")}
        count = 0
        clip = float(self.config["clip_param"])
        for _ in range(int(self.config["num_learning_epochs"])):
            permutation = torch.randperm(size, device=batch["value"].device)
            for indices in permutation.chunk(minibatches):
                mini = {key: value[indices] for key, value in flat.items()}
                evaluated = self.policy.evaluate(mini["policy"], mini["critic"], mini["latent"])
                with torch.no_grad():
                    kl = latent_normal_kl(mini["mean"], mini["std"], evaluated["mean"], evaluated["std"]).mean()
                    if not torch.isfinite(kl):
                        raise FloatingPointError("PPO 解析 KL 非有限")
                    if kl > 0.02:
                        self.learning_rate /= 1.5
                    elif 0.0 < kl < 0.005:
                        self.learning_rate *= 1.5
                    self.learning_rate = min(3e-4, max(1e-5, self.learning_rate))
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate
                ratio = torch.exp(evaluated["normal_log_prob"] - mini["normal_log_prob"])
                surrogate = torch.maximum(-mini["advantages"] * ratio,
                                          -mini["advantages"] * ratio.clamp(1.0 - clip, 1.0 + clip)).mean()
                value_error = (evaluated["value"] - mini["returns"]).square()
                clipped_value = mini["value"] + (evaluated["value"] - mini["value"]).clamp(-clip, clip)
                value_loss = torch.maximum(value_error, (clipped_value - mini["returns"]).square()).mean()
                entropy = evaluated["entropy"].mean()
                loss = surrogate + float(self.config["value_loss_coef"]) * value_loss - float(self.config["entropy_coef"]) * entropy
                if not torch.isfinite(loss) or not torch.isfinite(ratio).all():
                    raise FloatingPointError("PPO 损失或概率比非有限")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float(self.config["max_grad_norm"]), error_if_nonfinite=True)
                self.optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in self.policy.parameters()):
                    raise FloatingPointError("PPO 更新后参数非有限")
                measures = {"policy_loss": surrogate, "value_loss": value_loss, "entropy": entropy,
                            "kl": kl, "gradient_norm": norm,
                            "clip_fraction": ((ratio - 1.0).abs() > clip).float().mean()}
                for key, value in measures.items():
                    totals[key] += float(value.detach())
                count += 1
        result = {key: value / count for key, value in totals.items()}
        result.update(learning_rate=self.learning_rate, measured_minibatch_count=count,
                      latent_std=self.policy.std.detach().cpu().tolist())
        if any(not math.isfinite(value) for value in result.values() if isinstance(value, float)):
            raise FloatingPointError("PPO 更新统计非有限")
        return result

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "learning_rate": self.learning_rate,
                "schedule": {"minimum": 1e-5, "maximum": 3e-4, "kl_lower": 0.005, "kl_upper": 0.02}}

    def load_state_dict(self, state):
        if set(state) != {"optimizer", "learning_rate", "schedule"} or state["schedule"] != self.state_dict()["schedule"]:
            raise ValueError("PPO 恢复调度配置不一致")
        if not 1e-5 <= state["learning_rate"] <= 3e-4:
            raise ValueError("PPO 恢复学习率超出契约范围")
        self.optimizer.load_state_dict(state["optimizer"])
        self.learning_rate = float(state["learning_rate"])
        if any(group["lr"] != self.learning_rate for group in self.optimizer.param_groups):
            raise ValueError("PPO optimizer 学习率和调度状态不一致")
