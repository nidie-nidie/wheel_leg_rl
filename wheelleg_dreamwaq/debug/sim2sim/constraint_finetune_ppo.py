from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from wheelleg_dreamwaq.algorithms.dreamwaq.ppo import (
    DreamWaQPPO,
    _global_norm,
    _require_finite,
    masked_reconstruction_loss,
)

from .constraint_finetune_math import diagonal_gaussian_kl, validate_constraint_tensor


def _reference_parameter_drift(policy: nn.Module, reference: nn.Module) -> torch.Tensor:
    numerator = torch.zeros((), device=next(policy.parameters()).device)
    denominator = torch.zeros_like(numerator)
    for current, frozen in zip(policy.parameters(), reference.parameters(), strict=True):
        numerator = numerator + (current.detach() - frozen.detach()).square().sum()
        denominator = denominator + frozen.detach().square().sum()
    return (numerator / denominator.clamp_min(1.0e-12)).sqrt()


class ConstrainedDreamWaQPPO(DreamWaQPPO):
    def __init__(self, *args: Any, reference_kl_coef: float = 0.05, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not torch.isfinite(torch.as_tensor(reference_kl_coef)) or reference_kl_coef < 0.0:
            raise ValueError("reference_kl_coef must be finite and non-negative")
        self.reference_kl_coef = float(reference_kl_coef)
        self.reference_policy: nn.Module | None = None
        self._last_reference_kl = 0.0
        self._last_parameter_drift = 0.0

    def set_reference_from_policy(self) -> None:
        self.reference_policy = copy.deepcopy(self.policy).to(self.device).eval()
        for parameter in self.reference_policy.parameters():
            parameter.requires_grad_(False)
        self._last_reference_kl = 0.0
        self._last_parameter_drift = 0.0

    @staticmethod
    def reference_kl_from_distributions(
        current_mean: torch.Tensor,
        current_std: torch.Tensor,
        reference_mean: torch.Tensor,
        reference_std: torch.Tensor,
    ) -> torch.Tensor:
        return diagonal_gaussian_kl(current_mean, current_std, reference_mean, reference_std)

    def _reference_distribution(self, observations):
        if self.reference_policy is None:
            raise RuntimeError("reference policy is not initialized")
        with torch.no_grad():
            encoding = self.reference_policy._encoding(observations)
            mean = self.reference_policy._effective_mean(
                observations[self.reference_policy.current_obs_group], encoding
            )
            std = self.reference_policy._effective_std(mean)
        return mean, std

    def reference_kl(self, observations, current_mean: torch.Tensor, current_std: torch.Tensor) -> torch.Tensor:
        reference_mean, reference_std = self._reference_distribution(observations)
        result = self.reference_kl_from_distributions(
            current_mean, current_std, reference_mean, reference_std
        )
        validate_constraint_tensor(result, "reference policy KL")
        return result

    def update(self) -> dict[str, float]:
        if self.reference_policy is None:
            raise RuntimeError("reference policy must be initialized before update")
        rollout_context_std = self._finalize_context_rollout()
        if self.policy.is_recurrent:
            raise ValueError("DreamWaQPPO supports feed-forward policies only")
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        totals = {
            "value_function": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "velocity": 0.0,
            "reconstruction": 0.0,
            "kl": 0.0,
            "reference_policy_kl": 0.0,
            "total": 0.0,
            "ppo_gradient_norm": 0.0,
            "auxiliary_gradient_norm": 0.0,
            "auxiliary_to_ppo_gradient_norm_ratio": 0.0,
            "pre_clip_gradient_norm": 0.0,
        }
        parameters = [parameter for parameter in self.policy.parameters() if parameter.requires_grad]

        for batch in generator:
            advantages = batch.advantages
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

            output = self.policy.batch_forward(batch.observations, batch.actions)
            self._adapt_learning_rate(output.action_sigma, batch.old_sigma, batch.old_mu, output.action_mean)
            ratio = torch.exp(output.action_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(advantages) * ratio
            surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.target_values + (output.value - batch.target_values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (output.value - batch.returns).square()
                value_losses_clipped = (value_clipped - batch.returns).square()
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - output.value).square().mean()
            entropy_mean = output.entropy.mean()
            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_mean

            velocity_target = self.policy.get_velocity_target(batch.observations)
            velocity_loss = (output.estimated_velocity - velocity_target).square().mean()
            reconstruction_loss = masked_reconstruction_loss(
                output.predicted_next_proprio, batch.next_proprio_target, batch.reconstruction_mask
            )
            kl_loss = -0.5 * (
                1.0 + output.context_logvar - output.context_mu.square()
                - torch.exp(output.context_logvar)
            ).mean()
            auxiliary_loss = (
                self.velocity_coef * velocity_loss
                + self.reconstruction_coef * reconstruction_loss
                + self.kl_beta * kl_loss
            )
            reference_kl = self.reference_kl(batch.observations, output.action_mean, output.action_sigma).mean()
            total_loss = ppo_loss + auxiliary_loss + self.reference_kl_coef * reference_kl
            finite_values = {
                "ratio": ratio,
                "advantage": advantages,
                "surrogate loss": surrogate_loss,
                "value loss": value_loss,
                "entropy": entropy_mean,
                "PPO loss": ppo_loss,
                "velocity loss": velocity_loss,
                "reconstruction loss": reconstruction_loss,
                "KL exp(logvar)": torch.exp(output.context_logvar),
                "KL loss": kl_loss,
                "reference policy KL": reference_kl,
                "total loss": total_loss,
            }
            for name, value in finite_values.items():
                _require_finite(name, value)

            ppo_grads = torch.autograd.grad(ppo_loss, parameters, retain_graph=True, allow_unused=True)
            auxiliary_grads = torch.autograd.grad(auxiliary_loss, parameters, retain_graph=True, allow_unused=True)
            ppo_grad_norm = _global_norm(ppo_grads, device=total_loss.device)
            auxiliary_grad_norm = _global_norm(auxiliary_grads, device=total_loss.device)
            grad_ratio = auxiliary_grad_norm / ppo_grad_norm.clamp_min(1.0e-12)
            _require_finite("PPO gradient norm", ppo_grad_norm)
            _require_finite("auxiliary gradient norm", auxiliary_grad_norm)
            _require_finite("auxiliary/PPO gradient norm ratio", grad_ratio)

            self.optimizer.zero_grad()
            total_loss.backward()
            self._assert_gradients_finite("backward")
            if self.is_multi_gpu:
                self.reduce_parameters()
                self._assert_gradients_finite("distributed reduction")
            pre_clip_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            _require_finite("pre-clip gradient norm", pre_clip_norm)
            self._assert_gradients_finite("gradient clipping")
            self.optimizer.step()
            self._assert_optimizer_finite()

            values = {
                "value_function": value_loss,
                "surrogate": surrogate_loss,
                "entropy": entropy_mean,
                "velocity": velocity_loss,
                "reconstruction": reconstruction_loss,
                "kl": kl_loss,
                "reference_policy_kl": reference_kl,
                "total": total_loss,
                "ppo_gradient_norm": ppo_grad_norm,
                "auxiliary_gradient_norm": auxiliary_grad_norm,
                "auxiliary_to_ppo_gradient_norm_ratio": grad_ratio,
                "pre_clip_gradient_norm": pre_clip_norm,
            }
            for name, value in values.items():
                totals[name] += float(value.detach().item())

        update_count = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        self._last_reference_kl = totals["reference_policy_kl"] / update_count
        self._last_parameter_drift = float(_reference_parameter_drift(self.policy, self.reference_policy).item())
        result = {name: value / update_count for name, value in totals.items()}
        result["context_mu_feature_std_mean_rollout"] = rollout_context_std
        monitor = self.estimator_monitor_state
        final_mean = monitor.final_window_mean
        threshold = monitor.acceptance_threshold(
            self.context_mu_min_feature_std, self.context_mu_min_initial_std_ratio
        )
        result.update({
            "context_mu_feature_std_mean_final_window": float(final_mean),
            "context_mu_feature_std_threshold": threshold,
            "context_mu_gate_partial_pass": float(monitor.passes_context_gate(
                self.context_mu_min_feature_std,
                self.context_mu_min_initial_std_ratio,
                require_full_window=False
            )),
            "reference_policy_parameter_drift": self._last_parameter_drift,
        })
        return result

