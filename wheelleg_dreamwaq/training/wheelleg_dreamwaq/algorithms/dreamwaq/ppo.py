from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO

from wheelleg_dreamwaq.schemas.dreamwaq_manifest import EstimatorMonitorStateV1
from wheelleg_dreamwaq.schemas.observation import (
    PROPRIO_RECONSTRUCTION_TARGET_DIM,
    extract_proprio_reconstruction_target,
)

from .actor_critic import DreamWaQActorCritic
from .storage import DreamWaQRolloutStorage


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def context_mu_feature_std_mean(sum_: torch.Tensor, sum_sq: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0:
        raise ValueError("context monitor count must be positive")
    mean = sum_ / count
    variance = (sum_sq / count - mean.square()).clamp_min(0.0)
    return variance.sqrt().mean()


def masked_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("reconstruction prediction and target must have identical rank-2 shapes")
    if prediction.shape[-1] != PROPRIO_RECONSTRUCTION_TARGET_DIM:
        raise ValueError("reconstruction tensors must have 16 features")
    if mask.shape != (prediction.shape[0], 1) or mask.dtype != torch.bool:
        raise ValueError("reconstruction mask must be bool [B,1]")
    squared_error = (prediction - target).square()
    numerator = (squared_error * mask.to(dtype=squared_error.dtype)).sum()
    denominator = (mask.sum() * PROPRIO_RECONSTRUCTION_TARGET_DIM).clamp_min(1)
    return numerator / denominator


def velocity_zero_baseline_mse_ratio(
    estimated_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    denominator_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if estimated_velocity.shape != target_velocity.shape or estimated_velocity.shape[-1] != 3:
        raise ValueError("velocity tensors must have identical [B,3] shapes")
    mse = (estimated_velocity - target_velocity).square().mean()
    zero_mse = target_velocity.square().mean()
    ratio = mse / zero_mse.clamp_min(float(denominator_floor))
    return mse, zero_mse, ratio


def _global_norm(values: Iterable[torch.Tensor | None], *, device: torch.device) -> torch.Tensor:
    squares = [value.detach().float().square().sum() for value in values if value is not None]
    return torch.stack(squares).sum().sqrt() if squares else torch.zeros((), device=device)


class DreamWaQPPO(PPO):
    policy: DreamWaQActorCritic

    def __init__(
        self,
        policy: DreamWaQActorCritic,
        *,
        velocity_coef: float = 1.0,
        reconstruction_coef: float = 1.0,
        kl_beta: float = 1.0,
        strict_finite_checks: bool = True,
        context_mu_min_feature_std: float = 1.0e-3,
        context_mu_min_initial_std_ratio: float = 0.10,
        context_monitor_final_window_iterations: int = 10,
        velocity_mse_max_zero_baseline_ratio: float = 0.80,
        velocity_mse_denominator_floor: float = 1.0e-8,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        **ppo_kwargs: Any,
    ) -> None:
        if rnd_cfg is not None or symmetry_cfg is not None:
            raise ValueError("DreamWaQAlgorithmContractV1 requires rnd_cfg=None and symmetry_cfg=None")
        if (velocity_coef, reconstruction_coef, kl_beta) != (1.0, 1.0, 1.0):
            raise ValueError("DreamWaQ auxiliary coefficients must all equal 1.0")
        if not strict_finite_checks:
            raise ValueError("DreamWaQ strict finite checks cannot be disabled")
        if (
            context_mu_min_feature_std != 1.0e-3
            or context_mu_min_initial_std_ratio != 0.10
            or context_monitor_final_window_iterations != 10
            or velocity_mse_max_zero_baseline_ratio != 0.80
            or velocity_mse_denominator_floor != 1.0e-8
        ):
            raise ValueError("DreamWaQ estimator tripwire settings differ from the frozen contract")
        super().__init__(policy, rnd_cfg=None, symmetry_cfg=None, **ppo_kwargs)
        self.velocity_coef = velocity_coef
        self.reconstruction_coef = reconstruction_coef
        self.kl_beta = kl_beta
        self.strict_finite_checks = strict_finite_checks
        self.context_mu_min_feature_std = context_mu_min_feature_std
        self.context_mu_min_initial_std_ratio = context_mu_min_initial_std_ratio
        self.context_monitor_final_window_iterations = context_monitor_final_window_iterations
        self.velocity_mse_max_zero_baseline_ratio = velocity_mse_max_zero_baseline_ratio
        self.velocity_mse_denominator_floor = velocity_mse_denominator_floor
        self.transition = DreamWaQRolloutStorage.Transition()
        self.estimator_monitor_state = EstimatorMonitorStateV1()
        self._context_sum = torch.zeros(16, dtype=torch.float64, device=self.device)
        self._context_sum_sq = torch.zeros(16, dtype=torch.float64, device=self.device)
        self._context_count = 0

    def set_estimator_monitor_state(self, state: EstimatorMonitorStateV1) -> None:
        self.estimator_monitor_state = EstimatorMonitorStateV1.from_dict(state.to_dict())

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        self.storage = DreamWaQRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def _record_context(self, context_mu: torch.Tensor) -> None:
        detached = context_mu.detach().to(dtype=torch.float64)
        self._context_sum += detached.sum(dim=0)
        self._context_sum_sq += detached.square().sum(dim=0)
        self._context_count += detached.shape[0]

    def _finalize_context_rollout(self) -> float:
        value = context_mu_feature_std_mean(self._context_sum, self._context_sum_sq, self._context_count)
        _require_finite("context_mu_feature_std_mean", value)
        result = float(value.item())
        self.estimator_monitor_state.record_rollout(result)
        self._context_sum.zero_()
        self._context_sum_sq.zero_()
        self._context_count = 0
        return result

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        snapshot = obs.clone(recurse=True)
        actions, context_mu = self.policy.act_with_context(snapshot)
        self._record_context(context_mu)
        self.transition.actions = actions.detach()
        self.transition.values = self.policy.evaluate(snapshot).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = snapshot
        return self.transition.actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        self.policy.update_normalization(obs)
        self.transition.next_proprio_target = extract_proprio_reconstruction_target(obs["critic"])
        self.transition.reconstruction_mask = dones.eq(0).view(-1, 1)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def _adapt_learning_rate(
        self,
        sigma_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
        old_mu_batch: torch.Tensor,
        mu_batch: torch.Tensor,
    ) -> None:
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                / (2.0 * torch.square(sigma_batch))
                - 0.5,
                axis=-1,
            )
            kl_mean = torch.mean(kl)
            _require_finite("adaptive KL", kl_mean)
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
            if self.gpu_global_rank == 0:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
            if self.is_multi_gpu:
                lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(lr_tensor, src=0)
                self.learning_rate = lr_tensor.item()
            for parameter_group in self.optimizer.param_groups:
                parameter_group["lr"] = self.learning_rate

    def _assert_gradients_finite(self, stage: str) -> None:
        for name, parameter in self.policy.named_parameters():
            if parameter.grad is not None:
                _require_finite(f"{stage} gradient {name}", parameter.grad)

    def _assert_optimizer_finite(self) -> None:
        for name, parameter in self.policy.named_parameters():
            _require_finite(f"optimizer parameter {name}", parameter)
        for parameter_state in self.optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq", "step"):
                value = parameter_state.get(key)
                if isinstance(value, torch.Tensor):
                    _require_finite(f"Adam state {key}", value)

    def update(self) -> dict[str, float]:
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
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.target_values + (output.value - batch.target_values).clamp(
                    -self.clip_param,
                    self.clip_param,
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
                output.predicted_next_proprio,
                batch.next_proprio_target,
                batch.reconstruction_mask,
            )
            kl_loss = -0.5 * (
                1.0
                + output.context_logvar
                - output.context_mu.square()
                - torch.exp(output.context_logvar)
            ).mean()
            auxiliary_loss = (
                self.velocity_coef * velocity_loss
                + self.reconstruction_coef * reconstruction_loss
                + self.kl_beta * kl_loss
            )
            total_loss = ppo_loss + auxiliary_loss
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
        result = {name: value / update_count for name, value in totals.items()}
        monitor = self.estimator_monitor_state
        final_mean = monitor.final_window_mean
        threshold = monitor.acceptance_threshold(
            self.context_mu_min_feature_std,
            self.context_mu_min_initial_std_ratio,
        )
        result.update(
            {
                "context_mu_feature_std_mean_rollout": rollout_context_std,
                "context_mu_feature_std_mean_final_window": float(final_mean),
                "context_mu_feature_std_threshold": threshold,
                "context_mu_gate_partial_pass": float(
                    monitor.passes_context_gate(
                        self.context_mu_min_feature_std,
                        self.context_mu_min_initial_std_ratio,
                        require_full_window=False,
                    )
                ),
            }
        )
        return result
