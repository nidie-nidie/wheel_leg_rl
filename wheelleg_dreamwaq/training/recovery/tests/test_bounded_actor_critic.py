from pathlib import Path

import numpy as np
import pytest
import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from recovery.bounded_actor_critic import BoundedActorCritic, DeterministicStandingActor, latent_normal_kl, tanh_log_jacobian
from recovery.observations import build_actor_frame


PROJECT = Path(__file__).resolve().parents[3]
REFERENCE = PROJECT / "recovery/artifacts/standing-reference401-20261029"


def actual_observations():
    with np.load(REFERENCE / "recovery_contact_00.npz") as data:
        state = {key: torch.from_numpy(data[key][100:201].copy()).float() for key in data.files}
    previous = torch.cat((state["leg_target_rad"], state["wheel_target_control_rad_s"] / 30.0), -1)
    q_reference = torch.tensor([-0.33367134, 0.33367134, -0.33367134, 0.33367134])
    frame = build_actor_frame(state["angular_velocity_control_rad_s"], state["projected_gravity"],
                              state["q_control_rad"], state["dq_control_rad_s"], previous, q_reference)
    actor = torch.stack([frame[index:index + 5].flatten() for index in range(96)])
    clean = frame[4:100]
    critic = torch.cat((clean, (state["linear_velocity_control_m_s"][4:100] * 2).clamp(-6, 6),
                        ((state["height_m"][4:100, None] - 0.2) / 0.04).clamp(-5, 5),
                        state["motor_generalized_torque_nm"][4:100] / torch.tensor([18.] * 4 + [9.] * 2),
                        torch.zeros(96, 1), torch.ones(96, 1), torch.zeros(96, 2),
                        state["wheel_normal_force_n"][4:100] / 43.1273,
                        state["nonwheel_normal_force_n"][4:100, None] / 43.1273, torch.zeros(96, 2)), -1)
    with np.load(REFERENCE / "recovery_contact_paired_reference.npz") as data:
        median = torch.from_numpy(np.median(data["q_control_rad"][:, :4], axis=0)).float()
    assert actor.shape == (96, 110) and critic.shape == (96, 41)
    return actor, critic, median


def test_actual_reference_initialization_and_sampling():
    actor, critic, median = actual_observations()
    policy = BoundedActorCritic(median)
    sampled = policy.sample(actor, critic)
    assert torch.equal(sampled.action, torch.tanh(sampled.latent))
    assert sampled.action.abs().max() <= 1
    assert torch.allclose(policy.deterministic(actor)[:, :4], median.expand(96, 4), atol=1e-7)
    assert torch.equal(policy.deterministic(actor)[:, 4:], torch.zeros(96, 2))
    assert torch.allclose(policy.std, torch.tensor([0.15] * 4 + [0.10] * 2))
    evaluated = policy.evaluate(actor, critic, sampled.latent)
    assert torch.equal(torch.exp(evaluated["normal_log_prob"] - sampled.normal_log_prob), torch.ones(96))
    assert torch.equal(latent_normal_kl(sampled.mean, sampled.std, evaluated["mean"], evaluated["std"]), torch.zeros(96))


def test_probability_matches_torch_transform_distribution():
    actor, critic, median = actual_observations()
    policy = BoundedActorCritic(median)
    sampled = policy.sample(actor, critic)
    distribution = TransformedDistribution(policy.distribution(actor), [TanhTransform(cache_size=1)])
    expected = distribution.log_prob(sampled.action).sum(-1)
    assert torch.allclose(sampled.log_prob, expected, atol=2e-5, rtol=2e-5)
    old = Normal(sampled.mean, sampled.std)
    new = Normal(sampled.mean + 0.01, sampled.std * 1.1)
    expected_kl = torch.distributions.kl_divergence(old, new).sum(-1)
    assert torch.allclose(latent_normal_kl(old.mean, old.stddev, new.mean, new.stddev), expected_kl, atol=2e-6)


@pytest.mark.parametrize("magnitude", [-20.0, 20.0])
def test_extreme_latent_entropy_and_gradients_remain_finite(magnitude):
    actor, critic, median = actual_observations()
    policy = BoundedActorCritic(median)
    latent = torch.full((96, 6), magnitude)
    evaluated = policy.evaluate(actor, critic, latent)
    assert all(torch.isfinite(value).all() for value in evaluated.values())
    loss = -evaluated["log_prob"].mean() - evaluated["entropy"].mean() + evaluated["value"].square().mean()
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in policy.parameters())
    assert torch.isfinite(tanh_log_jacobian(latent)).all()


def test_bounded_std_and_scripted_deterministic_actor(tmp_path):
    actor, _, median = actual_observations()
    policy = BoundedActorCritic(median)
    with torch.no_grad():
        policy.raw_std.copy_(torch.tensor([-100., 100., -20., 20., -100., 100.]))
    assert (policy.std >= policy.std_min).all() and (policy.std <= policy.std_max).all()
    exported = torch.jit.script(DeterministicStandingActor(policy.actor))
    path = tmp_path / "actor.ts"
    exported.save(str(path))
    loaded = torch.jit.load(str(path))
    assert torch.equal(loaded(actor), policy.deterministic(actor))
    assert torch.equal(loaded(actor[:1]), policy.deterministic(actor[:1]))
