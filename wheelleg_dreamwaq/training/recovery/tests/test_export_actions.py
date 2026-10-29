from __future__ import annotations

import unittest

import torch
from torch import nn

from recovery.contracts import RecoveryContractV2
from recovery.export import RecoveryMeanActor, _build_mlp, _load_mean_actor


class ExportActionTests(unittest.TestCase):
    def test_eager_and_script_return_full_actor_means_for_all_batch_sizes(self):
        actor = nn.Sequential(nn.Linear(110, 6))
        with torch.no_grad():
            actor[0].weight.fill_(0.25)
            actor[0].bias.copy_(torch.tensor([11., -12., 13., -14., 15., -16.]))
        exported = RecoveryMeanActor(actor).eval()
        scripted = torch.jit.script(exported).eval()
        observations = torch.arange(32 * 110, dtype=torch.float32).reshape(32, 110) / 1000
        with torch.inference_mode():
            for size in (1, 3, 32):
                expected = actor(observations[:size])
                self.assertGreater(float(expected.abs().max()), 10.0)
                torch.testing.assert_close(exported(observations[:size]), expected, rtol=0, atol=0)
                torch.testing.assert_close(scripted(observations[:size]), expected, rtol=0, atol=0)

    def test_loaded_actorcritic_parameters_preserve_large_means(self):
        contract = RecoveryContractV2()
        actor = _build_mlp(contract.actor_input_dim, 6, contract.actor_hidden_dims)
        critic = _build_mlp(contract.critic_input_dim, 1, contract.critic_hidden_dims)
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.zero_()
            actor[-1].bias.copy_(torch.tensor([11., -12., 13., -14., 15., -16.]))
        state = {f"actor.{name}": value for name, value in actor.state_dict().items()}
        state.update({f"critic.{name}": value for name, value in critic.state_dict().items()})
        state["log_std"] = torch.zeros(6)
        exported = _load_mean_actor(state, contract)
        scripted = torch.jit.script(exported).eval()
        with torch.inference_mode():
            for size in (1, 3, 32):
                observations = torch.ones(size, contract.actor_input_dim)
                expected = actor(observations)
                self.assertGreater(float(expected.abs().min()), 10.0)
                torch.testing.assert_close(exported(observations), expected, rtol=0, atol=0)
                torch.testing.assert_close(scripted(observations), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
