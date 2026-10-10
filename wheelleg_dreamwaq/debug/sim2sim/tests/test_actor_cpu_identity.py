from __future__ import annotations

import numpy as np
import pytest
import torch

from debug.sim2sim.evaluate_actor_cpu import evaluate_actor_cpu
from debug.sim2sim.trace_schema import load_npz, save_npz
from debug.sim2sim.verify_actor_cpu_identity import evaluate_same_input


class _TinyActor(torch.nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return observation[:, :6] * 0.5


def test_same_serialized_input_has_exact_cpu_output(tmp_path) -> None:
    actor_path = tmp_path / "actor.ts"
    example = torch.zeros((1, 25), dtype=torch.float32)
    torch.jit.trace(_TinyActor(), example).save(str(actor_path))
    actor_input = np.arange(25, dtype=np.float32)
    first, second = evaluate_same_input(actor_path, actor_input)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, actor_input[:6] * 0.5)


def test_actor_identity_rejects_non_float32_input(tmp_path) -> None:
    with pytest.raises(ValueError, match="float32"):
        evaluate_same_input(tmp_path / "unused.ts", np.zeros(25, dtype=np.float64))


def test_actor_cpu_evaluator_writes_auditable_output(tmp_path) -> None:
    actor_path = tmp_path / "actor.ts"
    input_path = tmp_path / "input.npz"
    torch.jit.trace(_TinyActor(), torch.zeros((1, 25), dtype=torch.float32)).save(str(actor_path))
    save_npz(input_path, {"actor_obs_policy": np.arange(25, dtype=np.float32)})
    metadata = evaluate_actor_cpu(
        actor_path=actor_path,
        input_path=input_path,
        output_directory=tmp_path / "evaluation",
    )
    output = load_npz(tmp_path / "evaluation/actor_output.npz")["actor_output"]
    np.testing.assert_array_equal(output, np.arange(6, dtype=np.float32) * 0.5)
    assert metadata["device"] == "cpu"
    assert (tmp_path / "evaluation/file_hashes.json").is_file()
