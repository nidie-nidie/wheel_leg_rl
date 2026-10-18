from __future__ import annotations

import torch

from debug.sim2sim.constraint_finetune_env import QddotWindowTracker


def test_reset_clears_selected_environment() -> None:
    tracker = QddotWindowTracker(num_envs=3, window_steps=4, device="cpu")
    tracker.update(torch.ones(3, 6))
    tracker.reset(torch.tensor([1]))
    assert tracker.count.tolist() == [1, 0, 1]


def test_metric_uses_only_first_four_controlled_joints() -> None:
    tracker = QddotWindowTracker(num_envs=1, window_steps=2, device="cpu")
    tracker.update(torch.tensor([[3.0, 4.0, 0.0, 0.0, 100.0, 100.0]]))
    assert torch.allclose(tracker.rms, torch.tensor([2.5]), atol=1.0e-6)


def test_window_uses_only_available_samples() -> None:
    tracker = QddotWindowTracker(num_envs=1, window_steps=4, device="cpu")
    tracker.update(torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(tracker.rms, torch.tensor([1.0]))


