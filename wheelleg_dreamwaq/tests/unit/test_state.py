from __future__ import annotations

from dataclasses import fields

import torch

from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.state import WheelLegState


def _state(offset: float) -> WheelLegState:
    values = {
        field.name: torch.full((3, 2), offset + index, dtype=torch.float32)
        for index, field in enumerate(fields(WheelLegState))
    }
    return WheelLegState(**values)


def test_replace_rows_preserves_non_reset_transition_rows() -> None:
    transition = _state(0.0)
    reset = _state(100.0)

    merged = transition.replace_rows(reset, torch.tensor([1], dtype=torch.long))

    for field in fields(WheelLegState):
        transition_value = getattr(transition, field.name)
        reset_value = getattr(reset, field.name)
        merged_value = getattr(merged, field.name)
        assert torch.equal(merged_value[[0, 2]], transition_value[[0, 2]])
        assert torch.equal(merged_value[1], reset_value[1])
        assert merged_value.data_ptr() != transition_value.data_ptr()
