from __future__ import annotations

import torch


def all_finite(*values: torch.Tensor) -> torch.Tensor:
    if not values:
        raise ValueError("有限值检查必须提供张量")
    return torch.stack([torch.isfinite(value).all() for value in values]).all()
