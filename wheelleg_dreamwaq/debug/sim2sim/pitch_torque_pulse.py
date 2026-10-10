from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BasePitchTorquePulse:
    """End-exclusive body-local pitch torque expressed in ControlFrameV1."""

    magnitude_nm: float
    start_tick: int
    end_tick: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.magnitude_nm):
            raise ValueError("magnitude_nm must be finite")
        if self.magnitude_nm == 0.0:
            if self.start_tick != 0 or self.end_tick != 0:
                raise ValueError("A disabled pulse must use the empty interval [0, 0)")
            return
        if self.start_tick < 0 or self.end_tick <= self.start_tick:
            raise ValueError("A non-zero pulse requires 0 <= start_tick < end_tick")

    @classmethod
    def disabled(cls) -> "BasePitchTorquePulse":
        return cls(magnitude_nm=0.0, start_tick=0, end_tick=0)

    @property
    def enabled(self) -> bool:
        return self.magnitude_nm != 0.0

    @property
    def payload(self) -> dict[str, float | int]:
        return asdict(self)

    def validate_control_ticks(self, control_ticks: int) -> None:
        if control_ticks <= 0:
            raise ValueError("control_ticks must be positive")
        if self.enabled and self.end_tick > control_ticks:
            raise ValueError(
                f"Pitch torque interval ends at {self.end_tick}, outside the "
                f"{control_ticks}-tick collection window"
            )

    def value_at(self, tick: int) -> float:
        if tick < 0:
            raise ValueError("tick must be non-negative")
        return self.magnitude_nm if self.start_tick <= tick < self.end_tick else 0.0

    def control_vector(self, tick: int) -> np.ndarray:
        return np.asarray((0.0, self.value_at(tick), 0.0), dtype=np.float64)
