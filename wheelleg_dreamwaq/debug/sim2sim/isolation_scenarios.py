from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


ScenarioKind = Literal["fixed_base", "suspended", "grounded_pitch"]


@dataclass(frozen=True)
class IsolationScenario:
    kind: ScenarioKind
    control_ticks: int
    model_relative: str
    has_ground_contact: bool
    initial_pitch_deg: float = 0.0
    wheel_common_action: float = 0.0
    pulse_channel: int | None = None
    wheel_pulse_mode: Literal["common", "differential"] | None = None
    pulse_amplitude: float = 0.0
    pulse_start_tick: int = 0
    pulse_end_tick: int = 0

    def __post_init__(self) -> None:
        if self.control_ticks <= 0:
            raise ValueError("control_ticks must be positive")
        if self.kind == "grounded_pitch":
            if not 0.0 < abs(self.initial_pitch_deg) <= 30.0:
                raise ValueError("initial_pitch_deg must be non-zero and within +/-30 degrees")
            if not -1.0 <= self.wheel_common_action <= 1.0:
                raise ValueError("wheel_common_action must be within [-1, 1]")
            if self.pulse_channel is not None or self.wheel_pulse_mode is not None:
                raise ValueError("grounded_pitch does not use pulse selectors")
        else:
            if (self.pulse_channel is None) == (self.wheel_pulse_mode is None):
                raise ValueError("exactly one pulse selector must be provided")
            if self.pulse_channel is not None and not 0 <= self.pulse_channel < 6:
                raise ValueError("channel must be in [0, 5]")
            if self.wheel_pulse_mode not in {None, "common", "differential"}:
                raise ValueError("wheel pulse mode must be common or differential")
            if not -1.0 <= self.pulse_amplitude <= 1.0:
                raise ValueError("pulse amplitude must be within [-1, 1]")
            if not 0 <= self.pulse_start_tick < self.pulse_end_tick <= self.control_ticks:
                raise ValueError("pulse ticks must satisfy 0 <= start < end <= control_ticks")

    @classmethod
    def fixed_base_pulse(
        cls,
        *,
        control_ticks: int,
        channel: int,
        amplitude: float = 0.1,
        pulse_start_tick: int = 5,
        pulse_end_tick: int = 20,
    ) -> "IsolationScenario":
        return cls(
            kind="fixed_base",
            control_ticks=control_ticks,
            model_relative="debug/sim2sim/models/wheel_leg_urdf4_fixed_base_debug.xml",
            has_ground_contact=False,
            pulse_channel=channel,
            pulse_amplitude=amplitude,
            pulse_start_tick=pulse_start_tick,
            pulse_end_tick=pulse_end_tick,
        )

    @classmethod
    def suspended_pulse(
        cls,
        *,
        control_ticks: int,
        channel: int,
        amplitude: float = 0.1,
        pulse_start_tick: int = 5,
        pulse_end_tick: int = 20,
    ) -> "IsolationScenario":
        return cls(
            kind="suspended",
            control_ticks=control_ticks,
            model_relative="debug/sim2sim/models/wheel_leg_urdf4_suspended_debug.xml",
            has_ground_contact=False,
            pulse_channel=channel,
            pulse_amplitude=amplitude,
            pulse_start_tick=pulse_start_tick,
            pulse_end_tick=pulse_end_tick,
        )

    @classmethod
    def grounded_pitch(
        cls,
        *,
        control_ticks: int,
        initial_pitch_deg: float,
        wheel_common_action: float,
    ) -> "IsolationScenario":
        return cls(
            kind="grounded_pitch",
            control_ticks=control_ticks,
            model_relative="sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml",
            has_ground_contact=True,
            initial_pitch_deg=initial_pitch_deg,
            wheel_common_action=wheel_common_action,
        )

    @classmethod
    def suspended_wheel_pulse(
        cls,
        *,
        control_ticks: int,
        mode: Literal["common", "differential"],
        amplitude: float = 0.1,
        pulse_start_tick: int = 5,
        pulse_end_tick: int = 20,
    ) -> "IsolationScenario":
        return cls(
            kind="suspended",
            control_ticks=control_ticks,
            model_relative="debug/sim2sim/models/wheel_leg_urdf4_suspended_debug.xml",
            has_ground_contact=False,
            wheel_pulse_mode=mode,
            pulse_amplitude=amplitude,
            pulse_start_tick=pulse_start_tick,
            pulse_end_tick=pulse_end_tick,
        )

    @property
    def payload(self) -> dict:
        return asdict(self)

    def action_sequence(self) -> np.ndarray:
        sequence = np.zeros((self.control_ticks, 6), dtype=np.float32)
        if self.kind == "grounded_pitch":
            sequence[:, 4:6] = np.float32(self.wheel_common_action)
        elif self.wheel_pulse_mode == "common":
            sequence[self.pulse_start_tick : self.pulse_end_tick, 4:6] = np.float32(
                self.pulse_amplitude
            )
        elif self.wheel_pulse_mode == "differential":
            sequence[self.pulse_start_tick : self.pulse_end_tick, 4] = np.float32(
                self.pulse_amplitude
            )
            sequence[self.pulse_start_tick : self.pulse_end_tick, 5] = np.float32(
                -self.pulse_amplitude
            )
        else:
            sequence[self.pulse_start_tick : self.pulse_end_tick, self.pulse_channel] = np.float32(
                self.pulse_amplitude
            )
        return sequence
