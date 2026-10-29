import math
from dataclasses import dataclass


def _finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须为有限数值")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须为有限数值")
    return number


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class HeadingCommand:
    target_rad: float
    error_rad: float
    command_yaw_rate: float
    correction_yaw_rate: float
    mode: str
    saturated: bool = False


class HeadingController:
    def __init__(self, *, kp=0.5, kd=0.0, max_yaw_rate=1.0, enabled=True):
        self.kp = _finite(kp, "Kp")
        self.kd = _finite(kd, "Kd")
        self.max_yaw_rate = _finite(max_yaw_rate, "偏航速度上限")
        if self.kp < 0 or self.kd < 0 or self.max_yaw_rate <= 0:
            raise ValueError("Kp/Kd 必须非负，偏航速度上限必须为正数")
        if not isinstance(enabled, bool):
            raise ValueError("航向开关必须为布尔值")
        self.enabled = enabled
        self.target_rad = None
        self._was_turning = False

    def reset(self, yaw):
        self.target_rad = _wrap(_finite(yaw, "航向角"))
        self._was_turning = False

    def set_enabled(self, enabled, yaw):
        if not isinstance(enabled, bool):
            raise ValueError("航向开关必须为布尔值")
        self.reset(yaw)
        self.enabled = enabled

    def update(self, yaw, body_yaw_rate, requested_yaw_rate):
        yaw = _wrap(_finite(yaw, "航向角"))
        body_yaw_rate = _finite(body_yaw_rate, "机身 z 轴角速度")
        requested = _finite(requested_yaw_rate, "用户偏航速度")
        if abs(requested) > self.max_yaw_rate:
            raise ValueError("用户偏航速度超出模型命令范围")
        if self.target_rad is None:
            raise RuntimeError("航向控制器需要先 reset")
        if not self.enabled or requested != 0.0:
            self.target_rad = yaw
            self._was_turning = self.enabled and requested != 0.0
            return HeadingCommand(yaw, 0.0, requested, 0.0,
                                  "turn" if self.enabled else "disabled")
        if self._was_turning:
            self.reset(yaw)
        error = _wrap(self.target_rad - yaw)
        # D 项采用机身角速度阻尼；默认只使用航向比例反馈。
        raw = _finite(self.kp * error - self.kd * body_yaw_rate, "航向控制输出")
        command = max(-self.max_yaw_rate, min(self.max_yaw_rate, raw))
        return HeadingCommand(self.target_rad, error, command, command, "hold",
                              abs(raw) > self.max_yaw_rate)
