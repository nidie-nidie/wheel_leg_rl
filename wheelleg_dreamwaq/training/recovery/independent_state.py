from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class StandingGateConfig:
    control_dt_s: float = .02
    window_steps: int = 10
    qualification_steps: int = 50
    train_hold_steps: int = 100
    eval_hold_steps: int = 500
    getup_deadline_steps: int = 400
    train_horizon_steps: int = 500
    eval_horizon_steps: int = 900
    train_displacement_m: float = .10
    eval_displacement_m: float = .20
    tilt_rad: float = .10
    height_min_m: float = .185
    height_max_m: float = .215
    leg_error_rad: float = .10
    angular_speed_rad_s: float = .80
    leg_speed_rad_s: float = 1.50
    wheel_speed_rad_s: float = 6.
    horizontal_speed_m_s: float = .05
    wheel_min_weight_fraction: float = .10
    wheel_total_min_weight_fraction: float = .80
    wheel_total_max_weight_fraction: float = 1.20
    nonwheel_mean_max_weight_fraction: float = .01
    nonwheel_current_max_weight_fraction: float = .05

    def validate(self):
        if asdict(self) != asdict(StandingGateConfig()):
            raise ValueError("独立站立门限必须匹配已复核的固定契约")
        return self


REASON_LABELS = {0: "none", 1: "physical_failure", 2: "getup_timeout", 3: "hold_interrupted",
                 4: "horizon_timeout", 5: "success", 6: "nonfinite"}
REASON_CODES = {value: key for key, value in REASON_LABELS.items()}


class ContactWindow:
    def __init__(self, num_envs, device, length=10):
        if length != 10 or num_envs < 1:
            raise ValueError("独立站立接触窗口固定为 10 步")
        self.length, self.num_envs = length, num_envs
        self.buffer = torch.zeros((length, num_envs, 3), device=device)
        self.sum = torch.zeros((num_envs, 3), device=device)
        self.count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.cursor = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._ids = torch.arange(num_envs, device=device)

    def push(self, wheel_forces, nonwheel):
        if wheel_forces.shape != (self.num_envs, 2) or nonwheel.shape != (self.num_envs,):
            raise ValueError("真实接触力字段形状无效")
        values = torch.cat((wheel_forces, nonwheel[:, None]), dim=1)
        if not torch.isfinite(values).all() or torch.any(values < 0.):
            raise FloatingPointError("真实法向接触力必须有限且非负")
        self.buffer[self.cursor, self._ids] = values
        self.sum.copy_(self.buffer.sum(dim=0))
        self.count.add_(1).clamp_max_(self.length)
        self.cursor = (self.cursor + 1) % self.length

    @property
    def means(self):
        return self.sum / self.count.clamp_min(1)[:, None]

    @property
    def wheel_mean(self):
        return self.means[:, :2]

    @property
    def nonwheel_mean(self):
        return self.means[:, 2]

    def reset(self, env_ids):
        self.buffer[:, env_ids] = 0.
        self.sum[env_ids] = 0.
        self.count[env_ids] = 0
        self.cursor[env_ids] = 0

    def state_dict(self):
        return {"length": self.length, "num_envs": self.num_envs,
                **{name: getattr(self, name).clone() for name in ("buffer", "sum", "count", "cursor")}}

    def load_state_dict(self, state):
        if state["length"] != self.length or state["num_envs"] != self.num_envs:
            raise ValueError("接触窗口恢复维度不匹配")
        for name in ("buffer", "sum", "count", "cursor"):
            source, target = state[name], getattr(self, name)
            if source.shape != target.shape or source.dtype != target.dtype or not torch.isfinite(source).all():
                raise ValueError(f"接触窗口恢复字段 {name} 无效")
            target.copy_(source.to(target.device))
        if (torch.any(self.count < 0) or torch.any(self.count > self.length)
                or torch.any(self.cursor < 0) or torch.any(self.cursor >= self.length)
                or torch.any(self.buffer < 0.)):
            raise ValueError("接触窗口恢复计数或力无效")
        torch.testing.assert_close(self.sum, self.buffer.sum(dim=0), atol=1e-3, rtol=1e-5)


def evaluate_standing_gate(*, q, dq, omega, gravity, linear, height, q_reference,
                           wheel_mean, nonwheel_mean, nonwheel_current,
                           contact_count, weight_n, physical_failure, config=None):
    config = StandingGateConfig() if config is None else config.validate()
    weight = torch.as_tensor(weight_n, dtype=q.dtype, device=q.device)
    if (weight.ndim > 1 or (weight.ndim == 1 and weight.shape != (q.shape[0],))
            or not torch.isfinite(weight).all() or torch.any(weight <= 0.)):
        raise ValueError("站立物理门控重量无效")
    fields = (q, dq, omega, gravity, linear, height, q_reference,
              wheel_mean, nonwheel_mean, nonwheel_current)
    finite = torch.ones(q.shape[0], device=q.device, dtype=torch.bool)
    for value in fields:
        finite &= torch.isfinite(value).reshape(q.shape[0], -1).all(dim=1)
    total = wheel_mean.sum(dim=1)
    failures = {
        "tilt": torch.acos((-gravity[:, 2]).clamp(-1., 1.)) > config.tilt_rad,
        "height": (height < config.height_min_m) | (height > config.height_max_m),
        "leg_pose": (q - q_reference).abs().amax(dim=1) > config.leg_error_rad,
        "angular_velocity": omega.norm(dim=1) > config.angular_speed_rad_s,
        "leg_velocity": dq[:, :4].abs().amax(dim=1) > config.leg_speed_rad_s,
        "wheel_velocity": dq[:, 4:].abs().amax(dim=1) > config.wheel_speed_rad_s,
        "horizontal_velocity": linear[:, :2].norm(dim=1) > config.horizontal_speed_m_s,
        "wheel_support": wheel_mean.amin(dim=1) < config.wheel_min_weight_fraction * weight,
        "total_support": (total < config.wheel_total_min_weight_fraction * weight)
                         | (total > config.wheel_total_max_weight_fraction * weight),
        "nonwheel_mean": nonwheel_mean > config.nonwheel_mean_max_weight_fraction * weight,
        "nonwheel_current": nonwheel_current > config.nonwheel_current_max_weight_fraction * weight,
        "contact_window": contact_count < config.window_steps,
        "physical_failure": physical_failure.bool(),
        "nonfinite": ~finite,
    }
    qualified = ~torch.stack(list(failures.values())).any(dim=0)
    return qualified, failures


@dataclass(frozen=True)
class StandingStep:
    qualified: torch.Tensor
    success: torch.Tensor
    failure: torch.Tensor
    timeout: torch.Tensor
    reason_codes: torch.Tensor
    gate_failures: dict


class StandingTask:
    GETUP = 0
    HOLD = 1

    def __init__(self, num_envs, device, reference, weight_n, *, mode="train", config=None):
        self.config = (StandingGateConfig() if config is None else config).validate()
        if mode not in ("train", "eval"):
            raise ValueError("站立任务模式必须为 train 或 eval")
        self.mode, self.reference, self.weight_n = mode, reference, weight_n
        self.num_envs = num_envs
        self.contact_window = ContactWindow(num_envs, device, self.config.window_steps)
        self.qualification_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.hold_count = torch.zeros_like(self.qualification_count)
        self.phase = torch.zeros_like(self.qualification_count)
        self.elapsed_steps = torch.zeros_like(self.qualification_count)
        self.hold_origin = torch.zeros((num_envs, 2), device=device)
        self.position_delta = torch.zeros_like(self.hold_origin)
        self.terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.physical_reason = torch.zeros_like(self.qualification_count)
        self.first_qualified_step = torch.full_like(self.qualification_count, -1)
        self.hold_interruptions = torch.zeros_like(self.qualification_count)

    @property
    def hold_steps(self):
        return self.config.train_hold_steps if self.mode == "train" else self.config.eval_hold_steps

    @property
    def horizon_steps(self):
        return self.config.train_horizon_steps if self.mode == "train" else self.config.eval_horizon_steps

    @property
    def displacement_limit(self):
        return self.config.train_displacement_m if self.mode == "train" else self.config.eval_displacement_m

    @property
    def qualification_progress(self):
        return (self.qualification_count.float() / self.config.qualification_steps).clamp(0., 1.)

    @property
    def hold_progress(self):
        return (self.hold_count.float() / self.hold_steps).clamp(0., 1.)

    @property
    def remaining(self):
        return (1. - self.elapsed_steps.float() / self.horizon_steps).clamp(0., 1.)

    def reset(self, env_ids):
        self.contact_window.reset(env_ids)
        for value in (self.qualification_count, self.hold_count, self.phase, self.elapsed_steps,
                      self.hold_origin, self.position_delta, self.physical_reason, self.hold_interruptions):
            value[env_ids] = 0
        self.terminated[env_ids] = False
        self.first_qualified_step[env_ids] = -1

    def state_dict(self):
        names = ("qualification_count", "hold_count", "phase", "elapsed_steps", "hold_origin",
                 "position_delta", "terminated", "physical_reason", "first_qualified_step", "hold_interruptions")
        return {"schema_version": "IndependentStandingTaskStateV1", "mode": self.mode,
                "gate_config": asdict(self.config), "reference": self.reference.identity,
                "weight_n": torch.as_tensor(self.weight_n).cpu(),
                "contact_window": self.contact_window.state_dict(),
                **{name: getattr(self, name).clone() for name in names}}

    def load_state_dict(self, state):
        if (state["schema_version"] != "IndependentStandingTaskStateV1" or state["mode"] != self.mode
                or state["gate_config"] != asdict(self.config) or state["reference"] != self.reference.identity):
            raise ValueError("独立站立任务恢复身份不匹配")
        torch.testing.assert_close(torch.as_tensor(state["weight_n"]).cpu(),
                                   torch.as_tensor(self.weight_n).cpu(), rtol=0., atol=0.)
        self.contact_window.load_state_dict(state["contact_window"])
        names = ("qualification_count", "hold_count", "phase", "elapsed_steps", "hold_origin",
                 "position_delta", "terminated", "physical_reason", "first_qualified_step", "hold_interruptions")
        for name in names:
            source, target = state[name], getattr(self, name)
            if source.shape != target.shape or source.dtype != target.dtype or not torch.isfinite(source).all():
                raise ValueError(f"独立站立恢复字段 {name} 无效")
            target.copy_(source.to(target.device))
        if (torch.any((self.phase != self.GETUP) & (self.phase != self.HOLD))
                or torch.any(self.qualification_count < 0)
                or torch.any(self.qualification_count > self.config.qualification_steps)
                or torch.any(self.hold_count < 0) or torch.any(self.hold_count > self.hold_steps)
                or torch.any(self.elapsed_steps < 0) or torch.any(self.elapsed_steps > self.horizon_steps)):
            raise ValueError("独立站立恢复阶段或连续计数无效")

    def step(self, *, q, dq, omega, gravity, linear, height, wheel_forces, nonwheel_force,
             world_xy, physical_failure, physical_reason=None):
        if q.shape != (self.num_envs, 4) or world_xy.shape != (self.num_envs, 2):
            raise ValueError("站立任务角度或位置形状无效")
        self.contact_window.push(wheel_forces, nonwheel_force)
        match = self.reference.nearest(q, height)
        qualified, failures = evaluate_standing_gate(
            q=q, dq=dq, omega=omega, gravity=gravity, linear=linear, height=height,
            q_reference=match.q, wheel_mean=self.contact_window.wheel_mean,
            nonwheel_mean=self.contact_window.nonwheel_mean, nonwheel_current=nonwheel_force,
            contact_count=self.contact_window.count, weight_n=self.weight_n,
            physical_failure=physical_failure, config=self.config)
        active = ~self.terminated
        self.elapsed_steps += active.long()
        was_hold = active & (self.phase == self.HOLD)
        self.position_delta[was_hold] = world_xy[was_hold] - self.hold_origin[was_hold]
        displacement_bad = was_hold & (self.position_delta.norm(dim=1) > self.displacement_limit)
        failures["displacement"] = displacement_bad
        qualified &= ~displacement_bad
        physical = active & physical_failure.bool()
        if physical_reason is not None:
            self.physical_reason[physical] = physical_reason[physical]
        broken = was_hold & ~qualified & ~physical
        self.hold_interruptions += broken.long()
        self.phase[broken] = self.GETUP
        self.hold_count[broken] = 0
        self.qualification_count[broken] = 0
        self.hold_origin[broken] = 0.
        self.position_delta[broken] = 0.
        getup = active & (self.phase == self.GETUP) & ~physical & ~broken
        self.qualification_count[getup & qualified] += 1
        self.qualification_count[getup & ~qualified] = 0
        entered = getup & (self.qualification_count >= self.config.qualification_steps)
        entered &= self.elapsed_steps <= self.config.getup_deadline_steps
        self.phase[entered] = self.HOLD
        self.hold_origin[entered] = world_xy[entered]
        first = entered & (self.first_qualified_step < 0)
        self.first_qualified_step[first] = self.elapsed_steps[first]
        continuing = was_hold & qualified & ~physical
        self.hold_count[continuing] += 1
        success = continuing & (self.hold_count >= self.hold_steps)
        hold_failed = broken & ((self.mode == "eval") | (self.elapsed_steps > self.config.getup_deadline_steps))
        getup_timeout = active & (self.phase == self.GETUP)
        getup_timeout &= self.elapsed_steps >= self.config.getup_deadline_steps
        horizon_timeout = active & (self.elapsed_steps >= self.horizon_steps) & ~success
        timeout = (hold_failed | getup_timeout | horizon_timeout) & ~physical
        reason = torch.zeros_like(self.elapsed_steps)
        reason[horizon_timeout] = REASON_CODES["horizon_timeout"]
        reason[getup_timeout] = REASON_CODES["getup_timeout"]
        reason[hold_failed] = REASON_CODES["hold_interrupted"]
        reason[success] = REASON_CODES["success"]
        reason[physical] = REASON_CODES["physical_failure"]
        success &= ~physical
        self.terminated |= success | physical | timeout
        return StandingStep(qualified & active, success, physical | timeout, timeout, reason, failures)
