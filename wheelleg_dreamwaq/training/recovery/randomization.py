from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any

import torch

from .contracts import RecoveryContractV2, stable_hash
from .observations import ActorNoise


@dataclass(frozen=True)
class RecoveryRandomizationV1:
    name: str = "RecoveryTrainingRandomizationV1"
    enabled: bool = True
    friction_range: tuple[float, float] = (0.6, 1.4)
    friction_bucket_count: int = 64
    kp_scale_range: tuple[float, float] = (0.95, 1.05)
    kd_scale_range: tuple[float, float] = (0.95, 1.05)
    effort_limit_scale_range: tuple[float, float] = (0.95, 1.0)
    leg_reference_offset_range_rad: tuple[float, float] = (0.0, 0.0)
    reset_velocity_perturbation_enabled: bool = False
    angular_velocity_noise_rad_s: float = 0.20
    projected_gravity_noise: float = 0.05
    leg_position_noise_rad: float = 0.02
    joint_velocity_noise_rad_s: float = 1.50
    mass_scale: float = 1.0
    inertia_scale: float = 1.0
    com_offset_m: tuple[float, ...] = (0.0, 0.0, 0.0)
    action_delay_steps: int = 0
    measurement_delay_steps: int = 0
    dropout_probability: float = 0.0
    imu_bias: tuple[float, ...] = (0.0, 0.0, 0.0)
    joint_friction_randomization_enabled: bool = False
    passive_damping_randomization_enabled: bool = False
    armature_randomization_enabled: bool = False
    ground_tilt_randomization_enabled: bool = False
    external_push_enabled: bool = False

    def __post_init__(self) -> None:
        stable_hash(asdict(self))
        for name in ("friction_range", "kp_scale_range", "kd_scale_range", "effort_limit_scale_range"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or bounds[0] <= 0 or bounds[0] > bounds[1]:
                raise ValueError(f"随机化范围无效: {name}")
        if self.effort_limit_scale_range[1] > 1 or self.friction_bucket_count != 64:
            raise ValueError("起身随机化不得提高物理力矩上限，摩擦桶数固定为64")
        if tuple(self.leg_reference_offset_range_rad) != (0, 0) or self.reset_velocity_perturbation_enabled:
            raise ValueError("姿态库参考固定，禁止额外覆盖动态初态速度")
        if (self.mass_scale != 1 or self.inertia_scale != 1 or any(self.com_offset_m)
                or self.action_delay_steps or self.measurement_delay_steps or self.dropout_probability
                or any(self.imu_bias) or self.joint_friction_randomization_enabled
                or self.passive_damping_randomization_enabled or self.armature_randomization_enabled
                or self.ground_tilt_randomization_enabled or self.external_push_enabled):
            raise ValueError("当前契约关闭的动力学或测量注入不能静默启用")
        if any(value < 0 for value in (self.angular_velocity_noise_rad_s, self.projected_gravity_noise,
                                       self.leg_position_noise_rad, self.joint_velocity_noise_rad_s)):
            raise ValueError("观测增强幅值无效")


@dataclass(frozen=True)
class ParameterPlan:
    friction: float
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    effort_limits: tuple[float, ...]
    q_reference: tuple[float, ...]

    def __post_init__(self) -> None:
        stable_hash(asdict(self))
        if len(self.q_reference) != 4 or any(len(v) != 6 for v in (self.kp, self.kd, self.effort_limits)):
            raise ValueError("实际执行器计划维度无效")
        if self.friction <= 0 or any(v < 0 for v in (*self.kp, *self.kd)):
            raise ValueError("实际摩擦或增益无效")
        if any(v <= 0 or v > upper for v, upper in zip(self.effort_limits, (18, 18, 18, 18, 9, 9))):
            raise ValueError("实际力矩能力超出物理范围")
        if self.kp[4] != 0 or self.kp[5] != 0:
            raise ValueError("轮子必须使用速度反馈")

    @property
    def sha256(self) -> str:
        return stable_hash(asdict(self))


def derive_seed(master_seed: int, stream_name: str) -> int:
    if not isinstance(master_seed, int) or master_seed < 0 or not stream_name:
        raise ValueError("随机种子或流名称无效")
    payload = f"RecoverySeedV1|{master_seed}|{stream_name}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)
    return value or 1


class RecoveryRandomization:
    def __init__(self, num_envs: int, master_seed: int, device: str | torch.device = "cpu",
                 profile: RecoveryRandomizationV1 | None = None, q_nominal: Any = None) -> None:
        if num_envs <= 0:
            raise ValueError("随机化环境数量无效")
        self.profile = profile or RecoveryRandomizationV1()
        self.num_envs, self.master_seed, self.device = num_envs, master_seed, torch.device(device)
        names = ["material_rng", "actuator_gain_rng", "effort_limit_rng", "observation_noise_rng", "eval_case_rng"]
        names += [f"pose_rng/{index}" for index in range(num_envs)]
        self._generators = {}
        for name in names:
            stream_device = self.device if name == "observation_noise_rng" else torch.device("cpu")
            self._generators[name] = torch.Generator(device=stream_device).manual_seed(derive_seed(master_seed, name))
        cfg = RecoveryContractV2()
        nominal = cfg.q_nominal if q_nominal is None else tuple(float(v) for v in q_nominal)
        enabled = self.profile.enabled
        self.friction_buckets = self._uniform((64,), self.profile.friction_range, "material_rng") if enabled else torch.ones(1)
        self.friction_bucket_ids = (torch.randint(0, 64, (num_envs,), generator=self._generators["material_rng"])
                                    if enabled else torch.zeros(num_envs, dtype=torch.long))
        kp_scale = self._uniform((num_envs, 6), self.profile.kp_scale_range, "actuator_gain_rng") if enabled else torch.ones(num_envs, 6)
        kd_scale = self._uniform((num_envs, 6), self.profile.kd_scale_range, "actuator_gain_rng") if enabled else torch.ones(num_envs, 6)
        effort_scale = self._uniform((num_envs, 6), self.profile.effort_limit_scale_range, "effort_limit_rng") if enabled else torch.ones(num_envs, 6)
        kp = kp_scale * torch.tensor([cfg.leg_kp] * 4 + [0.0, 0.0])
        kd = kd_scale * torch.tensor([cfg.leg_kd] * 4 + [cfg.wheel_kd] * 2)
        effort = effort_scale * torch.tensor(cfg.effort_limits)
        if enabled:
            self.plans = [ParameterPlan(float(self.friction_buckets[self.friction_bucket_ids[i]]),
                                        tuple(kp[i].tolist()), tuple(kd[i].tolist()), tuple(effort[i].tolist()), nominal)
                          for i in range(num_envs)]
        else:
            # 名义参数身份使用契约数值；实际写入后再按仿真器精度读回检查。
            self.plans = [ParameterPlan(1.0, (cfg.leg_kp,) * 4 + (0.0, 0.0),
                                        (cfg.leg_kd,) * 4 + (cfg.wheel_kd,) * 2,
                                        cfg.effort_limits, nominal) for _ in range(num_envs)]
        self._refresh_tensors()

    def _uniform(self, shape: tuple[int, ...], bounds: tuple[float, float], stream: str) -> torch.Tensor:
        low, high = bounds
        if low == high:
            return torch.full(shape, low)
        return low + (high - low) * torch.rand(shape, generator=self._generators[stream], dtype=torch.float32)

    def _refresh_tensors(self) -> None:
        for attr, key in (("q_reference", "q_reference"), ("realized_stiffness", "kp"),
                          ("realized_damping", "kd"), ("realized_effort_limit", "effort_limits")):
            setattr(self, attr, torch.tensor([getattr(plan, key) for plan in self.plans], device=self.device))
        self.friction = torch.tensor([plan.friction for plan in self.plans], device=self.device)

    def plan(self, env_id: int) -> ParameterPlan:
        return self.plans[env_id]

    def pose_generator(self, env_id: int) -> torch.Generator:
        return self._generators[f"pose_rng/{env_id}"]

    def sample_actor_noise(self, count: int) -> ActorNoise | None:
        if not self.profile.enabled:
            return None
        if count <= 0:
            raise ValueError("增强采样数量必须为正")
        def noise(size: int, amplitude: float) -> torch.Tensor:
            if amplitude == 0:
                return torch.zeros(count, size, device=self.device)
            return (torch.rand(count, size, generator=self._generators["observation_noise_rng"],
                                device=self.device) * 2 - 1) * amplitude
        return ActorNoise(noise(3, self.profile.angular_velocity_noise_rad_s),
                          noise(3, self.profile.projected_gravity_noise),
                          noise(4, self.profile.leg_position_noise_rad),
                          noise(6, self.profile.joint_velocity_noise_rad_s))

    def manifest(self) -> dict[str, Any]:
        return {"schema_version": "RecoveryRandomizationV1", "seed_derivation_version": "RecoverySeedDerivationV1",
                "profile": asdict(self.profile), "profile_sha256": stable_hash(asdict(self.profile)),
                "num_envs": self.num_envs, "device": str(self.device), "master_seed": self.master_seed,
                "plans": [asdict(plan) for plan in self.plans],
                "parameter_plan_sha256": stable_hash([asdict(plan) for plan in self.plans]),
                "streams": {name: {"seed": derive_seed(self.master_seed, name), "device": str(generator.device)}
                            for name, generator in self._generators.items()}}

    def state_dict(self) -> dict[str, Any]:
        return {"manifest": self.manifest(), "rng_states": {name: generator.get_state().clone()
                                                              for name, generator in self._generators.items()}}

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload["manifest"] != self.manifest() or set(payload["rng_states"]) != set(self._generators):
            raise ValueError("恢复随机状态时参数计划、设备、种子或随机流不一致")
        for name, generator in self._generators.items():
            generator.set_state(payload["rng_states"][name].cpu())
