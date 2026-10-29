from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "recovery" / "artifacts"
CANONICAL_JOINT_ORDER = ("jIJ", "jIO", "jAB", "jAG", "jwheel_left", "jwheel_right")
CATEGORIES = ("front", "back", "left", "right", "supine", "near_upright")
WALKER_LEG_TARGET_LOWER = -1.0
WALKER_LEG_TARGET_UPPER = 1.0


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class RecoveryContractV2:
    schema_version: str = "RecoveryContractV2"
    control_dt_s: float = 0.02
    isaac_physics_dt_s: float = 0.005
    isaac_decimation: int = 4
    mujoco_physics_dt_s: float = 0.001
    mujoco_decimation: int = 20
    q_nominal: tuple[float, ...] = (-0.33367134, 0.33367134, -0.33367134, 0.33367134)
    leg_action_scale: float = 0.35
    wheel_action_scale: float = 25.0
    leg_kp: float = 120.0
    leg_kd: float = 4.0
    wheel_kd: float = 0.6
    effort_limits: tuple[float, ...] = (18.0, 18.0, 18.0, 18.0, 9.0, 9.0)
    history_length: int = 5
    actor_frame_dim: int = 22
    actor_input_dim: int = 110
    critic_input_dim: int = 34
    history_layout: str = "frame_major"
    angular_velocity_scale: float = 0.25
    angular_velocity_clip: float = 5.0
    projected_gravity_clip: float = 1.0
    leg_position_scale: float = 1.0
    leg_position_clip: float = 1.5
    joint_velocity_scale: float = 0.05
    joint_velocity_clip: float = 5.0
    root_linear_velocity_scale: float = 2.0
    root_linear_velocity_clip: float = 6.0
    nominal_base_height: float = 0.20
    height_command_span: float = 0.04
    base_height_clip: float = 5.0
    applied_torque_scale: float = 0.05
    applied_torque_clip: float = 5.0
    preserve_raw_previous_action: bool = True
    actor_hidden_dims: tuple[int, ...] = (256, 128, 64)
    critic_hidden_dims: tuple[int, ...] = (256, 128, 64)
    activation: str = "elu"
    gamma: float = 0.99
    max_recovery_steps: int = 400
    upright_weight: float = 2.0
    potential_weight: float = 2.0
    height_weight: float = 1.0
    stable_weight: float = 1.0
    torque_weight: float = 0.002
    leg_smooth_weight: float = 0.01
    wheel_smooth_weight: float = 0.01
    time_weight: float = 0.1
    success_reward: float = 10.0
    failure_reward: float = -10.0
    reward_gate_tilt_rad: float = 0.5
    height_target_m: float = 0.20
    height_sigma_m: float = 0.04
    pose_sigma_rad: float = 0.20
    angular_sigma_rad_s: float = 0.5
    wheel_sigma_rad_s: float = 2.0
    max_loop_error_m: float = 0.005

    def __post_init__(self) -> None:
        if (tuple(self.effort_limits) != (18.0, 18.0, 18.0, 18.0, 9.0, 9.0)
                or tuple(self.actor_hidden_dims) != (256, 128, 64)
                or tuple(self.critic_hidden_dims) != (256, 128, 64)):
            raise ValueError("V2的执行器名义能力和网络宽度固定，变化需要新契约版本")
        payload = asdict(self)
        stable_hash(payload)
        if self.schema_version != "RecoveryContractV2":
            raise ValueError("不支持的起身契约版本")
        if len(self.q_nominal) != 4 or len(self.effort_limits) != 6:
            raise ValueError("参考姿态或力矩范围维度错误")
        if self.actor_frame_dim != 22 or self.actor_input_dim != 110 or self.critic_input_dim != 34:
            raise ValueError("起身观测维度必须为22×5和34")
        if self.history_length != 5 or self.activation != "elu":
            raise ValueError("起身历史或网络激活不符合契约")
        for field in fields(self):
            if field.name not in ("q_nominal", "effort_limits", "actor_hidden_dims", "critic_hidden_dims"):
                if getattr(self, field.name) != field.default:
                    raise ValueError(f"固定契约字段变化需要更新设计和版本: {field.name}")
        if abs(self.isaac_physics_dt_s * self.isaac_decimation - self.control_dt_s) > 1e-12:
            raise ValueError("Isaac物理步长与控制周期不一致")
        if abs(self.mujoco_physics_dt_s * self.mujoco_decimation - self.control_dt_s) > 1e-12:
            raise ValueError("MuJoCo物理步长与控制周期不一致")
        if self.control_dt_s != 0.02 or self.max_recovery_steps != 400 or self.gamma != 0.99:
            raise ValueError("计时或折扣变化需要新的契约版本")
        if any(v <= 0 for v in self.effort_limits):
            raise ValueError("执行器范围无效")

    @property
    def sha256(self) -> str:
        return stable_hash(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RecoveryContractV2:
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise ValueError(f"起身契约字段不完整: missing={expected-set(payload)}, extra={set(payload)-expected}")
        values = dict(payload)
        for name in ("q_nominal", "effort_limits", "actor_hidden_dims", "critic_hidden_dims"):
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class GuidanceConfigV1:
    schema_version: str = "RecoveryForceGuidanceV1"
    force_enabled: bool = True
    torque_enabled: bool = False
    h_target: float = 0.20
    mu: float = 20.0
    max_force_body_weights: float = 0.5
    beta_force: float = 0.00001
    beta_torque: float = 0.0
    max_torque_nm: float = 0.0
    end_updates: int = 3000
    rollout_steps: int = 24
    gamma: float = 0.99
    gae_lambda: float = 0.95
    cost_scale_n: float = 43.127251624
    cost_learning_rate: float = 0.001
    cost_max_grad_norm: float = 1.0
    cost_scalarization: str = "L2_norm"
    substep_aggregation: str = "arithmetic_mean"
    height_domain: str = "nonnegative_deficit"
    formula_version: str = "FTSR_Eq5Eq6AdaptationV1"
    diagnostic_interval_steps: int = 24

    def __post_init__(self) -> None:
        for name in ("force_enabled", "torque_enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"辅助开关必须为布尔值: {name}")
        for name in ("end_updates", "rollout_steps", "diagnostic_interval_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"辅助计数必须为正整数: {name}")
        positive = ("h_target", "mu", "max_force_body_weights", "cost_scale_n",
                    "cost_learning_rate", "cost_max_grad_norm")
        nonnegative = ("beta_force", "beta_torque", "max_torque_nm")
        for name in positive + nonnegative + ("gamma", "gae_lambda"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"辅助参数必须为有限数值: {name}")
            if value < 0 or (name in positive and value == 0):
                raise ValueError(f"辅助参数范围无效: {name}")
        if not 0 < self.gamma < 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("辅助折扣或GAE参数无效")
        fixed = {
            "schema_version": "RecoveryForceGuidanceV1", "torque_enabled": False,
            "beta_torque": 0.0, "max_torque_nm": 0.0, "rollout_steps": 24,
            "cost_scalarization": "L2_norm", "substep_aggregation": "arithmetic_mean",
            "height_domain": "nonnegative_deficit", "formula_version": "FTSR_Eq5Eq6AdaptationV1",
        }
        if any(getattr(self, name) != value for name, value in fixed.items()):
            raise ValueError("辅助配置不符合已审阅的向上力计算版本")
        stable_hash(asdict(self))

    @property
    def end_control_steps(self) -> int:
        return self.end_updates * self.rollout_steps

    @property
    def sha256(self) -> str:
        return stable_hash(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GuidanceConfigV1:
        expected = {field.name for field in fields(cls)}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("辅助配置必须明确提供全部字段")
        return cls(**payload)


@dataclass(frozen=True)
class GuidanceTrainingContractV1:
    schema_version: str = "GuidanceTrainingContractV1"
    physical_contract_sha256: str = RecoveryContractV2().sha256
    actor_input_dim: int = 111
    critic_input_dim: int = 35
    external_input_dim: int = 110
    action_dim: int = 6
    auxiliary_observation: str = "alpha_tail"
    deployment_alpha: float = 0.0
    formula_version: str = "FTSR_Eq5Eq6AdaptationV1"
    cost_signal: str = "world_force_L2_N_substep_arithmetic_mean"
    advantage_normalization: str = "mixed_rollout_population_once"

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not type(field.default) or value != field.default:
                raise ValueError(f"辅助训练契约固定字段变化: {field.name}")

    @property
    def sha256(self) -> str:
        return stable_hash(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GuidanceTrainingContractV1:
        if not isinstance(payload, dict) or set(payload) != {field.name for field in fields(cls)}:
            raise ValueError("辅助训练契约字段不完整")
        return cls(**payload)


def default_resolved_config() -> dict[str, Any]:
    from .randomization import RecoveryRandomizationV1
    from .supervisor import SupervisorConfig

    return {
        "schema_version": "RecoveryResolvedConfigV2",
        "contract": asdict(RecoveryContractV2()),
        "randomization": asdict(RecoveryRandomizationV1()),
        "supervisor": asdict(SupervisorConfig()),
        "environment": {"seed": 42001, "num_envs": 4, "device": "cuda:0",
                        "env_spacing_m": 2.0, "ground_size_xy_m": [200.0, 200.0],
                        "ground_activity_margin_m": 10.0,
                        "pose_bank_path": None, "walker_export_dir": None, "run_dir": None,
                        "task_type": "interface_validation", "isaac_contact_model_sha256": None,
                        "physx_buffers": {"gpu_max_rigid_contact_count": 65536,
                                          "gpu_max_rigid_patch_count": 16384,
                                          "gpu_found_lost_pairs_capacity": 65536,
                                          "gpu_found_lost_aggregate_pairs_capacity": 65536,
                                          "gpu_total_aggregate_pairs_capacity": 65536}},
        "ppo": {"num_steps_per_env": 24, "save_interval": 50,
                "policy": {"actor_hidden_dims": [256, 128, 64], "critic_hidden_dims": [256, 128, 64],
                           "activation": "elu", "init_noise_std": 0.5, "noise_std_type": "log",
                           "actor_obs_normalization": False, "critic_obs_normalization": False,
                           "state_dependent_std": False},
                "algorithm": {"num_learning_epochs": 5, "num_mini_batches": 4,
                              "clip_param": 0.2, "gamma": 0.99, "lam": 0.95,
                              "value_loss_coef": 1.0, "entropy_coef": 0.01,
                              "learning_rate": 0.001, "max_grad_norm": 1.0,
                              "use_clipped_value_loss": True, "schedule": "adaptive", "desired_kl": 0.01,
                              "normalize_advantage_per_mini_batch": False, "rnd_cfg": None,
                              "symmetry_cfg": None, "multi_gpu_cfg": None}},
        "readiness": {"support_force_n": None, "target_difference_limits": None,
                      "torque_difference_limits": None, "calibration_path": None, "geometry_guard": None},
        "pose_sampling": {"final_probabilities": [0.18, 0.18, 0.18, 0.18, 0.18, 0.10],
                          "curriculum_path": None},
        "assessment": {"formal_evidence_path": None, "updates": None},
    }


def default_guidance_resolved_config() -> dict[str, Any]:
    payload = default_resolved_config()
    payload["schema_version"] = "RecoveryResolvedConfigV3"
    payload["guidance"] = asdict(GuidanceConfigV1())
    payload["training_contract"] = asdict(GuidanceTrainingContractV1())
    return payload


def load_resolved_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_resolved_config(payload)
    return payload


def validate_resolved_config(payload: dict[str, Any], *, check_paths: bool = True) -> None:
    if isinstance(payload, dict) and payload.get("schema_version") == "RecoveryIndependentStandV1":
        from .independent_config import validate_independent_config

        validate_independent_config(payload, check_paths=check_paths)
        return

    from .randomization import RecoveryRandomizationV1
    from .supervisor import SupervisorConfig

    if isinstance(payload, dict) and payload.get("schema_version") == "RecoveryResolvedConfigV3":
        if set(payload) != set(default_guidance_resolved_config()):
            raise ValueError("V3解析配置顶层字段不完整")
        guidance = GuidanceConfigV1.from_dict(payload["guidance"])
        training_contract = GuidanceTrainingContractV1.from_dict(payload["training_contract"])
        if training_contract.physical_contract_sha256 != stable_hash(payload["contract"]):
            raise ValueError("辅助训练与物理契约身份不同")
        if (guidance.rollout_steps != payload["ppo"]["num_steps_per_env"]
                or guidance.gamma != payload["ppo"]["algorithm"]["gamma"]
                or guidance.gae_lambda != payload["ppo"]["algorithm"]["lam"]
                or guidance.h_target != payload["contract"]["height_target_m"]):
            raise ValueError("辅助配置与任务高度、PPO折扣或采集长度不同")
        original = {key: value for key, value in payload.items() if key not in ("guidance", "training_contract")}
        original["schema_version"] = "RecoveryResolvedConfigV2"
        validate_resolved_config(original, check_paths=check_paths)
        stable_hash(payload)
        return
    expected = set(default_resolved_config())
    if set(payload) != expected or payload["schema_version"] != "RecoveryResolvedConfigV2":
        raise ValueError("解析配置的版本或顶层字段不完整")
    if set(payload["assessment"]) != {"formal_evidence_path", "updates"}:
        raise ValueError("训练验收配置字段不符合V2契约")
    contract = RecoveryContractV2.from_dict(payload["contract"])
    RecoveryRandomizationV1(**payload["randomization"])
    supervisor = SupervisorConfig(**payload["supervisor"])
    if supervisor.dt != contract.control_dt_s or supervisor.recovery_budget_s != 8 or supervisor.event_budget_s != 10:
        raise ValueError("监督器步长和累计预算必须符合起身契约")
    stable_hash(payload)
    environment = payload["environment"]
    if environment["task_type"] not in ("interface_validation", "bootstrap_training", "formal_training"):
        raise ValueError("必须显式区分接口验证、初始训练和正式能力训练")
    if not isinstance(environment["num_envs"], int) or environment["num_envs"] <= 0:
        raise ValueError("并行环境数量无效")
    if (not math.isfinite(environment["env_spacing_m"]) or environment["env_spacing_m"] <= 0
            or not math.isfinite(environment["ground_activity_margin_m"]) or environment["ground_activity_margin_m"] <= 0
            or len(environment["ground_size_xy_m"]) != 2
            or any(not math.isfinite(value) or value <= 2 * environment["ground_activity_margin_m"]
                   for value in environment["ground_size_xy_m"])):
        raise ValueError("地面尺寸、环境间距和活动余量必须为有效正有限数")
    if not isinstance(environment["seed"], int) or environment["seed"] < 0:
        raise ValueError("必须提供非负随机种子")
    policy, algorithm = payload["ppo"]["policy"], payload["ppo"]["algorithm"]
    if (tuple(policy["actor_hidden_dims"]) != contract.actor_hidden_dims
            or tuple(policy["critic_hidden_dims"]) != contract.critic_hidden_dims
            or policy["activation"] != "elu" or policy["noise_std_type"] != "log"
            or policy["actor_obs_normalization"] or policy["critic_obs_normalization"]
            or algorithm["gamma"] != contract.gamma or payload["ppo"]["num_steps_per_env"] != 24):
        raise ValueError("PPO网络、归一化、折扣或rollout长度不符合设计")
    if (policy["state_dependent_std"] or algorithm["normalize_advantage_per_mini_batch"]
            or any(algorithm[key] is not None for key in ("rnd_cfg", "symmetry_cfg", "multi_gpu_cfg"))):
        raise ValueError("起身V2仅允许普通固定方差PPO，关闭额外奖励、对称与多进程扩展")
    guard = payload["readiness"]["geometry_guard"]
    if guard is not None:
        from .geometry_guard import validate_guard_config

        validate_guard_config(guard, check_paths=check_paths)
    if environment["task_type"] == "bootstrap_training":
        if guard is None:
            raise ValueError("初始训练必须逐物理子步检查真实刚体几何")
        if (payload["pose_sampling"]["curriculum_path"] is None
                or not isinstance(payload["assessment"]["updates"], int) or payload["assessment"]["updates"] <= 0):
            raise ValueError("初始训练必须固定课程和总更新预算")
        if check_paths and not Path(payload["pose_sampling"]["curriculum_path"]).is_file():
            raise FileNotFoundError(payload["pose_sampling"]["curriculum_path"])
    if environment["task_type"] == "formal_training":
        if guard is None:
            raise ValueError("正式训练必须逐物理子步检查真实刚体几何")
        if guard["internal_clearance_mode"] != "strict":
            raise ValueError("正式训练不得使用仅诊断的内部间隙模型")
        isaac_contact_hash = environment["isaac_contact_model_sha256"]
        if not isinstance(isaac_contact_hash, str) or len(isaac_contact_hash) != 64:
            raise ValueError("正式训练缺少真实验收后的 Isaac 完整接触身份")
        readiness = payload["readiness"]
        required = (environment["pose_bank_path"], environment["walker_export_dir"],
                    environment["run_dir"], readiness["calibration_path"],
                    payload["pose_sampling"]["curriculum_path"],
                    payload["assessment"]["formal_evidence_path"])
        if any(value is None for value in required):
            raise ValueError("正式训练缺少姿态库、标定、课程或前置实验记录")
        if readiness["support_force_n"] is None or readiness["support_force_n"] <= 0:
            raise ValueError("正式训练缺少实测支撑力门限")
        for key in ("target_difference_limits", "torque_difference_limits"):
            values = readiness[key]
            if values is None or len(values) != 6 or any(value <= 0 for value in values):
                raise ValueError(f"正式训练缺少有效的六维{key}")
        if not isinstance(payload["assessment"]["updates"], int) or payload["assessment"]["updates"] <= 0:
            raise ValueError("正式更新预算必须在配置中固定")
        if check_paths:
            for filename in required[:2] + required[3:]:
                if not Path(filename).exists():
                    raise FileNotFoundError(filename)


def source_sha256(root: str | Path = PROJECT_ROOT) -> str:
    base = Path(root)
    paths = sorted(path for path in (base / "training" / "recovery").rglob("*.py")
                   if not set(path.relative_to(base / "training" / "recovery").parts).intersection({"artifacts", "runs", "data", "__pycache__"}))
    paths += sorted((base / "training" / "wheelleg_dreamwaq").rglob("*.py"))
    return stable_hash({path.relative_to(base).as_posix(): file_sha256(path) for path in paths})


def verify_walker_export(export_dir: str | Path, contract: RecoveryContractV2 | None = None) -> dict:
    import torch

    from wheelleg_dreamwaq.schemas.normalization import RawActionNormalizationV1
    from wheelleg_dreamwaq.schemas.frames import R_CONTROL_FROM_USD

    cfg = contract or RecoveryContractV2()
    directory = Path(export_dir)
    manifest = json.loads((directory / "policy_manifest.json").read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    if manifest.get("manifest_hash") != hashlib.sha256(encoded).hexdigest().upper():
        raise ValueError("冻结行走清单自身的SHA256无效")
    if manifest.get("schema_version") != "DreamWaQPolicyExportV1":
        raise ValueError("冻结行走导出版本无效")
    if manifest.get("dreamwaq_export_contract_version") != "DreamWaQExportContractV2RawHeading":
        raise ValueError("起身交接要求原始动作行走导出契约")
    for filename_key, hash_key, filename in (("actor_file", "actor_sha256", "actor.ts"),
                                           ("golden_vectors_file", "golden_vectors_sha256", "golden_vectors.pt")):
        if manifest.get(filename_key) != filename or file_sha256(directory / filename) != manifest.get(hash_key):
            raise ValueError(f"冻结行走导出文件身份无效: {filename}")
    network = manifest["network"]
    expected_network = {"input_dimension": 125, "output_dimension": 6, "history_length": 5,
                        "current_observation_dimension": 25, "history_layout": "frame_major", "dynamic_batch": True,
                        "runtime_action_clip": None}
    if any(network.get(key) != value for key, value in expected_network.items()):
        raise ValueError("冻结行走网络历史或原始动作接口不匹配")
    action, control, timing = manifest["action"], manifest["control"], manifest["timing"]
    if (action.get("clip_actions", 1) is not None or action.get("canonical_joint_order") != list(CANONICAL_JOINT_ORDER)
            or action.get("wheel_joint_sign_usd") != [1.0, -1.0]
            or control.get("leg_action_scale") != cfg.leg_action_scale
            or control.get("wheel_action_scale") != cfg.wheel_action_scale
            or control.get("leg_target_lower") != [WALKER_LEG_TARGET_LOWER] * 4
            or control.get("leg_target_upper") != [WALKER_LEG_TARGET_UPPER] * 4
            or timing.get("control_dt_s") != cfg.control_dt_s
            or timing.get("isaac_sim_dt_s") != cfg.isaac_physics_dt_s):
        raise ValueError("冻结行走执行器映射或控制时序不符合行走契约")
    if not torch.allclose(torch.tensor(manifest["q_nominal"], dtype=torch.float64),
                          torch.tensor(cfg.q_nominal, dtype=torch.float64), atol=1e-7, rtol=0):
        raise ValueError("冻结行走名义腿角与起身不一致")
    if manifest.get("normalization") != asdict(RawActionNormalizationV1()):
        raise ValueError("冻结行走归一化或上一原始动作接口不同")
    schemas = manifest.get("schemas", {})
    if schemas.get("action") != "RawActionV1" or schemas.get("normalization") != "RawActionNormalizationV1":
        raise ValueError("冻结行走原始动作schema缺失")
    rotation = torch.tensor(manifest["frames"]["r_control_from_usd"], dtype=torch.float64)
    if (rotation.shape != (3, 3) or not torch.isfinite(rotation).all()
            or not torch.allclose(rotation @ rotation.T, torch.eye(3, dtype=torch.float64), atol=1e-10, rtol=0)
            or abs(float(torch.linalg.det(rotation)) - 1) > 1e-10):
        raise ValueError("冻结行走控制坐标旋转无效")
    if not torch.equal(rotation, R_CONTROL_FROM_USD.to(torch.float64)):
        raise ValueError("冻结行走与起身代码使用不同的控制坐标")
    legs, wheels = manifest["actuators"]["legs"], manifest["actuators"]["wheels"]
    if (legs["stiffness"] != cfg.leg_kp or legs["damping"] != cfg.leg_kd
            or wheels["damping"] != cfg.wheel_kd or legs["effort_limit_sim"] != cfg.effort_limits[0]
            or wheels["effort_limit_sim"] != cfg.effort_limits[4]):
        raise ValueError("冻结行走名义反馈增益和力矩能力与起身不同")
    golden = torch.load(directory / "golden_vectors.pt", map_location="cpu", weights_only=False)
    inputs, expected = golden["history"], golden["expected_action_mean"]
    if inputs.shape != (32, 125) or expected.shape != (32, 6) or not torch.isfinite(inputs).all() or not torch.isfinite(expected).all():
        raise ValueError("冻结行走golden维度或数值无效")
    actor = torch.jit.load(str(directory / "actor.ts"), map_location="cpu").eval()
    with torch.inference_mode():
        for size in (1, 3, 32):
            actual = actor(inputs[:size])
            if (actual.shape != (size, 6) or not torch.isfinite(actual).all()
                    or (size == 32 and float((actual - expected).abs().max()) > 1e-7)):
                raise ValueError("冻结行走TorchScript golden验证失败")
    return manifest
