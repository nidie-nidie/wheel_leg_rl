from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

from .contracts import PROJECT_ROOT, RecoveryContractV2, file_sha256, stable_hash


SCHEMA = "RecoveryIndependentStandV1"


@dataclass(frozen=True)
class IndependentStandingContractV1(RecoveryContractV2):
    schema_version: str = "IndependentStandingContractV1"
    critic_input_dim: int = 41
    action_dim: int = 6
    leg_action_scale: float = 1.0
    wheel_action_scale: float = 30.0
    action_distribution: str = "tanh_diagonal_normal"
    leg_target_semantics: str = "absolute_control_joint_radians"
    previous_action_semantics: str = "executed_bounded_action"
    preserve_raw_previous_action: bool = False
    max_recovery_steps: int = 500
    success_reward: float = 20.0
    timeout_reward: float = -5.0
    reward_schema: str = "IndependentStandingRewardsV1"

    def __post_init__(self) -> None:
        for field in fields(self):
            expected = field.default
            actual = getattr(self, field.name)
            if isinstance(expected, tuple):
                actual = tuple(actual)
            if actual != expected:
                raise ValueError(f"独立站立固定契约字段变化: {field.name}")
        stable_hash(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict) -> IndependentStandingContractV1:
        if set(payload) != {field.name for field in fields(cls)}:
            raise ValueError("独立站立契约字段不完整")
        values = dict(payload)
        for key in ("q_nominal", "effort_limits", "actor_hidden_dims", "critic_hidden_dims"):
            values[key] = tuple(values[key])
        return cls(**values)


def default_independent_config() -> dict:
    from .independent_state import StandingGateConfig
    from .randomization import RecoveryRandomizationV1

    data = PROJECT_ROOT / "recovery/artifacts/standing-reference401-20261029"
    bank = data / "recovery_contact_paired_reference.npz"
    manifest = data / "manifest.json"
    return {
        "schema_version": SCHEMA,
        "contract": asdict(IndependentStandingContractV1()),
        "reference": {"manifest_path": str(manifest), "manifest_sha256": file_sha256(manifest),
                      "bank_path": str(bank), "bank_sha256": file_sha256(bank)},
        "standing": asdict(StandingGateConfig()),
        "randomization": asdict(RecoveryRandomizationV1(enabled=False)),
        "initialization": {"fallen_fraction": 0.6, "standing_fraction": 0.2, "perturbed_fraction": 0.2,
                           "train_ids": {}, "test_ids": {}},
        "environment": {"seed": 42002, "num_envs": 4, "device": "cuda:0",
                        "env_spacing_m": 2.0, "ground_size_xy_m": [200.0, 200.0],
                        "ground_activity_margin_m": 10.0,
                        "pose_bank_path": None,
                        "run_dir": None, "task_type": "interface_validation", "isaac_contact_model_sha256": None,
                        "physx_buffers": {"gpu_max_rigid_contact_count": 65536,
                                          "gpu_max_rigid_patch_count": 16384,
                                          "gpu_found_lost_pairs_capacity": 65536,
                                          "gpu_found_lost_aggregate_pairs_capacity": 65536,
                                          "gpu_total_aggregate_pairs_capacity": 65536}},
        "ppo": {"num_steps_per_env": 24, "save_interval": 50, "max_iterations": 2,
                "learning_rate": 3e-4, "gamma": .99, "lam": .95, "clip_param": .2,
                "num_learning_epochs": 5, "num_mini_batches": 4, "entropy_coef": .001,
                "value_loss_coef": 1., "max_grad_norm": 1.},
        "geometry_guard": None,
        "assessment": {"calibration_path": None, "calibration_sha256": None,
                       "evidence_path": None, "evaluation_interval": 500,
                       "evaluation_pose_count": 12, "trajectory_quota_bytes": 512 * 1024 ** 2},
    }


def validate_independent_config(config: dict, *, check_paths: bool = True) -> None:
    from .geometry_guard import validate_guard_config
    from .independent_state import StandingGateConfig
    from .randomization import RecoveryRandomizationV1

    required = {"schema_version", "contract", "reference", "standing", "randomization",
                "initialization", "environment", "ppo", "geometry_guard", "assessment"}
    if set(config) != required or config["schema_version"] != SCHEMA:
        raise ValueError("独立站立配置版本或字段不完整")
    IndependentStandingContractV1.from_dict(config["contract"])
    StandingGateConfig(**config["standing"]).validate()
    if RecoveryRandomizationV1(**config["randomization"]).enabled:
        raise ValueError("本次独立站立使用名义物理参数")
    stable_hash(config)
    env, ppo = config["environment"], config["ppo"]
    if env["task_type"] not in ("interface_validation", "formal_training"):
        raise ValueError("独立站立必须区分接口验证和正式训练")
    if env["device"] != "cuda:0":
        raise ValueError("本次独立站立实验训练设备固定为cuda:0")
    if type(env["seed"]) is not int or env["seed"] != 42002:
        raise ValueError("独立站立实验seed固定为42002")
    if type(env["num_envs"]) is not int or env["num_envs"] <= 0:
        raise ValueError("并行环境数必须为正整数")
    if type(ppo["max_iterations"]) is not int or ppo["max_iterations"] <= 0:
        raise ValueError("必须固定正整数更新预算")
    fixed_ppo = {"num_steps_per_env": 24, "save_interval": 50, "learning_rate": 3e-4,
                 "gamma": .99, "lam": .95, "clip_param": .2, "num_learning_epochs": 5,
                 "num_mini_batches": 4, "entropy_coef": .001, "value_loss_coef": 1., "max_grad_norm": 1.}
    if set(ppo) != set(fixed_ppo) | {"max_iterations"} or any(ppo[k] != v for k, v in fixed_ppo.items()):
        raise ValueError("PPO配置必须符合独立站立方案")
    if env["task_type"] == "formal_training" and (env["num_envs"] != 4096 or ppo["max_iterations"] != 4000):
        raise ValueError("正式预算固定4096环境和4000次更新")
    if env["task_type"] == "interface_validation" and (env["num_envs"] > 4 or ppo["max_iterations"] > 4):
        raise ValueError("接口验证上限为4环境4次更新")
    if (not math.isfinite(env["env_spacing_m"]) or env["env_spacing_m"] <= 0
            or len(env["ground_size_xy_m"]) != 2 or any(v <= 20 or not math.isfinite(v) for v in env["ground_size_xy_m"])):
        raise ValueError("地面尺寸或环境间距无效")
    initial = config["initialization"]
    if [initial[k] for k in ("fallen_fraction", "standing_fraction", "perturbed_fraction")] != [.6, .2, .2]:
        raise ValueError("独立站立初态比例固定为60/20/20")
    train_ids = {record for group in initial["train_ids"].values() for record in group}
    test_ids = {record for group in initial["test_ids"].values() for record in group}
    if train_ids & test_ids:
        raise ValueError("训练和测评初态ID必须分离")
    if config["geometry_guard"] is None:
        raise ValueError("独立站立必须提供真实逐子步几何检查")
    validate_guard_config(config["geometry_guard"], check_paths=check_paths)
    ref = config["reference"]
    if set(ref) != {"manifest_path", "manifest_sha256", "bank_path", "bank_sha256"}:
        raise ValueError("参考必须绑定manifest与成对参考文件身份")
    for key in ("manifest", "bank"):
        sha = ref[f"{key}_sha256"]
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError("参考文件必须提供SHA256")
        if check_paths and file_sha256(ref[f"{key}_path"]) != sha:
            raise ValueError(f"参考文件身份变化: {key}")
    if check_paths and (not env["pose_bank_path"] or not Path(env["pose_bank_path"]).is_file()):
        raise FileNotFoundError(env["pose_bank_path"])
    assessment = config["assessment"]
    if (assessment["evaluation_interval"] != 500 or assessment["evaluation_pose_count"] != 12
            or assessment["trajectory_quota_bytes"] != 512 * 1024 ** 2):
        raise ValueError("独立测评周期、姿态数与诊断轨迹配额必须固定")
    if not assessment["calibration_path"] or not assessment["calibration_sha256"]:
        raise ValueError("新任务必须绑定完整站立校准报告")
    if check_paths and file_sha256(assessment["calibration_path"]) != assessment["calibration_sha256"]:
        raise ValueError("站立校准报告身份变化")


def load_independent_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_independent_config(config)
    return config
