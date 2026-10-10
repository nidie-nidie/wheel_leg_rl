from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .versions import ACTION_ADAPTER_VERSION, HISTORY_ADAPTER_VERSION, OBSERVATION_ADAPTER_VERSION
from .model_semantics import stable_hash
from .physics import PHYSICS_SCHEMA_VERSION, validate_policy_physics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _stable_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest().upper()


@dataclass(frozen=True)
class AdapterContract:
    q_nominal: np.ndarray
    action_clip: float
    leg_action_scale: float
    wheel_action_scale: float
    leg_target_lower: np.ndarray
    leg_target_upper: np.ndarray
    wheel_signs: np.ndarray
    r_control_from_mujoco: np.ndarray
    normalization: dict[str, float]
    leg_kp: float
    leg_kd: float
    wheel_kd: float
    effort_limits: np.ndarray
    physics_dt_s: float
    physics_steps_per_action: int
    control_dt_s: float
    policy_input_dimension: int = 25
    history_length: int = 1
    history_layout: str = "current_frame"
    isaac_physics_dt_s: float = 0.005

    @classmethod
    def from_policy_manifest(cls, payload: dict) -> "AdapterContract":
        validate_policy_physics(payload)
        actuators = payload["actuators"]
        legs = actuators["legs"]
        wheels = actuators["wheels"]
        timing = payload["timing"]
        action = payload["action"]
        control = payload["control"]
        return cls(
            q_nominal=np.asarray(payload["q_nominal"], dtype=np.float64),
            action_clip=float(action["clip_actions"]),
            leg_action_scale=float(control["leg_action_scale"]),
            wheel_action_scale=float(control["wheel_action_scale"]),
            leg_target_lower=np.asarray(control["leg_target_lower"], dtype=np.float64),
            leg_target_upper=np.asarray(control["leg_target_upper"], dtype=np.float64),
            wheel_signs=np.asarray(action["wheel_joint_sign_usd"], dtype=np.float64),
            r_control_from_mujoco=np.asarray(payload["frames"]["r_control_from_usd"], dtype=np.float64),
            normalization={key: float(value) for key, value in payload["normalization"].items()},
            leg_kp=float(legs["stiffness"]),
            leg_kd=float(legs["damping"]),
            wheel_kd=float(wheels["damping"]),
            effort_limits=np.asarray(
                [float(legs["effort_limit_sim"])] * 4 + [float(wheels["effort_limit_sim"])] * 2,
                dtype=np.float64,
            ),
            physics_dt_s=float(timing["mujoco_physics_dt_s"]),
            physics_steps_per_action=int(timing["mujoco_physics_steps_per_action"]),
            control_dt_s=float(timing["control_dt_s"]),
            policy_input_dimension=int(payload["network"]["input_dimension"]),
            history_length=int(payload["network"].get("history_length", 1)),
            history_layout=str(payload["network"].get("history_layout", "current_frame")),
            isaac_physics_dt_s=float(timing["isaac_sim_dt_s"]),
        )


def load_policy_contract(
    manifest_path: Path,
    actor_path: Path,
    model_manifest_path: Path,
) -> tuple[AdapterContract, dict, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedded_hash = manifest.get("manifest_hash")
    calculated_hash = _stable_hash({key: value for key, value in manifest.items() if key != "manifest_hash"})
    if embedded_hash != calculated_hash:
        raise ValueError(f"Policy manifest hash mismatch: {embedded_hash!r} != {calculated_hash!r}")
    expected_schemas = {
        "action": "ActionV1",
        "actor_observation": "ActorObsV1",
        "command": "CommandV1",
        "command_sampling": "CommandSamplingV2",
        "control_frame": "ControlFrameV1",
        "critic_observation": "CriticObsV1",
        "normalization": "NormalizationV2",
        "physics": PHYSICS_SCHEMA_VERSION,
        "reward": "RewardSchemaV2",
        "virtual_leg_kinematics": "VirtualLegKinematicsV1",
    }
    schemas = manifest.get("schemas")
    schema_version = manifest.get("schema_version")
    if (
        schema_version not in {"PpoActorExportV1", "DreamWaQPolicyExportV1"}
        or not isinstance(schemas, dict)
        or any(schemas.get(key) != value for key, value in expected_schemas.items())
    ):
        raise ValueError("Unsupported policy manifest schema")
    if manifest.get("phase1_contract_version") == "Phase1RandomizedContractV1":
        if schemas.get("randomization") != "RandomizationSchemaV1":
            raise ValueError("Randomized policy manifest is missing RandomizationSchemaV1")
    if manifest["network"]["output_dimension"] != 6:
        raise ValueError("Policy manifest output dimension does not match ActionV1")
    if schema_version == "PpoActorExportV1":
        if manifest["network"]["input_dimension"] != 25:
            raise ValueError("PPO policy manifest input dimension does not match ActorObsV1")
    else:
        network = manifest["network"]
        if (
            network.get("input_dimension") != 125
            or network.get("history_length") != 5
            or network.get("current_observation_dimension") != 25
            or network.get("history_layout") != "frame_major"
            or network.get("dynamic_batch") is not True
            or manifest.get("history_adapter_version") != HISTORY_ADAPTER_VERSION
        ):
            raise ValueError("DreamWaQ policy manifest history contract is invalid")
        required_contract_fields = {
            "base_task_contract_hash",
            "dreamwaq_algorithm_contract_hash",
            "dreamwaq_export_contract_hash",
        }
        if any(not isinstance(manifest.get(field), str) or len(manifest[field]) != 64 for field in required_contract_fields):
            raise ValueError("DreamWaQ policy manifest contract hashes are missing")
        golden_name = manifest.get("golden_vectors_file")
        if golden_name != "golden_vectors.pt":
            raise ValueError("DreamWaQ policy manifest golden-vector filename is invalid")
        golden_path = manifest_path.parent / golden_name
        if not golden_path.is_file() or sha256_file(golden_path) != manifest.get("golden_vectors_sha256"):
            raise ValueError("DreamWaQ golden-vector artifact hash differs from the policy manifest")
        if {path.name for path in manifest_path.parent.iterdir()} != {
            "actor.ts",
            "policy_manifest.json",
            "golden_vectors.pt",
        }:
            raise ValueError("DreamWaQ export directory must contain exactly the frozen three artifacts")
    if manifest["action"]["canonical_joint_order"] != [
        "jIJ", "jIO", "jAB", "jAG", "jwheel_left", "jwheel_right"
    ]:
        raise ValueError("Policy manifest canonical joint order is invalid")
    if sha256_file(actor_path) != manifest["actor_sha256"]:
        raise ValueError("TorchScript actor hash does not match policy manifest")

    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(model_manifest_path) != manifest["mujoco_model"]["model_manifest_sha256"]:
        raise ValueError("MuJoCo model manifest hash does not match policy manifest")
    if model_manifest["model_xml"]["sha256"] != manifest["mujoco_model"]["model_xml_sha256"]:
        raise ValueError("MuJoCo XML identity does not match policy manifest")
    if model_manifest["dynamics_semantics_hash"] != manifest["mujoco_model"]["dynamics_semantics_hash"]:
        raise ValueError("MuJoCo dynamics semantic identity does not match policy manifest")
    if model_manifest["dynamics_semantics_hash"] != stable_hash(model_manifest["dynamics_semantics"]):
        raise ValueError("MuJoCo dynamics semantic record has an invalid hash")
    if model_manifest["asset_bundle_hash"] != manifest["asset_bundle_hash"]:
        raise ValueError("MuJoCo model and policy use different asset identities")
    if model_manifest["r_control_from_mujoco"] != manifest["frames"]["r_control_from_usd"]:
        raise ValueError("MuJoCo and policy control-frame rotations differ")
    adapter_files = {
        "observation": (
            OBSERVATION_ADAPTER_VERSION,
            Path(__file__).with_name("observation.py"),
        ),
        "action": (
            ACTION_ADAPTER_VERSION,
            Path(__file__).with_name("control.py"),
        ),
    }
    for name, (version, implementation_path) in adapter_files.items():
        adapter = model_manifest["adapters"][name]
        if adapter["version"] != version:
            raise ValueError(f"MuJoCo {name} adapter version differs from the model manifest")
        if adapter["implementation_sha256"] != sha256_file(implementation_path):
            raise ValueError(f"MuJoCo {name} adapter implementation differs from the model manifest")
    mesh_root = model_manifest_path.parent / "models" / "meshes"
    for mesh in model_manifest["meshes"]:
        mesh_path = mesh_root / mesh["name"]
        if not mesh_path.is_file() or sha256_file(mesh_path) != mesh["sha256"]:
            raise ValueError(f"MuJoCo mesh identity differs from the model manifest: {mesh['name']}")

    contract = AdapterContract.from_policy_manifest(manifest)
    if abs(contract.physics_dt_s * contract.physics_steps_per_action - contract.control_dt_s) > 1.0e-12:
        raise ValueError("Policy timing manifest is internally inconsistent")
    if abs(model_manifest["physics_dt_s"] - contract.physics_dt_s) > 1.0e-15:
        raise ValueError("MuJoCo physics timestep differs from policy manifest")
    semantics = model_manifest["dynamics_semantics"]
    if semantics["controlled_joint_order"] != manifest["action"]["canonical_joint_order"]:
        raise ValueError("MuJoCo controlled joint order differs from ActionV1")
    reset = semantics["reset"]
    initial_state = manifest["initial_state"]
    if not np.allclose(reset["base_quaternion"], initial_state["rot"], rtol=0.0, atol=1.0e-8):
        raise ValueError("MuJoCo reset base quaternion differs from the policy contract")
    if abs(float(reset["base_position"][2]) - float(initial_state["pos"][2])) > 1.0e-8:
        raise ValueError("MuJoCo reset base height differs from the policy contract")
    if set(reset["hinge_positions"]) != set(initial_state["joint_pos"]):
        raise ValueError("MuJoCo reset hinge set differs from the policy contract")
    for joint_name, expected in initial_state["joint_pos"].items():
        if abs(float(reset["hinge_positions"][joint_name]) - float(expected)) > 1.0e-8:
            raise ValueError(f"MuJoCo reset position differs for joint {joint_name}")
    actuator_by_name = {record["name"]: record for record in semantics["actuators"]}
    expected_effort_limits = manifest["mujoco_model"]["actuator_effort_limits_nm"]
    for actuator_name, limit in expected_effort_limits.items():
        record = actuator_by_name.get(actuator_name)
        if record is None or not np.allclose(record["control_range"], [-limit, limit], rtol=0.0, atol=1.0e-12):
            raise ValueError(f"MuJoCo actuator effort range differs for {actuator_name}")
    return contract, manifest, model_manifest
