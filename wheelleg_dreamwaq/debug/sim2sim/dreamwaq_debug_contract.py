from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import torch


ACTOR_OBS_DIM = 25
DREAMWAQ_HISTORY_LENGTH = 5
DREAMWAQ_HISTORY_DIM = ACTOR_OBS_DIM * DREAMWAQ_HISTORY_LENGTH
CENET_VELOCITY_DIM = 3
CENET_CONTEXT_DIM = 16


@dataclass(frozen=True)
class DebugTimingProfile:
    name: str
    physics_dt_s: float
    physics_steps_per_action: int
    torque_refresh_substeps: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Timing profile name must be non-empty")
        if self.physics_dt_s <= 0.0 or self.physics_steps_per_action <= 0:
            raise ValueError("Timing profile values must be positive")
        if self.torque_refresh_substeps <= 0:
            raise ValueError("Torque refresh cadence must be positive")
        if self.physics_steps_per_action % self.torque_refresh_substeps:
            raise ValueError("Torque refresh cadence must divide the physics substeps")
        if abs(self.physics_dt_s * self.physics_steps_per_action - 0.020) > 1.0e-12:
            raise ValueError("Every debug timing profile must retain the 20 ms control period")

    def refreshes_at(self, substep_index: int) -> bool:
        if not 0 <= substep_index < self.physics_steps_per_action:
            raise IndexError(substep_index)
        return substep_index % self.torque_refresh_substeps == 0


TIMING_PROFILES = {
    "formal_1ms": DebugTimingProfile("formal_1ms", 0.001, 20, 1),
    "isaac_sync_5ms": DebugTimingProfile("isaac_sync_5ms", 0.005, 4, 1),
    "hold_5ms": DebugTimingProfile("hold_5ms", 0.001, 20, 5),
}


@dataclass(frozen=True)
class DebugPolicyOutput:
    raw_action: np.ndarray
    estimated_velocity: np.ndarray
    context_mu: np.ndarray
    context_logvar: np.ndarray


class DebugPolicyAdapter:
    """Validated PPO/DreamWaQ TorchScript adapter used only by debug collectors."""

    def __init__(
        self,
        *,
        actor_path: str | Path,
        manifest_path: str | Path,
        model_manifest_path: str | Path,
        load_mujoco_contract: bool = True,
    ) -> None:
        self.actor_path = Path(actor_path).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.model_manifest_path = Path(model_manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.model_manifest = json.loads(self.model_manifest_path.read_text(encoding="utf-8"))
        if _sha256_file(self.actor_path) != self.manifest.get("actor_sha256"):
            raise ValueError("TorchScript actor hash differs from the policy manifest")
        if _sha256_file(self.model_manifest_path) != self.manifest["mujoco_model"].get(
            "model_manifest_sha256"
        ):
            raise ValueError("Model manifest hash differs from the policy manifest")
        embedded_hash = self.manifest.get("manifest_hash")
        calculated_hash = hashlib.sha256(
            json.dumps(
                {key: value for key, value in self.manifest.items() if key != "manifest_hash"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest().upper()
        if embedded_hash != calculated_hash:
            raise ValueError("Policy manifest embedded hash is invalid")
        self.contract = None
        if load_mujoco_contract:
            from wheelleg_mujoco.contract import load_policy_contract

            self.contract, _, _ = load_policy_contract(
                self.manifest_path,
                self.actor_path,
                self.model_manifest_path,
            )
        self.actor = torch.jit.load(str(self.actor_path), map_location="cpu").eval()
        self.input_dimension = int(self.manifest["network"]["input_dimension"])
        if self.input_dimension not in (ACTOR_OBS_DIM, DREAMWAQ_HISTORY_DIM):
            raise ValueError(f"Unsupported policy input dimension: {self.input_dimension}")
        self.policy_kind = "dreamwaq" if self.input_dimension == DREAMWAQ_HISTORY_DIM else "ppo"

    @staticmethod
    def _current_observation(values: np.ndarray) -> np.ndarray:
        current = np.asarray(values)
        if current.shape != (ACTOR_OBS_DIM,):
            raise ValueError(f"Current ActorObsV1 must have shape (25,), got {current.shape}")
        if current.dtype != np.float32:
            raise TypeError(f"Current ActorObsV1 must use float32, got {current.dtype}")
        if not np.isfinite(current).all():
            raise FloatingPointError("Current ActorObsV1 contains NaN or Inf")
        return current

    def initialize_policy_input(self, current_observation: np.ndarray) -> np.ndarray:
        current = self._current_observation(current_observation)
        if self.policy_kind == "ppo":
            return current.copy()
        return np.repeat(current[None, :], DREAMWAQ_HISTORY_LENGTH, axis=0).reshape(-1)

    def advance_policy_input(
        self,
        policy_input: np.ndarray,
        next_current_observation: np.ndarray,
        *,
        done: bool = False,
    ) -> np.ndarray:
        current = self._current_observation(next_current_observation)
        previous = np.asarray(policy_input)
        if previous.shape != (self.input_dimension,) or previous.dtype != np.float32:
            raise ValueError(
                f"Policy input must have shape ({self.input_dimension},) and dtype float32"
            )
        if self.policy_kind == "ppo":
            return current.copy()
        if done:
            return self.initialize_policy_input(current)
        history = previous.reshape(DREAMWAQ_HISTORY_LENGTH, ACTOR_OBS_DIM)
        advanced = np.empty_like(history)
        advanced[:-1] = history[1:]
        advanced[-1] = current
        return advanced.reshape(-1)

    def current_observation(self, policy_input: np.ndarray) -> np.ndarray:
        values = np.asarray(policy_input)
        if values.shape != (self.input_dimension,):
            raise ValueError(f"Policy input has invalid shape: {values.shape}")
        return values[-ACTOR_OBS_DIM:].copy()

    def infer(self, policy_input: np.ndarray) -> DebugPolicyOutput:
        values = np.asarray(policy_input)
        if values.shape != (self.input_dimension,) or values.dtype != np.float32:
            raise ValueError(
                f"Policy input must have shape ({self.input_dimension},) and dtype float32"
            )
        tensor = torch.from_numpy(values).unsqueeze(0)
        with torch.inference_mode():
            action = self.actor(tensor)
            if self.policy_kind == "dreamwaq":
                encoded = self.actor.encoder(tensor)
                estimated_velocity = encoded[:, :CENET_VELOCITY_DIM]
                context_mu = encoded[:, CENET_VELOCITY_DIM : CENET_VELOCITY_DIM + CENET_CONTEXT_DIM]
                context_logvar = encoded[:, CENET_VELOCITY_DIM + CENET_CONTEXT_DIM :]
            else:
                estimated_velocity = torch.full((1, CENET_VELOCITY_DIM), torch.nan)
                context_mu = torch.full((1, CENET_CONTEXT_DIM), torch.nan)
                context_logvar = torch.full((1, CENET_CONTEXT_DIM), torch.nan)
        arrays = tuple(
            value.detach().cpu().numpy().reshape(-1).astype(np.float64)
            for value in (action, estimated_velocity, context_mu, context_logvar)
        )
        if arrays[0].shape != (6,) or not np.isfinite(arrays[0]).all():
            raise FloatingPointError("TorchScript actor returned an invalid action")
        return DebugPolicyOutput(*arrays)


def contract_for_timing(
    contract: Any,
    timing: DebugTimingProfile,
) -> AdapterContract:
    return replace(
        contract,
        physics_dt_s=timing.physics_dt_s,
        physics_steps_per_action=timing.physics_steps_per_action,
        control_dt_s=0.020,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_debug_timing_model_copy(
    source_model: str | Path,
    output_model: str | Path,
    timing: DebugTimingProfile,
) -> Path:
    source = Path(source_model).resolve()
    output = Path(output_model).resolve()
    if source == output:
        raise ValueError("Debug timing model must not overwrite the formal MJCF")
    if "debug" not in {part.lower() for part in output.parts}:
        raise ValueError("Debug timing model output must be inside a debug directory")
    tree = ET.parse(source)
    root = tree.getroot()
    option = root.find("option")
    compiler = root.find("compiler")
    if option is None or compiler is None:
        raise ValueError("Formal MJCF lacks compiler or option configuration")
    mesh_directory = compiler.get("meshdir")
    if not mesh_directory:
        raise ValueError("Formal MJCF compiler lacks meshdir")
    compiler.set("meshdir", (source.parent / mesh_directory).resolve().as_posix())
    option.set("timestep", format(timing.physics_dt_s, ".17g"))
    root.set("model", f"{root.get('model', 'wheelleg')}_debug_{timing.name}")
    root.insert(0, ET.Comment(" DEBUG-ONLY timing copy; not eligible for formal evaluation or ranking "))
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    sidecar = {
        "schema_version": "DebugTimingModelCopyV1",
        "debug_only": True,
        "formal_ranking_eligible": False,
        "source_model": str(source),
        "timing_profile": {
            "name": timing.name,
            "physics_dt_s": timing.physics_dt_s,
            "physics_steps_per_action": timing.physics_steps_per_action,
            "torque_refresh_substeps": timing.torque_refresh_substeps,
            "control_dt_s": 0.020,
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
