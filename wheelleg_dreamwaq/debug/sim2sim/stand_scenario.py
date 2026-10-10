from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .trace_schema import save_npz, sha256_file, stable_payload_hash, write_json


FROZEN_FILES: dict[str, tuple[str, str]] = {
    "checkpoint": (
        "logs/rsl_rl/wheelleg_flat_ppo/2026-10-05_07-11-36_rtx5070_v4-suite-20261005-071132-run01-seed250509479/model_999.pt",
        "04E040272EE3D2682DEFA053062D1EBE9AB3D13652DBC8D4760773A23A58197E",
    ),
    "actor": (
        "artifacts/phase1_v4/formal-4x1000-20261005/exports/run-01/actor.ts",
        "5B714AF635AE3C93BD86853BD9141F0B8888BB49C0F275CBCCD531B6AF53B9BC",
    ),
    "policy_manifest": (
        "artifacts/phase1_v4/formal-4x1000-20261005/exports/run-01/policy_manifest.json",
        "EC9F42FD7BD5176D1B6865D0A3D67A6B49103899FE44BB056F16E9D048BA90F1",
    ),
    "source_run_manifest": (
        "logs/rsl_rl/wheelleg_flat_ppo/2026-10-05_07-11-36_rtx5070_v4-suite-20261005-071132-run01-seed250509479/run_manifest.json",
        "CD3DA566179885DD628A09DC66BD57B17F177C00018FA1EF4F93EABD867A87C7",
    ),
    "model_manifest": (
        "sim2sim/mujoco/model_manifest.json",
        "C3DB4FE2797F6AC3628F0A9CB5E0B166FFD0CE802776A335922C474A045F7AE0",
    ),
    "model_xml": (
        "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml",
        "691C607271ED85EFFA388237E00A7B8F8F41CB66F6BA9C49EA64ABD186D803D1",
    ),
}


@dataclass(frozen=True)
class StandScenarioV1:
    scenario_version: str = "StandScenarioV1"
    name: str = "nominal_stand_debug_v1"
    num_envs: int = 1
    control_ticks: int = 500
    control_dt_s: float = 0.020
    command: tuple[float, float, float] = (0.0, 0.0, 0.20)
    hold_command: bool = True
    domain_randomization: bool = False
    external_disturbance: bool = False
    initial_yaw_rad: float = 0.0
    random_seed: int = 0
    previous_action: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def payload(self) -> dict:
        return asdict(self)

    @property
    def payload_hash(self) -> str:
        return stable_payload_hash(self.payload)


def verify_frozen_files(project_root: str | Path, frozen_files: Mapping[str, tuple[str, str]] = FROZEN_FILES) -> dict:
    root = Path(project_root)
    records: dict[str, dict[str, str]] = {}
    for name, (relative_path, expected_hash) in frozen_files.items():
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Frozen {name} hash mismatch: {actual_hash} != {expected_hash}")
        records[name] = {"path": relative_path.replace("\\", "/"), "sha256": actual_hash}
    return records


def build_action_sequences(scenario: StandScenarioV1) -> dict[str, np.ndarray]:
    sequences: dict[str, np.ndarray] = {
        "zero_action": np.zeros((100, 6), dtype=np.float32),
    }
    for channel in range(6):
        pulse = np.zeros((100, 6), dtype=np.float32)
        pulse[25:50, channel] = 0.1
        pulse[75:100, channel] = -0.1
        sequences[f"channel_pulse_{channel}"] = pulse
    return sequences


def write_scenario_bundle(project_root: str | Path, run_directory: str | Path) -> dict:
    scenario = StandScenarioV1()
    identities = verify_frozen_files(project_root)
    payload = scenario.payload | {"scenario_hash": scenario.payload_hash, "frozen_files": identities}
    output = Path(run_directory)
    write_json(output / "scenario.json", payload)
    save_npz(output / "action_sequences.npz", build_action_sequences(scenario))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the frozen WheelLeg standing debug scenario.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = write_scenario_bundle(args.project_root.resolve(), args.output.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
