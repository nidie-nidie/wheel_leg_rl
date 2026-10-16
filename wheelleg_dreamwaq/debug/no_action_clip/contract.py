from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


EXPERIMENT_ID = "DebugNoActionClipV2"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
BASELINE_EXPORT = PROJECT_ROOT / "artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/exports/run-03"
ACTION_CONTRACT = {
    "id": EXPERIMENT_ID,
    "runtime_clip": None,
    "previous_action": "raw_six_dimensional_action",
    "decoder_action": "detached_raw_six_dimensional_action",
    "reward_action": "raw_six_dimensional_action",
    "leg_target_clip_rad": [-1.0, 1.0],
    "leg_scale": 0.35,
    "wheel_scale": 25.0,
    "effective_mean_clip": [-20.0, 20.0],
    "history": "five_frames_t_minus_4_through_t",
}
GEOMETRY_GUARD_CONTRACT = {
    "schema_version": "DebugFkDiagnosticOnlyV1",
    "isaac_runtime": {
        "fk_disagreement": "diagnostic_only",
        "fk_valid_flag": "diagnostic_only",
        "min_actual_leg_length_m": 0.05,
        "max_actual_loop_error_m": 0.005,
        "finite_body_vectors_and_state": "required",
    },
    "initialization_and_reset": "unchanged_formal_validation",
    "mujoco_runtime": "unchanged_actual_geometry_metrics",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def verify_formal_files() -> int:
    records = json.loads((EXPERIMENT_ROOT / "formal_files_before.json").read_text(encoding="utf-8-sig"))
    for record in records:
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"正式文件指纹发生变化：{path}")
    return len(records)
