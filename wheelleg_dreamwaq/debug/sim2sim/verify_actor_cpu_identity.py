from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .compare_traces import compare_metadata_identity, load_verified_engine_run
from .trace_schema import (
    build_file_hashes,
    load_json,
    load_npz,
    save_npz,
    sha256_array,
    sha256_file,
    stable_payload_hash,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTOR_RELATIVE = Path("artifacts/phase1_v4/formal-4x1000-20261005/exports/run-01/actor.ts")


def evaluate_same_input(
    actor_path: str | Path,
    actor_input: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(actor_input)
    if values.shape != (25,) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError(f"Actor identity input must be finite float32[25], got {values.dtype} {values.shape}")
    tensor = torch.from_numpy(values.copy()).unsqueeze(0)
    first_actor = torch.jit.load(str(actor_path), map_location="cpu").eval()
    second_actor = torch.jit.load(str(actor_path), map_location="cpu").eval()
    with torch.inference_mode():
        first = first_actor(tensor).detach().cpu().numpy().reshape(-1)
        second = second_actor(tensor).detach().cpu().numpy().reshape(-1)
    if first.shape != (6,) or second.shape != (6,):
        raise ValueError(f"Actor identity output must be [6], got {first.shape} and {second.shape}")
    return first, second


def _load_verified_evaluation(
    directory: Path,
    *,
    actor_path: Path,
    input_path: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    metadata_path = directory / "metadata.json"
    output_path = directory / "actor_output.npz"
    metadata = load_json(metadata_path)
    expected_hashes = load_json(directory / "file_hashes.json")
    required_paths = {
        "actor": actor_path,
        "input": input_path,
        "actor_output": output_path,
        "metadata": metadata_path,
        "evaluator": Path(__file__).with_name("evaluate_actor_cpu.py"),
    }
    missing = sorted(set(required_paths) - set(expected_hashes))
    if missing:
        raise ValueError(f"Actor evaluation hash manifest is missing fields: {missing}")
    mismatches = {
        name: {"expected": expected_hashes[name], "actual": sha256_file(path)}
        for name, path in required_paths.items()
        if expected_hashes[name] != sha256_file(path)
    }
    if mismatches:
        raise ValueError(f"Actor evaluation artifact hash mismatch: {mismatches}")
    if metadata.get("schema_version") != "ActorCpuEvaluationV1":
        raise ValueError("Unsupported Actor CPU evaluation schema")
    actor_output = load_npz(output_path).get("actor_output")
    if actor_output is None or actor_output.shape != (6,) or actor_output.dtype != np.float32:
        raise ValueError("Actor CPU evaluation output must be float32[6]")
    if metadata.get("output_array_sha256") != sha256_array(actor_output):
        raise ValueError("Actor CPU evaluation output content hash mismatch")
    return metadata, actor_output


def verify_actor_cpu_identity(
    *,
    project_root: str | Path,
    isaac_directory: str | Path,
    mujoco_directory: str | Path,
    output_directory: str | Path,
    tick: int = 0,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    isaac = load_verified_engine_run(isaac_directory)
    mujoco = load_verified_engine_run(mujoco_directory)
    isaac_metadata = isaac["metadata"]
    mujoco_metadata = mujoco["metadata"]
    actor_path = root / ACTOR_RELATIVE
    actor_hash = sha256_file(actor_path)
    if isaac_metadata["actor_sha256"] != actor_hash or mujoco_metadata["actor_sha256"] != actor_hash:
        raise ValueError("Run metadata does not identify the frozen Actor file")
    run_identity = compare_metadata_identity(isaac_metadata, mujoco_metadata)

    isaac_inputs = np.asarray(isaac["control"]["actor_obs_policy_pre_step"])
    mujoco_inputs = np.asarray(mujoco["control"]["actor_obs_policy_pre_step"])
    if tick < 0 or tick >= len(isaac_inputs) or tick >= len(mujoco_inputs):
        raise IndexError(f"Actor identity tick {tick} is outside one of the traces")
    canonical_input = np.asarray(isaac_inputs[tick], dtype=np.float32)
    serialized_path = save_npz(output / "serialized_actor_input.npz", {"actor_obs_policy": canonical_input})
    reloaded_input = load_npz(serialized_path)["actor_obs_policy"]
    isaac_evaluation = output / "isaac_environment"
    mujoco_evaluation = output / "mujoco_environment"
    evaluator_arguments = [
        "-m",
        "debug.sim2sim.evaluate_actor_cpu",
        "--actor",
        str(actor_path),
        "--input",
        str(serialized_path),
    ]
    subprocess.run(
        [str(root / ".venv/Scripts/python.exe"), *evaluator_arguments, "--output", str(isaac_evaluation)],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(root / "sim2sim/mujoco"),
            "python",
            *evaluator_arguments,
            "--output",
            str(mujoco_evaluation),
        ],
        cwd=root,
        check=True,
    )
    isaac_evaluation_metadata, isaac_result = _load_verified_evaluation(
        isaac_evaluation,
        actor_path=actor_path,
        input_path=serialized_path,
    )
    mujoco_evaluation_metadata, mujoco_result = _load_verified_evaluation(
        mujoco_evaluation,
        actor_path=actor_path,
        input_path=serialized_path,
    )
    evaluation_identity_fields = (
        "schema_version",
        "device",
        "dtype",
        "actor_sha256",
        "input_file_sha256",
        "input_array_sha256",
    )
    evaluation_identity_failures = [
        name
        for name in evaluation_identity_fields
        if isaac_evaluation_metadata.get(name) != mujoco_evaluation_metadata.get(name)
    ]
    exact_input_output_error = float(
        np.max(np.abs(isaac_result.astype(np.float64) - mujoco_result.astype(np.float64)))
    )

    mujoco_input = np.asarray(mujoco_inputs[tick], dtype=np.float32)
    input_difference = np.abs(canonical_input.astype(np.float64) - mujoco_input.astype(np.float64))
    stored_output_difference = np.abs(
        np.asarray(isaac["control"]["actor_output_raw"][tick], dtype=np.float64)
        - np.asarray(mujoco["control"]["actor_output_raw"][tick], dtype=np.float64)
    )
    report: dict[str, Any] = {
        "schema_version": "ActorCpuIdentityGateV1",
        "actor_sha256": actor_hash,
        "tick": tick,
        "device": "cpu",
        "dtype": "float32",
        "serialized_input_sha256": sha256_array(reloaded_input),
        "input_run_hashes": {
            "isaac_metadata": sha256_file(Path(isaac_directory).resolve() / "metadata.json"),
            "isaac_control_trace": sha256_file(Path(isaac_directory).resolve() / "control_trace.npz"),
            "mujoco_metadata": sha256_file(Path(mujoco_directory).resolve() / "metadata.json"),
            "mujoco_control_trace": sha256_file(Path(mujoco_directory).resolve() / "control_trace.npz"),
        },
        "same_input_output_max_abs_error": exact_input_output_error,
        "same_input_limit": 1.0e-6,
        "evaluation_identity_failures": evaluation_identity_failures,
        "run_identity_failures": run_identity["failures"],
        "isaac_evaluation_versions": {
            "python": isaac_evaluation_metadata["python_version"],
            "numpy": isaac_evaluation_metadata["numpy_version"],
            "torch": isaac_evaluation_metadata["torch_version"],
        },
        "mujoco_evaluation_versions": {
            "python": mujoco_evaluation_metadata["python_version"],
            "numpy": mujoco_evaluation_metadata["numpy_version"],
            "torch": mujoco_evaluation_metadata["torch_version"],
        },
        "cross_engine_input_max_abs_error": float(np.max(input_difference)),
        "cross_engine_input_rms_error": float(np.sqrt(np.mean(np.square(input_difference)))),
        "stored_runtime_output_max_abs_error": float(np.max(stored_output_difference)),
        "passed": (
            exact_input_output_error <= 1.0e-6
            and not evaluation_identity_failures
            and run_identity["passed"]
        ),
    }
    report["report_hash"] = stable_payload_hash(report)
    write_json(output / "summary.json", report)
    write_json(
        output / "file_hashes.json",
        build_file_hashes(
            {
                "actor": actor_path,
                "serialized_actor_input": serialized_path,
                "summary": output / "summary.json",
                "verifier": Path(__file__),
                "evaluator": Path(__file__).with_name("evaluate_actor_cpu.py"),
                "isaac_evaluation": isaac_evaluation / "actor_output.npz",
                "isaac_evaluation_metadata": isaac_evaluation / "metadata.json",
                "isaac_evaluation_hashes": isaac_evaluation / "file_hashes.json",
                "mujoco_evaluation": mujoco_evaluation / "actor_output.npz",
                "mujoco_evaluation_metadata": mujoco_evaluation / "metadata.json",
                "mujoco_evaluation_hashes": mujoco_evaluation / "file_hashes.json",
            }
        ),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exact CPU TorchScript output for one serialized input.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--isaac-run", type=Path, required=True)
    parser.add_argument("--mujoco-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tick", type=int, default=0)
    args = parser.parse_args()
    report = verify_actor_cpu_identity(
        project_root=args.project_root,
        isaac_directory=args.isaac_run,
        mujoco_directory=args.mujoco_run,
        output_directory=args.output,
        tick=args.tick,
    )
    print(json.dumps({"passed": report["passed"]}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
