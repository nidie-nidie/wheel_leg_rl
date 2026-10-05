"""Fail-closed wrapper around the unchanged official PACE fitter."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence, cast

import torch


DATA_KEYS = ("time", "dof_pos", "des_dof_pos")
SAMPLE_COUNT = 10_000
DT_S = 0.002
A1_TASK = "Isaac-Pace-A1-v0"
A1_SYNTHETIC_TASK = "Isaac-Pace-A1-Synthetic-v0"
ALLOWED_TASKS = frozenset((A1_TASK, A1_SYNTHETIC_TASK))
A1_DATA_DIR = Path("a1/chirp_data.pt")
A1_ROBOT_NAME = "a1"
A1_PACE_JOINT_ORDER = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)
ALLOWED_SYNTHETIC_DELAYS = frozenset((0, 1, 5, 9, 10))
MEAN_PATTERN = re.compile(r"^mean_([0-9]+)\.pt$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
OFFICIAL_SOURCE_SHA256 = {
    "fit.py": "970f97fb066b8bf519d25e035c32b3c0273a85107f37a75f3d5e109d8ed4131f",
    "cma_es.py": "cd9ee989d88b19c96126fdaebb581b4e4055eb568ec8eae607b3b67cbc59ec24",
    "pace_actuator.py": "c9d5204b6ae962de87b756ff0f8a8302f709dd49de09f2c72a52ee1c7bedd18d",
    "pace_sim2real_env_cfg.py": "fcefdfb3a990077e843366985963831ec2704f5a2bcefe86de8846fa8068d658",
}
REAL_CONTRACT_SHA256 = {
    "raw_schema_sha256": "7d48dddca82a7077462aef8aeae5b95487a39d44b7d078791fd26252a830fdd1",
    "trial_plan_schema_sha256": "8df3d4b4a1263225ed88abfe8fe9ddb2c54250d5765bbf292bab978431e718a7",
    "columns_sha256": "fd09fd5df97a18c584e319d42273911690838ab9b3101e0b16b2fd3fa2b98515",
    "golden_sha256": "6bb3c77bca3f76ce78ec5f8e216a52d5881eb3a325e94d68dc227ebae1077791",
}
PINNED_GOGO_TASK8_COMMIT = "ae6802d92fec7a25bfb39514230c10345d18a09b"
GOGO_CONVERTER_SOURCE_SHA256 = {
    "gogo_learn/pace/__init__.py": (
        "4edae17e78b99960a83a453ec48babeccaa3e8d1768ccb166cc12ae864ccb005"
    ),
    "gogo_learn/pace/contract.py": (
        "447a3d581ea6513dd8134a0e8d07e72f3cf47141f5cdfb39185523c92af4654b"
    ),
    "gogo_learn/pace/convert_trial.py": (
        "fba1b5969a779fc998d0258853b2006d98418d2836da2d4a64872e1b42841e74"
    ),
}
ALLOWED_GOGO_UNTRACKED = frozenset(
    {
        "contracts/a1_pace_selection_v1.schema.json",
        "gogo_learn/pace/validate_selection.py",
        "tests/test_a1_pace_selection.py",
    }
)

# Reviewed A1 collection/provenance release required for real fitting.
PINNED_A1_REAL_COLLECTOR_COMMIT = "64c818c61825d13b5034699a377846129f1bea29"
A1_REAL_COLLECTOR_SOURCE_SHA256 = {
    "real/unitree_sdk_rl/a1_pace_collect_real.cpp": (
        "166e1e6967d26335602533fd3aff831a2f18f2210e4cb7e6f0ee476222f8b83e"
    ),
    "real/unitree_sdk_rl/a1_pace_collection_io.cpp": (
        "abfd909b498dc4349ea70db3d5d94f49febb8bf63472c21448c03be6f3405e8d"
    ),
    "core/src/pace_collector.cpp": (
        "d887ce20cebebc76d339b3bd48a6fa616bd5ffb818ddf88bc91fac0e92aa080a"
    ),
    "core/include/a1_base/pace_collector.hpp": (
        "3e0de7758be3911cf8636337fe41f3fdb5da84292761f438150b3878c141809b"
    ),
    "scripts/29_a1_pace_collector_execute.sh": (
        "c39a596318010c7bba3c48307c0ff9fc7f9829fc3c2410ca4e18b024d8aeae41"
    ),
    "scripts/a1_pace_ownership_watchdog.py": (
        "8f3c6d05ffb70f18cec39693107b3819cd1a4fa534386a318d78018b40254110"
    ),
    "tools/build_a1_pace_real_provenance.py": (
        "953ab3299116121f46a98d3af3c8dcd817979882d88b53a82f42ce257bb1410e"
    ),
    "scripts/30_a1_pace_finalize_trial.sh": (
        "6d2e831ff762713aef0cfbf574d9c72f98e71d52a3abc9e5d6a8644638aa4479"
    ),
}


class FitGuardError(ValueError):
    """Raised when an input or fitted artifact cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FitGuardError(message)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _require_finite_json(value: Any, label: str = "JSON") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_json(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_json(child, f"{label}[{index}]")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise FitGuardError(f"cannot read strict JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    _require_finite_json(value, str(path))
    return value


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise FitGuardError(f"cannot safely load torch artifact {path}: {exc}") from exc


def _validate_data(data: Any) -> dict[str, torch.Tensor]:
    _require(isinstance(data, dict), "PACE data must be a tensor dictionary")
    _require(
        tuple(data.keys()) == DATA_KEYS or set(data) == set(DATA_KEYS),
        f"PACE data keys must be exactly {list(DATA_KEYS)}",
    )

    typed: dict[str, torch.Tensor] = {}
    for key in DATA_KEYS:
        value = data[key]
        _require(
            isinstance(value, torch.Tensor),
            f"PACE data value {key} must be a torch.Tensor",
        )
        typed[key] = value

    time_tensor = typed["time"]
    measured = typed["dof_pos"]
    desired = typed["des_dof_pos"]
    _require(time_tensor.dtype == torch.float64, "PACE time must use torch.float64")
    _require(measured.dtype == torch.float32, "PACE dof_pos must use torch.float32")
    _require(desired.dtype == torch.float32, "PACE des_dof_pos must use torch.float32")
    _require(
        time_tensor.shape == (SAMPLE_COUNT,),
        f"PACE time must have shape ({SAMPLE_COUNT},)",
    )
    expected_position_shape = (SAMPLE_COUNT, 12)
    _require(
        measured.shape == expected_position_shape,
        f"PACE dof_pos must have shape {expected_position_shape}",
    )
    _require(
        desired.shape == expected_position_shape,
        f"PACE des_dof_pos must have shape {expected_position_shape}",
    )
    for key, value in typed.items():
        _require(value.device.type == "cpu", f"PACE tensor {key} must be on CPU")
        _require(value.is_contiguous(), f"PACE tensor {key} must be contiguous")
        _require(
            bool(torch.isfinite(value).all().item()),
            f"PACE tensor {key} contains a non-finite value",
        )
    expected_time = torch.arange(SAMPLE_COUNT, dtype=torch.float64) * DT_S
    _require(
        torch.equal(time_tensor.cpu(), expected_time),
        "PACE time must equal time[k] = k * 0.002 exactly",
    )

    # Reuse the public Task 5 contract when the installed extension is available.
    try:
        from pace_sim2real.tasks.manager_based.pace.a1_replay import (
            validate_a1_pace_data,
        )
    except ImportError:
        validate_a1_pace_data = None
    if validate_a1_pace_data is not None:
        validate_a1_pace_data(typed)
    return typed


def tensor_payload_sha256(data: Mapping[str, Any]) -> str:
    """Hash the canonical three-key tensor payload independently of torch.save."""

    _require(
        set(data) == set(DATA_KEYS),
        f"tensor payload keys must be exactly {list(DATA_KEYS)}",
    )
    digest = hashlib.sha256()
    for key in DATA_KEYS:
        value = data[key]
        _require(
            isinstance(value, torch.Tensor),
            f"tensor payload value {key} must be a torch.Tensor",
        )
        tensor = value.detach().cpu().contiguous()
        _require(
            bool(torch.isfinite(tensor).all().item()),
            f"tensor payload value {key} contains a non-finite value",
        )
        shape = json.dumps(list(tensor.shape), separators=(",", ":"))
        digest.update(key.encode("ascii") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(shape.encode("ascii") + b"\0")
        try:
            digest.update(tensor.numpy().tobytes(order="C"))
        except (TypeError, RuntimeError) as exc:
            raise FitGuardError(
                f"tensor payload value {key} cannot be encoded canonically: {exc}"
            ) from exc
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _validate_synthetic_excitation(
    value: Any, desired: torch.Tensor
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "synthetic excitation contract is missing")
    excitation = cast(Mapping[str, Any], value)
    expected_keys = {
        "schema_version",
        "formula",
        "dt_s",
        "duration_s",
        "f0_hz",
        "f1_hz",
        "center_rad",
        "nominal_amplitude_rad",
        "amplitude_scale",
        "realized_amplitude_rad",
        "direction",
        "phase_rad",
        "joint_order",
    }
    _require(set(excitation) == expected_keys, "synthetic excitation keys mismatch")
    _require(
        excitation["schema_version"] == "a1_pace_excitation/v1",
        "synthetic excitation schema mismatch",
    )
    _require(
        excitation["dt_s"] == DT_S
        and excitation["duration_s"] == SAMPLE_COUNT * DT_S
        and excitation["f0_hz"] == 0.1
        and excitation["f1_hz"] == 10.0,
        "synthetic excitation rate/frequency contract mismatch",
    )
    _require(
        tuple(excitation["joint_order"]) == A1_PACE_JOINT_ORDER,
        "synthetic excitation joint order mismatch",
    )
    centers = excitation["center_rad"]
    nominal = excitation["nominal_amplitude_rad"]
    realized = excitation["realized_amplitude_rad"]
    directions = excitation["direction"]
    phases = excitation["phase_rad"]
    scale = excitation["amplitude_scale"]
    _require(
        centers == [0.0] * 4 + [0.8] * 4 + [-1.5] * 4
        and nominal == [0.12] * 4 + [0.18] * 4 + [0.2] * 4,
        "synthetic excitation center/amplitude contract mismatch",
    )
    _require(scale in (0.5, 1.0), "synthetic amplitude scale mismatch")
    _require(
        isinstance(realized, list)
        and len(realized) == 12
        and all(
            float(value) == float(base) * float(scale)
            for value, base in zip(realized, nominal, strict=True)
        ),
        "synthetic realized amplitudes mismatch",
    )
    _require(
        isinstance(directions, list)
        and len(directions) == 12
        and all(value in (-1.0, 1.0) for value in directions),
        "synthetic excitation directions mismatch",
    )
    _require(
        isinstance(phases, list)
        and len(phases) == 12
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in phases
        ),
        "synthetic excitation phases are invalid",
    )
    time_tensor = torch.arange(SAMPLE_COUNT, dtype=torch.float64) * DT_S
    beta = (10.0 - 0.1) / (SAMPLE_COUNT * DT_S)
    phase_tensor = 2.0 * math.pi * (
        0.1 * time_tensor + 0.5 * beta * time_tensor.square()
    )
    reconstructed = torch.tensor(centers, dtype=torch.float64)[None, :] + (
        torch.tensor(realized, dtype=torch.float64)[None, :]
        * torch.tensor(directions, dtype=torch.float64)[None, :]
        * torch.sin(
            phase_tensor[:, None]
            + torch.tensor(phases, dtype=torch.float64)[None, :]
        )
    )
    _require(
        torch.equal(reconstructed.to(torch.float32), desired.detach().cpu()),
        "synthetic targets do not implement the declared A1_Base chirp",
    )
    return dict(excitation)


def validate_capacity_report(
    report_path: str | Path,
    *,
    repository_root: Path,
    num_envs: int,
) -> dict[str, Any]:
    """Bind a fit population to the current isolated-process capacity proof."""

    path = Path(report_path).resolve()
    _require(path.is_file() and not path.is_symlink(), "capacity report is missing")
    report = _load_json(path)
    _require(
        report.get("schema_version") == "a1_pace_capacity/v1",
        "capacity report schema mismatch",
    )
    _require(report.get("status") == "PASS", "capacity report is not PASS")
    _require(report.get("task") == A1_TASK, "capacity report task mismatch")
    _require(
        type(num_envs) is int and num_envs > 0,
        "fit population must be a positive integer",
    )
    _require(
        report.get("formal_num_envs") == num_envs,
        "fit population differs from capacity formal_num_envs",
    )
    script_path = repository_root / "scripts" / "pace" / "a1_capacity_smoke.py"
    _require(script_path.is_file(), "capacity script is missing")
    _require(
        report.get("capacity_script_sha256") == _sha256_path(script_path),
        "capacity report does not match the current script",
    )
    _require(
        report.get("repository_revision") == _git_head(repository_root),
        "capacity report repository revision differs from the fit runtime",
    )
    _require(
        report.get("a1_extension_source_sha256")
        == _a1_extension_source_hashes(repository_root),
        "capacity report A1 extension sources differ from the fit runtime",
    )
    contract = report.get("fit_buffer_contract")
    _require(
        contract
        == {
            "sample_count": SAMPLE_COUNT,
            "joint_count": 12,
            "parameter_count": 49,
            "save_optimization_process": True,
        },
        "capacity fit-buffer contract mismatch",
    )
    repeats = report.get("repeats")
    records = report.get("records")
    _require(
        type(repeats) is int and repeats >= 2,
        "capacity report requires at least two cold-process repeats",
    )
    _require(isinstance(records, list), "capacity report records are missing")
    records = cast(list[Any], records)
    repeats = cast(int, repeats)
    selected = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("candidate") == num_envs
    ]
    _require(
        len(selected) == repeats
        and {record.get("repeat") for record in selected} == set(range(repeats))
        and all(
            record.get("status") == "PASS" and record.get("exit_code") == 0
            for record in selected
        ),
        "formal capacity candidate did not pass every isolated repeat",
    )
    return {
        "report_sha256": _sha256_path(path),
        "formal_num_envs": num_envs,
        "repeats": repeats,
        "capacity_script_sha256": report["capacity_script_sha256"],
    }


def validate_sensitivity_report(
    report_path: str | Path,
    *,
    repository_root: Path,
    runtime_bounds: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    """Bind fitting to the current 49-dimensional numerical sensitivity proof."""

    path = Path(report_path).resolve()
    _require(path.is_file() and not path.is_symlink(), "sensitivity report is missing")
    report = _load_json(path)
    _require(
        report.get("schema_version") == "a1_pace_sensitivity_gate/v1",
        "sensitivity report schema mismatch",
    )
    _require(
        report.get("status") == "PASS" and report.get("accepted") is True,
        "sensitivity report is not PASS",
    )
    _require(report.get("task") == A1_TASK, "sensitivity report task mismatch")
    _require(
        report.get("scope") == "synthetic_numerical_only"
        and report.get("physical_motion_authorized") is False,
        "sensitivity report must remain explicitly non-authorizing",
    )
    script_path = repository_root / "scripts" / "pace" / "a1_sensitivity_gate.py"
    prior_path = repository_root / "tests" / "data" / "a1_parameter_priors.json"
    _require(script_path.is_file(), "sensitivity script is missing")
    _require(prior_path.is_file(), "A1 parameter-prior envelope is missing")
    _require(
        report.get("script_sha256") == _sha256_path(script_path),
        "sensitivity report does not match the current script",
    )
    _require(
        report.get("repository_revision") == _git_head(repository_root),
        "sensitivity report repository revision differs from the fit runtime",
    )
    _require(
        report.get("a1_extension_source_sha256")
        == _a1_extension_source_hashes(repository_root),
        "sensitivity report A1 extension sources differ from the fit runtime",
    )
    _require(
        report.get("prior_sha256") == _sha256_path(prior_path),
        "sensitivity report does not match the current prior envelope",
    )
    dimensions = report.get("dimensions")
    _require(
        isinstance(dimensions, list)
        and len(dimensions) == 49
        and all(
            isinstance(record, Mapping)
            and record.get("index") == index
            and record.get("status") == "PASS"
            for index, record in enumerate(dimensions)
        ),
        "sensitivity report does not PASS all 49 ordered dimensions",
    )
    _require(report.get("failures") == [], "sensitivity report contains failures")
    registered = report.get("registered_task_contract")
    _require(
        isinstance(registered, Mapping)
        and registered.get("bounds") == [list(pair) for pair in runtime_bounds],
        "sensitivity bounds differ from the runtime fit task",
    )
    return {
        "report_sha256": _sha256_path(path),
        "script_sha256": report["script_sha256"],
        "prior_sha256": report["prior_sha256"],
        "scope": report["scope"],
        "physical_motion_authorized": False,
    }


def validate_synthetic_manifest(
    manifest: Mapping[str, Any],
    output_path: str | Path,
    data: Mapping[str, Any],
    task: str,
    *,
    strict_provenance: bool = False,
    known_mean_path: str | Path | None = None,
    runtime_contract: Mapping[str, Any] | None = None,
) -> None:
    """Validate an accepted synthetic producer manifest and its exact output."""

    output_path = Path(output_path)
    _require(output_path.is_file(), f"synthetic output does not exist: {output_path}")
    _require(
        manifest.get("schema_version") == "a1_pace_synthetic/v1",
        "synthetic manifest schema_version mismatch",
    )
    _require(
        manifest.get("scope") == "synthetic_software_gate"
        and manifest.get("physical_motion_authorized") is False,
        "synthetic manifest must remain explicitly non-authorizing",
    )
    acceptance = manifest.get("acceptance")
    _require(
        isinstance(acceptance, Mapping),
        "synthetic manifest acceptance must be an object",
    )
    acceptance = cast(Mapping[str, Any], acceptance)
    _require(acceptance.get("accepted") is True, "synthetic manifest is not accepted")
    _require(
        acceptance.get("status") == "PASS", "synthetic manifest status is not PASS"
    )
    _require(manifest.get("task") == task, "synthetic manifest task mismatch")
    _require(
        manifest.get("num_samples") == SAMPLE_COUNT,
        "synthetic manifest sample count mismatch",
    )
    dt_s = manifest.get("dt_s")
    _require(
        isinstance(dt_s, (int, float))
        and not isinstance(dt_s, bool)
        and math.isfinite(float(dt_s))
        and float(dt_s) == DT_S,
        "synthetic manifest dt mismatch",
    )

    validated = _validate_data(dict(data))
    _require(
        _require_sha256(manifest.get("output_sha256"), "synthetic output SHA")
        == _sha256_path(output_path),
        "synthetic output SHA mismatch",
    )
    expected_payload_hash = tensor_payload_sha256(validated)
    _require(
        _require_sha256(
            manifest.get("tensor_payload_sha256"), "synthetic tensor payload SHA"
        )
        == expected_payload_hash,
        "synthetic tensor payload SHA mismatch",
    )

    reloaded = _validate_data(_load_torch(output_path))
    _require(
        tensor_payload_sha256(reloaded) == expected_payload_hash,
        "synthetic output tensors differ from the validated payload",
    )

    if not strict_provenance:
        return
    _require(
        task == A1_SYNTHETIC_TASK,
        f"strict synthetic provenance accepts only {A1_SYNTHETIC_TASK}",
    )
    _require(
        known_mean_path is not None,
        "strict synthetic provenance requires the known-mean artifact",
    )
    _require(
        runtime_contract is not None,
        "strict synthetic provenance requires the runtime task contract",
    )
    required_hashes = (
        "parameter_file_sha256",
        "target_tensor_sha256",
        "known_mean_sha256",
        "generator_sha256",
    )
    for key in required_hashes:
        _require_sha256(manifest.get(key), f"synthetic {key}")
    target = validated["des_dof_pos"].detach().cpu().contiguous()
    target_sha256 = hashlib.sha256(target.numpy().tobytes(order="C")).hexdigest()
    _require(
        manifest.get("target_tensor_sha256") == target_sha256,
        "synthetic target tensor SHA mismatch",
    )
    _validate_synthetic_excitation(manifest.get("excitation"), target)
    delay = manifest.get("delay_steps")
    _require(
        type(delay) is int and delay in ALLOWED_SYNTHETIC_DELAYS,
        "synthetic delay_steps is not an allowed gate delay",
    )
    vector = manifest.get("physical_parameter_vector")
    _require(
        isinstance(vector, list) and len(vector) == 49,
        "synthetic effective physical parameter vector must have length 49",
    )
    vector = cast(list[Any], vector)
    _require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in vector
        ),
        "synthetic effective physical parameter vector must be finite",
    )
    known_mean_path = Path(cast(str | Path, known_mean_path))
    _require(
        known_mean_path.is_file(),
        f"synthetic known mean does not exist: {known_mean_path}",
    )
    _require(
        manifest.get("known_mean_sha256") == _sha256_path(known_mean_path),
        "synthetic known-mean SHA mismatch",
    )
    known_mean = _load_torch(known_mean_path)
    _require(
        isinstance(known_mean, torch.Tensor)
        and known_mean.shape == (49,)
        and known_mean.dtype == torch.float32
        and bool(torch.isfinite(known_mean).all().item()),
        "synthetic known mean must be a finite float32 tensor with shape (49,)",
    )
    manifest_vector = torch.tensor(vector, dtype=torch.float32)
    _require(
        torch.equal(known_mean.detach().cpu().contiguous(), manifest_vector),
        "synthetic known mean differs from the manifest physical vector",
    )
    runtime_contract = cast(Mapping[str, Any], runtime_contract)
    runtime_bounds = runtime_contract.get("bounds")
    _require(
        isinstance(runtime_bounds, list) and len(runtime_bounds) == 49,
        "runtime task contract is missing 49 parameter bounds",
    )
    runtime_bounds = cast(list[Any], runtime_bounds)
    for index, (value, pair) in enumerate(
        zip(vector, runtime_bounds, strict=True)
    ):
        _require(
            isinstance(pair, list)
            and len(pair) == 2
            and all(
                isinstance(bound, (int, float))
                and not isinstance(bound, bool)
                and math.isfinite(float(bound))
                for bound in pair
            ),
            f"runtime bounds[{index}] is invalid",
        )
        low, high = float(pair[0]), float(pair[1])
        _require(
            low <= float(value) <= high,
            f"synthetic physical vector[{index}] is outside runtime bounds",
        )
    _require(
        int(float(vector[48])) == delay and float(vector[48]) == float(delay),
        "synthetic effective vector delay does not match delay_steps",
    )
    _require(
        tuple(manifest.get("pace_joint_order", ())) == A1_PACE_JOINT_ORDER,
        "synthetic manifest PACE joint order mismatch",
    )
    producer_environment = manifest.get("environment")
    _require(
        isinstance(producer_environment, Mapping),
        "synthetic manifest is missing environment provenance",
    )
    producer_environment = cast(Mapping[str, Any], producer_environment)
    environment_pairs = {
        "dt_s": "sim_dt_s",
        "decimation": "decimation",
        "data_dir": "data_dir",
        "joint_order": "joint_order",
        "fix_root_link": "fix_root_link",
        "root_initial_position": "root_initial_position",
        "action_scale": "action_scale",
        "use_default_offset": "use_default_offset",
    }
    for producer_key, runtime_key in environment_pairs.items():
        _require(
            producer_environment.get(producer_key)
            == runtime_contract.get(runtime_key),
            f"synthetic producer/runtime environment mismatch: {producer_key}",
        )
    expected_control = {
        "class_type": "pace_sim2real.utils.pace_actuator.PaceDCMotor",
        "saturation_effort_nm": 33.5,
        "effort_limit_nm": 33.5,
        "velocity_limit_rad_s": 21.0,
        "stiffness_values": [25.0],
        "damping_values": [0.5],
    }
    producer_control = producer_environment.get("actuator_control")
    runtime_control = runtime_contract.get("actuator_control")
    _require(
        isinstance(producer_control, Mapping)
        and isinstance(runtime_control, Mapping),
        "synthetic actuator control provenance is missing",
    )
    producer_control = cast(Mapping[str, Any], producer_control)
    runtime_control = cast(Mapping[str, Any], runtime_control)
    for key, expected in expected_control.items():
        _require(
            producer_control.get(key) == expected
            and runtime_control.get(key) == expected,
            f"synthetic actuator control mismatch: {key}",
        )
    _require(
        producer_control.get("configured_delay_steps") == delay
        and runtime_control.get("configured_delay_steps") == 10,
        "synthetic actuator delay contract mismatch",
    )
    extension_hashes = manifest.get("a1_extension_source_sha256")
    _require(
        isinstance(extension_hashes, Mapping)
        and len(extension_hashes) == 4
        and all(
            isinstance(key, str)
            and _require_sha256(value, f"synthetic A1 extension hash {key}")
            for key, value in extension_hashes.items()
        ),
        "synthetic A1 extension source hashes are invalid",
    )
    _require(
        isinstance(manifest.get("revisions"), Mapping),
        "synthetic manifest is missing repository revisions",
    )
    _require(
        isinstance(manifest.get("exact_command"), (str, list)),
        "synthetic manifest is missing exact_command",
    )
    seed = manifest.get("seed")
    _require(
        type(seed) is int and seed >= 0,
        "synthetic manifest seed must be a non-negative integer",
    )


def validate_synthetic_fit_role(manifest: Mapping[str, Any]) -> dict[str, float]:
    """Require the fitter input to be the frozen full-amplitude excitation."""

    excitation = manifest.get("excitation")
    _require(isinstance(excitation, Mapping), "synthetic fit excitation is missing")
    excitation = cast(Mapping[str, Any], excitation)
    _require(
        excitation.get("amplitude_scale") == 1.0,
        "synthetic fit requires the full-amplitude profile",
    )
    return {"source_amplitude_scale": 1.0}


def _validate_frozen_real_contracts(contracts_dir: Path) -> dict[str, str]:
    paths = {
        "raw_schema_sha256": contracts_dir / "a1_pace_raw_v1.schema.json",
        "trial_plan_schema_sha256": contracts_dir
        / "a1_pace_trial_plan_v1.schema.json",
        "golden_sha256": contracts_dir / "a1_pace_golden_v1.csv",
    }
    for key, path in paths.items():
        _require(path.is_file() and not path.is_symlink(), f"missing frozen contract: {path}")
        _require(
            _sha256_path(path) == REAL_CONTRACT_SHA256[key],
            f"frozen real contract hash mismatch: {path.name}",
        )
    hash_manifest_path = contracts_dir / "a1_pace_contract_v1.hashes.json"
    hash_manifest = _load_json(hash_manifest_path)
    _require(
        hash_manifest.get("schema_version") == "a1_pace_contract_hashes/v1"
        and hash_manifest.get("hash_algorithm") == "sha256",
        "frozen contract hash-manifest metadata mismatch",
    )
    for key, expected in REAL_CONTRACT_SHA256.items():
        _require(
            hash_manifest.get(key) == expected,
            f"frozen contract hash-manifest value mismatch: {key}",
        )
    return {
        **REAL_CONTRACT_SHA256,
        "hash_manifest_sha256": _sha256_path(hash_manifest_path),
    }


def _validate_frozen_control(control: Any, label: str) -> None:
    _require(isinstance(control, Mapping), f"{label} control is missing")
    control = cast(Mapping[str, Any], control)
    expected = {
        "kp": 25.0,
        "kd": 0.5,
        "dq_des": 0.0,
        "tau_ff": 0.0,
        "effort_limit_nm": 33.5,
        "velocity_limit_rad_s": 21.0,
        "q_target_slew_limit_rad_s": 20.0,
        "power_protect_factor": 1,
    }
    for key, value in expected.items():
        _require(control.get(key) == value, f"{label} control mismatch: {key}")


def _gogo_converter_source_hashes(gogo_learn_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in GOGO_CONVERTER_SOURCE_SHA256.items():
        path = gogo_learn_root / relative
        _require(
            path.is_file() and not path.is_symlink(),
            f"Gogo converter source is missing: {relative}",
        )
        observed[relative] = _sha256_path(path)
        _require(
            observed[relative] == expected,
            f"Gogo Task 8 converter source hash mismatch: {relative}",
        )
    return observed


def _a1_real_collector_source_hashes(a1_base_root: Path) -> dict[str, str]:
    _require(
        PINNED_A1_REAL_COLLECTOR_COMMIT is not None
        and bool(A1_REAL_COLLECTOR_SOURCE_SHA256),
        "real collector pins are unavailable until reviewed Task 10 completion",
    )
    observed: dict[str, str] = {}
    for relative, expected in A1_REAL_COLLECTOR_SOURCE_SHA256.items():
        path = a1_base_root / relative
        _require(
            path.is_file() and not path.is_symlink(),
            f"A1 real collector source is missing: {relative}",
        )
        observed[relative] = _sha256_path(path)
        _require(
            observed[relative] == expected,
            f"A1 real collector source hash mismatch: {relative}",
        )
    return observed


def _rerun_real_conversion(
    *,
    raw_manifest_path: Path,
    conversion_report_path: Path,
    raw_csv_path: Path,
    trial_plan_path: Path,
    contracts_dir: Path,
    data_path: Path,
    data: Mapping[str, Any],
    gogo_learn_root: Path,
    python_executable: Path,
    jsonschema_path: Path | None,
) -> dict[str, Any]:
    """Rebuild Task 8 output with the pinned converter and compare exact payloads."""

    gogo_learn_root = gogo_learn_root.resolve()
    python_executable = python_executable.resolve()
    _require(gogo_learn_root.is_dir(), "Gogo repository does not exist")
    _require(python_executable.is_file(), "converter Python executable is missing")
    converter_revision = _git_head(gogo_learn_root)
    converter_sources = _gogo_converter_source_hashes(gogo_learn_root)
    contract_paths = tuple(
        contracts_dir / name
        for name in (
            "a1_pace_raw_v1.schema.json",
            "a1_pace_trial_plan_v1.schema.json",
            "a1_pace_golden_v1.csv",
            "a1_pace_contract_v1.hashes.json",
        )
    )
    immutable_paths = (
        raw_manifest_path,
        conversion_report_path,
        raw_csv_path,
        trial_plan_path,
        data_path,
        *contract_paths,
        *(
            gogo_learn_root / relative
            for relative in GOGO_CONVERTER_SOURCE_SHA256
        ),
    )
    for path in immutable_paths:
        _require(
            path.is_file() and not path.is_symlink(),
            f"real provenance input must be a regular non-symlink file: {path}",
        )
    immutable_hashes = {path.resolve(): _sha256_path(path) for path in immutable_paths}

    supplied_report = _load_json(conversion_report_path)
    supplied_data = _validate_data(dict(data))
    supplied_payload_sha = tensor_payload_sha256(supplied_data)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if jsonschema_path is not None:
        resolved_jsonschema = jsonschema_path.resolve()
        _require(
            resolved_jsonschema.is_dir(),
            "A1 PACE jsonschema fallback directory does not exist",
        )
        environment["A1_PACE_JSONSCHEMA_PATH"] = str(resolved_jsonschema)

    with tempfile.TemporaryDirectory(prefix="a1-pace-reconvert-") as directory:
        temporary_root = Path(directory)
        runtime_root = temporary_root / "runtime"
        input_root = runtime_root / "input"
        snapshot_contracts = runtime_root / "contracts"
        input_root.mkdir(parents=True)
        snapshot_contracts.mkdir(parents=True)
        source_basenames = {
            raw_manifest_path.name,
            raw_csv_path.name,
            trial_plan_path.name,
        }
        _require(
            len(source_basenames) == 3,
            "raw manifest, CSV, and trial plan basenames must be distinct",
        )
        snapshot_manifest = input_root / raw_manifest_path.name
        snapshot_csv = input_root / raw_csv_path.name
        snapshot_trial_plan = input_root / trial_plan_path.name
        for source, destination in (
            (raw_manifest_path, snapshot_manifest),
            (raw_csv_path, snapshot_csv),
            (trial_plan_path, snapshot_trial_plan),
        ):
            shutil.copy2(source, destination)
            _require(
                _sha256_path(destination) == immutable_hashes[source.resolve()],
                f"immutable conversion snapshot mismatch: {source.name}",
            )
        for source in contract_paths:
            destination = snapshot_contracts / source.name
            shutil.copy2(source, destination)
            _require(
                _sha256_path(destination) == immutable_hashes[source.resolve()],
                f"immutable contract snapshot mismatch: {source.name}",
            )
        for relative in GOGO_CONVERTER_SOURCE_SHA256:
            source = gogo_learn_root / relative
            destination = runtime_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            _require(
                _sha256_path(destination) == immutable_hashes[source.resolve()],
                f"immutable converter snapshot mismatch: {relative}",
            )
        regenerated_data_path = temporary_root / "chirp_data.pt"
        regenerated_report_path = temporary_root / "conversion.report.json"
        inherited_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(runtime_root)
            if not inherited_pythonpath
            else str(runtime_root) + os.pathsep + inherited_pythonpath
        )
        command = [
            str(python_executable),
            "-B",
            "-m",
            "gogo_learn.pace.convert_trial",
            "--raw-csv",
            str(snapshot_csv),
            "--manifest",
            str(snapshot_manifest),
            "--trial-plan",
            str(snapshot_trial_plan),
            "--output",
            str(regenerated_data_path),
            "--report",
            str(regenerated_report_path),
            "--contracts-dir",
            str(snapshot_contracts),
        ]
        child = subprocess.run(
            command,
            cwd=runtime_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        details = (child.stderr or child.stdout).strip()[-4000:]
        _require(
            child.returncode == 0,
            f"Task 8 converter replay failed with code {child.returncode}: {details}",
        )
        _require(
            regenerated_data_path.is_file() and regenerated_report_path.is_file(),
            "Task 8 converter replay did not publish both outputs",
        )
        regenerated_data = _validate_data(_load_torch(regenerated_data_path))
        for key in DATA_KEYS:
            _require(
                torch.equal(regenerated_data[key], supplied_data[key]),
                f"Task 8 converter replay tensor mismatch: {key}",
            )
        regenerated_payload_sha = tensor_payload_sha256(regenerated_data)
        _require(
            regenerated_payload_sha == supplied_payload_sha,
            "Task 8 converter replay payload SHA mismatch",
        )
        regenerated_report = _load_json(regenerated_report_path)
        _require(
            regenerated_report.get("output_sha256")
            == _sha256_path(regenerated_data_path),
            "Task 8 regenerated report does not bind its output",
        )
        supplied_comparable = dict(supplied_report)
        regenerated_comparable = dict(regenerated_report)
        supplied_comparable.pop("output_sha256", None)
        regenerated_comparable.pop("output_sha256", None)
        _require(
            regenerated_comparable == supplied_comparable,
            "Task 8 converter replay report differs from the supplied report",
        )
        regenerated_data_sha = _sha256_path(regenerated_data_path)
        regenerated_report_sha = _sha256_path(regenerated_report_path)

    _require(
        all(_sha256_path(path) == digest for path, digest in immutable_hashes.items()),
        "real provenance input or converter source changed during replay",
    )
    _require(
        _git_head(gogo_learn_root) == converter_revision,
        "Gogo repository revision changed during converter replay",
    )
    return {
        "converter_repository_revision": converter_revision,
        "converter_source_sha256": converter_sources,
        "converter_replay_command": command,
        "regenerated_output_sha256": regenerated_data_sha,
        "regenerated_report_sha256": regenerated_report_sha,
        "regenerated_tensor_payload_sha256": regenerated_payload_sha,
    }


def _validate_real_provenance(
    raw_manifest_path: Path,
    conversion_report_path: Path,
    raw_csv_path: Path,
    trial_plan_path: Path,
    contracts_dir: Path,
    data_path: Path,
    data: Mapping[str, Any],
    *,
    gogo_learn_root: Path,
    a1_base_root: Path,
    python_executable: Path = Path(sys.executable),
    jsonschema_path: Path | None = None,
    required_amplitude_scale: float = 1.0,
) -> dict[str, Any]:
    raw = _load_json(raw_manifest_path)
    conversion = _load_json(conversion_report_path)
    trial_plan = _load_json(trial_plan_path)
    contract_hashes = _validate_frozen_real_contracts(contracts_dir)
    _require(
        raw.get("schema_version") == "a1_pace_raw/v1",
        "raw manifest schema_version mismatch",
    )
    acceptance = raw.get("acceptance")
    _require(
        isinstance(acceptance, dict)
        and acceptance.get("accepted") is True
        and acceptance.get("status") == "PASS",
        "raw manifest is not accepted",
    )
    _require(
        raw.get("fit_window_sample_count") == SAMPLE_COUNT,
        "raw manifest fit sample count mismatch",
    )
    control = raw.get("control")
    _require(
        isinstance(control, Mapping)
        and control.get("dt_s") == DT_S
        and control.get("duration_s") == 20.0
        and control.get("f0_hz") == 0.1
        and control.get("f1_hz") == 10.0,
        "raw manifest excitation timing mismatch",
    )
    _validate_frozen_control(control, "raw manifest")
    _require(
        trial_plan.get("schema_version") == "a1_pace_trial_plan/v1"
        and trial_plan.get("realized") is True
        and trial_plan.get("robot_model") == "A1"
        and trial_plan.get("trial_kind") == "full"
        and trial_plan.get("dt_s") == DT_S
        and trial_plan.get("duration_s") == 20.0
        and trial_plan.get("sample_count") == SAMPLE_COUNT
        and trial_plan.get("f0_hz") == 0.1
        and trial_plan.get("f1_hz") == 10.0
        and trial_plan.get("amplitude_scale") == required_amplitude_scale,
        "real fit trial-plan identity/excitation contract mismatch",
    )
    _validate_frozen_control(trial_plan.get("control"), "trial plan")
    _require(
        tuple(trial_plan.get("pace_joint_order", ())) == A1_PACE_JOINT_ORDER
        and tuple(trial_plan.get("active_pace_joints", ()))
        == A1_PACE_JOINT_ORDER,
        "real fit trial plan does not activate all PACE joints in order",
    )

    _require(
        conversion.get("schema_version") == "a1_pace_conversion_report/v1",
        "conversion report schema_version mismatch",
    )
    _require(conversion.get("status") == "PASS", "conversion report status is not PASS")
    _require(
        conversion.get("manifest_sha256") == _sha256_path(raw_manifest_path),
        "conversion report raw-manifest SHA mismatch",
    )
    _require(
        conversion.get("output_sha256") == _sha256_path(data_path),
        "conversion report output SHA mismatch",
    )
    _require(
        conversion.get("raw_csv_sha256") == _sha256_path(raw_csv_path),
        "conversion report raw CSV SHA mismatch",
    )
    _require(
        conversion.get("trial_plan_sha256") == _sha256_path(trial_plan_path),
        "conversion report trial-plan SHA mismatch",
    )
    for key, expected in REAL_CONTRACT_SHA256.items():
        _require(
            conversion.get(key) == expected,
            f"conversion report frozen contract mismatch: {key}",
        )
    _require(
        conversion.get("fit_sample_count") == SAMPLE_COUNT,
        "conversion report fit sample count mismatch",
    )
    _require(
        conversion.get("pairing") == "q[k]_before_same_row_final_sent_q[k]",
        "conversion report sample-pairing contract mismatch",
    )
    _require(
        conversion.get("time_grid") == "time[k]=k*0.002",
        "conversion report time-grid contract mismatch",
    )
    _require(
        conversion.get("interpolation_or_repair") is False,
        "conversion report used interpolation or repair",
    )
    _require(
        tuple(raw.get("pace_joint_order", ())) == A1_PACE_JOINT_ORDER,
        "raw manifest PACE joint order mismatch",
    )
    _require(
        tuple(conversion.get("pace_joint_order", ())) == A1_PACE_JOINT_ORDER,
        "conversion report PACE joint order mismatch",
    )
    artifacts = raw.get("artifacts")
    _require(isinstance(artifacts, dict), "raw manifest artifacts must be an object")
    artifacts = cast(dict[str, Any], artifacts)
    _require(
        artifacts.get("csv_sha256") == conversion.get("raw_csv_sha256"),
        "raw manifest and conversion report CSV SHAs differ",
    )
    _require(
        raw.get("trial_plan_sha256") == conversion.get("trial_plan_sha256"),
        "raw manifest and conversion report trial-plan SHAs differ",
    )
    _require(
        artifacts.get("csv_sha256") == _sha256_path(raw_csv_path)
        and raw.get("trial_plan_sha256") == _sha256_path(trial_plan_path),
        "raw manifest does not bind the supplied CSV/trial plan",
    )
    rejection_counts = raw.get("rejection_counts")
    _require(
        isinstance(rejection_counts, Mapping)
        and bool(rejection_counts)
        and all(value == 0 for value in rejection_counts.values()),
        "raw manifest contains a nonzero rejection count",
    )
    _require(
        conversion.get("tensor_dtypes")
        == {
            "time": "torch.float64",
            "dof_pos": "torch.float32",
            "des_dof_pos": "torch.float32",
        },
        "conversion report tensor dtype contract mismatch",
    )
    raw_provenance = raw.get("provenance")
    trial_revisions = trial_plan.get("code_revisions")
    _require(
        isinstance(raw_provenance, Mapping)
        and isinstance(trial_revisions, Mapping),
        "real provenance repository revisions are missing",
    )
    raw_provenance = cast(Mapping[str, Any], raw_provenance)
    trial_revisions = cast(Mapping[str, Any], trial_revisions)
    current_a1_revision = _git_head(a1_base_root.resolve())
    current_gogo_revision = _git_head(gogo_learn_root.resolve())
    pinned_a1_commit = PINNED_A1_REAL_COLLECTOR_COMMIT
    _require(
        isinstance(pinned_a1_commit, str)
        and _git_commit_is_ancestor(
            a1_base_root.resolve(), pinned_a1_commit, current_a1_revision
        ),
        "A1_Base checkout does not descend from the reviewed real collector commit",
    )
    _require(
        _git_commit_is_ancestor(
            gogo_learn_root.resolve(),
            PINNED_GOGO_TASK8_COMMIT,
            current_gogo_revision,
        ),
        "Gogo checkout does not descend from the reviewed Task 8 commit",
    )
    a1_collector_sources = _a1_real_collector_source_hashes(
        a1_base_root.resolve()
    )
    _require(
        _git_worktree_is_clean(a1_base_root.resolve()),
        "A1_Base checkout must be clean for real provenance",
    )
    _require(
        _git_worktree_is_clean(
            gogo_learn_root.resolve(),
            allowed_untracked=ALLOWED_GOGO_UNTRACKED,
        ),
        "Gogo checkout must be clean for real provenance",
    )
    _require(
        raw_provenance.get("a1_base_commit")
        == trial_revisions.get("a1_base_commit")
        == current_a1_revision,
        "real provenance A1_Base revision differs from the current checkout",
    )
    _require(
        raw_provenance.get("gogo_learn_commit")
        == trial_revisions.get("gogo_learn_commit")
        == current_gogo_revision,
        "real provenance Gogo revision differs from the current checkout",
    )
    converter_replay = _rerun_real_conversion(
        raw_manifest_path=raw_manifest_path,
        conversion_report_path=conversion_report_path,
        raw_csv_path=raw_csv_path,
        trial_plan_path=trial_plan_path,
        contracts_dir=contracts_dir,
        data_path=data_path,
        data=data,
        gogo_learn_root=gogo_learn_root,
        python_executable=python_executable,
        jsonschema_path=jsonschema_path,
    )
    _require(
        _git_worktree_is_clean(a1_base_root.resolve())
        and _git_worktree_is_clean(
            gogo_learn_root.resolve(),
            allowed_untracked=ALLOWED_GOGO_UNTRACKED,
        ),
        "A1_Base or Gogo checkout changed during real provenance replay",
    )
    return {
        "raw_manifest_sha256": _sha256_path(raw_manifest_path),
        "conversion_report_sha256": _sha256_path(conversion_report_path),
        "raw_csv_sha256": _require_sha256(
            conversion.get("raw_csv_sha256"), "conversion raw CSV SHA"
        ),
        "trial_plan_sha256": _require_sha256(
            conversion.get("trial_plan_sha256"), "conversion trial-plan SHA"
        ),
        "frozen_contract_sha256": contract_hashes,
        "tensor_payload_sha256": tensor_payload_sha256(data),
        "a1_base_repository_revision": current_a1_revision,
        "gogo_repository_revision": current_gogo_revision,
        "pinned_a1_real_collector_commit": pinned_a1_commit,
        "pinned_gogo_task8_commit": PINNED_GOGO_TASK8_COMMIT,
        "a1_real_collector_source_sha256": a1_collector_sources,
        "required_amplitude_scale": required_amplitude_scale,
        "converter_replay": converter_replay,
    }


def build_official_fit_command(
    *,
    isaaclab_root: str | Path,
    repository_root: str | Path,
    task: str,
    num_envs: int,
    headless: bool,
) -> list[str]:
    """Build the argv for the unchanged upstream fit.py entry point."""

    _require(
        type(num_envs) is int and num_envs > 0, "num_envs must be a positive integer"
    )
    launcher = (Path(isaaclab_root) / "isaaclab.sh").as_posix()
    fit_script = (Path(repository_root) / "scripts" / "pace" / "fit.py").as_posix()
    command = [
        launcher,
        "-p",
        fit_script,
        "--task",
        task,
        "--num_envs",
        str(num_envs),
    ]
    if headless:
        command.append("--headless")
    return command


def select_single_new_run(before: set[str], after: set[str]) -> str:
    """Return the only newly created run name, never using timestamps."""

    _require(before <= after, "fit removed or renamed a pre-existing run")
    created = after - before
    _require(len(created) == 1, "fit must create exactly one new run directory")
    return next(iter(created))


def select_final_mean(run_dir: str | Path) -> Path:
    """Select the numerically largest official mean_NNN.pt checkpoint."""

    run_dir = Path(run_dir)
    _require(
        run_dir.is_dir() and not run_dir.is_symlink(),
        f"fit run is not a real directory: {run_dir}",
    )
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.iterdir():
        match = MEAN_PATTERN.fullmatch(path.name)
        if match is not None and path.is_file() and not path.is_symlink():
            candidates.append((int(match.group(1)), path))
    _require(bool(candidates), "fit run contains no numeric mean_NNN.pt")
    largest = max(index for index, _ in candidates)
    winners = [path for index, path in candidates if index == largest]
    _require(
        len(winners) == 1,
        "fit run contains ambiguous names for the final mean iteration",
    )
    return winners[0]


def classify_cma_termination(
    scores_buffer: torch.Tensor,
    *,
    populated_count: int,
    max_iterations: int,
    epsilon: float,
) -> dict[str, Any]:
    """Prove that a shortened official run satisfied its relative-spread rule."""

    _require(
        isinstance(scores_buffer, torch.Tensor)
        and scores_buffer.ndim == 2
        and scores_buffer.shape[0] == max_iterations,
        "CMA termination scores have an invalid shape",
    )
    _require(
        type(populated_count) is int and 1 <= populated_count <= max_iterations,
        "CMA populated generation count is invalid",
    )
    last_scores = scores_buffer[populated_count - 1]
    minimum = float(last_scores.min().item())
    maximum = float(last_scores.max().item())
    relative_spread = (maximum - minimum) / minimum if minimum > 0.0 else None
    if populated_count < max_iterations:
        _require(
            relative_spread is not None and relative_spread < epsilon,
            "short CMA run does not satisfy the official epsilon stop rule",
        )
        reason = "epsilon_relative_score_spread"
    else:
        reason = "max_iteration"
    return {
        "reason": reason,
        "populated_generation_count": populated_count,
        "last_min_score": minimum,
        "last_max_score": maximum,
        "last_relative_score_spread": relative_spread,
        "epsilon": epsilon,
    }


def _directory_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    _require(
        path.is_dir() and not path.is_symlink(),
        f"fit log root is not a real directory: {path}",
    )
    return {
        child.name
        for child in path.iterdir()
        if child.is_dir() and not child.is_symlink()
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FitGuardError(f"another fit guard owns lock {path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        path.unlink(missing_ok=True)


def _stage_files(pairs: Sequence[tuple[Path, Path, str]]) -> None:
    destinations = [destination.resolve() for _, destination, _ in pairs]
    _require(
        len(destinations) == len(set(destinations)),
        "staging destinations must be unique",
    )
    temporary_by_destination: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for source, destination, expected_hash in pairs:
            _require(source.is_file(), f"staging source does not exist: {source}")
            _require(
                _sha256_path(source) == expected_hash,
                f"staging source changed before copy: {source}",
            )
            if source.resolve() == destination.resolve():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".stage", dir=destination.parent
            )
            temporary = Path(temporary_name)
            with (
                source.open("rb") as input_stream,
                os.fdopen(descriptor, "wb") as output_stream,
            ):
                shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            _require(
                _sha256_path(temporary) == expected_hash,
                f"staged copy hash mismatch: {destination}",
            )
            _require(
                _sha256_path(source) == expected_hash,
                f"staging source changed during copy: {source}",
            )
            temporary_by_destination[destination] = temporary

        for destination in temporary_by_destination:
            if destination.exists():
                descriptor, backup_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".backup",
                    dir=destination.parent,
                )
                os.close(descriptor)
                backup = Path(backup_name)
                backup.unlink()
                os.replace(destination, backup)
                backups[destination] = backup
            os.replace(temporary_by_destination[destination], destination)
            installed.append(destination)
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    except Exception:
        for destination in reversed(installed):
            destination.unlink(missing_ok=True)
            if destination in backups:
                backup = backups.pop(destination)
                os.replace(backup, destination)
        for destination, backup in backups.items():
            if not destination.exists():
                os.replace(backup, destination)
        raise
    finally:
        for temporary in temporary_by_destination.values():
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _probe_task_contract(
    task: str,
    *,
    isaaclab_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Read the registered task contract in a disposable Isaac process."""

    probe_source = """
import json
from pathlib import Path
import sys
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
try:
    import pace_sim2real.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    cfg = parse_env_cfg(sys.argv[2], device="cpu", num_envs=1)
    actuator = cfg.scene.robot.actuators["base_legs"]
    def numeric_values(value):
        values = value.values() if isinstance(value, dict) else (value,)
        return sorted({float(item) for item in values})
    class_type = actuator.class_type
    result = {
        "task": sys.argv[2],
        "data_dir": str(cfg.sim2real.data_dir),
        "robot_name": str(cfg.sim2real.robot_name),
        "sim_dt_s": float(cfg.sim.dt),
        "decimation": int(cfg.decimation),
        "joint_order": list(cfg.sim2real.joint_order),
        "save_optimization_process": bool(
            cfg.sim2real.cmaes.save_optimization_process
        ),
        "max_iteration": int(cfg.sim2real.cmaes.max_iteration),
        "save_interval": int(cfg.sim2real.cmaes.save_interval),
        "epsilon": float(cfg.sim2real.cmaes.epsilon),
        "sigma": float(cfg.sim2real.cmaes.sigma),
        "action_scale": float(cfg.actions.joint_pos.scale),
        "use_default_offset": bool(cfg.actions.joint_pos.use_default_offset),
        "fix_root_link": bool(
            cfg.scene.robot.spawn.articulation_props.fix_root_link
        ),
        "root_initial_position": list(cfg.scene.robot.init_state.pos),
        "bounds_shape": list(cfg.sim2real.bounds_params.shape),
        "bounds": cfg.sim2real.bounds_params.detach().cpu().tolist(),
        "actuator_control": {
            "class_type": f"{class_type.__module__}.{class_type.__qualname__}",
            "saturation_effort_nm": float(actuator.saturation_effort),
            "effort_limit_nm": float(actuator.effort_limit),
            "velocity_limit_rad_s": float(actuator.velocity_limit),
            "stiffness_values": numeric_values(actuator.stiffness),
            "damping_values": numeric_values(actuator.damping),
            "configured_delay_steps": int(actuator.max_delay),
        },
    }
    Path(sys.argv[1]).write_text(
        json.dumps(result, sort_keys=True) + "\\n", encoding="ascii"
    )
finally:
    app.close()
"""
    with tempfile.TemporaryDirectory(prefix=".a1-fit-contract-") as directory:
        root = Path(directory)
        probe_path = root / "probe.py"
        result_path = root / "result.json"
        probe_path.write_text(probe_source, encoding="ascii", newline="\n")
        environment = os.environ.copy()
        environment["PACE_ROOT"] = str(repository_root)
        child = subprocess.run(
            [
                str(isaaclab_root / "isaaclab.sh"),
                "-p",
                "-B",
                str(probe_path),
                str(result_path),
                task,
            ],
            cwd=repository_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        _require(
            child.returncode == 0 and result_path.is_file(),
            "runtime task-contract probe failed: "
            + (child.stderr + child.stdout)[-4000:],
        )
        return _load_json(result_path)


def _git_head(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNAVAILABLE"
    value = result.stdout.strip()
    return value if re.fullmatch(r"[0-9a-f]{40,64}", value) else "UNAVAILABLE"


def _git_worktree_is_clean(
    path: Path, *, allowed_untracked: frozenset[str] = frozenset()
) -> bool:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "-z",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    for record in result.stdout.split("\0"):
        if not record:
            continue
        if record.startswith("?? ") and record[3:] in allowed_untracked:
            continue
        return False
    return True


def _git_commit_is_ancestor(path: Path, ancestor: str, descendant: str) -> bool:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _validate_final_mean(path: Path, repository_root: Path) -> torch.Tensor:
    value = _load_torch(path)
    _require(isinstance(value, torch.Tensor), "final mean must be a torch.Tensor")
    _require(value.shape == (49,), "final mean must have shape (49,)")
    _require(torch.is_floating_point(value), "final mean must use a floating dtype")
    _require(
        bool(torch.isfinite(value).all().item()),
        "final mean contains a non-finite value",
    )
    parser_path = (
        repository_root
        / "source"
        / "pace_sim2real"
        / "pace_sim2real"
        / "tasks"
        / "manager_based"
        / "pace"
        / "a1_mean.py"
    )
    _require(parser_path.is_file(), "Task 5 A1 mean parser is missing")
    spec = importlib.util.spec_from_file_location("a1_pace_mean_contract", parser_path)
    _require(spec is not None and spec.loader is not None, "cannot load A1 mean parser")
    if spec is None or spec.loader is None:
        raise FitGuardError("cannot load A1 mean parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.parse_a1_mean(value)
    return value.detach().cpu().to(dtype=torch.float64).contiguous()


def synthetic_parameter_recovery(
    fitted: torch.Tensor,
    known: Sequence[Any],
    bounds: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    """Report normalized truth error without claiming unique identifiability."""

    _require(fitted.shape == (49,), "fitted mean must have shape (49,)")
    _require(len(known) == 49 and len(bounds) == 49, "recovery contract needs 49 values")
    families = (
        ("armature", range(0, 12)),
        ("viscous_friction", range(12, 24)),
        ("coulomb_friction", range(24, 36)),
        ("encoder_bias", range(36, 48)),
        ("global_delay", range(48, 49)),
    )
    rows: list[dict[str, Any]] = []
    normalized: list[float] = []
    for index, (truth_value, pair) in enumerate(zip(known, bounds, strict=True)):
        _require(
            not isinstance(truth_value, bool)
            and isinstance(pair, Sequence)
            and not isinstance(pair, (str, bytes))
            and len(pair) == 2,
            f"recovery input {index} is invalid",
        )
        try:
            low, high = float(pair[0]), float(pair[1])
            truth = float(truth_value)
            estimate = float(fitted[index])
        except (TypeError, ValueError) as exc:
            raise FitGuardError(f"recovery input {index} is invalid") from exc
        _require(
            all(math.isfinite(value) for value in (low, high, truth, estimate))
            and low < high,
            f"recovery input {index} is invalid",
        )
        error = abs(estimate - truth)
        normalized_error = error / (high - low)
        normalized.append(normalized_error)
        rows.append(
            {
                "index": index,
                "truth": truth,
                "estimate": estimate,
                "absolute_error": error,
                "normalized_bound_error": normalized_error,
            }
        )
    family_rows: dict[str, Any] = {}
    for family, indices in families:
        values = [normalized[index] for index in indices]
        family_rows[family] = {
            "mean_normalized_bound_error": math.fsum(values) / len(values),
            "max_normalized_bound_error": max(values),
        }
    recovered_delay = int(float(fitted[48]))
    expected_delay = int(float(known[48]))
    return {
        "interpretation": (
            "diagnostic_only_parameters_may_be_non_unique; held-out trajectory "
            "replay is the behavioral gate"
        ),
        "per_parameter": rows,
        "families": family_rows,
        "mean_normalized_bound_error": math.fsum(normalized) / len(normalized),
        "max_normalized_bound_error": max(normalized),
        "expected_delay_steps": expected_delay,
        "recovered_delay_steps": recovered_delay,
        "delay_exact": recovered_delay == expected_delay,
    }


def _official_source_hashes(repository_root: Path) -> dict[str, str]:
    paths = {
        "fit.py": repository_root / "scripts" / "pace" / "fit.py",
        "cma_es.py": repository_root
        / "source"
        / "pace_sim2real"
        / "pace_sim2real"
        / "optim"
        / "cma_es.py",
        "pace_actuator.py": repository_root
        / "source"
        / "pace_sim2real"
        / "pace_sim2real"
        / "utils"
        / "pace_actuator.py",
        "pace_sim2real_env_cfg.py": repository_root
        / "source"
        / "pace_sim2real"
        / "pace_sim2real"
        / "tasks"
        / "manager_based"
        / "pace"
        / "pace_sim2real_env_cfg.py",
    }
    observed: dict[str, str] = {}
    for name, path in paths.items():
        _require(path.is_file(), f"official PACE source is missing: {path}")
        observed[name] = _sha256_path(path)
        _require(
            observed[name] == OFFICIAL_SOURCE_SHA256[name],
            f"official PACE v0.1.2 source changed: {name}",
        )
    return observed


def _a1_extension_source_hashes(repository_root: Path) -> dict[str, str]:
    relative_paths = (
        "source/pace_sim2real/pace_sim2real/tasks/manager_based/__init__.py",
        "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/a1_mean.py",
        "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/a1_pace_env_cfg.py",
        "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/a1_replay.py",
    )
    observed: dict[str, str] = {}
    for relative in relative_paths:
        path = repository_root / relative
        _require(path.is_file(), f"A1 extension source is missing: {relative}")
        observed[relative] = _sha256_path(path)
    return observed


def run_fit_guard(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = args.repository_root.resolve()
    isaaclab_root = args.isaaclab_root.resolve()
    source_data = args.data.resolve()
    fit_manifest_output = args.fit_manifest_output.resolve()
    fit_guard_path = Path(__file__).resolve()
    fit_guard_sha = _sha256_path(fit_guard_path)
    _require(args.task in ALLOWED_TASKS, "fit guard task is not an A1 PACE task")
    _require(source_data.is_file(), f"PACE data does not exist: {source_data}")
    _require(
        repository_root.is_dir(), f"PACE repository does not exist: {repository_root}"
    )
    fit_script = repository_root / "scripts" / "pace" / "fit.py"
    isaaclab_script = isaaclab_root / "isaaclab.sh"
    _require(fit_script.is_file(), f"official fit.py does not exist: {fit_script}")
    _require(
        isaaclab_script.is_file(),
        f"Isaac Lab launcher does not exist: {isaaclab_script}",
    )
    official_source_hashes = _official_source_hashes(repository_root)
    a1_extension_source_hashes = _a1_extension_source_hashes(repository_root)

    synthetic_mode = args.manifest is not None
    real_mode = args.raw_manifest is not None or args.conversion_report is not None
    _require(
        synthetic_mode != real_mode,
        "choose exactly one provenance mode: --manifest or real sidecars",
    )
    _require(
        not real_mode
        or (args.raw_manifest is not None and args.conversion_report is not None),
        "real provenance requires both --raw-manifest and --conversion-report",
    )
    _require(
        (synthetic_mode and args.task == A1_SYNTHETIC_TASK)
        or (real_mode and args.task == A1_TASK),
        "synthetic provenance requires the bounded synthetic task; real provenance requires Isaac-Pace-A1-v0",
    )
    _require(
        synthetic_mode == (args.known_mean is not None),
        "synthetic provenance requires --known-mean; real provenance forbids it",
    )
    real_evidence_args = (
        args.raw_csv,
        args.trial_plan,
        args.contracts_dir,
    )
    _require(
        (real_mode and all(value is not None for value in real_evidence_args))
        or (synthetic_mode and all(value is None for value in real_evidence_args)),
        "real provenance requires raw CSV, trial plan, and contracts directory; synthetic provenance forbids them",
    )

    task_contract = _probe_task_contract(
        args.task,
        isaaclab_root=isaaclab_root,
        repository_root=repository_root,
    )
    _require(
        task_contract["data_dir"] == A1_DATA_DIR.as_posix(),
        "runtime env_cfg.sim2real.data_dir does not match a1/chirp_data.pt",
    )
    _require(
        task_contract["robot_name"] == A1_ROBOT_NAME,
        "runtime env_cfg.sim2real.robot_name is not a1",
    )
    _require(
        task_contract["sim_dt_s"] == DT_S and task_contract["decimation"] == 1,
        "runtime A1 task is not the frozen 500 Hz contract",
    )
    _require(
        task_contract["save_optimization_process"] is True,
        "runtime A1 task does not save the optimization process",
    )
    expected_iterations = 200
    expected_save_interval = 10
    _require(
        task_contract["max_iteration"] == expected_iterations
        and task_contract["save_interval"] == expected_save_interval
        and task_contract["epsilon"] == 0.01
        and task_contract["sigma"] == 0.5,
        "runtime A1 task CMA-ES schedule differs from the frozen contract",
    )
    _require(
        task_contract["action_scale"] == 1.0
        and task_contract["use_default_offset"] is False,
        "runtime A1 task is not using absolute joint targets",
    )
    _require(
        task_contract["fix_root_link"] is True
        and task_contract["root_initial_position"] == [0.0, 0.0, 1.0]
        and task_contract["bounds_shape"] == [49, 2],
        "runtime A1 task fixed-base/bounds contract mismatch",
    )
    _require(
        tuple(task_contract["joint_order"]) == A1_PACE_JOINT_ORDER,
        "runtime A1 task PACE joint order mismatch",
    )
    capacity_path = args.capacity_report.resolve()
    sensitivity_path = args.sensitivity_report.resolve()
    capacity_evidence = validate_capacity_report(
        capacity_path,
        repository_root=repository_root,
        num_envs=args.num_envs,
    )
    sensitivity_evidence = validate_sensitivity_report(
        sensitivity_path,
        repository_root=repository_root,
        runtime_bounds=task_contract["bounds"],
    )

    data = _validate_data(_load_torch(source_data))
    source_data_sha = _sha256_path(source_data)
    payload_sha = tensor_payload_sha256(data)
    provenance: dict[str, Any]
    source_sidecars: list[tuple[Path, str, str]]
    additional_inputs: list[Path] = [capacity_path, sensitivity_path]
    if synthetic_mode:
        source_manifest = args.manifest.resolve()
        known_mean = args.known_mean.resolve()
        source_manifest_sha = _sha256_path(source_manifest)
        known_mean_sha = _sha256_path(known_mean)
        manifest = _load_json(source_manifest)
        validate_synthetic_manifest(
            manifest,
            source_data,
            data,
            args.task,
            strict_provenance=True,
            known_mean_path=known_mean,
            runtime_contract=task_contract,
        )
        fit_role = validate_synthetic_fit_role(manifest)
        _require(
            _sha256_path(source_manifest) == source_manifest_sha,
            "synthetic manifest changed during validation",
        )
        provenance = {
            "mode": "synthetic",
            "source_manifest_sha256": source_manifest_sha,
            "parameter_file_sha256": manifest["parameter_file_sha256"],
            "effective_parameter_vector": manifest["physical_parameter_vector"],
            "delay_steps": manifest["delay_steps"],
            "known_mean_sha256": known_mean_sha,
            "source_seed": manifest["seed"],
            "source_target_sha256": manifest["target_tensor_sha256"],
            **fit_role,
        }
        generator_path = (
            repository_root / "scripts" / "pace" / "a1_generate_synthetic.py"
        )
        _require(
            generator_path.is_file(),
            f"synthetic generator does not exist: {generator_path}",
        )
        _require(
            _sha256_path(generator_path) == manifest["generator_sha256"],
            "synthetic generator SHA does not match the current source",
        )
        _require(
            manifest.get("a1_extension_source_sha256")
            == a1_extension_source_hashes,
            "synthetic generator A1 extension sources differ from fit runtime",
        )
        revisions = manifest["revisions"]
        _require(
            revisions.get("pace_sim2real") == _git_head(repository_root)
            and revisions.get("isaaclab") == _git_head(isaaclab_root),
            "synthetic generator revisions differ from the fit runtime",
        )
        source_sidecars = [
            (source_manifest, "chirp_data.manifest.json", source_manifest_sha),
            (known_mean, "chirp_data.known_mean.pt", known_mean_sha),
        ]
    else:
        raw_manifest = args.raw_manifest.resolve()
        conversion_report = args.conversion_report.resolve()
        raw_csv = args.raw_csv.resolve()
        trial_plan = args.trial_plan.resolve()
        contracts_dir = args.contracts_dir.resolve()
        raw_manifest_sha = _sha256_path(raw_manifest)
        conversion_report_sha = _sha256_path(conversion_report)
        provenance = {
            "mode": "real",
            **_validate_real_provenance(
                raw_manifest,
                conversion_report,
                raw_csv,
                trial_plan,
                contracts_dir,
                source_data,
                data,
                gogo_learn_root=args.gogo_learn_root.resolve(),
                a1_base_root=args.a1_base_root.resolve(),
                python_executable=Path(sys.executable),
                jsonschema_path=(
                    args.jsonschema_path.resolve()
                    if args.jsonschema_path is not None
                    else None
                ),
            ),
        }
        _require(
            _sha256_path(raw_manifest) == raw_manifest_sha,
            "raw manifest changed during validation",
        )
        _require(
            _sha256_path(conversion_report) == conversion_report_sha,
            "conversion report changed during validation",
        )
        source_sidecars = [
            (
                raw_manifest,
                "chirp_data.raw.manifest.json",
                raw_manifest_sha,
            ),
            (
                conversion_report,
                "chirp_data.conversion.json",
                conversion_report_sha,
            ),
        ]
        additional_inputs.extend(
            [
                raw_csv,
                trial_plan,
                contracts_dir / "a1_pace_raw_v1.schema.json",
                contracts_dir / "a1_pace_trial_plan_v1.schema.json",
                contracts_dir / "a1_pace_golden_v1.csv",
                contracts_dir / "a1_pace_contract_v1.hashes.json",
                *(
                    args.gogo_learn_root.resolve() / relative
                    for relative in GOGO_CONVERTER_SOURCE_SHA256
                ),
                *(
                    args.a1_base_root.resolve() / relative
                    for relative in A1_REAL_COLLECTOR_SOURCE_SHA256
                ),
            ]
        )

    staged_data = repository_root / "data" / A1_DATA_DIR
    staged_sidecars = [
        (source, staged_data.parent / basename, expected_hash)
        for source, basename, expected_hash in source_sidecars
    ]
    all_inputs = {
        source_data,
        *additional_inputs,
        *(source for source, _, _ in source_sidecars),
    }
    additional_input_hashes = {
        path: _sha256_path(path) for path in additional_inputs
    }
    _require(
        fit_manifest_output not in all_inputs,
        "fit manifest output aliases a source input",
    )
    _require(
        fit_manifest_output != staged_data,
        "fit manifest output aliases the staged data",
    )
    _require(
        fit_manifest_output
        not in {destination.resolve() for _, destination, _ in staged_sidecars},
        "fit manifest output aliases a staged provenance sidecar",
    )

    log_root = repository_root / "logs" / "pace" / A1_ROBOT_NAME
    command = build_official_fit_command(
        isaaclab_root=isaaclab_root,
        repository_root=repository_root,
        task=args.task,
        num_envs=args.num_envs,
        headless=args.headless,
    )
    fit_script_sha = _sha256_path(fit_script)
    lock_path = staged_data.parent / ".a1-fit-guard.lock"
    with _exclusive_lock(lock_path):
        _stage_files(
            [
                (source_data, staged_data, source_data_sha),
                *staged_sidecars,
            ]
        )
        _require(
            _sha256_path(staged_data) == source_data_sha,
            "staged PACE data SHA mismatch",
        )
        staged_data_payload = tensor_payload_sha256(
            _validate_data(_load_torch(staged_data))
        )
        _require(
            staged_data_payload == payload_sha, "staged PACE tensor payload mismatch"
        )
        for _, destination, expected_hash in staged_sidecars:
            _require(
                _sha256_path(destination) == expected_hash,
                f"staged provenance SHA mismatch: {destination}",
            )
        _require(
            _sha256_path(capacity_path) == capacity_evidence["report_sha256"]
            and _sha256_path(sensitivity_path)
            == sensitivity_evidence["report_sha256"],
            "pre-fit gate report changed before official fitting",
        )
        _require(
            all(
                _sha256_path(path) == digest
                for path, digest in additional_input_hashes.items()
            ),
            "an unstaged provenance input changed before official fitting",
        )
        if real_mode:
            _require(
                _git_head(args.a1_base_root.resolve())
                == provenance["a1_base_repository_revision"]
                and _git_head(args.gogo_learn_root.resolve())
                == provenance["gogo_repository_revision"]
                and _git_worktree_is_clean(args.a1_base_root.resolve())
                and _git_worktree_is_clean(
                    args.gogo_learn_root.resolve(),
                    allowed_untracked=ALLOWED_GOGO_UNTRACKED,
                ),
                "real producer/converter checkout changed before official fitting",
            )

        before = _directory_names(log_root)
        environment = os.environ.copy()
        environment["PACE_ROOT"] = str(repository_root)
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=repository_root,
            env=environment,
            check=False,
        )
        duration_s = time.monotonic() - started
        _require(
            result.returncode == 0,
            f"official fit.py exited with code {result.returncode}",
        )
        _require(
            _sha256_path(fit_script) == fit_script_sha,
            "official fit.py changed while fitting",
        )
        _require(
            _official_source_hashes(repository_root) == official_source_hashes,
            "official PACE source set changed while fitting",
        )
        _require(
            _a1_extension_source_hashes(repository_root)
            == a1_extension_source_hashes,
            "A1 extension source set changed while fitting",
        )
        _require(
            _sha256_path(fit_guard_path) == fit_guard_sha,
            "fit guard source changed while fitting",
        )
        _require(
            _sha256_path(staged_data) == source_data_sha,
            "staged PACE data changed while fitting",
        )
        for _, destination, expected_hash in staged_sidecars:
            _require(
                _sha256_path(destination) == expected_hash,
                f"staged provenance changed while fitting: {destination}",
            )
        _require(
            _sha256_path(capacity_path) == capacity_evidence["report_sha256"]
            and _sha256_path(sensitivity_path)
            == sensitivity_evidence["report_sha256"],
            "pre-fit gate report changed while fitting",
        )
        _require(
            all(
                _sha256_path(path) == digest
                for path, digest in additional_input_hashes.items()
            ),
            "an unstaged provenance input changed while fitting",
        )
        if real_mode:
            _require(
                _git_head(args.a1_base_root.resolve())
                == provenance["a1_base_repository_revision"]
                and _git_head(args.gogo_learn_root.resolve())
                == provenance["gogo_repository_revision"]
                and _git_worktree_is_clean(args.a1_base_root.resolve())
                and _git_worktree_is_clean(
                    args.gogo_learn_root.resolve(),
                    allowed_untracked=ALLOWED_GOGO_UNTRACKED,
                ),
                "real producer/converter checkout changed while fitting",
            )

        after = _directory_names(log_root)
        run_basename = select_single_new_run(before, after)
        run_dir = log_root / run_basename
        _require(
            run_dir.is_dir() and not run_dir.is_symlink(),
            "new fit run is not a real directory",
        )
        config_path = run_dir / "config.pt"
        config = _load_torch(config_path)
        _require(isinstance(config, dict), "fit config.pt must be a dictionary")
        _require(
            set(config)
            == {"bounds", "joint_order", "dof_pos", "des_dof_pos", "time"},
            "fit config.pt keys mismatch",
        )
        config_data = _validate_data({key: config.get(key) for key in DATA_KEYS})
        config_payload_sha = tensor_payload_sha256(config_data)
        _require(
            config_payload_sha == staged_data_payload,
            "fit config tensors do not equal the staged input tensors",
        )
        config_bounds = config["bounds"]
        _require(
            isinstance(config_bounds, torch.Tensor)
            and config_bounds.shape == (49, 2)
            and config_bounds.detach().cpu().tolist() == task_contract["bounds"],
            "fit config bounds differ from the runtime task",
        )
        _require(
            tuple(config["joint_order"]) == A1_PACE_JOINT_ORDER,
            "fit config joint order mismatch",
        )

        mean_path = select_final_mean(run_dir)
        final_mean = _validate_final_mean(mean_path, repository_root)
        recovery: dict[str, Any] | None = None
        if synthetic_mode:
            recovery = synthetic_parameter_recovery(
                final_mean,
                provenance["effective_parameter_vector"],
                task_contract["bounds"],
            )
            _require(
                recovery["delay_exact"] is True,
                "official fit did not recover the known integer delay",
            )
        progress_path = run_dir / "progress.pt"
        progress = _load_torch(progress_path)
        _require(isinstance(progress, dict), "fit progress.pt must be a dictionary")
        _require(
            set(progress) == {"params_buffer", "scores_buffer"},
            "fit progress.pt keys mismatch",
        )
        params_buffer = progress["params_buffer"]
        scores_buffer = progress["scores_buffer"]
        _require(
            isinstance(params_buffer, torch.Tensor)
            and params_buffer.shape == (expected_iterations, args.num_envs, 49)
            and bool(torch.isfinite(params_buffer).all().item()),
            "fit progress params_buffer shape/finite contract mismatch",
        )
        _require(
            isinstance(scores_buffer, torch.Tensor)
            and scores_buffer.shape == (expected_iterations, args.num_envs)
            and bool(torch.isfinite(scores_buffer).all().item())
            and bool((scores_buffer >= 0.0).all().item()),
            "fit progress scores_buffer shape/finite contract mismatch",
        )
        populated_mask = torch.any(scores_buffer != 0.0, dim=1)
        populated_count = int(populated_mask.sum().item())
        _require(populated_count >= 1, "fit progress contains no populated generation")
        _require(
            bool(populated_mask[:populated_count].all().item())
            and not bool(populated_mask[populated_count:].any().item()),
            "fit progress populated generations are not a contiguous prefix",
        )
        termination = classify_cma_termination(
            scores_buffer,
            populated_count=populated_count,
            max_iterations=expected_iterations,
            epsilon=float(task_contract["epsilon"]),
        )
        bounds_tensor = config_bounds.detach().cpu().to(dtype=params_buffer.dtype)
        populated_params = params_buffer[:populated_count].detach().cpu()
        _require(
            bool(
                (
                    (populated_params >= bounds_tensor[:, 0])
                    & (populated_params <= bounds_tensor[:, 1])
                ).all().item()
            ),
            "fit progress contains a parameter outside runtime bounds",
        )
        mean_match = MEAN_PATTERN.fullmatch(mean_path.name)
        _require(mean_match is not None, "selected final mean basename is invalid")
        if mean_match is None:
            raise FitGuardError("selected final mean basename is invalid")
        _require(
            int(mean_match.group(1)) == populated_count - 1,
            "selected final mean does not match the last populated generation",
        )
        best_trajectory_path = run_dir / "best_trajectory.pt"
        best_trajectory = _load_torch(best_trajectory_path)
        _require(
            isinstance(best_trajectory, torch.Tensor)
            and best_trajectory.shape == (SAMPLE_COUNT, 12)
            and bool(torch.isfinite(best_trajectory).all().item()),
            "fit best_trajectory.pt shape/finite contract mismatch",
        )
        event_files = sorted(run_dir.glob("events.out.tfevents.*"))
        _require(
            bool(event_files)
            and all(path.is_file() and path.stat().st_size > 0 for path in event_files),
            "fit run has no non-empty TensorBoard event file",
        )
        report: dict[str, Any] = {
            "schema_version": "a1_pace_fit_manifest/v1",
            "status": "PASS",
            "physical_motion_authorized": False,
            "task": args.task,
            "num_envs": args.num_envs,
            "command": command,
            "exit_code": result.returncode,
            "duration_s": duration_s,
            "provenance": provenance,
            "pre_fit_gates": {
                "capacity": capacity_evidence,
                "sensitivity": sensitivity_evidence,
            },
            "source_data_sha256": source_data_sha,
            "source_tensor_payload_sha256": payload_sha,
            "staged_data_path": str(staged_data),
            "staged_data_sha256": _sha256_path(staged_data),
            "staged_tensor_payload_sha256": staged_data_payload,
            "staged_provenance": {
                str(destination): _sha256_path(destination)
                for _, destination, _ in staged_sidecars
            },
            "config_basename": config_path.name,
            "config_sha256": _sha256_path(config_path),
            "config_tensor_payload_sha256": config_payload_sha,
            "run_basename": run_basename,
            "mean_basename": mean_path.name,
            "mean_sha256": _sha256_path(mean_path),
            "synthetic_parameter_recovery": recovery,
            "progress_basename": progress_path.name,
            "progress_sha256": _sha256_path(progress_path),
            "populated_generation_count": populated_count,
            "termination": termination,
            "best_trajectory_basename": best_trajectory_path.name,
            "best_trajectory_sha256": _sha256_path(best_trajectory_path),
            "tensorboard_event_sha256": {
                path.name: _sha256_path(path) for path in event_files
            },
            "official_source_sha256": official_source_hashes,
            "a1_extension_source_sha256": a1_extension_source_hashes,
            "fit_guard_sha256": fit_guard_sha,
            "environment_contract": task_contract,
            "repository_revisions": {
                "pace": _git_head(repository_root),
                "isaaclab": _git_head(isaaclab_root),
                "a1_base": _git_head(args.a1_base_root.resolve()),
                "gogo_learn": _git_head(args.gogo_learn_root.resolve()),
            },
        }
        _atomic_json(fit_manifest_output, report)
        return report


def _build_parser() -> argparse.ArgumentParser:
    repository_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Validate, stage, and launch unchanged official A1 PACE fit.py"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--known-mean", type=Path)
    parser.add_argument("--raw-manifest", type=Path)
    parser.add_argument("--conversion-report", type=Path)
    parser.add_argument("--raw-csv", type=Path)
    parser.add_argument("--trial-plan", type=Path)
    parser.add_argument("--contracts-dir", type=Path)
    parser.add_argument("--capacity-report", type=Path, required=True)
    parser.add_argument("--sensitivity-report", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--fit-manifest-output", type=Path, required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=repository_default,
    )
    parser.add_argument(
        "--isaaclab-root",
        type=Path,
        default=Path(os.environ.get("ISAACLAB_ROOT", "/home/changba01/IsaacLab")),
    )
    parser.add_argument(
        "--a1-base-root",
        type=Path,
        default=Path(
            os.environ.get("A1_BASE_ROOT", "/home/changba01/worktrees/A1_Base-a1-pace")
        ),
    )
    parser.add_argument(
        "--gogo-learn-root",
        type=Path,
        default=Path(
            os.environ.get(
                "GOGO_LEARN_ROOT", "/home/changba01/worktrees/gogo-learn-a1-pace"
            )
        ),
    )
    parser.add_argument(
        "--jsonschema-path",
        type=Path,
        default=(
            Path(os.environ["A1_PACE_JSONSCHEMA_PATH"])
            if "A1_PACE_JSONSCHEMA_PATH" in os.environ
            else None
        ),
        help="Optional jsonschema fallback directory for the Task 8 converter",
    )
    parser.add_argument("--headless", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = run_fit_guard(args)
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"[A1 PACE FIT GUARD] FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
