"""Fail-closed per-parameter sensitivity gate for the registered A1 PACE task.

The coordinator intentionally has no Isaac Lab imports.  It launches one fresh
Python process for task-contract discovery, several repeated all-mid probes for
the numerical-noise estimate, and then one process for every low/mid/high probe
of every parameter dimension.  A worker starts Isaac only after command-line
parsing and writes one float32 trajectory plus a small JSON result.

The pure :func:`validate_prior_contract` and
:func:`evaluate_sensitivity_records` functions are importable by ordinary unit
tests without starting an ``AppLauncher``.

Prior JSON contract (``a1_pace_parameter_priors/v1``)::

    {
      "schema_version": "a1_pace_parameter_priors/v1",
      "robot": "Unitree A1",
      "joint_order": ["FL_hip_joint", "FR_hip_joint", "..."],
      "parameters": [
        {
          "index": 0,
          "name": "armature.FL_hip_joint",
          "family": "armature",
          "joint": "FL_hip_joint",
          "unit": "kg*m^2",
          "prior": [0.0, 0.1],
          "source": {
            "url": "https://example.invalid/a1-document",
            "document": "Document title, revision, section/table"
          }
        }
      ]
    }

All 49 entries are mandatory and ordered.  The final entry is named
``global_delay``, belongs to family ``delay``, and has ``joint: null``.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit


PRIOR_SCHEMA = "a1_pace_parameter_priors/v1"
RECORD_SCHEMA = "a1_pace_sensitivity_records/v1"
REPORT_SCHEMA = "a1_pace_sensitivity_gate/v1"
WORKER_SPEC_SCHEMA = "a1_pace_sensitivity_worker_spec/v1"
WORKER_RESULT_SCHEMA = "a1_pace_sensitivity_worker_result/v1"
ROBOT_NAME = "Unitree A1"
DT_S = 0.002
DEFAULT_SAMPLE_COUNT = 2500
MIN_SAMPLE_COUNT = 2000
MAX_SAMPLE_COUNT = 5000
DEFAULT_NOISE_REPEATS = 3
ABSOLUTE_NOISE_FLOOR_RAD = 1.0e-9

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

PARAMETER_FAMILIES = (
    ("armature", "kg*m^2"),
    ("viscous_friction", "N*m*s/rad"),
    ("coulomb_friction", "1"),
    ("encoder_bias", "rad"),
)


def _parameter_contract() -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for family, expected_unit in PARAMETER_FAMILIES:
        for joint in A1_PACE_JOINT_ORDER:
            result.append(
                {
                    "index": len(result),
                    "name": f"{family}.{joint}",
                    "family": family,
                    "joint": joint,
                    "expected_unit": expected_unit,
                }
            )
    result.append(
        {
            "index": 48,
            "name": "global_delay",
            "family": "delay",
            "joint": None,
            "expected_unit": "step",
        }
    )
    return tuple(result)


PARAMETER_CONTRACT = _parameter_contract()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: Any, field: str) -> float:
    _require(_is_number(value), f"{field} must be a number")
    result = float(value)
    _require(math.isfinite(result), f"{field} must be finite")
    return result


def _integer(value: Any, field: str, *, minimum: int | None = None) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{field} must be an integer")
    if minimum is not None:
        _require(value >= minimum, f"{field} must be >= {minimum}")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    _require(
        actual == expected,
        f"{field} keys differ: missing={sorted(expected - actual)}, "
        f"unexpected={sorted(actual - expected)}",
    )


def _strict_json_load(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {path}: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    _require(isinstance(value, dict), f"{path} must contain one JSON object")

    def reject_nonfinite(item: Any) -> None:
        if isinstance(item, float):
            _require(math.isfinite(item), f"{path} contains a non-finite number")
        elif isinstance(item, dict):
            for child in item.values():
                reject_nonfinite(child)
        elif isinstance(item, list):
            for child in item:
                reject_nonfinite(child)

    reject_nonfinite(value)
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def _a1_extension_hashes(repository_root: Path) -> dict[str, str]:
    relative_paths = (
        "source/pace_sim2real/pace_sim2real/tasks/manager_based/__init__.py",
        "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/a1_mean.py",
        "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/a1_pace_env_cfg.py",
        "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/a1_replay.py",
    )
    return {
        relative: _sha256_path(repository_root / relative)
        for relative in relative_paths
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_bytes(value: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value: Mapping[str, Any], destination: Path) -> None:
    encoded = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    _atomic_bytes(encoded, destination)


def validate_prior_contract(priors: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate and normalize the explicit 49-dimensional A1 prior contract.

    This function validates provenance syntax but deliberately does not fetch a
    URL.  Repository audit evidence must pin the cited document independently.
    Bounds coverage is evaluated by :func:`evaluate_sensitivity_records` because
    the registered task, rather than the prior file, owns optimizer bounds.
    """

    _require(isinstance(priors, Mapping), "priors must be a mapping")
    _exact_keys(
        priors,
        {"schema_version", "robot", "joint_order", "parameters"},
        "priors",
    )
    _require(priors["schema_version"] == PRIOR_SCHEMA, "prior schema version mismatch")
    _require(priors["robot"] == ROBOT_NAME, "prior robot must be Unitree A1")
    joint_order = priors["joint_order"]
    _require(isinstance(joint_order, list), "prior joint_order must be a list")
    _require(tuple(joint_order) == A1_PACE_JOINT_ORDER, "prior joint order mismatch")
    parameters = priors["parameters"]
    _require(isinstance(parameters, list), "prior parameters must be a list")
    _require(len(parameters) == 49, "prior contract must contain exactly 49 parameters")

    normalized: list[dict[str, Any]] = []
    for index, (entry, expected) in enumerate(zip(parameters, PARAMETER_CONTRACT, strict=True)):
        field = f"priors.parameters[{index}]"
        _require(isinstance(entry, Mapping), f"{field} must be an object")
        _exact_keys(
            entry,
            {"index", "name", "family", "joint", "unit", "prior", "source"},
            field,
        )
        _require(_integer(entry["index"], f"{field}.index") == index, f"{field}.index is out of order")
        for key in ("name", "family", "joint"):
            _require(entry[key] == expected[key], f"{field}.{key} mismatch")

        unit = entry["unit"]
        _require(
            isinstance(unit, str) and unit.strip() == unit and bool(unit),
            f"{field}.unit must be non-empty",
        )
        _require(unit == expected["expected_unit"], f"{field}.unit must be {expected['expected_unit']!r}")

        prior = entry["prior"]
        _require(isinstance(prior, list) and len(prior) == 2, f"{field}.prior must be [low, high]")
        prior_low = _finite_number(prior[0], f"{field}.prior[0]")
        prior_high = _finite_number(prior[1], f"{field}.prior[1]")
        _require(prior_low <= prior_high, f"{field}.prior is reversed")
        if index == 48:
            _require(prior_low.is_integer() and prior_high.is_integer(), "delay prior endpoints must be integer steps")

        source = entry["source"]
        _require(isinstance(source, Mapping), f"{field}.source must be an object")
        _exact_keys(source, {"url", "document"}, f"{field}.source")
        source_url = source["url"]
        source_document = source["document"]
        _require(isinstance(source_url, str), f"{field}.source.url must be a string")
        parsed_url = urlsplit(source_url)
        _require(parsed_url.scheme == "https" and bool(parsed_url.netloc), f"{field}.source.url must be an absolute HTTPS URL")
        _require(
            isinstance(source_document, str)
            and source_document.strip() == source_document
            and len(source_document) >= 3,
            f"{field}.source.document must identify a document and section",
        )
        normalized.append(
            {
                "index": index,
                "name": expected["name"],
                "family": expected["family"],
                "joint": expected["joint"],
                "unit": unit,
                "prior_low": prior_low,
                "prior_high": prior_high,
                "source_url": source_url,
                "source_document": source_document,
            }
        )
    return normalized


def _validate_bounds(bounds: Sequence[Sequence[Any]]) -> list[list[float]]:
    _require(
        isinstance(bounds, Sequence) and not isinstance(bounds, (str, bytes)),
        "bounds must be a sequence",
    )
    _require(len(bounds) == 49, "bounds must have shape (49, 2)")
    normalized: list[list[float]] = []
    for index, pair in enumerate(bounds):
        _require(
            isinstance(pair, Sequence)
            and not isinstance(pair, (str, bytes))
            and len(pair) == 2,
            f"bounds[{index}] must contain [low, high]",
        )
        low = _finite_number(pair[0], f"bounds[{index}][0]")
        high = _finite_number(pair[1], f"bounds[{index}][1]")
        _require(low < high, f"bounds[{index}] must have low < high")
        normalized.append([low, high])
    return normalized


def _metric(value: Mapping[str, Any], field: str) -> dict[str, float]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    _exact_keys(value, {"rmse_rad", "max_abs_rad"}, field)
    rmse = _finite_number(value["rmse_rad"], f"{field}.rmse_rad")
    max_abs = _finite_number(value["max_abs_rad"], f"{field}.max_abs_rad")
    _require(rmse >= 0.0 and max_abs >= 0.0, f"{field} metrics must be nonnegative")
    _require(rmse <= max_abs + 1.0e-15, f"{field}.rmse_rad cannot exceed max_abs_rad")
    return {"rmse_rad": rmse, "max_abs_rad": max_abs}


def evaluate_sensitivity_records(
    bounds: Sequence[Sequence[Any]],
    priors: Mapping[str, Any],
    records: Mapping[str, Any],
    noise_multiplier: float,
) -> dict[str, Any]:
    """Purely evaluate prior coverage and 49 low/mid/high response records.

    ``records`` must use ``a1_pace_sensitivity_records/v1`` and contain a
    global repeated-mid noise estimate plus 49 ordered dimension records.  Each
    response metric compares a low/high trajectory with that dimension's own
    independently generated mid trajectory.  ``mid_vs_noise_reference`` proves
    that this independent mid remains inside the repeated-mid noise envelope.
    """

    normalized_bounds = _validate_bounds(bounds)
    normalized_priors = validate_prior_contract(priors)
    multiplier = _finite_number(noise_multiplier, "noise_multiplier")
    _require(multiplier > 1.0, "noise_multiplier must be greater than one")
    _require(isinstance(records, Mapping), "records must be a mapping")
    _exact_keys(records, {"schema_version", "noise", "dimensions"}, "records")
    _require(records["schema_version"] == RECORD_SCHEMA, "record schema version mismatch")

    noise = records["noise"]
    _require(isinstance(noise, Mapping), "records.noise must be an object")
    _exact_keys(
        noise,
        {"repeat_count", "pair_count", "rmse_rad", "max_abs_rad", "absolute_floor_rad"},
        "records.noise",
    )
    repeat_count = _integer(noise["repeat_count"], "records.noise.repeat_count", minimum=2)
    pair_count = _integer(noise["pair_count"], "records.noise.pair_count", minimum=1)
    _require(
        pair_count == repeat_count * (repeat_count - 1) // 2,
        "records.noise.pair_count is inconsistent with repeat_count",
    )
    noise_rmse = _finite_number(noise["rmse_rad"], "records.noise.rmse_rad")
    noise_max_abs = _finite_number(noise["max_abs_rad"], "records.noise.max_abs_rad")
    absolute_floor = _finite_number(
        noise["absolute_floor_rad"], "records.noise.absolute_floor_rad"
    )
    _require(noise_rmse >= 0.0 and noise_max_abs >= 0.0, "noise metrics must be nonnegative")
    _require(noise_rmse <= noise_max_abs + 1.0e-15, "noise RMSE cannot exceed maximum")
    _require(absolute_floor > 0.0, "absolute noise floor must be positive")
    rmse_threshold = max(absolute_floor, multiplier * noise_rmse)
    max_abs_threshold = max(absolute_floor, multiplier * noise_max_abs)

    dimensions = records["dimensions"]
    _require(isinstance(dimensions, list), "records.dimensions must be a list")
    _require(len(dimensions) == 49, "records must contain exactly 49 dimensions")
    evaluated: list[dict[str, Any]] = []
    failures: list[str] = []

    for index, (entry, prior, bound, expected) in enumerate(
        zip(dimensions, normalized_priors, normalized_bounds, PARAMETER_CONTRACT, strict=True)
    ):
        field = f"records.dimensions[{index}]"
        _require(isinstance(entry, Mapping), f"{field} must be an object")
        _exact_keys(entry, {"index", "name", "probes", "responses"}, field)
        _require(_integer(entry["index"], f"{field}.index") == index, f"{field}.index is out of order")
        _require(entry["name"] == expected["name"], f"{field}.name mismatch")

        probes = entry["probes"]
        _require(isinstance(probes, Mapping), f"{field}.probes must be an object")
        _exact_keys(probes, {"low", "mid", "high"}, f"{field}.probes")
        expected_values = {
            "low": bound[0],
            "mid": (bound[0] + bound[1]) / 2.0,
            "high": bound[1],
        }
        normalized_probes: dict[str, dict[str, Any]] = {}
        for level in ("low", "mid", "high"):
            probe = probes[level]
            _require(isinstance(probe, Mapping), f"{field}.probes.{level} must be an object")
            _require("value" in probe, f"{field}.probes.{level}.value is required")
            value = _finite_number(probe["value"], f"{field}.probes.{level}.value")
            _require(
                math.isclose(value, expected_values[level], rel_tol=0.0, abs_tol=1.0e-12),
                f"{field}.probes.{level}.value is not the registered bound/midpoint",
            )
            vector = probe.get("vector")
            _require(
                isinstance(vector, list) and len(vector) == 49,
                f"{field}.probes.{level}.vector must contain 49 values",
            )
            normalized_vector = [
                _finite_number(item, f"{field}.probes.{level}.vector[{vector_index}]")
                for vector_index, item in enumerate(vector)
            ]
            for vector_index, vector_value in enumerate(normalized_vector):
                vector_bound = normalized_bounds[vector_index]
                expected_vector_value = (
                    expected_values[level]
                    if vector_index == index
                    else (vector_bound[0] + vector_bound[1]) / 2.0
                )
                _require(
                    math.isclose(
                        vector_value,
                        expected_vector_value,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    ),
                    f"{field}.probes.{level}.vector[{vector_index}] is not "
                    "the selected probe value or midpoint",
                )
            normalized_probes[level] = dict(probe)
            normalized_probes[level]["value"] = value
            normalized_probes[level]["vector"] = normalized_vector

        responses = entry["responses"]
        _require(isinstance(responses, Mapping), f"{field}.responses must be an object")
        _exact_keys(
            responses,
            {"low_vs_mid", "high_vs_mid", "mid_vs_noise_reference"},
            f"{field}.responses",
        )
        low_response = _metric(responses["low_vs_mid"], f"{field}.responses.low_vs_mid")
        high_response = _metric(responses["high_vs_mid"], f"{field}.responses.high_vs_mid")
        mid_response = _metric(
            responses["mid_vs_noise_reference"],
            f"{field}.responses.mid_vs_noise_reference",
        )

        prior_covered = prior["prior_low"] >= bound[0] and prior["prior_high"] <= bound[1]
        mid_repeatable = (
            mid_response["rmse_rad"] <= rmse_threshold
            and mid_response["max_abs_rad"] <= max_abs_threshold
        )
        low_sensitive = (
            low_response["rmse_rad"] > rmse_threshold
            and low_response["max_abs_rad"] > max_abs_threshold
        )
        high_sensitive = (
            high_response["rmse_rad"] > rmse_threshold
            and high_response["max_abs_rad"] > max_abs_threshold
        )
        accepted = prior_covered and mid_repeatable and low_sensitive and high_sensitive
        reasons: list[str] = []
        if not prior_covered:
            reasons.append("registered bounds do not cover the documented prior")
        if not mid_repeatable:
            reasons.append("independent midpoint exceeds the numerical-noise envelope")
        if not low_sensitive:
            reasons.append("low-to-mid response is not above the noise threshold")
        if not high_sensitive:
            reasons.append("high-to-mid response is not above the noise threshold")
        failures.extend(f"dimension {index} ({expected['name']}): {reason}" for reason in reasons)
        evaluated.append(
            {
                "index": index,
                "name": expected["name"],
                "family": expected["family"],
                "joint": expected["joint"],
                "unit": prior["unit"],
                "bound": bound,
                "prior": [prior["prior_low"], prior["prior_high"]],
                "source": {
                    "url": prior["source_url"],
                    "document": prior["source_document"],
                },
                "probes": normalized_probes,
                "responses": {
                    "low_vs_mid": low_response,
                    "high_vs_mid": high_response,
                    "mid_vs_noise_reference": mid_response,
                },
                "thresholds": {
                    "rmse_rad": rmse_threshold,
                    "max_abs_rad": max_abs_threshold,
                },
                "checks": {
                    "prior_covered": prior_covered,
                    "mid_repeatable": mid_repeatable,
                    "low_sensitive": low_sensitive,
                    "high_sensitive": high_sensitive,
                },
                "accepted": accepted,
                "status": "PASS" if accepted else "FAIL",
                "failure_reasons": reasons,
            }
        )

    accepted = not failures
    return {
        "status": "PASS" if accepted else "FAIL",
        "accepted": accepted,
        "noise_multiplier": multiplier,
        "noise": {
            "repeat_count": repeat_count,
            "pair_count": pair_count,
            "rmse_rad": noise_rmse,
            "max_abs_rad": noise_max_abs,
            "absolute_floor_rad": absolute_floor,
        },
        "thresholds": {
            "rmse_rad": rmse_threshold,
            "max_abs_rad": max_abs_threshold,
        },
        "dimensions": evaluated,
        "failures": failures,
    }


def _build_probe_targets(torch: Any, *, seed: int, sample_count: int) -> Any:
    """Build a safe deterministic 4-10 Hz-rich fixed-base A1 trajectory."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    phase_offsets = torch.rand(12, generator=generator, dtype=torch.float64) * (
        2.0 * math.pi
    )
    directions = torch.where(
        torch.rand(12, generator=generator) >= 0.5,
        torch.ones(12),
        -torch.ones(12),
    ).to(torch.float64)
    centers = torch.tensor([0.0] * 4 + [0.8] * 4 + [-1.5] * 4, dtype=torch.float64)
    amplitudes = torch.tensor([0.10] * 4 + [0.16] * 4 + [0.18] * 4, dtype=torch.float64)
    time = torch.arange(sample_count, dtype=torch.float64) * DT_S
    duration = sample_count * DT_S
    f0 = torch.linspace(0.25, 0.55, 12, dtype=torch.float64)
    f1 = torch.linspace(8.0, 10.0, 12, dtype=torch.float64)
    beta = (f1 - f0) / duration
    chirp_phase = 2.0 * math.pi * (
        time[:, None] * f0[None, :]
        + 0.5 * time[:, None].square() * beta[None, :]
    )
    low_phase = 2.0 * math.pi * time[:, None] * (0.45 + 0.025 * torch.arange(12))
    excitation = (
        0.78 * torch.sin(chirp_phase + phase_offsets[None, :])
        + 0.22 * torch.sin(low_phase + 0.5 * phase_offsets[None, :])
    )
    return (
        centers[None, :] + amplitudes[None, :] * directions[None, :] * excitation
    ).to(torch.float32).contiguous()


def _runtime_bounds(env_cfg: Any) -> list[list[float]]:
    return _validate_bounds(env_cfg.sim2real.bounds_params.detach().cpu().tolist())


def _validate_worker_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(spec, Mapping), "worker spec must be an object")
    mode = spec.get("mode")
    if mode == "discover":
        _exact_keys(
            spec,
            {"schema_version", "mode", "task", "seed", "sample_count"},
            "worker spec",
        )
    elif mode == "probe":
        _exact_keys(
            spec,
            {
                "schema_version",
                "mode",
                "task",
                "seed",
                "sample_count",
                "probe_id",
                "parameter_index",
                "level",
                "parameter_value",
                "vector",
                "trajectory_output",
            },
            "worker spec",
        )
    else:
        raise ValueError("worker spec mode must be discover or probe")
    _require(spec["schema_version"] == WORKER_SPEC_SCHEMA, "worker spec schema mismatch")
    _require(
        isinstance(spec["task"], str) and bool(spec["task"]),
        "worker task is required",
    )
    _integer(spec["seed"], "worker seed", minimum=0)
    sample_count = _integer(spec["sample_count"], "worker sample_count", minimum=MIN_SAMPLE_COUNT)
    _require(sample_count <= MAX_SAMPLE_COUNT, f"worker sample_count must be <= {MAX_SAMPLE_COUNT}")
    return dict(spec)


def _run_worker(spec_path: Path, result_path: Path, *, device: str) -> dict[str, Any]:
    """Run after AppLauncher startup; all Isaac imports remain inside here."""

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import isaaclab_tasks  # noqa: F401
    import pace_sim2real.tasks  # noqa: F401
    from pace_sim2real.tasks.manager_based.pace.a1_mean import parse_a1_mean
    from pace_sim2real.tasks.manager_based.pace.a1_replay import (
        inject_a1_actuator_before_make,
        make_a1_pace_actuator_cfg,
    )

    spec_hash = _sha256_path(spec_path)
    spec = _validate_worker_spec(_strict_json_load(spec_path))
    env_cfg = parse_env_cfg(spec["task"], device=device, num_envs=1)
    bounds = _runtime_bounds(env_cfg)
    _require(tuple(env_cfg.sim2real.joint_order) == A1_PACE_JOINT_ORDER, "registered task joint order mismatch")
    _require(abs(float(env_cfg.sim.dt) - DT_S) <= 1.0e-12, "registered task dt is not 0.002 s")
    runtime = {
        "task": spec["task"],
        "device": device,
        "dt_s": float(env_cfg.sim.dt),
        "decimation": int(env_cfg.decimation),
        "num_envs": int(env_cfg.scene.num_envs),
        "joint_order": list(env_cfg.sim2real.joint_order),
        "bounds": bounds,
        "fix_root_link": bool(env_cfg.scene.robot.spawn.articulation_props.fix_root_link),
        "action_scale": float(env_cfg.actions.joint_pos.scale),
        "use_default_offset": bool(env_cfg.actions.joint_pos.use_default_offset),
    }
    if spec["mode"] == "discover":
        result = {
            "schema_version": WORKER_RESULT_SCHEMA,
            "mode": "discover",
            "pid": os.getpid(),
            "spec_sha256": spec_hash,
            "runtime": runtime,
        }
        _require(_sha256_path(spec_path) == spec_hash, "worker spec changed during discovery")
        _atomic_json(result, result_path)
        return result

    vector_values = spec["vector"]
    _require(isinstance(vector_values, list) and len(vector_values) == 49, "probe vector must have 49 values")
    vector = torch.tensor(
        [_finite_number(value, f"probe vector[{index}]") for index, value in enumerate(vector_values)],
        dtype=torch.float32,
    )
    _require(bool(torch.isfinite(vector).all()), "probe vector must be finite")
    for index, (value, bound) in enumerate(zip(vector.tolist(), bounds, strict=True)):
        _require(bound[0] <= value <= bound[1], f"probe vector[{index}] is outside runtime bounds")
    parameter_index = _integer(spec["parameter_index"], "worker parameter_index")
    _require(-1 <= parameter_index < 49, "worker parameter_index is outside [-1, 48]")
    _require(
        isinstance(spec["level"], str) and bool(spec["level"]),
        "worker level is required",
    )
    parameter_value = _finite_number(spec["parameter_value"], "worker parameter_value")
    if parameter_index >= 0:
        _require(
            math.isclose(float(vector[parameter_index]), parameter_value, rel_tol=0.0, abs_tol=1.0e-7),
            "probe parameter value differs from vector",
        )

    mean = parse_a1_mean(vector)
    inject_a1_actuator_before_make(env_cfg, make_a1_pace_actuator_cfg(mean))
    env_cfg.seed = spec["seed"]
    targets = _build_probe_targets(
        torch, seed=spec["seed"], sample_count=spec["sample_count"]
    )
    env = gym.make(spec["task"], cfg=env_cfg)
    try:
        unwrapped = cast(Any, env.unwrapped)
        _require(unwrapped.num_envs == 1, "sensitivity worker requires one environment")
        articulation = unwrapped.scene["robot"]
        try:
            joint_indices = [
                articulation.joint_names.index(name) for name in A1_PACE_JOINT_ORDER
            ]
        except ValueError as exc:
            raise ValueError("A1 articulation is missing a PACE joint") from exc
        _require(len(set(joint_indices)) == 12, "PACE joint mapping is not one-to-one")
        joint_ids = torch.tensor(joint_indices, device=unwrapped.device, dtype=torch.long)
        dtype = articulation.data.joint_pos.dtype
        targets_device = targets.to(device=unwrapped.device, dtype=dtype)
        bias = torch.tensor(
            [mean.encoder_bias[name] for name in A1_PACE_JOINT_ORDER],
            device=unwrapped.device,
            dtype=dtype,
        )
        env.reset(seed=spec["seed"])
        initial_position = (targets_device[0] + bias).unsqueeze(0)
        articulation.write_joint_position_to_sim(initial_position, joint_ids=joint_ids)
        articulation.write_joint_velocity_to_sim(
            torch.zeros_like(initial_position), joint_ids=joint_ids
        )
        samples = torch.empty_like(targets_device)
        with torch.inference_mode():
            for step in range(spec["sample_count"]):
                samples[step] = articulation.data.joint_pos[0, joint_ids] - bias
                action_shape = cast(Sequence[int], env.action_space.shape)
                action = torch.zeros(
                    action_shape, device=unwrapped.device, dtype=dtype
                )
                action[:, joint_ids] = targets_device[step].unsqueeze(0)
                env.step(action)
    finally:
        env.close()

    trajectory = samples.detach().cpu().to(torch.float32).contiguous()
    _require(
        tuple(trajectory.shape) == (spec["sample_count"], 12),
        "worker trajectory shape mismatch",
    )
    _require(bool(torch.isfinite(trajectory).all()), "worker trajectory is non-finite")
    trajectory_bytes = trajectory.numpy().astype("<f4", copy=False).tobytes(order="C")
    trajectory_output = Path(spec["trajectory_output"]).resolve()
    _require(trajectory_output not in {spec_path.resolve(), result_path.resolve()}, "worker outputs alias")
    _atomic_bytes(trajectory_bytes, trajectory_output)
    target_bytes = targets.numpy().astype("<f4", copy=False).tobytes(order="C")
    result = {
        "schema_version": WORKER_RESULT_SCHEMA,
        "mode": "probe",
        "pid": os.getpid(),
        "spec_sha256": spec_hash,
        "runtime": runtime,
        "probe_id": spec["probe_id"],
        "parameter_index": parameter_index,
        "level": spec["level"],
        "parameter_value": parameter_value,
        "seed": spec["seed"],
        "sample_count": spec["sample_count"],
        "vector": [float(value) for value in vector_values],
        "vector_sha256": _canonical_sha256(
            [float(value) for value in vector_values]
        ),
        "target_sha256": hashlib.sha256(target_bytes).hexdigest(),
        "trajectory": {
            "path": str(trajectory_output),
            "format": "float32-little-endian-row-major",
            "shape": [spec["sample_count"], 12],
            "byte_count": len(trajectory_bytes),
            "sha256": hashlib.sha256(trajectory_bytes).hexdigest(),
        },
    }
    _require(_sha256_path(spec_path) == spec_hash, "worker spec changed during probe")
    trajectory_metadata = cast(dict[str, Any], result["trajectory"])
    _require(
        _sha256_path(trajectory_output) == trajectory_metadata["sha256"],
        "published trajectory changed",
    )
    _atomic_json(result, result_path)
    return result


def _worker_main(argv: Sequence[str]) -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Internal A1 sensitivity worker")
    parser.add_argument("--_worker-spec", type=Path, required=True)
    parser.add_argument("--_worker-result", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    launcher = AppLauncher(args)
    simulation_app = launcher.app
    try:
        result = _run_worker(
            args._worker_spec.resolve(), args._worker_result.resolve(), device=args.device
        )
        print(json.dumps({"mode": result["mode"], "status": "PASS"}, sort_keys=True))
        return 0
    finally:
        simulation_app.close()


def _run_child(
    *,
    python_executable: Path,
    script_path: Path,
    spec_path: Path,
    result_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    device: str,
    headless: bool,
    timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        str(python_executable),
        str(script_path),
        "--_worker-spec",
        str(spec_path),
        "--_worker-result",
        str(result_path),
        "--device",
        device,
    ]
    if headless:
        command.append("--headless")
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started_ns = __import__("time").monotonic_ns()
    with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
        try:
            completed = subprocess.run(
                command,
                cwd=script_path.parents[2],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"worker timed out after {timeout_s}s; logs: {stdout_path}, {stderr_path}"
            ) from exc
    duration_s = (__import__("time").monotonic_ns() - started_ns) / 1.0e9
    _require(
        completed.returncode == 0,
        f"worker exited {completed.returncode}; logs: {stdout_path}, {stderr_path}",
    )
    _require(result_path.is_file(), f"worker did not publish {result_path}")
    result = _strict_json_load(result_path)
    audit = {
        "command": shlex.join(command),
        "exit_code": completed.returncode,
        "duration_s": duration_s,
        "result_path": str(result_path),
        "result_sha256": _sha256_path(result_path),
        "stdout_path": str(stdout_path),
        "stdout_sha256": _sha256_path(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": _sha256_path(stderr_path),
    }
    return result, audit


def _read_trajectory(result: Mapping[str, Any]) -> list[float]:
    _require(result.get("schema_version") == WORKER_RESULT_SCHEMA, "worker result schema mismatch")
    _require(result.get("mode") == "probe", "expected a probe worker result")
    trajectory = result.get("trajectory")
    _require(isinstance(trajectory, Mapping), "worker trajectory metadata is missing")
    trajectory = cast(Mapping[str, Any], trajectory)
    _require(
        trajectory.get("format") == "float32-little-endian-row-major",
        "worker trajectory format mismatch",
    )
    shape = trajectory.get("shape")
    _require(
        isinstance(shape, list)
        and len(shape) == 2
        and shape[1] == 12
        and shape[0] == result.get("sample_count"),
        "worker trajectory shape metadata mismatch",
    )
    shape = cast(list[Any], shape)
    path = Path(str(trajectory.get("path"))).resolve()
    payload = path.read_bytes()
    expected_bytes = int(shape[0]) * 12 * 4
    _require(len(payload) == expected_bytes, "worker trajectory byte count mismatch")
    _require(trajectory.get("byte_count") == expected_bytes, "worker byte_count mismatch")
    _require(hashlib.sha256(payload).hexdigest() == trajectory.get("sha256"), "worker trajectory hash mismatch")
    values = array.array("f")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    result_values = list(values)
    _require(all(math.isfinite(value) for value in result_values), "worker trajectory is non-finite")
    return result_values


def _difference_metric(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    _require(len(left) == len(right) and len(left) > 0, "trajectory lengths differ")
    differences = [float(a) - float(b) for a, b in zip(left, right, strict=True)]
    return {
        "rmse_rad": math.sqrt(math.fsum(value * value for value in differences) / len(differences)),
        "max_abs_rad": max(abs(value) for value in differences),
    }


def _response_matrix_diagnostic(
    signatures: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Quantify cross-parameter collinearity; this is diagnostic, not authorization."""

    import numpy as np

    matrix = np.asarray(signatures, dtype=np.float64)
    _require(
        matrix.ndim == 2
        and matrix.shape[0] == 49
        and matrix.shape[1] > 49
        and bool(np.isfinite(matrix).all()),
        "sensitivity response matrix is invalid",
    )
    norms = np.linalg.norm(matrix, axis=1)
    _require(bool(np.all(norms > 0.0)), "sensitivity response matrix has a zero row")
    normalized = matrix / norms[:, None]
    correlation = normalized @ normalized.T
    np.fill_diagonal(correlation, 0.0)
    pair_index = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = (
        np.finfo(np.float64).eps
        * max(matrix.shape)
        * float(singular_values[0])
    )
    numerical_rank = int(np.sum(singular_values > tolerance))
    smallest = float(singular_values[-1])
    condition_number = (
        float(singular_values[0] / smallest) if smallest > 0.0 else None
    )
    return {
        "interpretation": (
            "synthetic response-matrix diagnostic only; real encoder noise and "
            "repeat trials remain mandatory"
        ),
        "matrix_shape": [int(value) for value in matrix.shape],
        "matrix_sha256": hashlib.sha256(
            matrix.astype("<f8", copy=False).tobytes(order="C")
        ).hexdigest(),
        "numerical_rank": numerical_rank,
        "full_rank": numerical_rank == 49,
        "condition_number": condition_number,
        "largest_singular_value": float(singular_values[0]),
        "smallest_singular_value": smallest,
        "max_absolute_row_correlation": float(abs(correlation[pair_index])),
        "max_correlation_parameter_indices": [
            int(pair_index[0]),
            int(pair_index[1]),
        ],
    }


def _probe_metadata(
    result: Mapping[str, Any], audit: Mapping[str, Any], *, value: float
) -> dict[str, Any]:
    trajectory = result["trajectory"]
    return {
        "value": value,
        "pid": result["pid"],
        "probe_id": result["probe_id"],
        "vector": result["vector"],
        "vector_sha256": result["vector_sha256"],
        "target_sha256": result["target_sha256"],
        "trajectory_path": trajectory["path"],
        "trajectory_sha256": trajectory["sha256"],
        "worker_result_path": audit["result_path"],
        "worker_result_sha256": audit["result_sha256"],
        "stdout_path": audit["stdout_path"],
        "stdout_sha256": audit["stdout_sha256"],
        "stderr_path": audit["stderr_path"],
        "stderr_sha256": audit["stderr_sha256"],
        "duration_s": audit["duration_s"],
    }


def _check_worker_result(
    result: Mapping[str, Any],
    *,
    spec_path: Path,
    expected_mode: str,
    expected_bounds: Sequence[Sequence[float]] | None = None,
) -> None:
    _require(result.get("schema_version") == WORKER_RESULT_SCHEMA, "worker result schema mismatch")
    _require(result.get("mode") == expected_mode, "worker result mode mismatch")
    _require(result.get("spec_sha256") == _sha256_path(spec_path), "worker spec hash mismatch")
    runtime = result.get("runtime")
    _require(isinstance(runtime, Mapping), "worker runtime contract missing")
    runtime = cast(Mapping[str, Any], runtime)
    _require(tuple(runtime.get("joint_order", ())) == A1_PACE_JOINT_ORDER, "worker joint order mismatch")
    runtime_dt = _finite_number(runtime.get("dt_s"), "worker runtime dt_s")
    _require(
        math.isclose(runtime_dt, DT_S, rel_tol=0.0, abs_tol=1.0e-12),
        "worker dt mismatch",
    )
    _require(runtime.get("num_envs") == 1, "worker did not use one environment")
    _require(runtime.get("fix_root_link") is True, "worker task is not fixed-base")
    _require(runtime.get("action_scale") == 1.0, "worker action scale mismatch")
    _require(runtime.get("use_default_offset") is False, "worker action offset mismatch")
    bounds_value = runtime.get("bounds")
    _require(isinstance(bounds_value, Sequence), "worker runtime bounds missing")
    runtime_bounds = _validate_bounds(cast(Sequence[Sequence[Any]], bounds_value))
    if expected_bounds is not None:
        _require(runtime_bounds == list(expected_bounds), "runtime bounds changed between workers")


def _coordinator_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed 49-dimensional A1 PACE sensitivity gate"
    )
    parser.add_argument("--task", default="Isaac-Pace-A1-v0")
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--noise-repeats", type=int, default=DEFAULT_NOISE_REPEATS)
    parser.add_argument("--noise-multiplier", type=float, default=5.0)
    parser.add_argument("--worker-timeout-s", type=float, default=600.0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _coordinator_main(argv: Sequence[str]) -> int:
    args = _coordinator_parser().parse_args(argv)
    _integer(args.seed, "seed", minimum=0)
    _require(MIN_SAMPLE_COUNT <= args.sample_count <= MAX_SAMPLE_COUNT, f"sample_count must be in [{MIN_SAMPLE_COUNT}, {MAX_SAMPLE_COUNT}]")
    _integer(args.noise_repeats, "noise_repeats", minimum=2)
    _require(math.isfinite(args.noise_multiplier) and args.noise_multiplier > 1.0, "noise_multiplier must be > 1")
    _require(math.isfinite(args.worker_timeout_s) and args.worker_timeout_s > 0.0, "worker_timeout_s must be positive")

    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[2]
    priors_path = args.priors.resolve()
    output_path = args.output.resolve()
    work_dir = args.work_dir.resolve()
    python_executable = args.python_executable.resolve()
    _require(priors_path.is_file(), f"prior file does not exist: {priors_path}")
    _require(python_executable.is_file(), f"Python executable does not exist: {python_executable}")
    _require(output_path not in {priors_path, script_path}, "output aliases an input")
    if work_dir.exists():
        _require(not any(work_dir.iterdir()), "work-dir must be absent or empty")
    work_dir.mkdir(parents=True, exist_ok=True)

    prior_hash = _sha256_path(priors_path)
    script_hash = _sha256_path(script_path)
    repository_revision = _git_head(repository_root)
    extension_hashes = _a1_extension_hashes(repository_root)
    priors = _strict_json_load(priors_path)
    validate_prior_contract(priors)

    def child_paths(probe_id: str) -> tuple[Path, Path, Path, Path]:
        return (
            work_dir / f"{probe_id}.spec.json",
            work_dir / f"{probe_id}.result.json",
            work_dir / f"{probe_id}.stdout.log",
            work_dir / f"{probe_id}.stderr.log",
        )

    def run_spec(spec: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path]:
        probe_id = str(spec.get("probe_id", spec["mode"]))
        spec_path, result_path, stdout_path, stderr_path = child_paths(probe_id)
        _atomic_json(spec, spec_path)
        result, audit = _run_child(
            python_executable=python_executable,
            script_path=script_path,
            spec_path=spec_path,
            result_path=result_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            device=args.device,
            headless=args.headless,
            timeout_s=args.worker_timeout_s,
        )
        return result, audit, spec_path

    discovery_spec = {
        "schema_version": WORKER_SPEC_SCHEMA,
        "mode": "discover",
        "task": args.task,
        "seed": args.seed,
        "sample_count": args.sample_count,
    }
    discovery, discovery_audit, discovery_spec_path = run_spec(discovery_spec)
    _check_worker_result(discovery, spec_path=discovery_spec_path, expected_mode="discover")
    bounds = _validate_bounds(discovery["runtime"]["bounds"])
    mid_vector = [(low + high) / 2.0 for low, high in bounds]

    def run_probe(
        probe_id: str, parameter_index: int, level: str, vector: list[float]
    ) -> tuple[dict[str, Any], dict[str, Any], list[float]]:
        parameter_value = (
            vector[parameter_index] if parameter_index >= 0 else float("nan")
        )
        # JSON cannot carry NaN; all-mid noise probes identify their value with delay-mid.
        if parameter_index < 0:
            parameter_value = vector[48]
        trajectory_path = work_dir / f"{probe_id}.trajectory.f32"
        spec = {
            "schema_version": WORKER_SPEC_SCHEMA,
            "mode": "probe",
            "task": args.task,
            "seed": args.seed,
            "sample_count": args.sample_count,
            "probe_id": probe_id,
            "parameter_index": parameter_index,
            "level": level,
            "parameter_value": parameter_value,
            "vector": vector,
            "trajectory_output": str(trajectory_path),
        }
        result, audit, spec_path = run_spec(spec)
        _check_worker_result(
            result,
            spec_path=spec_path,
            expected_mode="probe",
            expected_bounds=bounds,
        )
        _require(result.get("probe_id") == probe_id, "worker probe_id mismatch")
        _require(result.get("parameter_index") == parameter_index, "worker parameter_index mismatch")
        _require(result.get("level") == level, "worker level mismatch")
        _require(result.get("seed") == args.seed, "worker seed mismatch")
        _require(result.get("sample_count") == args.sample_count, "worker sample count mismatch")
        _require(result.get("vector_sha256") == _canonical_sha256(vector), "worker vector hash mismatch")
        return result, audit, _read_trajectory(result)

    noise_runs: list[dict[str, Any]] = []
    noise_trajectories: list[list[float]] = []
    target_hash: str | None = None
    for repeat in range(args.noise_repeats):
        probe_id = f"noise_mid_{repeat:02d}"
        result, audit, trajectory = run_probe(
            probe_id, -1, "noise_mid", list(mid_vector)
        )
        if target_hash is None:
            target_hash = result["target_sha256"]
        _require(result["target_sha256"] == target_hash, "fixed targets changed between workers")
        noise_runs.append(_probe_metadata(result, audit, value=mid_vector[48]))
        noise_trajectories.append(trajectory)

    noise_pair_metrics: list[dict[str, Any]] = []
    for left_index in range(len(noise_trajectories)):
        for right_index in range(left_index + 1, len(noise_trajectories)):
            metric = _difference_metric(
                noise_trajectories[left_index], noise_trajectories[right_index]
            )
            noise_pair_metrics.append(
                {"left": left_index, "right": right_index, **metric}
            )
    noise_record = {
        "repeat_count": args.noise_repeats,
        "pair_count": len(noise_pair_metrics),
        "rmse_rad": max(metric["rmse_rad"] for metric in noise_pair_metrics),
        "max_abs_rad": max(metric["max_abs_rad"] for metric in noise_pair_metrics),
        "absolute_floor_rad": ABSOLUTE_NOISE_FLOOR_RAD,
    }
    noise_reference = noise_trajectories[0]

    dimension_records: list[dict[str, Any]] = []
    response_signatures: list[list[float]] = []
    all_probe_commands: list[str] = []
    for index, expected in enumerate(PARAMETER_CONTRACT):
        low, high = bounds[index]
        values = {"low": low, "mid": (low + high) / 2.0, "high": high}
        probe_results: dict[str, Mapping[str, Any]] = {}
        probe_audits: dict[str, Mapping[str, Any]] = {}
        trajectories: dict[str, list[float]] = {}
        for level in ("low", "mid", "high"):
            vector = list(mid_vector)
            vector[index] = values[level]
            probe_id = f"dimension_{index:02d}_{level}"
            result, audit, trajectory = run_probe(probe_id, index, level, vector)
            _require(result["target_sha256"] == target_hash, "fixed targets changed between probes")
            probe_results[level] = result
            probe_audits[level] = audit
            trajectories[level] = trajectory
            all_probe_commands.append(audit["command"])
        response_signatures.append(
            [
                high_value - low_value
                for low_value, high_value in zip(
                    trajectories["low"], trajectories["high"], strict=True
                )
            ]
        )
        dimension_records.append(
            {
                "index": index,
                "name": expected["name"],
                "probes": {
                    level: _probe_metadata(
                        probe_results[level], probe_audits[level], value=values[level]
                    )
                    for level in ("low", "mid", "high")
                },
                "responses": {
                    "low_vs_mid": _difference_metric(
                        trajectories["low"], trajectories["mid"]
                    ),
                    "high_vs_mid": _difference_metric(
                        trajectories["high"], trajectories["mid"]
                    ),
                    "mid_vs_noise_reference": _difference_metric(
                        trajectories["mid"], noise_reference
                    ),
                },
            }
        )

    records = {
        "schema_version": RECORD_SCHEMA,
        "noise": noise_record,
        "dimensions": dimension_records,
    }
    evaluation = evaluate_sensitivity_records(
        bounds, priors, records, args.noise_multiplier
    )
    response_matrix = _response_matrix_diagnostic(response_signatures)
    matrix_failures = (
        []
        if response_matrix["full_rank"]
        else ["49-dimensional response matrix is numerically rank deficient"]
    )
    accepted = bool(evaluation["accepted"] and not matrix_failures)
    _require(_sha256_path(priors_path) == prior_hash, "prior file changed during gate")
    _require(_sha256_path(script_path) == script_hash, "gate script changed during gate")
    _require(
        _git_head(repository_root) == repository_revision
        and _a1_extension_hashes(repository_root) == extension_hashes,
        "PACE revision or A1 extension source changed during sensitivity gate",
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS" if accepted else "FAIL",
        "accepted": accepted,
        "scope": "synthetic_numerical_only",
        "physical_motion_authorized": False,
        "task": args.task,
        "robot": ROBOT_NAME,
        "seed": args.seed,
        "sample_count": args.sample_count,
        "dt_s": DT_S,
        "target_sha256": target_hash,
        "noise_multiplier": args.noise_multiplier,
        "prior_path": str(priors_path),
        "prior_sha256": prior_hash,
        "script_path": str(script_path),
        "script_sha256": script_hash,
        "repository_revision": repository_revision,
        "a1_extension_source_sha256": extension_hashes,
        "python_executable": str(python_executable),
        "device": args.device,
        "headless": args.headless,
        "registered_task_contract": discovery["runtime"],
        "discovery_process": discovery_audit,
        "noise_runs": noise_runs,
        "noise_pairs": noise_pair_metrics,
        "noise": evaluation["noise"],
        "thresholds": evaluation["thresholds"],
        "response_matrix_diagnostic": response_matrix,
        "dimensions": evaluation["dimensions"],
        "failures": [*evaluation["failures"], *matrix_failures],
        "process_contract": {
            "discovery_process_count": 1,
            "noise_process_count": args.noise_repeats,
            "dimension_process_count": 49 * 3,
            "total_child_process_count": 1 + args.noise_repeats + 49 * 3,
            "each_probe_has_fresh_app_process": True,
            "dimension_probe_commands_sha256": _canonical_sha256(all_probe_commands),
        },
        "exact_command": shlex.join(sys.argv),
    }
    _atomic_json(report, output_path)
    reloaded = _strict_json_load(output_path)
    _require(reloaded == report, "published sensitivity report changed on reload")
    print(
        json.dumps(
            {
                "status": report["status"],
                "accepted": report["accepted"],
                "output": str(output_path),
                "output_sha256": _sha256_path(output_path),
                "failed_dimensions": len(report["failures"]),
            },
            sort_keys=True,
        )
    )
    return 0 if report["accepted"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if "--_worker-spec" in arguments:
            return _worker_main(arguments)
        return _coordinator_main(arguments)
    except Exception as exc:
        print(f"A1 sensitivity gate failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
