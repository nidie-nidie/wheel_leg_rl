"""Replay one explicitly named A1 PACE mean and report encoder-frame error."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, cast

import torch


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
CHECKOUT_NAMES = ("pace", "isaaclab", "a1_base", "gogo_learn")
NEGATIVE_CONTROL_DELAYS = frozenset((0, 1, 5, 9))
REAL_DIAGNOSTIC_PROFILES = {
    "real-stock-diagnostic": "stock",
    "real-identified-diagnostic": "identified",
}
REAL_VALIDATION_NAMES = frozenset(("validation_a", "validation_b"))
REAL_SELECTION_SCHEMA_VERSION = "a1_pace_selection/v1"
REAL_SELECTION_INDEPENDENCE_FIELDS = (
    "data_sha256",
    "tensor_payload_sha256",
    "raw_csv_sha256",
    "trial_id",
    "target_tensor_sha256",
    "seed_phase_family",
    "output_path",
)
SYNTHETIC_GATE_LIMITS = {
    "synthetic-heldout-half": {
        "max_rmse": 0.005,
        "max_p95": 0.01,
        "max_joint_rmse": 0.015,
        "max_joint_p95": 0.03,
        "max_joint_abs": 0.015,
    },
    "synthetic-heldout-full": {
        "max_rmse": 0.005,
        "max_p95": 0.01,
        "max_joint_rmse": 0.015,
        "max_joint_p95": 0.03,
        "max_joint_abs": 0.015,
    },
    "synthetic-known-positive": {
        "max_rmse": 0.005,
        "max_p95": 0.01,
        "max_joint_rmse": 0.015,
        "max_joint_p95": 0.03,
        "max_joint_abs": 0.015,
    },
    "synthetic-negative-dynamics": {
        "max_rmse": 0.005,
        "max_p95": 0.01,
        "max_joint_rmse": 0.015,
        "max_joint_p95": 0.03,
        "max_joint_abs": 0.015,
    },
    "synthetic-negative-delay": {
        "max_rmse": 0.005,
        "max_p95": 0.01,
        "max_joint_rmse": 0.015,
        "max_joint_p95": 0.03,
        "max_joint_abs": 0.015,
    },
    "synthetic-endpoint": {
        "max_rmse": 0.0001,
        "max_p95": 0.0002,
        "max_joint_rmse": 0.0002,
        "max_joint_p95": 0.0004,
        "max_joint_abs": 0.002,
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256",
    )
    return cast(str, value)


def _encoder_bias_values(
    encoder_bias_reference: Mapping[str, Any],
) -> list[float]:
    _require(
        set(encoder_bias_reference) == set(A1_PACE_JOINT_ORDER),
        "encoder bias reference must contain the exact A1 joint set",
    )
    values = [encoder_bias_reference[name] for name in A1_PACE_JOINT_ORDER]
    _require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        ),
        "encoder bias reference must contain finite numbers",
    )
    return [float(value) for value in values]


def encoder_bias_reference_evidence(
    encoder_bias_reference: Mapping[str, Any], profile: str
) -> dict[str, Any]:
    """Serialize the exact encoder frame used by one replay."""

    _require(
        profile
        in {
            "zero_stock_reference",
            "fitted_mean_reference",
            "effective_mean_reference",
        },
        "unsupported encoder bias profile",
    )
    values = _encoder_bias_values(encoder_bias_reference)
    payload = json.dumps(
        {"joint_order": list(A1_PACE_JOINT_ORDER), "values_rad": values},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256(b"a1_pace_encoder_bias_reference/v1\0" + payload)
    return {
        "profile": profile,
        "joint_order": list(A1_PACE_JOINT_ORDER),
        "values_rad": values,
        "sha256": digest.hexdigest(),
    }


def target_tensor_sha256(data: Mapping[str, Any]) -> str:
    """Hash real validation targets with Gogo's frozen domain contract."""

    target = data.get("des_dof_pos")
    _require(
        isinstance(target, torch.Tensor)
        and torch.is_floating_point(target)
        and target.ndim == 2
        and target.shape[1:] == (len(A1_PACE_JOINT_ORDER),)
        and bool(torch.isfinite(target).all()),
        "target tensor is invalid",
    )
    target = cast(torch.Tensor, target)
    target = target.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"a1_pace_target_tensor/v1\0")
    digest.update(str(target.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(target.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(target.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _finite_threshold(value: Any, name: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) >= 0.0
        and math.isfinite(float(value)),
        f"{name} must be a finite non-negative number",
    )
    return float(value)


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(
        path.read_text(encoding="ascii"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")

    def require_finite(item: Any) -> None:
        if isinstance(item, float):
            _require(math.isfinite(item), f"JSON contains a non-finite value: {path}")
        elif isinstance(item, dict):
            for child in item.values():
                require_finite(child)
        elif isinstance(item, list):
            for child in item:
                require_finite(child)

    require_finite(value)
    return value


def resolve_regular_input_path(path: Path | str, label: str) -> Path:
    """Reject lexical symlinks before resolving one immutable real input."""

    lexical = Path(path)
    _require(os.path.lexists(lexical), f"{label} does not exist")
    _require(not lexical.is_symlink(), f"{label} must not be a symbolic link")
    _require(lexical.is_file(), f"{label} must be a regular file")
    resolved = lexical.resolve(strict=True)
    _require(
        resolved.is_file() and not resolved.is_symlink(),
        f"{label} must resolve to a regular non-symlink file",
    )
    return resolved


def evaluate_replay_thresholds(
    metrics: Mapping[str, Any],
    *,
    max_rmse: float,
    max_p95: float,
    max_joint_rmse: float,
    max_joint_p95: float,
    max_joint_abs: float,
) -> dict[str, Any]:
    """Apply aggregate and worst-joint gates without averaging away a bad motor."""

    limits = {
        "aggregate_rmse_rad": _finite_threshold(max_rmse, "max_rmse"),
        "aggregate_p95_rad": _finite_threshold(max_p95, "max_p95"),
        "per_joint_rmse_rad": _finite_threshold(max_joint_rmse, "max_joint_rmse"),
        "per_joint_p95_rad": _finite_threshold(max_joint_p95, "max_joint_p95"),
        "per_joint_max_abs_rad": _finite_threshold(max_joint_abs, "max_joint_abs"),
    }
    aggregate_rmse = _finite_threshold(
        metrics.get("aggregate_rmse_rad"), "aggregate_rmse_rad"
    )
    aggregate_p95 = _finite_threshold(
        metrics.get("aggregate_p95_rad"), "aggregate_p95_rad"
    )
    per_joint = metrics.get("per_joint")
    _require(
        isinstance(per_joint, list) and len(per_joint) == 12,
        "replay metrics must contain exactly 12 per-joint records",
    )
    per_joint = cast(list[Any], per_joint)
    failures: list[str] = []
    rmse_pass = aggregate_rmse <= limits["aggregate_rmse_rad"]
    p95_pass = aggregate_p95 <= limits["aggregate_p95_rad"]
    if not rmse_pass:
        failures.append("aggregate RMSE exceeds threshold")
    if not p95_pass:
        failures.append("aggregate P95 exceeds threshold")

    per_joint_rmse_pass = True
    per_joint_p95_pass = True
    per_joint_abs_pass = True
    for index, (record, expected_name) in enumerate(
        zip(per_joint, A1_PACE_JOINT_ORDER, strict=True)
    ):
        _require(isinstance(record, Mapping), f"per_joint[{index}] must be an object")
        record = cast(Mapping[str, Any], record)
        _require(
            record.get("joint") == expected_name,
            f"per_joint[{index}] joint order mismatch",
        )
        joint_rmse = _finite_threshold(
            record.get("rmse_rad"), f"per_joint[{index}].rmse_rad"
        )
        joint_p95 = _finite_threshold(
            record.get("p95_rad"), f"per_joint[{index}].p95_rad"
        )
        joint_abs = _finite_threshold(
            record.get("max_abs_rad"), f"per_joint[{index}].max_abs_rad"
        )
        if joint_rmse > limits["per_joint_rmse_rad"]:
            per_joint_rmse_pass = False
            failures.append(f"{expected_name} RMSE exceeds per-joint threshold")
        if joint_p95 > limits["per_joint_p95_rad"]:
            per_joint_p95_pass = False
            failures.append(f"{expected_name} P95 exceeds per-joint threshold")
        if joint_abs > limits["per_joint_max_abs_rad"]:
            per_joint_abs_pass = False
            failures.append(f"{expected_name} max-abs exceeds per-joint threshold")

    accepted = (
        rmse_pass
        and p95_pass
        and per_joint_rmse_pass
        and per_joint_p95_pass
        and per_joint_abs_pass
    )
    return {
        "status": "PASS" if accepted else "FAIL",
        "accepted": accepted,
        "limits": limits,
        "aggregate_rmse_pass": rmse_pass,
        "aggregate_p95_pass": p95_pass,
        "per_joint_rmse_pass": per_joint_rmse_pass,
        "per_joint_p95_pass": per_joint_p95_pass,
        "per_joint_max_abs_pass": per_joint_abs_pass,
        "failures": failures,
    }


def build_negative_control_vector(truth: torch.Tensor, mode: str) -> torch.Tensor:
    """Derive one frozen, conspicuously wrong A1 parameter vector from truth."""

    _require(isinstance(truth, torch.Tensor), "negative-control truth must be a tensor")
    _require(
        truth.ndim == 1
        and truth.shape == (49,)
        and torch.is_floating_point(truth)
        and bool(torch.isfinite(truth).all()),
        "negative-control truth must be one finite floating 49-vector",
    )
    control = truth.detach().to(device="cpu").clone()
    negative_modes = {
        "synthetic-negative-dynamics",
        "synthetic-negative-delay",
    }
    _require(mode in negative_modes, f"unsupported negative-control mode: {mode}")
    delay_value = float(control[48].item())
    delay_steps = int(delay_value)
    _require(
        delay_value == float(delay_steps) and delay_steps in NEGATIVE_CONTROL_DELAYS,
        "negative-control truth delay must be one of the gated delays 0/1/5/9",
    )
    if mode == "synthetic-negative-dynamics":
        control[:12] = 1.0e-5
        control[12:36] = 0.0
        control[36:48] = 0.0
    elif mode == "synthetic-negative-delay":
        control[48] = float(delay_steps + 1)
    _require(
        not torch.equal(control, truth.detach().to(device="cpu")),
        "negative-control vector does not differ from truth",
    )
    return control.contiguous()


def interpret_negative_control_decision(
    threshold_decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Pass a negative control only when the positive-model thresholds reject it."""

    accepted = threshold_decision.get("accepted")
    _require(isinstance(accepted, bool), "threshold decision accepted flag is invalid")
    rejected = not accepted
    return {
        "status": "PASS" if rejected else "FAIL",
        "accepted": rejected,
        "negative_control_rejected": rejected,
        "threshold_decision": dict(threshold_decision),
        "failures": (
            []
            if rejected
            else ["negative control met the frozen positive-model thresholds"]
        ),
    }


def validate_replay_argument_contract(args: argparse.Namespace) -> str | None:
    """Validate mode-specific inputs and derive the real actuator profile."""

    mode = getattr(args, "mode", None)
    if mode in REAL_DIAGNOSTIC_PROFILES:
        _require(
            getattr(args, "fit_manifest", None) is not None,
            "real replay requires a fit manifest",
        )
        _require(
            getattr(args, "selection_manifest", None) is not None,
            "real replay requires a selection manifest",
        )
        validation_name = getattr(args, "validation_name", None)
        _require(
            validation_name is not None,
            "real replay requires a validation name",
        )
        _require(
            validation_name in REAL_VALIDATION_NAMES,
            "real replay validation name must be validation_a or validation_b",
        )
        _require(
            getattr(args, "expected_delay", None) is None
            and getattr(args, "data_manifest", None) is None
            and getattr(args, "data_known_mean", None) is None,
            "real replay forbids synthetic truth arguments",
        )
        return REAL_DIAGNOSTIC_PROFILES[mode]
    _require(mode in SYNTHETIC_GATE_LIMITS, f"unsupported replay mode: {mode}")
    _require(
        getattr(args, "selection_manifest", None) is None
        and getattr(args, "validation_name", None) is None,
        "synthetic replay forbids real selection arguments",
    )
    return None


def _validate_selection_trial_record(
    record: Any, *, name: str, expected_role: str
) -> Mapping[str, Any]:
    _require(isinstance(record, Mapping), f"selection {name} record is invalid")
    record = cast(Mapping[str, Any], record)
    _require(record.get("role") == expected_role, f"selection {name} role mismatch")
    for field in (
        "data_sha256",
        "tensor_payload_sha256",
        "raw_csv_sha256",
        "target_tensor_sha256",
        "seed_phase_family",
    ):
        _require_sha256(record.get(field), f"selection {name} {field}")
    for field in ("trial_id",):
        value = record.get(field)
        _require(
            isinstance(value, str) and bool(value.strip()),
            f"selection {name} {field} is invalid",
        )
    output_path = record.get("output_path")
    _require(
        isinstance(output_path, str) and Path(output_path).is_absolute(),
        f"selection {name} output_path must be absolute",
    )
    return record


def validate_real_selection_binding(
    selection_manifest: Mapping[str, Any],
    *,
    selection_manifest_path: Path,
    validation_name: str,
    data_path: Path,
    data_payload_sha256: str,
    data_target_sha256: str,
    fit_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one held-out real tensor to its selected, independent fit trial."""

    _require(
        selection_manifest_path.is_file() and not selection_manifest_path.is_symlink(),
        "selection manifest must be a regular non-symlink file",
    )
    _require(
        _load_json(selection_manifest_path) == dict(selection_manifest),
        "selection manifest changed after loading",
    )
    _require(
        selection_manifest.get("schema_version") == REAL_SELECTION_SCHEMA_VERSION,
        "selection manifest schema is not a1_pace_selection/v1",
    )
    _require(
        selection_manifest.get("status") == "PASS",
        "selection manifest status must be PASS",
    )
    _require(
        validation_name in REAL_VALIDATION_NAMES,
        "selection validation name must be validation_a or validation_b",
    )
    selected_fit = _validate_selection_trial_record(
        selection_manifest.get("selected_fit"),
        name="selected_fit",
        expected_role="fit",
    )
    validation_trials = selection_manifest.get("validation_trials")
    _require(
        isinstance(validation_trials, Mapping)
        and set(validation_trials) == REAL_VALIDATION_NAMES,
        "selection manifest must contain both validation records",
    )
    validation_trials = cast(Mapping[str, Any], validation_trials)
    validated_trials = {
        name: _validate_selection_trial_record(
            validation_trials[name], name=name, expected_role="validation"
        )
        for name in sorted(REAL_VALIDATION_NAMES)
    }
    records = {"selected_fit": selected_fit, **validated_trials}
    for field in REAL_SELECTION_INDEPENDENCE_FIELDS:
        values = [record[field] for record in records.values()]
        _require(
            len(set(values)) == len(values),
            f"fit and validation trials are not independent in {field}",
        )

    _require(
        fit_manifest.get("schema_version") == "a1_pace_fit_manifest/v1"
        and fit_manifest.get("status") == "PASS",
        "selection binding requires an accepted fit manifest",
    )
    fit_provenance = fit_manifest.get("provenance")
    _require(
        isinstance(fit_provenance, Mapping) and fit_provenance.get("mode") == "real",
        "selection binding requires real fit provenance",
    )
    fit_provenance = cast(Mapping[str, Any], fit_provenance)
    _require(
        fit_manifest.get("source_data_sha256") == selected_fit["data_sha256"]
        and fit_manifest.get("source_tensor_payload_sha256")
        == selected_fit["tensor_payload_sha256"]
        and fit_provenance.get("tensor_payload_sha256")
        == selected_fit["tensor_payload_sha256"]
        and fit_provenance.get("raw_csv_sha256") == selected_fit["raw_csv_sha256"],
        "selection selected fit does not match the fit manifest",
    )

    validation = validated_trials[validation_name]
    _require(
        data_path.is_file() and not data_path.is_symlink(),
        "validation data must be a regular non-symlink file",
    )
    data_sha256 = _sha256_path(data_path)
    _require(
        validation["data_sha256"] == data_sha256,
        "validation data SHA does not match selection",
    )
    _require(
        validation["tensor_payload_sha256"]
        == _require_sha256(data_payload_sha256, "validation payload SHA"),
        "validation payload SHA does not match selection",
    )
    _require(
        validation["target_tensor_sha256"]
        == _require_sha256(data_target_sha256, "validation target SHA"),
        "validation target SHA does not match selection",
    )
    _require(
        Path(cast(str, validation["output_path"])).resolve() == data_path.resolve(),
        "validation data path does not match selection",
    )
    return {
        "selection_manifest_sha256": _sha256_path(selection_manifest_path),
        "selection_schema_version": REAL_SELECTION_SCHEMA_VERSION,
        "validation_name": validation_name,
        "validation_data_sha256": data_sha256,
        "validation_tensor_payload_sha256": data_payload_sha256,
        "validation_target_tensor_sha256": data_target_sha256,
        "selected_fit_data_sha256": selected_fit["data_sha256"],
        "selected_fit_trial_id": selected_fit["trial_id"],
        "validation_trial_id": validation["trial_id"],
        "independence_fields": list(REAL_SELECTION_INDEPENDENCE_FIELDS),
        "independence_status": "PASS",
    }


def enforce_replay_decision(mode: str, decision: Mapping[str, Any]) -> None:
    """Raise for every non-diagnostic replay whose mode-specific gate failed."""

    if mode in REAL_DIAGNOSTIC_PROFILES:
        return
    failures = decision.get("failures")
    _require(isinstance(failures, list), "replay decision failures are invalid")
    failures = cast(list[Any], failures)
    _require(
        decision.get("accepted") is True,
        "; ".join(str(failure) for failure in failures),
    )


def validate_fitted_mean_binding(
    fit_manifest: Mapping[str, Any],
    *,
    fit_manifest_path: Path,
    mean_path: Path,
    data_manifest: Mapping[str, Any] | None,
    repository_root: Path,
    task: str,
    expected_delay: int | None,
    require_held_out: bool,
    expected_heldout_amplitude_scale: float | None,
    current_revisions: Mapping[str, str],
) -> dict[str, Any]:
    """Prove that a mean is the exact guard-selected artifact for this replay."""

    _require(fit_manifest_path.is_file(), "fit manifest does not exist")
    _require(
        fit_manifest.get("schema_version") == "a1_pace_fit_manifest/v1"
        and fit_manifest.get("status") == "PASS",
        "fit manifest is not an accepted A1 PACE fit",
    )
    _require(fit_manifest.get("task") == task, "fit manifest task mismatch")
    manifest_revisions = fit_manifest.get("repository_revisions")
    _require(
        isinstance(manifest_revisions, Mapping)
        and set(manifest_revisions) == set(CHECKOUT_NAMES)
        and all(
            isinstance(value, str)
            and 40 <= len(value) <= 64
            and all(character in "0123456789abcdef" for character in value)
            for value in manifest_revisions.values()
        ),
        "fit manifest repository revisions are missing or invalid",
    )
    manifest_revisions = cast(Mapping[str, str], manifest_revisions)
    _require(
        fit_manifest.get("repository_cleanliness")
        == {name: True for name in CHECKOUT_NAMES},
        "fit manifest repository cleanliness proof is missing",
    )
    _require(
        set(current_revisions) == set(CHECKOUT_NAMES)
        and dict(manifest_revisions) == dict(current_revisions),
        "fit manifest differs from current checkout revisions",
    )
    run_basename = fit_manifest.get("run_basename")
    mean_basename = fit_manifest.get("mean_basename")
    _require(
        isinstance(run_basename, str)
        and run_basename not in ("", ".", "..")
        and Path(run_basename).name == run_basename,
        "fit manifest run basename is invalid",
    )
    _require(
        isinstance(mean_basename, str) and Path(mean_basename).name == mean_basename,
        "fit manifest mean basename is invalid",
    )
    run_basename = cast(str, run_basename)
    mean_basename = cast(str, mean_basename)
    expected_path = (
        repository_root / "logs" / "pace" / "a1" / run_basename / mean_basename
    )
    _require(
        mean_path.absolute() == expected_path.absolute(),
        "mean is not at the guard-selected run path",
    )
    _require(
        mean_path.is_file() and not mean_path.is_symlink(),
        "guard-selected mean must be a regular non-symlink file",
    )
    expected_mean_sha = fit_manifest.get("mean_sha256")
    _require(
        isinstance(expected_mean_sha, str)
        and len(expected_mean_sha) == 64
        and _sha256_path(mean_path) == expected_mean_sha,
        "mean SHA does not match the fit manifest",
    )
    provenance = fit_manifest.get("provenance")
    _require(isinstance(provenance, Mapping), "fit manifest provenance is missing")
    provenance = cast(Mapping[str, Any], provenance)
    if expected_delay is not None:
        _require(
            provenance.get("delay_steps") == expected_delay,
            "fit provenance delay mismatch",
        )
    if data_manifest is not None:
        heldout_revisions = data_manifest.get("revisions")
        _require(
            heldout_revisions
            == {
                "pace_sim2real": manifest_revisions["pace"],
                "isaaclab": manifest_revisions["isaaclab"],
            },
            "held-out repository revisions differ from the fitted checkout",
        )
        _require(
            provenance.get("mode") == "synthetic",
            "synthetic replay requires synthetic fit provenance",
        )
        _require(
            provenance.get("source_amplitude_scale") == 1.0,
            "synthetic fit provenance is not full-amplitude",
        )
        _require(
            data_manifest.get("task") == task,
            "replay data manifest task mismatch",
        )
        _require(
            data_manifest.get("delay_steps") == provenance.get("delay_steps"),
            "replay data and fit delay differ",
        )
        _require(
            data_manifest.get("physical_parameter_vector")
            == provenance.get("effective_parameter_vector"),
            "replay data and fit physical truth differ",
        )
        if require_held_out:
            excitation = data_manifest.get("excitation")
            _require(
                isinstance(excitation, Mapping)
                and expected_heldout_amplitude_scale in (0.5, 1.0)
                and excitation.get("amplitude_scale")
                == expected_heldout_amplitude_scale,
                "synthetic held-out amplitude profile mismatch",
            )
            _require(
                data_manifest.get("seed") != provenance.get("source_seed")
                and data_manifest.get("target_tensor_sha256")
                != provenance.get("source_target_sha256"),
                "synthetic gate requires genuinely held-out replay targets",
            )
    return {
        "fit_manifest_sha256": _sha256_path(fit_manifest_path),
        "run_basename": run_basename,
        "mean_basename": mean_basename,
        "mean_sha256": expected_mean_sha,
        "repository_revisions": dict(manifest_revisions),
    }


def compute_encoder_replay_metrics(
    physical_samples: torch.Tensor,
    data: dict[str, torch.Tensor],
    encoder_bias_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare replayed physical q-bias against measured encoder q."""

    measured = data["dof_pos"].detach().to(device="cpu", dtype=torch.float64)
    _require(
        isinstance(physical_samples, torch.Tensor)
        and physical_samples.shape == measured.shape,
        "physical replay samples must match measured shape",
    )
    _require(
        bool(torch.isfinite(physical_samples).all()),
        "physical replay samples must be finite",
    )
    bias = torch.tensor(
        _encoder_bias_values(encoder_bias_reference),
        dtype=torch.float64,
    )
    encoder_samples = (
        physical_samples.detach().to(device="cpu", dtype=torch.float64) - bias
    )
    error = encoder_samples - measured
    absolute = error.abs()
    per_joint_rmse = torch.sqrt(torch.mean(error.square(), dim=0))
    per_joint_p95 = torch.quantile(absolute, 0.95, dim=0)
    per_joint_max_abs = torch.max(absolute, dim=0).values
    aggregate_rmse = torch.sqrt(torch.mean(error.square()))
    aggregate_p95 = torch.quantile(absolute.flatten(), 0.95)
    aggregate_max_abs = torch.max(absolute)
    return {
        "sample_count": int(measured.shape[0]),
        "aggregate_rmse_rad": float(aggregate_rmse),
        "aggregate_p95_rad": float(aggregate_p95),
        "aggregate_max_abs_rad": float(aggregate_max_abs),
        "per_joint": [
            {
                "joint": name,
                "rmse_rad": float(per_joint_rmse[index]),
                "p95_rad": float(per_joint_p95[index]),
                "max_abs_rad": float(per_joint_max_abs[index]),
            }
            for index, name in enumerate(A1_PACE_JOINT_ORDER)
        ],
    }


def _atomic_json(value: dict[str, Any], destination: Path) -> None:
    _require(
        not os.path.lexists(destination),
        f"replay output already exists: {destination}",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".json", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("w", encoding="ascii", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ValueError(f"replay output already exists: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _exclusive_output_lock(destination: Path) -> Iterator[None]:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.a1-pace.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"replay destination is locked: {destination}") from exc
    try:
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        finally:
            os.close(descriptor)
    except Exception:
        lock_path.unlink(missing_ok=True)
        raise
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def replay(args: argparse.Namespace) -> dict[str, Any]:
    actuator_profile = validate_replay_argument_contract(args)
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import isaaclab_tasks  # noqa: F401
    import pace_sim2real.tasks  # noqa: F401
    from pace_sim2real.tasks.manager_based.pace.a1_mean import (
        load_a1_mean,
        parse_a1_mean,
    )
    from pace_sim2real.tasks.manager_based.pace.a1_replay import (
        inject_a1_actuator_before_make,
        inject_a1_stock_actuator_before_make,
        make_a1_pace_actuator_cfg,
        replay_absolute_targets,
        validate_a1_pace_data,
        zero_a1_encoder_bias_reference,
    )
    from pace_sim2real.utils import PaceDCMotor

    import importlib.util

    guard_path = Path(__file__).with_name("a1_fit_guard.py")
    spec = importlib.util.spec_from_file_location("a1_fit_guard_runtime", guard_path)
    _require(
        spec is not None and spec.loader is not None,
        "cannot load the sibling A1 fit guard",
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the sibling A1 fit guard")
    guard_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard_module)
    _a1_extension_source_hashes = guard_module._a1_extension_source_hashes
    _freeze_checkout_state = guard_module._freeze_checkout_state
    _validate_frozen_checkout_state = guard_module._validate_frozen_checkout_state
    tensor_payload_sha256 = guard_module.tensor_payload_sha256
    validate_synthetic_manifest = guard_module.validate_synthetic_manifest

    repository_root = Path(args.repository_root).resolve()
    checkout_roots = {
        "pace": repository_root,
        "isaaclab": Path(args.isaaclab_root).resolve(),
        "a1_base": Path(args.a1_base_root).resolve(),
        "gogo_learn": Path(args.gogo_learn_root).resolve(),
    }
    real_input_paths: dict[str, Path] = {}
    if actuator_profile is not None:
        real_input_paths = {
            "data": resolve_regular_input_path(args.data, "validation data"),
            "mean": resolve_regular_input_path(args.mean, "final mean"),
            "fit_manifest": resolve_regular_input_path(
                args.fit_manifest, "fit manifest"
            ),
            "selection_manifest": resolve_regular_input_path(
                args.selection_manifest, "selection manifest"
            ),
        }
        data_path = real_input_paths["data"]
        mean_path = real_input_paths["mean"]
    else:
        data_path = Path(args.data).resolve()
        mean_path = Path(args.mean).resolve()
    output_path = Path(args.output).resolve()
    _require(data_path.is_file(), "replay data does not exist")
    _require(mean_path.is_file(), "replay mean does not exist")
    provenance_paths = (
        {
            real_input_paths["fit_manifest"],
            real_input_paths["selection_manifest"],
        }
        if actuator_profile is not None
        else {
            Path(value).resolve()
            for value in (
                args.fit_manifest,
                args.data_manifest,
                args.data_known_mean,
            )
            if value is not None
        }
    )
    _require(
        output_path not in {data_path, mean_path, *provenance_paths},
        "replay output aliases an input",
    )
    data_sha256 = _sha256_path(data_path)
    mean_sha256 = _sha256_path(mean_path)
    provenance_hashes = {str(path): _sha256_path(path) for path in provenance_paths}
    replay_source = Path(__file__).resolve()
    replay_source_sha256 = _sha256_path(replay_source)
    data = torch.load(data_path, map_location="cpu", weights_only=True)
    validate_a1_pace_data(data)
    source_mean = load_a1_mean(mean_path)
    mean = source_mean
    negative_control: dict[str, Any] | None = None

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    _require(abs(float(env_cfg.sim.dt) - 0.002) <= 1.0e-12, "replay dt mismatch")
    _require(env_cfg.decimation == 1, "replay decimation mismatch")
    base_actuator = env_cfg.scene.robot.actuators["base_legs"]

    def numeric_values(value: Any) -> list[float]:
        values = value.values() if isinstance(value, dict) else (value,)
        return sorted({float(item) for item in values})

    class_type = base_actuator.class_type
    runtime_contract = {
        "task": args.task,
        "data_dir": str(env_cfg.sim2real.data_dir),
        "robot_name": str(env_cfg.sim2real.robot_name),
        "sim_dt_s": float(env_cfg.sim.dt),
        "decimation": int(env_cfg.decimation),
        "joint_order": list(env_cfg.sim2real.joint_order),
        "save_optimization_process": bool(
            env_cfg.sim2real.cmaes.save_optimization_process
        ),
        "max_iteration": int(env_cfg.sim2real.cmaes.max_iteration),
        "action_scale": float(env_cfg.actions.joint_pos.scale),
        "use_default_offset": bool(env_cfg.actions.joint_pos.use_default_offset),
        "fix_root_link": bool(
            env_cfg.scene.robot.spawn.articulation_props.fix_root_link
        ),
        "root_initial_position": list(env_cfg.scene.robot.init_state.pos),
        "bounds_shape": list(env_cfg.sim2real.bounds_params.shape),
        "bounds": env_cfg.sim2real.bounds_params.detach().cpu().tolist(),
        "actuator_control": {
            "class_type": f"{class_type.__module__}.{class_type.__qualname__}",
            "saturation_effort_nm": float(base_actuator.saturation_effort),
            "effort_limit_nm": float(base_actuator.effort_limit),
            "velocity_limit_rad_s": float(base_actuator.velocity_limit),
            "stiffness_values": numeric_values(base_actuator.stiffness),
            "damping_values": numeric_values(base_actuator.damping),
            "configured_delay_steps": int(base_actuator.max_delay),
        },
    }
    binding: dict[str, Any]
    heldout_profiles = {
        "synthetic-heldout-half": 0.5,
        "synthetic-heldout-full": 1.0,
    }
    truth_control_modes = {
        "synthetic-endpoint",
        "synthetic-known-positive",
        "synthetic-negative-dynamics",
        "synthetic-negative-delay",
    }
    fitted_checkout_state: Mapping[str, Mapping[str, Any]] | None = None
    current_revisions: Mapping[str, str] = {}
    if args.mode in (*heldout_profiles, *REAL_DIAGNOSTIC_PROFILES):
        observed_checkout_state = cast(
            Mapping[str, Mapping[str, Any]],
            _freeze_checkout_state(checkout_roots),
        )
        fitted_checkout_state = observed_checkout_state
        current_revisions = cast(
            Mapping[str, str], observed_checkout_state["revisions"]
        )
    if args.mode in (*heldout_profiles, *truth_control_modes):
        _require(
            args.expected_delay is not None
            and args.data_manifest is not None
            and args.data_known_mean is not None,
            "synthetic replay requires delay, data manifest, and data known mean",
        )
        _require(
            source_mean.delay_steps == args.expected_delay,
            "selected source mean does not match the expected integer delay",
        )
        data_manifest_path = Path(args.data_manifest).resolve()
        data_known_mean_path = Path(args.data_known_mean).resolve()
        data_manifest = _load_json(data_manifest_path)
        validate_synthetic_manifest(
            data_manifest,
            data_path,
            data,
            args.task,
            strict_provenance=True,
            known_mean_path=data_known_mean_path,
            runtime_contract=runtime_contract,
        )
        _require(
            data_manifest.get("a1_extension_source_sha256")
            == _a1_extension_source_hashes(repository_root),
            "replay data A1 extension hashes differ from current source",
        )
        if args.mode in heldout_profiles:
            _require(
                args.fit_manifest is not None, "held-out replay needs fit manifest"
            )
            fit_manifest_path = Path(args.fit_manifest).resolve()
            fit_manifest = _load_json(fit_manifest_path)
            binding = validate_fitted_mean_binding(
                fit_manifest,
                fit_manifest_path=fit_manifest_path,
                mean_path=mean_path,
                data_manifest=data_manifest,
                repository_root=repository_root,
                task=args.task,
                expected_delay=args.expected_delay,
                require_held_out=True,
                expected_heldout_amplitude_scale=heldout_profiles[args.mode],
                current_revisions=current_revisions,
            )
            binding.update(
                {
                    "data_manifest_sha256": _sha256_path(data_manifest_path),
                    "data_known_mean_sha256": _sha256_path(data_known_mean_path),
                    "data_target_sha256": data_manifest["target_tensor_sha256"],
                }
            )
        else:
            _require(
                args.fit_manifest is None,
                "synthetic truth/control replay forbids fit manifest",
            )
            _require(
                mean_path == data_known_mean_path
                and data_manifest.get("known_mean_sha256") == mean_sha256,
                "synthetic truth/control replay must use its exact generated known mean",
            )
            excitation = data_manifest.get("excitation")
            _require(
                isinstance(excitation, Mapping)
                and excitation.get("amplitude_scale") == 1.0,
                "synthetic truth/control replay requires full-amplitude data",
            )
            binding = {
                "data_manifest_sha256": _sha256_path(data_manifest_path),
                "known_mean_sha256": mean_sha256,
            }
            if args.mode.startswith("synthetic-negative-"):
                source_mean_tensor = torch.load(
                    mean_path, map_location="cpu", weights_only=True
                )
                effective_tensor = build_negative_control_vector(
                    source_mean_tensor, args.mode
                )
                mean = parse_a1_mean(effective_tensor)
                tensor_bytes = effective_tensor.numpy().tobytes(order="C")
                negative_control = {
                    "mode": args.mode,
                    "source_truth_sha256": mean_sha256,
                    "effective_tensor_dtype": str(effective_tensor.dtype),
                    "effective_tensor_bytes_sha256": hashlib.sha256(
                        tensor_bytes
                    ).hexdigest(),
                    "effective_parameter_vector": [
                        float(value) for value in effective_tensor.tolist()
                    ],
                    "source_truth_delay_steps": source_mean.delay_steps,
                    "effective_delay_steps": mean.delay_steps,
                }
                binding["negative_control"] = negative_control
    else:
        _require(
            actuator_profile is not None
            and args.fit_manifest is not None
            and args.selection_manifest is not None
            and args.validation_name is not None,
            "real replay argument validation was not applied",
        )
        fit_manifest_path = real_input_paths["fit_manifest"]
        fit_manifest = _load_json(fit_manifest_path)
        binding = validate_fitted_mean_binding(
            fit_manifest,
            fit_manifest_path=fit_manifest_path,
            mean_path=mean_path,
            data_manifest=None,
            repository_root=repository_root,
            task=args.task,
            expected_delay=None,
            require_held_out=False,
            expected_heldout_amplitude_scale=None,
            current_revisions=current_revisions,
        )
        selection_manifest_path = real_input_paths["selection_manifest"]
        selection_manifest = _load_json(selection_manifest_path)
        binding.update(
            validate_real_selection_binding(
                selection_manifest,
                selection_manifest_path=selection_manifest_path,
                validation_name=args.validation_name,
                data_path=data_path,
                data_payload_sha256=tensor_payload_sha256(data),
                data_target_sha256=target_tensor_sha256(data),
                fit_manifest=fit_manifest,
            )
        )

    if actuator_profile == "stock":
        encoder_bias_reference = zero_a1_encoder_bias_reference()
        encoder_bias_profile = "zero_stock_reference"
    elif actuator_profile == "identified":
        encoder_bias_reference = dict(mean.encoder_bias)
        encoder_bias_profile = "fitted_mean_reference"
    else:
        encoder_bias_reference = dict(mean.encoder_bias)
        encoder_bias_profile = "effective_mean_reference"
    encoder_bias_evidence = encoder_bias_reference_evidence(
        encoder_bias_reference, encoder_bias_profile
    )

    if actuator_profile == "stock":
        inject_a1_stock_actuator_before_make(env_cfg)
    else:
        inject_a1_actuator_before_make(env_cfg, make_a1_pace_actuator_cfg(mean))
    effective_actuator = env_cfg.scene.robot.actuators["base_legs"]
    if actuator_profile == "stock":
        _require(
            effective_actuator.class_type is not PaceDCMotor,
            "stock replay unexpectedly uses PaceDCMotor",
        )
    elif actuator_profile == "identified":
        _require(
            effective_actuator.class_type is PaceDCMotor,
            "identified replay does not use PaceDCMotor",
        )
    configured_delay = getattr(effective_actuator, "max_delay", None)
    effective_actuator_control = {
        "class_type": (
            f"{effective_actuator.class_type.__module__}."
            f"{effective_actuator.class_type.__qualname__}"
        ),
        "saturation_effort_nm": float(effective_actuator.saturation_effort),
        "effort_limit_nm": float(effective_actuator.effort_limit),
        "velocity_limit_rad_s": float(effective_actuator.velocity_limit),
        "stiffness_values": numeric_values(effective_actuator.stiffness),
        "damping_values": numeric_values(effective_actuator.damping),
        "configured_delay_steps": (
            int(configured_delay) if configured_delay is not None else None
        ),
    }
    env = gym.make(args.task, cfg=env_cfg)
    try:
        physical_samples = replay_absolute_targets(env, data, encoder_bias_reference)
    finally:
        env.close()
    _require(
        _sha256_path(data_path) == data_sha256, "replay data changed during simulation"
    )
    _require(
        _sha256_path(mean_path) == mean_sha256, "replay mean changed during simulation"
    )
    _require(
        all(
            _sha256_path(Path(path)) == digest
            for path, digest in provenance_hashes.items()
        ),
        "replay provenance changed during simulation",
    )
    _require(
        _sha256_path(replay_source) == replay_source_sha256,
        "replay source changed during simulation",
    )
    if fitted_checkout_state is not None:
        _validate_frozen_checkout_state(checkout_roots, fitted_checkout_state)
    metrics = compute_encoder_replay_metrics(
        physical_samples, data, encoder_bias_reference
    )
    decision: dict[str, Any]
    if args.mode in REAL_DIAGNOSTIC_PROFILES:
        decision = {
            "status": "METRICS_ONLY",
            "accepted": None,
            "failures": [],
        }
    else:
        _require(
            args.mode in SYNTHETIC_GATE_LIMITS,
            "synthetic replay mode has no frozen acceptance profile",
        )
        limits = SYNTHETIC_GATE_LIMITS[args.mode]
        threshold_decision = evaluate_replay_thresholds(
            metrics,
            max_rmse=limits["max_rmse"],
            max_p95=limits["max_p95"],
            max_joint_rmse=limits["max_joint_rmse"],
            max_joint_p95=limits["max_joint_p95"],
            max_joint_abs=limits["max_joint_abs"],
        )
        decision = (
            interpret_negative_control_decision(threshold_decision)
            if args.mode.startswith("synthetic-negative-")
            else threshold_decision
        )
    report = {
        "schema_version": "a1_pace_replay/v2",
        "mode": args.mode,
        "actuator_profile": actuator_profile,
        "actuator_control": effective_actuator_control,
        "encoder_bias_profile": encoder_bias_profile,
        "encoder_bias_reference": encoder_bias_evidence,
        "task": args.task,
        "expected_delay_steps": args.expected_delay,
        "recovered_delay_steps": mean.delay_steps,
        "source_truth_delay_steps": (
            source_mean.delay_steps if args.mode in truth_control_modes else None
        ),
        "data_sha256": data_sha256,
        "mean_sha256": mean_sha256,
        "mean_role": (
            "source_truth_for_deterministic_negative_control"
            if negative_control is not None
            else (
                "provenance_only_stock_baseline"
                if actuator_profile == "stock"
                else "simulated_parameters"
            )
        ),
        "negative_control": negative_control,
        "replay_source_sha256": replay_source_sha256,
        "binding": binding,
        "decision": decision,
        "exact_command": shlex.join(sys.argv),
        **metrics,
        "status": decision["status"],
    }
    _atomic_json(report, output_path)
    enforce_replay_decision(args.mode, decision)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay one explicitly named final A1 PACE mean"
    )
    parser.add_argument(
        "--mode",
        choices=(
            "synthetic-heldout-half",
            "synthetic-heldout-full",
            "synthetic-known-positive",
            "synthetic-negative-dynamics",
            "synthetic-negative-delay",
            "synthetic-endpoint",
            "real-stock-diagnostic",
            "real-identified-diagnostic",
        ),
        required=True,
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mean", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument(
        "--validation-name", choices=tuple(sorted(REAL_VALIDATION_NAMES))
    )
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--data-known-mean", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-delay", type=int)
    parser.add_argument(
        "--repository-root", type=Path, default=Path(__file__).resolve().parents[2]
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    output_path = Path(args.output).absolute()
    _require(
        not os.path.lexists(output_path),
        f"replay output already exists: {output_path}",
    )
    args.output = output_path
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        with _exclusive_output_lock(Path(args.output)):
            report = replay(args)
        print(json.dumps(report, sort_keys=True))
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
