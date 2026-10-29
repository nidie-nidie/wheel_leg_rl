"""Immutable capture-time calibration and source binding for a raw recording."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import pathlib

from .manifest import MotorManifest, MotorManifestEntry

METADATA_VERSION = "wheel_leg_motor_pace/session/v2"


def content_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sidecar_path(raw_path):
    return pathlib.Path(raw_path).with_suffix(".session_manifest.json")


def quality_path(raw_path):
    return pathlib.Path(raw_path).with_suffix(".quality.json")


def manifest_snapshot(manifest):
    return {
        "version": manifest.version,
        "manifest_hash": manifest.manifest_hash,
        "direction_source": "user_confirmed_existing_code",
        "zero_calibration_verified": all(motor.calibration_verified for motor in manifest.motors),
        "motors": [dataclasses.asdict(motor) for motor in manifest.motors],
    }


def manifest_from_snapshot(snapshot):
    motors = [MotorManifestEntry(**entry) for entry in snapshot["motors"]]
    result = MotorManifest(int(snapshot["version"]), int(snapshot["manifest_hash"]), motors, pathlib.Path("<capture-snapshot>"))
    result.validate()
    return result


def _wire_values(frame):
    if frame is None:
        return None
    values = dataclasses.asdict(frame)
    values.pop("raw", None)
    def stable(value):
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        if isinstance(value, dict):
            return {key: stable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [stable(item) for item in value]
        return value
    return stable(values)


def _verified_binary(plan):
    declaration = (plan or {}).get("firmware_binary")
    if not declaration:
        return None, None
    record = {"verified_at_capture": False}
    if not isinstance(declaration, dict):
        record["error"] = "firmware_binary declaration must include path and sha256"
        return None, record
    record.update(path=declaration.get("path"), declared_sha256=declaration.get("sha256"))
    declared = declaration.get("sha256")
    if not isinstance(declared, str) or re.fullmatch(r"[0-9a-fA-F]{64}", declared) is None:
        record["error"] = "declared firmware binary SHA-256 is invalid"
        return None, record
    try:
        binary = pathlib.Path(declaration["path"])
        actual = hashlib.sha256(binary.read_bytes()).hexdigest()
    except (KeyError, TypeError, OSError) as error:
        record["error"] = f"firmware binary could not be verified: {error}"
        return None, record
    record.update(path=str(binary.resolve()), sha256=actual)
    if actual != declared.lower():
        record["error"] = "firmware binary differs from planned SHA-256"
        return None, record
    record["verified_at_capture"] = True
    return actual, record


def build_session_metadata(decoded, snapshot, *, capture_state="complete", plan=None, firmware_sources=None):
    binary_sha, binary_record = _verified_binary(plan)
    result = {
        "metadata_version": METADATA_VERSION,
        "raw_sha256": hashlib.sha256(decoded.raw_stream).hexdigest(),
        "capture_state": capture_state,
        "fixture": "mounted_body_fixed_wheels_suspended",
        "calibration_snapshot": snapshot,
        "calibration_sha256": content_hash(snapshot),
        "firmware_sources": firmware_sources or {},
        "firmware_binary_sha256": binary_sha,
        "firmware_binary": binary_record,
        "expected": plan or {},
        "online": {
            "header": _wire_values(decoded.header),
            "stages": [_wire_values(stage) for stage in decoded.stage_configs.values()],
            "footer": _wire_values(decoded.footer),
        },
        "timestamp_semantics": "TIM5 HAL callback service time; CAN Tx confirmation does not prove motor execution",
    }
    result["metadata_sha256"] = content_hash(result)
    return result


def load_session_metadata(raw_path):
    path = sidecar_path(raw_path)
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise ValueError(f"invalid session sidecar {path}: {error}") from error
    return result


def write_session_metadata(raw_path, metadata):
    path = sidecar_path(raw_path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path


def metadata_errors(decoded):
    metadata = decoded.metadata
    if not isinstance(metadata, dict):
        return ["missing capture-time session metadata"]
    errors = []
    if metadata.get("metadata_version") != METADATA_VERSION:
        errors.append("unsupported session metadata version")
    body = dict(metadata)
    stated = body.pop("metadata_sha256", None)
    try:
        matching_hash = stated == content_hash(body)
    except (TypeError, ValueError):
        matching_hash = False
    if not matching_hash:
        errors.append("session metadata content hash mismatch")
    if metadata.get("raw_sha256") != hashlib.sha256(decoded.raw_stream).hexdigest():
        errors.append("raw SHA-256 differs from immutable sidecar")
    snapshot = metadata.get("calibration_snapshot")
    if not isinstance(snapshot, dict):
        errors.append("missing calibration snapshot")
    else:
        try:
            snapshot_hash = content_hash(snapshot)
        except (TypeError, ValueError):
            snapshot_hash = None
        if snapshot_hash is None or metadata.get("calibration_sha256") != snapshot_hash:
            errors.append("calibration snapshot content hash mismatch")
        try:
            manifest = manifest_from_snapshot(snapshot)
        except (KeyError, TypeError, ValueError):
            errors.append("invalid calibration snapshot")
        else:
            if decoded.header and (manifest.version != decoded.header.motor_order_version or manifest.manifest_hash != decoded.header.scale_manifest_hash):
                errors.append("capture-time manifest differs from wire declaration")
    online = {"header": _wire_values(decoded.header), "stages": [_wire_values(stage) for stage in decoded.stage_configs.values()], "footer": _wire_values(decoded.footer)}
    if metadata.get("online") != json.loads(json.dumps(online)):
        errors.append("sidecar online values differ from raw")
    if metadata.get("capture_state") != "complete":
        errors.append("capture did not finish normally")
    binary = metadata.get("firmware_binary")
    binary_sha = metadata.get("firmware_binary_sha256")
    if binary and not isinstance(binary, dict):
        errors.append("firmware binary verification record is invalid")
    elif binary:
        if binary.get("error"):
            errors.append(str(binary["error"]))
        if binary_sha and (not binary.get("verified_at_capture") or binary.get("sha256") != binary_sha
                           or str(binary.get("declared_sha256", "")).lower() != binary_sha):
            errors.append("firmware binary identity was not verified at capture")
    elif binary_sha:
        errors.append("firmware binary identity has no capture verification record")
    if binary_sha is not None and (not isinstance(binary_sha, str) or re.fullmatch(r"[0-9a-f]{64}", binary_sha) is None):
        errors.append("firmware binary SHA-256 is invalid")
    expected = metadata.get("expected", {})
    if isinstance(expected, dict) and "online" in expected and expected["online"] != metadata.get("online"):
        errors.append("planned online contract differs from captured declaration")
    if isinstance(expected, dict):
        if expected.get("session_type") and decoded.header:
            named_type = {"commissioning": 1, "final": 2}.get(expected["session_type"], expected["session_type"])
            if named_type != decoded.header.session_type:
                errors.append("planned session type differs from wire declaration")
        if expected.get("fixture") and expected["fixture"] != metadata.get("fixture"):
            errors.append("planned fixture differs from capture declaration")
        planned_manifest = expected.get("motor_manifest")
        if isinstance(planned_manifest, dict) and isinstance(snapshot, dict):
            if any(planned_manifest.get(key) != snapshot.get(key) for key in ("version", "manifest_hash", "motors")):
                errors.append("planned calibration differs from capture snapshot")
        contract = expected.get("collection_contract", {})
        if isinstance(contract, dict):
            if decoded.header:
                for key in ("sample_rate_hz", "control_rate_hz"):
                    if key in contract and contract[key] != getattr(decoded.header, key):
                        errors.append(f"planned {key} differs from wire declaration")
            if contract.get("fit_and_independent_validation"):
                families = (("leg_dm", range(4), (2,), (3,)), ("wheel_torque", (4,5), (4,), (4,)), ("wheel_velocity", (4,5), (5,), (5,)))
                for family, indices, fit_stages, validation_stages in families:
                    for index in indices:
                        for role, stages in ((1,fit_stages),(2,validation_stages)):
                            if not any(config.experiment_role==role and config.stage_id in stages and config.group_mask & (1<<index)
                                       for config in decoded.stage_configs.values()):
                                errors.append(f"planned independent FIT/VALIDATION coverage missing for {family} motor {index}")
            for config in decoded.stage_configs.values():
                if "action_rate_hz" in contract and contract["action_rate_hz"] != config.action_update_rate_hz:
                    errors.append("planned action rate differs from stage declaration")
                if "command_hold" in contract and (contract["command_hold"] != "zero_order_hold" or config.command_hold_method != 1):
                    errors.append("planned hold method differs from stage declaration")
                for key, raw_values, quantum in (("dm_kp", config.dm_kp_raw, 500./4095.), ("dm_kd", config.dm_kd_raw, 5./4095.)):
                    if key in contract and any(abs(float(contract[key])-raw*quantum)>1.5*quantum for raw in raw_values):
                        errors.append(f"planned {key} differs from stage controller gains")
    return list(dict.fromkeys(errors))
