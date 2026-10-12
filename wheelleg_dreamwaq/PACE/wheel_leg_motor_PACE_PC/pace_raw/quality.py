"""Shared quality gate for every fitting and formal-export entry point."""
from __future__ import annotations

import dataclasses
import numpy as np

from .session_metadata import metadata_errors


def quality_report(value):
    decoded = getattr(value, "decoded", value)
    errors = [f"{issue.code}: {issue.message}" for issue in decoded.issues]
    header, footer = decoded.header, decoded.footer
    if header is None:
        errors.append("missing session header")
    if footer is None:
        errors.append("missing session footer")
    elif footer.overflow or footer.dropped_frames or not footer.statistics_complete or footer.stop_reason not in (0, 1):
        errors.append("footer reports incomplete/faulted capture")
    version = header.header.version if header else None
    if version != 2:
        errors.append("legacy V1 is available for diagnosis, not verified fitting")
    errors.extend(metadata_errors(decoded))
    if any(sample.safety_flags or sample.can_error_flags for sample in decoded.samples):
        errors.append("samples report a safety/CAN fault")
    if hasattr(value, "manifest") and isinstance(decoded.metadata, dict):
        snapshot = decoded.metadata.get("calibration_snapshot", {})
        normalized = [dataclasses.asdict(motor) for motor in value.manifest.motors]
        if (not isinstance(snapshot, dict) or normalized != snapshot.get("motors") or value.manifest.version != snapshot.get("version")
                or value.manifest.manifest_hash != snapshot.get("manifest_hash")):
            errors.append("normalized calibration differs from immutable capture snapshot")
    periods = np.diff([sample.extended_time_us for sample in decoded.samples if sample.extended_time_us is not None])
    period = 1e6 / header.sample_rate_hz if header and header.sample_rate_hz > 0 else None
    age_values = [block.rx_age_us for sample in decoded.samples for block in (*sample.dm, *sample.lk)]
    report = {
        "protocol_version": version,
        "fit_quality_passed": not errors,
        "errors": errors,
        "sample_count": len(decoded.samples),
        "sample_fit_eligible_count": len(decoded.fit_samples),
        "sample_interval_us": {
            "expected": period,
            "minimum": float(np.min(periods)) if periods.size else None,
            "maximum": float(np.max(periods)) if periods.size else None,
            "max_jitter": float(np.max(np.abs(periods - period))) if periods.size and period else None,
        },
        "feedback_age_us_max": max(age_values, default=0),
        "saturated_sample_count": sum(bool(sample.saturation_mask) for sample in decoded.samples),
        "fault_sample_count": sum(bool(sample.safety_flags or sample.can_error_flags) for sample in decoded.samples),
        "status_count": len(decoded.statuses),
        "stop_reason": footer.stop_reason if footer else None,
        "metadata_present": decoded.metadata is not None,
        "segments": [
            {"segment_id": stage.segment_id, "config_seq": stage.config_seq, "stage_id": stage.stage_id,
             "role": stage.experiment_role, "group_mask": stage.group_mask,
             "action_update_rate_hz": stage.action_update_rate_hz, "hold_method": stage.command_hold_method}
            for stage in decoded.stage_configs.values()
        ],
    }
    if header and header.session_type == 2:
        metadata = decoded.metadata or {}
        calibration = metadata.get("calibration_snapshot", {})
        if not isinstance(calibration, dict) or not calibration.get("zero_calibration_verified"):
            report["errors"].append("final model requires verified zero calibration")
        if not metadata.get("firmware_binary_sha256"):
            report["errors"].append("final model requires firmware binary identity")
        report["fit_quality_passed"] = not report["errors"]
    return report


def require_fit_quality(value):
    report = quality_report(value)
    if not report["fit_quality_passed"]:
        raise ValueError(f"raw session is not fit-eligible: {report['errors']}")
    return report
