from __future__ import annotations

import argparse
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fit.fit_leg_dm import fit_leg_dm
from fit.fit_wheel_lk import fit_wheel_lk
from fit.model_manifest import write_model_manifest
from pace_raw.decoder import decode_file
from pace_raw.normalize import normalize_session


CRITICAL_DECODE_ISSUES = {
    "crc_error",
    "malformed_frame",
    "sequence_gap",
    "sequence_reorder",
    "timestamp_discontinuity",
    "timestamp_backwards",
    "missing_stage_config",
    "stage_mismatch",
    "missing_session_header",
    "missing_footer",
    "invalid_session_footer",
    "footer_frame_count",
    "config_hash_mismatch",
    "can_enqueue_failure",
    "tx_event_loss",
    "can_error",
    "can_state_fault",
    "header_endianness",
    "header_rate",
    "header_sample_size",
    "zero_config_hash",
    "duplicate_header",
    "duplicate_config",
    "duplicate_footer",
    "stage_before_header",
    "invalid_length",
    "truncated_header",
    "truncated_frame",
    "trailing_bytes",
    "resync",
}


def _require_fit_quality(decoded) -> None:
    critical = [issue for issue in decoded.issues if issue.code in CRITICAL_DECODE_ISSUES]
    if critical:
        counts = {}
        for issue in critical:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        raise ValueError(f"raw session is not fit-eligible: {counts}")
    if decoded.footer is None:
        raise ValueError("raw session has no footer")
    if decoded.footer.overflow or decoded.footer.dropped_frames:
        raise ValueError("raw session footer reports dropped data")
    if not decoded.footer.statistics_complete:
        raise ValueError("raw session footer reports incomplete statistics")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the provisional/final six-channel fit pipeline")
    parser.add_argument("raw_session", type=pathlib.Path)
    parser.add_argument("output_manifest", type=pathlib.Path)
    parser.add_argument("--normalized-csv", type=pathlib.Path)
    parser.add_argument("--normalized-npz", type=pathlib.Path)
    parser.add_argument("--max-delay-steps", type=int, default=10)
    args = parser.parse_args()
    decoded = decode_file(args.raw_session)
    _require_fit_quality(decoded)
    dataset = normalize_session(decoded)
    if args.normalized_csv:
        dataset.export_csv(args.normalized_csv)
    if args.normalized_npz:
        dataset.export_npz(args.normalized_npz)
    model = fit_leg_dm(dataset, max_delay_steps=args.max_delay_steps)
    model = fit_wheel_lk(dataset, model=model, max_delay_steps=args.max_delay_steps)
    write_model_manifest(model, args.output_manifest)
    print(
        json.dumps(
            {
                "output_manifest": str(args.output_manifest),
                "model_maturity": model["model_maturity"],
                "canonical_order": model["canonical_order"],
                "aggregate_metrics": model["aggregate_metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
