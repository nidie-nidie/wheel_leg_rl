from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .compare_isaac_equivalence import (
    FLOAT_FIELDS,
    SHARED_IDENTITY_FIELDS,
    compare_equivalence_runs,
    independent_run_report,
)
from .compare_traces import load_verified_engine_run
from .trace_schema import build_file_hashes, sha256_array, sha256_file, stable_payload_hash, write_json


REPLAY_FLOAT_FIELDS = FLOAT_FIELDS + (
    "target_command_canonical",
    "target_command_engine_native",
)
REPLAY_SHARED_IDENTITY_FIELDS = tuple(
    name for name in SHARED_IDENTITY_FIELDS if name not in {"scenario_variant", "action_sequence_identity"}
) + ("configured_episode_length_s",)


def _identity_report(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    failures: list[str] = []
    for name in REPLAY_SHARED_IDENTITY_FIELDS:
        values = [item.get(name) for item in metadata]
        serialized = [json.dumps(value, sort_keys=True, separators=(",", ":")) for value in values]
        passed = all(value is not None for value in values) and len(set(serialized)) == 1
        checks[name] = {"passed": passed, "values": values if not passed else [values[0]]}
        if not passed:
            failures.append(name)
    return {"checks": checks, "failures": failures, "passed": not failures}


def compare_isaac_replay_runs(
    original_directories: list[Path],
    replay_directories: list[Path],
) -> dict[str, Any]:
    original_loaded = [load_verified_engine_run(path) for path in original_directories]
    replay_loaded = [load_verified_engine_run(path) for path in replay_directories]
    original_metadata = [item["metadata"] for item in original_loaded]
    replay_metadata = [item["metadata"] for item in replay_loaded]
    original_independence = independent_run_report(
        original_directories,
        original_metadata,
        trace_filename="control_trace.npz",
    )
    replay_independence = independent_run_report(
        replay_directories,
        replay_metadata,
        trace_filename="control_trace.npz",
    )
    original_paths = {str(path.resolve()).casefold() for path in original_directories}
    replay_paths = {str(path.resolve()).casefold() for path in replay_directories}
    cross_path_overlap = sorted(original_paths & replay_paths)
    original_ids = {item.get("collection_id") for item in original_metadata}
    replay_ids = {item.get("collection_id") for item in replay_metadata}
    cross_collection_id_overlap = sorted(
        value for value in original_ids & replay_ids if value is not None
    )
    variant_failures = [
        *(f"original[{index}]" for index, item in enumerate(original_metadata) if item.get("scenario_variant") != "closed_loop"),
        *(f"replay[{index}]" for index, item in enumerate(replay_metadata) if item.get("scenario_variant") != "isaac_policy_replay"),
    ]
    observer_failures = [
        *(f"original[{index}]" for index, item in enumerate(original_metadata) if item.get("observer_mode") != "debug_step_subclass"),
        *(f"replay[{index}]" for index, item in enumerate(replay_metadata) if item.get("observer_mode") != "debug_step_subclass"),
    ]

    original_action_hashes_by_trace = {
        sha256_file(Path(path).resolve() / "control_trace.npz"): sha256_array(
            loaded["control"]["action_clipped"]
        )
        for path, loaded in zip(original_directories, original_loaded, strict=True)
    }
    replay_identities = [item.get("action_sequence_identity") for item in replay_metadata]
    replay_identity_serialized = {
        json.dumps(identity, sort_keys=True, separators=(",", ":")) for identity in replay_identities
    }
    replay_identity_failures: list[str] = []
    if len(replay_identity_serialized) != 1:
        replay_identity_failures.append("replay_action_identity_not_constant")
    for index, identity in enumerate(replay_identities):
        if not isinstance(identity, dict) or identity.get("source") != "isaac_policy_replay_npz":
            replay_identity_failures.append(f"replay[{index}].source")
            continue
        source_trace_hash = identity.get("source_control_trace_sha256")
        if source_trace_hash not in original_action_hashes_by_trace:
            replay_identity_failures.append(f"replay[{index}].source_control_trace_sha256")
        elif identity.get("content_sha256") != original_action_hashes_by_trace[source_trace_hash]:
            replay_identity_failures.append(f"replay[{index}].content_sha256")

    report = compare_equivalence_runs(
        [item["control"] for item in original_loaded],
        [item["control"] for item in replay_loaded],
        float_fields=REPLAY_FLOAT_FIELDS,
    )
    identity = _identity_report(original_metadata + replay_metadata)
    report.update(
        {
            "schema_version": "IsaacPolicyReplayEquivalenceV1",
            "identity": identity,
            "independence": {
                "original": original_independence,
                "replay": replay_independence,
                "cross_path_overlap": cross_path_overlap,
                "cross_collection_id_overlap": cross_collection_id_overlap,
            },
            "variant_failures": variant_failures,
            "observer_failures": observer_failures,
            "replay_identity_failures": replay_identity_failures,
        }
    )
    report["passed"] = (
        report["passed"]
        and identity["passed"]
        and original_independence["passed"]
        and replay_independence["passed"]
        and not cross_path_overlap
        and not cross_collection_id_overlap
        and not variant_failures
        and not observer_failures
        and not replay_identity_failures
    )
    report["report_hash"] = stable_payload_hash(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare closed-loop Isaac traces with fresh-reset replay traces.")
    parser.add_argument("--original", type=Path, nargs="+", required=True)
    parser.add_argument("--replay", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    if len(args.original) != len(args.replay):
        parser.error("Original and replay run counts must match")
    if not args.allow_smoke and len(args.original) < 5:
        parser.error("The formal replay gate requires at least five runs per path")
    if args.allow_smoke and len(args.original) < 2:
        parser.error("A smoke replay envelope still requires at least two runs per path")
    report = compare_isaac_replay_runs(args.original, args.replay)
    report["input_artifact_hashes"] = {
        "original": [
            {
                "metadata": sha256_file(path.resolve() / "metadata.json"),
                "control_trace": sha256_file(path.resolve() / "control_trace.npz"),
                "substep_trace": sha256_file(path.resolve() / "substep_trace.npz"),
            }
            for path in args.original
        ],
        "replay": [
            {
                "metadata": sha256_file(path.resolve() / "metadata.json"),
                "control_trace": sha256_file(path.resolve() / "control_trace.npz"),
                "substep_trace": sha256_file(path.resolve() / "substep_trace.npz"),
                "input_action_sequence": sha256_file(path.resolve() / "input_action_sequence.npz"),
            }
            for path in args.replay
        ],
    }
    report["gate_mode"] = "smoke" if args.allow_smoke else "formal"
    report["formal_gate_passed"] = (
        report["passed"] and not args.allow_smoke and len(args.original) >= 5
    )
    report["report_hash"] = stable_payload_hash(
        {name: value for name, value in report.items() if name != "report_hash"}
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", report)
    write_json(
        output / "file_hashes.json",
        build_file_hashes(
            {
                "summary": output / "summary.json",
                "comparison": Path(__file__),
                "equivalence_core": Path(__file__).with_name("compare_isaac_equivalence.py"),
                "trace_schema": Path(__file__).with_name("trace_schema.py"),
            }
        ),
    )
    print(json.dumps({"passed": report["passed"], "failures": report["failures"]}, sort_keys=True))
    gate_passed = report["passed"] if args.allow_smoke else report["formal_gate_passed"]
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
