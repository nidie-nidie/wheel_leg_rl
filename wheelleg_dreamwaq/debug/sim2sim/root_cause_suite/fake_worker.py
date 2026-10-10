from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from .bootstrap_runtime import finalize_bootstrap
from .contracts import canonical_json_bytes, sha256_file, stable_hash
from .trace_contract import FieldSpec, write_trace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--fast-exit", action="store_true")
    parser.add_argument("--replay-source", action="store_true")
    parser.add_argument("--terminal-adapter-bug", action="store_true")
    parser.add_argument("--terminal-adapter-marker", type=Path)
    parser.add_argument("--repeatability-unavailable", action="store_true")
    return parser


def _write_fake_replay_source(output: Path) -> dict[str, object]:
    source = output / "replay_source"
    source.mkdir()
    actions = np.arange(12, dtype=np.float32).reshape(2, 6) / 100.0
    action_path = source / "actions.npz"
    np.savez(
        action_path,
        action_sequence=actions,
        environment_action_sequence=actions[:, None, :],
    )
    trace = write_trace(
        source / "trace",
        {
            "time_s": np.asarray([0.0, 0.02], dtype=np.float64),
            "value": np.asarray([[0.0], [1.0]], dtype=np.float64),
        },
        {
            "time_s": FieldSpec("s", "simulation", "pre_step", "time"),
            "value": FieldSpec("1", "canonical", "pre_step", "fake_replay"),
        },
    )
    replay_identity = {"fixture": "fake_worker", "horizon": 2, "environment_count": 1}
    source_payload = {
        "schema_version": "RootCauseIsaacReplaySourceV1",
        "replay_source_identity": replay_identity,
        "replay_source_identity_hash": stable_hash(replay_identity),
        "action_sequence_file": action_path.name,
        "clipped_action_file_sha256": sha256_file(action_path),
        "clipped_action_sequence_sha256": hashlib.sha256(
            np.ascontiguousarray(actions).tobytes()
        ).hexdigest().upper(),
        "action_count": int(actions.shape[0]),
        "first_action": actions[0].tolist(),
        "source_trace_sha256": trace.trace_sha256,
        "trace_metadata_sha256": trace.metadata_sha256,
    }
    source_result = source / "result.json"
    source_result.write_bytes(canonical_json_bytes(source_payload) + b"\n")
    return {
        "relative_path": source.relative_to(output).as_posix(),
        "result_sha256": sha256_file(source_result),
        "actions_file": action_path.name,
        "actions_sha256": sha256_file(action_path),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.fail:
        print(f"intentional fake failure: {args.stage}", file=sys.stderr)
        return 7
    result: dict[str, object] = {"stage": args.stage, "status": "ok"}
    terminal_adapter_bug = bool(
        args.terminal_adapter_bug
        or (
            args.terminal_adapter_marker is not None
            and args.terminal_adapter_marker.is_file()
        )
    )
    if terminal_adapter_bug:
        if args.stage != "G02_adapter":
            raise ValueError("Terminal adapter fixture requires G02_adapter")
        result.update(
            {
                "schema_version": "RootCauseCoreStageResultV1",
                "passed": True,
                "terminal_adapter_bug": True,
                "diagnostic_gate_passed": False,
                "adapter_gate": {
                    "schema_version": "RootCauseAdapterGateV1",
                    "passed": False,
                    "evidence_valid": True,
                    "digital_chain_passed": False,
                    "terminal_adapter_bug": True,
                    "classification": "SIM2SIM_ADAPTER_BUG/adapter_chain",
                },
            }
        )
    if args.repeatability_unavailable:
        if args.stage != "G01_repeatability":
            raise ValueError("Repeatability fixture requires G01_repeatability")
        result.update(
            {
                "schema_version": "RootCauseCoreStageResultV1",
                "passed": True,
                "repeatability_usable": False,
                "exact_executed_coverage": True,
                "unusable_reasons": ["nondeterministic:fixture"],
            }
        )
    write_trace(
        args.output / "trace",
        {
            "time_s": np.asarray([0.0, 0.005], dtype=np.float64),
            "value": np.asarray([[0.0], [1.0]], dtype=np.float64),
        },
        {
            "time_s": FieldSpec("s", "simulation", "post_step", "scalar"),
            "value": FieldSpec("1", "canonical", "post_step", "fake"),
        },
    )
    if args.replay_source:
        result["replay_source"] = _write_fake_replay_source(args.output)
    (args.output / "result.json").write_text(
        json.dumps(result, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if args.fast_exit:
        finalize_bootstrap(0)
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
