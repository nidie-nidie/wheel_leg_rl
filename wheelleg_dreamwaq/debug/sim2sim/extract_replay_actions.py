from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .trace_schema import load_npz, save_npz, sha256_file, write_json


def extract_replay_actions(
    control_trace_path: str | Path,
    output_path: str | Path,
    *,
    key: str = "isaac_policy_replay",
) -> dict:
    source = Path(control_trace_path).resolve()
    output = Path(output_path).resolve()
    trace = load_npz(source)
    if "action_clipped" not in trace:
        raise ValueError("Control trace does not contain action_clipped")
    actions = np.asarray(trace["action_clipped"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 6 or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite clipped actions with shape (N, 6), got {actions.shape}")
    save_npz(output, {key: actions})
    metadata = {
        "schema_version": "IsaacPolicyReplayActionsV1",
        "key": key,
        "control_ticks": len(actions),
        "source_control_trace": str(source),
        "source_control_trace_sha256": sha256_file(source),
        "output_sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".json"), metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract clipped Isaac actions for open-loop replay.")
    parser.add_argument("control_trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--key", default="isaac_policy_replay")
    args = parser.parse_args()
    metadata = extract_replay_actions(args.control_trace, args.output, key=args.key)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
