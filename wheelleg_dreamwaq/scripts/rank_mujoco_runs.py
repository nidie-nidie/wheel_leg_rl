from __future__ import annotations

import argparse
import json
from pathlib import Path

from wheelleg_mujoco.evaluation import build_ranking_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank WheelLeg MuJoCo evaluation summaries.")
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = []
    evaluation_paths = []
    for path in args.evaluation:
        resolved = path.resolve()
        reports.append(json.loads(resolved.read_text(encoding="utf-8")))
        evaluation_paths.append(str(resolved))
    summary = build_ranking_summary(reports, evaluation_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
