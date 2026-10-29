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


from pace_raw.quality import require_fit_quality

_require_fit_quality = require_fit_quality


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
