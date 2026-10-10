from __future__ import annotations

import argparse
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace_raw.decoder import decode_file
from pace_raw.normalize import normalize_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode and normalize one PACE raw session")
    parser.add_argument("raw_session", type=pathlib.Path)
    parser.add_argument("--csv", type=pathlib.Path)
    parser.add_argument("--npz", type=pathlib.Path)
    parser.add_argument("--pt", type=pathlib.Path)
    parser.add_argument("--summary", type=pathlib.Path)
    args = parser.parse_args()
    decoded = decode_file(args.raw_session)
    dataset = normalize_session(decoded)
    if args.csv:
        dataset.export_csv(args.csv)
    if args.npz:
        dataset.export_npz(args.npz)
    if args.pt:
        dataset.export_pt(args.pt)
    summary = decoded.summary()
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
