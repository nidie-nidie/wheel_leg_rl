from __future__ import annotations

import argparse
from pathlib import Path


WRAPPER_HEAD = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A1 fixed-mean replay cross-validation</title>
  <style>
    :root {
      --foreground: #111827;
      --muted-foreground: #667085;
      --border: #d0d5dd;
      --ring: #2563eb;
      --viz-series-1: #2563eb;
      --viz-series-2: #dc6803;
      --viz-series-3: #16a34a;
    }
    body {
      margin: 24px;
      font-family: Arial, Helvetica, sans-serif;
      background: #ffffff;
      color: #111827;
    }
    h1 {
      margin: 0 0 16px;
      font-size: 22px;
      font-weight: 650;
    }
    .viz-controls {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }
    .form-label {
      display: inline-flex;
      gap: 6px;
      align-items: center;
    }
    .form-select {
      padding: 4px 8px;
    }
    .viz-row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }
    .viz-badge {
      font-size: 12px;
      color: #344054;
    }
    .text-small {
      font-size: 11px;
    }
  </style>
</head>
<body>
<h1>A1 fixed-mean replay cross-validation</h1>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fragment = args.fragment.read_text(encoding="utf-8")
    args.output.write_text(
        WRAPPER_HEAD + fragment + "\n</body>\n</html>\n",
        encoding="utf-8",
        newline="\n",
    )
    print(args.output)
    print(args.output.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
