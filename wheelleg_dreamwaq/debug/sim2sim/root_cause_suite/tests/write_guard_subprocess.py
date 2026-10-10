from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from debug.sim2sim.root_cause_suite.python_write_guard import PythonWriteGuard


def main() -> None:
    operation = sys.argv[1]
    run_root = Path(sys.argv[2]).resolve()
    outside = Path(sys.argv[3]).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    outside.parent.mkdir(parents=True, exist_ok=True)

    if operation == "temporary_outside":
        outside.write_bytes(b"keep")

    guard = PythonWriteGuard(run_root)
    result: dict[str, object] = {}
    try:
        with guard:
            if operation == "inside_write":
                (run_root / "inside.txt").write_text("ok", encoding="utf-8")
            elif operation == "outside_write":
                outside.write_text("bad", encoding="utf-8")
            elif operation == "delay_handler":
                logging.FileHandler(outside, delay=True)
            elif operation == "inside_handler":
                handler = logging.FileHandler(run_root / "handler.log", delay=False)
                handler.close()
            elif operation == "temporary_outside":
                flags = os.O_RDONLY | os.O_TEMPORARY
                descriptor = os.open(outside, flags)
                os.close(descriptor)
            elif operation == "nonce":
                result["installed"] = guard.installed
                result["probe_count"] = guard.probe_count
            else:
                raise ValueError(operation)
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
    result["outside_exists"] = outside.exists()
    result["outside_bytes"] = outside.read_bytes().decode("ascii") if outside.exists() else None
    result["created_handlers"] = len(guard.created_lifetime_handlers)
    result["capabilities"] = guard.capability_manifest
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
