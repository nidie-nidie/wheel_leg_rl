from __future__ import annotations

import os
from pathlib import Path
import sys

from .contracts import (
    BootstrapError,
    SUITE_ROOT,
    canonical_json_bytes,
    require_bootstrap,
    require_path_within,
)
from .python_write_guard import PythonWriteGuard


def main() -> int:
    try:
        require_bootstrap()
    except BootstrapError as error:
        print(str(error), file=sys.stderr)
        return 2
    run_root_value = os.environ.get("WHEELLEG_ROOT_CAUSE_RUN_ROOT")
    if not run_root_value:
        print("WHEELLEG_ROOT_CAUSE_RUN_ROOT is missing", file=sys.stderr)
        return 2
    try:
        run_root = require_path_within(
            Path(run_root_value).resolve(strict=True), SUITE_ROOT, label="parent run root"
        )
        prefix = os.environ.get("PYTHONPYCACHEPREFIX")
        if not prefix:
            raise BootstrapError("PYTHONPYCACHEPREFIX is missing")
        require_path_within(prefix, run_root, label="parent pycache prefix")
    except (BootstrapError, ValueError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return 2

    guard = PythonWriteGuard(run_root)
    with guard:
        from .cli import execute as cli_execute

        execution = cli_execute()
        if execution.run_identity_accepted:
            payload = {
                "schema_version": "RootCauseParentBootstrapV1",
                "argv": sys.argv,
                "run_root": str(run_root),
                "return_code": execution.return_code,
                "write_guard": {
                    "installed": guard.installed,
                    "probe_count": guard.probe_count,
                    "pre_handlers": guard.pre_handlers,
                    "created_lifetime_handlers": guard.created_lifetime_handlers,
                    "post_handlers": guard.post_handlers(),
                    "capability_manifest": guard.capability_manifest,
                    "ledger": guard.ledger,
                    "ledger_cutoff": "immediately_before_parent_identity_write",
                },
            }
            (run_root / "parent_guard.json").write_bytes(
                canonical_json_bytes(payload) + b"\n"
            )
        return execution.return_code


if __name__ == "__main__":
    raise SystemExit(main())
