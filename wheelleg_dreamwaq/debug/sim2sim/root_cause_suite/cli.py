from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys

from .contracts import (
    ConfigurationError,
    EvidenceIntegrityError,
    IdentityDriftError,
    StageExecutionError,
    SUITE_ROOT,
    validate_design_identity,
)


@dataclass(frozen=True)
class CliExecution:
    return_code: int
    run_identity_accepted: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WheelLeg Sim2Sim root-cause suite")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="execute the frozen Core V1 pipeline")
    run.add_argument("--resume", metavar="RUN_ID")
    verify = subparsers.add_parser("verify", help="verify one completed run")
    verify.add_argument("run_id")
    report = subparsers.add_parser("report", help="rebuild reports from verified analysis")
    report.add_argument("run_id")
    return parser


def execute(argv: list[str] | None = None) -> CliExecution:
    args = _parser().parse_args(argv)
    orchestrator = None

    def result(return_code: int) -> CliExecution:
        return CliExecution(
            return_code=return_code,
            run_identity_accepted=bool(
                orchestrator is not None and orchestrator.run_identity_accepted
            ),
        )

    try:
        validate_design_identity()
        from .orchestrator import RootCauseOrchestrator

        orchestrator = RootCauseOrchestrator(SUITE_ROOT)
        if args.command == "run":
            run_id = args.resume or os.environ.get("WHEELLEG_ROOT_CAUSE_RUN_ID")
            orchestrator.run(run_id=run_id, resume=args.resume is not None)
        elif args.command == "verify":
            orchestrator.verify(args.run_id)
        else:
            orchestrator.report(args.run_id)
        return result(0)
    except IdentityDriftError as error:
        print(str(error), file=sys.stderr)
        return CliExecution(return_code=3, run_identity_accepted=False)
    except EvidenceIntegrityError as error:
        print(str(error), file=sys.stderr)
        return result(5)
    except StageExecutionError as error:
        print(str(error), file=sys.stderr)
        return result(4)
    except (ConfigurationError, ValueError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return result(2)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return result(4)


def main(argv: list[str] | None = None) -> int:
    return execute(argv).return_code
