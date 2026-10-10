"""Final read-only audit of source identities and all recorded evaluations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
sys.path.insert(0, str(PROJECT / "sim2sim/mujoco"))
from wheelleg_mujoco.evaluation import build_evaluation_source_fingerprint, validate_evaluation_reports_for_ranking


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    analysis = read(ROOT / "analysis.json")
    assert analysis["passed"] and analysis["tests_passed"] == 40
    for path, expected in analysis["provenance_sha256"].items():
        assert digest(path) == expected, path
    assert analysis["evaluation_implementation"] == build_evaluation_source_fingerprint()
    reports = [read(ROOT / f"mujoco-evaluation/run-{i:02d}/summary.json") for i in range(1, 5)]
    validate_evaluation_reports_for_ranking(reports)
    assert all(r["aggregate"]["completed_scenarios"] == 8 for r in reports)
    assert all(r["golden_max_abs_error"] == 0. for r in analysis["policies"])
    assert analysis["initial_state_max_errors"] == {"baseline": 0., "closure_off": 0.}
    assert analysis["candidate_reproduction_max_errors"] == {"baseline": 0., "closure_off": 0.}
    assert digest(ROOT / "physx-5.6.1-forwardDynamic2.cu") == "FD19446E148E29F74BF937A44C8032860658E60640AD0925EAF480751388B2B8"
    plan = (PROJECT / "docs/superpowers/plans/2026-10-10-mujoco-angular-limit-alignment.md").read_text(encoding="utf-8")
    assert "- [ ]" not in plan
    report = (ROOT / "report.md").read_text(encoding="utf-8")
    assert analysis["architecture_sha256"] in report
    assert "32/32" in report and "站立仍有漂移" in report
    artifacts = {str(p): digest(p) for p in sorted(ROOT.glob("*.py"))}
    for name in ("analysis.json", "aligned-execution.json", "candidate-analysis.json", "candidate-execution.json",
                 "mujoco-candidate-execution.json", "mujoco-ranking-summary.json", "production-changes.diff",
                 "report.md", "physx-5.6.1-forwardDynamic2.cu", "protected-before.json"):
        p = ROOT / name
        artifacts[str(p)] = digest(p)
    result = {"schema_version": "AngularAlignmentVerificationV1", "passed": True,
              "verified_at": datetime.now(timezone.utc).isoformat(), "tests_passed": 40,
              "policy_scenarios_completed": 32, "golden_max_abs_errors": [0., 0., 0., 0.],
              "immutable_old_artifacts": analysis["immutable_input_count"], "architecture_sha256": analysis["architecture_sha256"],
              "analysis_provenance_files_verified": len(analysis["provenance_sha256"]), "artifacts_sha256": artifacts}
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL_VERIFIED", result["tests_passed"], "tests", result["policy_scenarios_completed"], "scenarios",
          result["immutable_old_artifacts"], "immutable artifacts", "golden=0", flush=True)


if __name__ == "__main__":
    main()
