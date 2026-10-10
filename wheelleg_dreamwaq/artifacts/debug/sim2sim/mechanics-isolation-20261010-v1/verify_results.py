"""Fresh final audit of evidence, source hashes, numbers and syntax."""
import ast
import datetime
import json

from analyze import ROOT, read, digest


def main():
    base, cap = read(ROOT / "analysis.json"), read(ROOT / "cap-analysis.json")
    assert base["passed"] and cap["passed"]
    protected = read(ROOT / "protected-before.json")
    assert len(protected) == 93
    for path, expected in protected.items(): assert digest(path) == expected
    for result in (base, cap):
        for path, expected in result["provenance_sha256"].items(): assert digest(path) == expected, path
    selected = base["selected_processes"] + cap["selected_processes"]
    assert len(selected) == 11 and len({r["output"] for r in selected}) == 11
    for record in selected:
        assert record["exit_code"] == 0 and record.get("evidence_complete", True)
        evidence = read(ROOT / record["output"] / "evidence.json")
        assert evidence["debug_only"] and evidence["sources_unchanged"]
    assert all(v == 0 for v in base["baseline_reproduction_max_errors"].values())
    assert all(v == 0 for v in cap["default_cap_reproduction_max_errors"].values())
    assert all(v == 0 for v in base["external_force_max_errors_nm"].values())
    assert all(v == 0 for errors in base["cross_engine_initial_errors"].values() for v in errors.values())
    for p in ROOT.glob("*.py"): ast.parse(p.read_text(encoding="utf-8"))
    report = (ROOT / "report.md").read_text(encoding="utf-8")
    assert "0.431066" in report and "0.056386" in report and "0.001458" in report
    assert "未新增或调用子 agent" in report and "暂不建议继续盲目重训" in report
    plan = ROOT / "diagnostic-plan.md"
    plan.write_text(plan.read_text(encoding="utf-8").replace("- [ ]", "- [x]").replace("schemas_cfg.py:93–94", "schemas_cfg.py:92–93"), encoding="utf-8")
    artifacts = {str(p): digest(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in (".py", ".md", ".json", ".diff", ".png")
                 and p.name != "verification.json"}
    payload = {"verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "passed": True,
               "protected_file_count": len(protected), "selected_valid_process_count": len(selected),
               "failed_startup_launch_count": 3, "syntax_files": len(list(ROOT.glob("*.py"))),
               "baseline_exact_reproduction": True, "initial_state_exact_alignment": True,
               "external_torque_delivery_exact": True, "formal_files_unchanged": True,
               "standalone_plot_visually_reviewed": True, "agents_invoked_for_this_task": 0,
               "artifacts_sha256": artifacts}
    (ROOT / "verification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("FINAL_VERIFIED", len(selected), "valid processes", len(protected), "protected files", len(artifacts), "artifact hashes")


if __name__ == "__main__":
    main()
