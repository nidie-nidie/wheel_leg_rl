from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from recovery.contracts import PROJECT_ROOT, file_sha256, source_sha256, write_json


MUJOCO_FILES = ("test_geometry_guard.py", "test_mujoco_recovery.py", "test_independent_mujoco.py")


def test_inventory():
    directory = PROJECT_ROOT / "training/recovery/tests"
    files = sorted(path.name for path in directory.glob("test_*.py"))
    return {"main": [name for name in files if name not in MUJOCO_FILES], "mujoco": list(MUJOCO_FILES)}


def verification_commands(directory):
    main = [str(PROJECT_ROOT / ".venv/Scripts/python.exe"), "-m", "pytest", "training/recovery/tests",
            *(f"--ignore=training/recovery/tests/{name}" for name in MUJOCO_FILES), "-q",
            f"--junitxml={directory / 'main.xml'}"]
    main[0] = sys.executable
    mujoco = [str(PROJECT_ROOT / "sim2sim/mujoco/.venv/Scripts/python.exe"), "-m", "pytest",
              *(f"training/recovery/tests/{name}" for name in MUJOCO_FILES), "-q",
              f"--junitxml={directory / 'mujoco.xml'}"]
    return {"main": main, "mujoco": mujoco}


def junit_statistics(path, selected_files):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    modules = sorted({component for case in cases for component in case.get("classname", "").split(".")
                      if component.startswith("test_")})
    counts = {"tests": len(cases), "failures": sum(len(case.findall("failure")) for case in cases),
              "errors": sum(len(case.findall("error")) for case in cases),
              "skipped": sum(len(case.findall("skipped")) for case in cases)}
    suites = list(root.iter("testsuite"))
    declared = {key: sum(int(suite.get(key, "0")) for suite in suites) for key in counts}
    # 同时核验声明计数和实际testcase，防止空报告或收集阶段错误被当成通过。
    required = {Path(name).stem for name in selected_files}
    if (not cases or declared["tests"] < counts["tests"]
            or any(counts[field] != declared[field] for field in ("failures", "errors", "skipped"))
            or counts["failures"] or counts["errors"] or counts["skipped"]
            or not required <= set(modules)):
        raise ValueError(f"单元验收JUnit未完整通过: {path}; actual={counts}; declared={declared}; modules={modules}")
    # pytest-subtests把通过的子测试计入tests属性，但不另建testcase元素。
    return {**declared, "testcase_elements": counts["tests"],
            "passed_subtests": declared["tests"] - counts["tests"],
            "modules": modules, "selected_test_files": list(selected_files)}


def reference(path):
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def main():
    parser = argparse.ArgumentParser(description="运行两个真实Python环境的完整单元验收并记录JUnit证据")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = source_sha256()
    inventory = test_inventory()
    report = {"schema_version": "RecoveryUnitVerificationV1", "source_sha256": before,
              "created_at": datetime.now(timezone.utc).isoformat(), "all_tests_verified": False,
              "passed": False, "runs": {}, "suites": []}
    for name, command in verification_commands(output).items():
        stdout, stderr = output / f"{name}.stdout.log", output / f"{name}.stderr.log"
        with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
            process = subprocess.run(command, cwd=PROJECT_ROOT, stdout=out, stderr=err)
        run = {"command": command, "returncode": process.returncode,
               "stdout": reference(stdout), "stderr": reference(stderr), "passed": False}
        xml = output / f"{name}.xml"
        if xml.is_file():
            run["xml"] = reference(xml)
        try:
            run["statistics"] = junit_statistics(xml, inventory[name])
            run["passed"] = process.returncode == 0
        except (OSError, ValueError, ET.ParseError) as error:
            run["error"] = str(error)
        report["runs"][name] = run
        if "statistics" in run:
            report["suites"].append({"name": name, **run["statistics"]})
    report["source_after_sha256"] = source_sha256()
    report["all_tests_verified"] = all(run["passed"] for run in report["runs"].values())
    report["passed"] = report["all_tests_verified"] and before == report["source_after_sha256"]
    destination = output / "verification.json"
    write_json(destination, report)
    print(json.dumps({"passed": report["passed"], "path": str(destination), "sha256": file_sha256(destination)}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
