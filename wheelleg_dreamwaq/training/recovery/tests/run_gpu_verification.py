from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from recovery.contracts import PROJECT_ROOT, file_sha256, write_json


def main():
    parser = argparse.ArgumentParser(description="真实GPU起身接口与性能验收")
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--interface", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--import-compatible", type=Path)
    parser.add_argument("--import-evidence", type=Path)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(arguments.source_config.read_text(encoding="utf-8"))
    config["environment"].update({"num_envs": arguments.num_envs, "run_dir": str(output / "run"),
        "task_type": "interface_validation" if arguments.interface else "bootstrap_training"})
    config["assessment"]["updates"] = arguments.updates
    path = output / "resolved_config.json"
    write_json(path, config)
    command = [sys.executable, "-u", "-m", "recovery.train", "--config", str(path),
               "--headless", "--device", config["environment"]["device"]]
    if arguments.interface:
        command += ["--verify-interface", "--verify-resume"]
    if arguments.profile:
        command.append("--profile")
    if arguments.import_compatible is not None:
        command += ["--import-compatible", str(arguments.import_compatible.resolve()),
                    "--import-evidence", str(arguments.import_evidence.resolve())]
    write_json(output / "command.json", {"command": command, "source_config_sha256": file_sha256(arguments.source_config)})
    with (output / "stdout.log").open("w", encoding="utf-8") as stdout, (output / "stderr.log").open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr)
    if result.returncode:
        raise RuntimeError(f"真实GPU验证退出={result.returncode}，日志目录={output}")
    report_path = output / "run/training_verification.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report["parameters_changed"] or (arguments.interface and not report["resume_and_continued_optimization_verified"]):
        raise RuntimeError("真实参数更新或恢复继续优化验收未通过")
    print(json.dumps({"passed": True, "report": str(report_path), "sha256": file_sha256(report_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
