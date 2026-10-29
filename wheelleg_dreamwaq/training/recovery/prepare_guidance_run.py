from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .contracts import (
    ARTIFACT_ROOT, GuidanceConfigV1, GuidanceTrainingContractV1, file_sha256,
    load_resolved_config, source_sha256, stable_hash, validate_resolved_config, write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_path = ARTIFACT_ROOT / "unclipped-actions-20261024/training_config.json"
    config = deepcopy(load_resolved_config(source_path))
    config["schema_version"] = "RecoveryResolvedConfigV3"
    config["guidance"] = asdict(GuidanceConfigV1())
    config["training_contract"] = asdict(GuidanceTrainingContractV1())
    config["environment"].update(num_envs=4096, seed=42001, run_dir=str(output / "training4096x4000"))
    config["assessment"]["updates"] = 4000
    config["ppo"]["save_interval"] = 50
    interface = deepcopy(config)
    interface["environment"].update(num_envs=4, task_type="interface_validation", run_dir=str(output / "interface4/run"))
    interface["assessment"]["updates"] = 1
    interface["ppo"]["save_interval"] = 1
    for name, selected in (("training_config.json", config), ("interface_config.json", interface)):
        validate_resolved_config(selected)
        destination = output / name
        if destination.exists() or Path(selected["environment"]["run_dir"]).exists():
            raise FileExistsError(destination)
    write_json(output / "training_config.json", config)
    write_json(output / "interface_config.json", interface)
    write_json(output / "configuration_preparation.json", {
        "source_config_path": str(source_path), "source_config_file_sha256": file_sha256(source_path),
        "source_sha256_at_preparation": source_sha256(), "training_config_sha256": stable_hash(config),
        "guidance_config_sha256": stable_hash(config["guidance"]),
        "training_contract_sha256": stable_hash(config["training_contract"]),
        "note": "配置身份固定；最终受验源码身份由独立复核及实际启动另行记录",
    })
    print(str(output / "training_config.json"), flush=True)


if __name__ == "__main__":
    main()
