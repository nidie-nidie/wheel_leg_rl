from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from debug.no_action_clip.contract import write_json
from debug.no_action_clip.experiment import load_experiment_checkpoint


def main(source: Path):
    payload, _ = load_experiment_checkpoint(source)
    target = source.parent / "artifact-integrity"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "params/env.yaml", "params/agent.yaml", "closed_chain_reset_cache.pt"):
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.parent / name, destination)
    checkpoint = target / source.name
    shutil.copyfile(source, checkpoint)
    load_experiment_checkpoint(checkpoint)
    checked = []
    for name in ("params/env.yaml", "params/agent.yaml", "closed_chain_reset_cache.pt"):
        path = target / name
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        try:
            load_experiment_checkpoint(checkpoint)
        except ValueError as error:
            if "配置或缓存指纹不符" not in str(error):
                raise
            checked.append(name)
        else:
            raise AssertionError(f"实验接受了已损坏文件：{name}")
        finally:
            path.write_bytes(original)
    load_experiment_checkpoint(checkpoint)
    geometry_checks = []
    damaged = dict(payload)
    damaged["infos"] = {**payload["infos"], "geometry_guard": {}}
    torch.save(damaged, checkpoint)
    try:
        load_experiment_checkpoint(checkpoint)
    except ValueError:
        geometry_checks.append("checkpoint_infos")
    else:
        raise AssertionError("实验接受了不同几何契约的 checkpoint")
    finally:
        shutil.copyfile(source, checkpoint)
    manifest_path = target / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    changed_manifest = json.loads(original_manifest)
    changed_manifest["geometry_guard"] = {}
    write_json(manifest_path, changed_manifest)
    try:
        load_experiment_checkpoint(checkpoint)
    except ValueError:
        geometry_checks.append("training_manifest")
    else:
        raise AssertionError("实验接受了不同几何契约的 manifest")
    finally:
        manifest_path.write_bytes(original_manifest)
    load_experiment_checkpoint(checkpoint)
    result = {"completed_iterations": payload["infos"]["completed_iterations"],
              "corrupted_files_rejected": checked, "geometry_contract_rejections": geometry_checks,
              "intact_checkpoint_verified": True}
    write_json(target / "verification.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    main(parser.parse_args().checkpoint)
