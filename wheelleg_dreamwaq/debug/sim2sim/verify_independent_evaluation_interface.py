from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from recovery.bounded_actor_critic import BoundedActorCritic, DeterministicStandingActor
from recovery.independent_evaluate import evaluate_actor, final_suite_cases
from recovery.independent_poses import IndependentPoseBank
from recovery.standing_reference import StandingReference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    bank = IndependentPoseBank.load(config["environment"]["pose_bank_path"])
    reference_cfg = config["reference"]
    reference = StandingReference.load(reference_cfg["manifest_path"], reference_cfg["bank_path"], reference_cfg["bank_sha256"])
    torch.manual_seed(42002)
    model = BoundedActorCritic(reference.q_median)
    actor = DeterministicStandingActor(model.actor).eval()
    suite = final_suite_cases(bank)
    cases = [suite[400], suite[600]]
    report = evaluate_actor(config, actor, case_specs=cases, output_dir=args.output)
    if len(report["episodes"]) != 2 or report["final_suite_complete"] or report["nominal_behavior_targets_met"] is not None:
        raise ValueError("小规模真实接口检查被错误记为最终行为验收")
    print(json.dumps({"passed": True, "episodes": len(report["episodes"]),
                      "actual_initialization_checks": [e["initialization"]["physical_validation_passed"] for e in report["episodes"]],
                      "policy_scope": "fresh_initialized_network_for_interface_check_only"}))


if __name__ == "__main__":
    main()
