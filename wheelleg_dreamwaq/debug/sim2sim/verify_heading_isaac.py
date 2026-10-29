from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

from scripts.export_dreamwaq_actor import _build_policy
from wheelleg_dreamwaq.algorithms.dreamwaq.history_wrapper import DreamWaQHistoryVecEnvWrapper
from wheelleg_dreamwaq.schemas.heading_control import yaw_from_quaternion
from wheelleg_dreamwaq.schemas.randomization import FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env import WheelLegFlatEnv
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.env_cfg import WheelLegFlatEnvCfg
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import apply_task_profile, task_profile_from_contract
from wheelleg_dreamwaq.training.dreamwaq_checkpoint import load_dreamwaq_checkpoint_artifact


def main():
    checkpoint, manifest, metadata = load_dreamwaq_checkpoint_artifact(args.checkpoint)
    cfg = WheelLegFlatEnvCfg()
    cfg.seed = metadata["seed"]
    cfg.scene.num_envs = 8
    cfg.sim.device = args.device
    cfg.randomization = FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1
    apply_task_profile(cfg, task_profile_from_contract(manifest["base_task_contract"]))
    direct = WheelLegFlatEnv(cfg)
    env = DreamWaQHistoryVecEnvWrapper(direct, clip_actions=None)
    policy = _build_policy(checkpoint["model_state_dict"], raw_actions=True).to(args.device)
    try:
        observations = env.get_observations()
        torch.testing.assert_close(observations["policy_history"].view(8, 5, 25),
                                   observations["policy"][:, None, :].expand(-1, 5, -1), rtol=0, atol=0)
        with torch.no_grad():
            for _ in range(24):
                observations, _, _, _ = env.step(policy.act_inference(observations))
        selected = torch.tensor([0, 3], device=direct.device)
        remaining = torch.tensor([1, 2, 4, 5, 6, 7], device=direct.device)
        reference, turning = direct._heading.reference.clone(), direct._heading.was_turning.clone()
        source, effective = direct._source_commands.clone(), direct._commands.clone()
        timestamp, tick = direct.robot.data._sim_timestamp, direct.common_step_counter
        direct._reset_idx(selected)
        assert timestamp == direct.robot.data._sim_timestamp and tick == direct.common_step_counter
        for before, after in ((reference, direct._heading.reference), (turning, direct._heading.was_turning),
                              (source, direct._source_commands), (effective, direct._commands)):
            assert torch.equal(before[remaining], after[remaining])
        yaw = yaw_from_quaternion(direct.robot.data.root_link_quat_w)
        reset_error = (direct._heading.reference[selected] - yaw[selected]).abs().max().item()
        assert reset_error < 1e-6 and not direct._heading.was_turning[selected].any()
        reference = direct._heading.reference.clone()
        direct._get_observations()
        direct._get_observations()
        assert torch.equal(reference, direct._heading.reference)
        # 真实 timeout 路径验证部分 done 后的 history 首帧填充。
        direct.episode_length_buf[selected] = direct.max_episode_length - 1
        raw = torch.tensor([[2., -2., 1.5, -1.5, 2., 2.]], device=direct.device).expand(8, -1)
        observations, _, dones, _ = env.step(raw)
        assert dones[selected].all()
        history = observations["policy_history"].view(8, 5, 25)
        assert torch.equal(history[selected], observations["policy"][selected, None, :].expand(-1, 5, -1))
        active = dones == 0
        assert active.any()
        assert torch.equal(observations["policy"][active, 19:25], raw[active])
        assert torch.equal(observations["critic"][:, 6:9], cfg.normalization.normalize_command(direct._commands))
        report = {"num_envs": 8, "interface_control_steps": 25, "completed_iterations": metadata["completed_iterations"],
                  "partial_reset_reference_error_rad": reset_error, "other_rows_unchanged": True,
                  "reset_no_clock_advance": True, "repeated_observation_preserves_reference": True,
                  "partial_done_history_first_frame": True, "raw_previous_action_preserved": True,
                  "critic_effective_command_matches": True}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        env.close()


try:
    main()
finally:
    launcher.app.close()
