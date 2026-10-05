"""A1 locomotion task registrations for gogo-learn."""

import gymnasium as gym

from . import agents


def _register_dreamwaq_rsl_rl_classes() -> None:
    try:
        import rsl_rl.runners.on_policy_runner as on_policy_runner
        from gogo_learn.dreamwaq import DreamWaQActorCritic
        from gogo_learn.dreamwaq.rsl_rl import DreamWaQPPO
    except ModuleNotFoundError:
        return

    on_policy_runner.DreamWaQActorCritic = DreamWaQActorCritic
    on_policy_runner.DreamWaQPPO = DreamWaQPPO


_register_dreamwaq_rsl_rl_classes()


gym.register(
    id="Gogo-A1-PACE-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace_flat_env_cfg:GogoA1PaceFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1FlatStudentPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-PACE-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace_flat_env_cfg:GogoA1PaceFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1FlatStudentPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-PACE-Paper-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace_flat_env_cfg:GogoA1PacePaperFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1FlatStudentPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-PACE-Paper-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace_flat_env_cfg:GogoA1PacePaperFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1FlatStudentPPORunnerCfg",
    },
)


gym.register(
    id="Gogo-A1-DreamWaQ-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQPPORunnerCfg",
    },
)


gym.register(
    id="Gogo-A1-DreamWaQ-Warmup-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQWarmupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQWarmupPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-Warmup-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQWarmupEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQWarmupPPORunnerCfg",
    },
)


gym.register(
    id="Gogo-A1-DreamWaQ-Forward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQForwardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQForwardPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-Forward-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQForwardEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQForwardPPORunnerCfg",
    },
)


gym.register(
    id="Gogo-A1-DreamWaQ-MidRough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQMidRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQMidRoughPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-MidRough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQMidRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQMidRoughPPORunnerCfg",
    },
)


gym.register(
    id="Gogo-A1-DreamWaQ-HardRough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQHardRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQHardRoughPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-HardRough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQHardRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQHardRoughPPORunnerCfg",
    },
)


gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeo-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoPPORunnerCfg",
    },
)


gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoHard-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoHardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoHardPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoHard-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoHardEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoHardPPORunnerCfg",
    },
)


gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoMidHigh-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoMidHighEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoMidHighPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoMidHigh-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoMidHighEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoMidHighPPORunnerCfg",
    },
)


gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoMidLow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoMidLowEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoMidLowPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoMidLow-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoMidLowEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoMidLowPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoMid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoMidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoMidPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoMid-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoMidEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoMidPPORunnerCfg",
    },
)


gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoDownStairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoDownStairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoDownStairsPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoDownStairs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoDownStairsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoDownStairsPPORunnerCfg",
    },
)


gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoDownStairsSlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoDownStairsSlowEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoDownStairsSlowPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoDownStairsSlow-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoDownStairsSlowEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoDownStairsSlowPPORunnerCfg",
    },
)


gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoDownStairsMidGeom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoDownStairsMidGeomEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoDownStairsMidGeomPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoDownStairsMidGeom-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoDownStairsMidGeomEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoDownStairsMidGeomPPORunnerCfg",
    },
)


gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoInvStairsMix-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoInvStairsMixEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoInvStairsMixPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoInvStairsMix-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoInvStairsMixEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoInvStairsMixPPORunnerCfg",
    },
)


gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoTableIHeight-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoTableIHeightEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoTableIHeightPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoTableIHeightA1Target-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoTableIHeightA1TargetEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoTableIHeightA1TargetPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoPositiveReward-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoPositiveRewardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoPositiveRewardPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoPositiveRewardA1Height-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoPositiveRewardA1HeightEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoPositiveRewardA1HeightPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoPublicTerrainL3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoPublicTerrainLevel3EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoPublicTerrainLevel3PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoPublicTerrainL5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoPublicTerrainLevel5EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoPublicTerrainLevel5PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLike-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQFromScratchPaperLikePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLike-Play-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQFromScratchPaperLikePPORunnerCfg"
        ),
    },
)

for _stage in (0, 1, 3, 5):
    gym.register(
        id=f"Gogo-A1-DreamWaQ-FromScratchPaperLikeStage{_stage}-v0",
        entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeStage{_stage}EnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:"
                f"GogoA1DreamWaQFromScratchPaperLikeStage{_stage}PPORunnerCfg"
            ),
        },
    )

    gym.register(
        id=f"Gogo-A1-DreamWaQ-FromScratchPaperLikeStage{_stage}-Play-v0",
        entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeStage{_stage}EnvCfg_PLAY"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:"
                f"GogoA1DreamWaQFromScratchPaperLikeStage{_stage}PPORunnerCfg"
            ),
        },
    )

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullState-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStatePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullState-Play-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStatePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateStage0-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateStage0EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateStage0PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateStage0-Play-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateStage0EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateStage0PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20-Play-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateStage0A1PD20-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateStage0A1PD20-Play-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20DofLimit-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20DofLimit-Play-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateStage0A1PD20DofLimit-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateStage0A1PD20DofLimit-Play-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20DofLimitHfStairs-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20DofLimitHfStairs-Play-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPublicRecon-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPublicReconPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPublicRecon-Play-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPublicReconPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPublicSample-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPublicSamplePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPublicSample-Play-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPublicSamplePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPublicCENet-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPublicCENetPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPublicCENet-Play-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPublicCENetPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPublicCENetNoAdaBoot-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPublicCENetNoAdaBootPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPublicCENetNoAdaBoot-Play-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsPublicCENetNoAdaBootPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20DofLimitHfStairsUpBias-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsUpBiasEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsUpBiasPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20DofLimitHfStairsUpBias-Play-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsUpBiasEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsUpBiasPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20DofLimitHfStairsLowUp-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsLowUpEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsLowUpPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateA1PD20DofLimitHfStairsLowUp-Play-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsLowUpEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateA1PD20DofLimitHfStairsLowUpPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateStage0A1PD20DofLimitHfStairs-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitHfStairsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitHfStairsPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateStage0A1PD20DofLimitHfStairs-Play-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitHfStairsEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitHfStairsPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateStage0A1PD20DofLimitHfStairsPublicCENetNoAdaBoot-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitHfStairsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitHfStairsPublicCENetNoAdaBootPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FromScratchPaperLikeFullStateStage0A1PD20DofLimitHfStairsPublicCENetNoAdaBoot-Play-v0",
    entry_point=f"{__name__}.envs:DreamWaQPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitHfStairsEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "GogoA1DreamWaQFromScratchPaperLikeFullStateStage0A1PD20DofLimitHfStairsPublicCENetNoAdaBootPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoTableIFootClearance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoTableIFootClearanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoTableIFootClearancePPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoTableIFull-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoTableIFullEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoTableIFullPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-RoughGeoTableIHeightInvStairsMix-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQRoughGeoTableIHeightInvStairsMixEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQRoughGeoTableIHeightInvStairsMixPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Gogo-A1-DreamWaQ-FlatStart-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQFlatStartEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQFlatStartPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-DreamWaQ-FlatStart-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1DreamWaQFlatStartEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1DreamWaQFlatStartPPORunnerCfg",
    },
)


gym.register(
    id="Gogo-A1-Flat-Student-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1FlatStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1FlatStudentPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Flat-Student-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1FlatStudentEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1FlatStudentPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-MLP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatMLPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatMLPPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-MLP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatMLPEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatMLPPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Fast-Elegant-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatFastElegantEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatFastElegantPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Fast-Elegant-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatFastElegantEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatFastElegantPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Fast-Agile-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatFastAgileEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatFastAgilePPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Fast-Agile-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatFastAgileEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatFastAgilePPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Forward-Warmup-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatForwardWarmupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatForwardWarmupPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Forward-Warmup-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatForwardWarmupEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatForwardWarmupPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Stable-Step-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatStableStepEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatStableStepPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Stable-Step-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatStableStepEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatStableStepPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Torque-Safe-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatTorqueSafeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatTorqueSafePPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Torque-Safe-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatTorqueSafeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatTorqueSafePPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Torque-Safe-Stage2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatTorqueSafeStage2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatTorqueSafeStage2PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Torque-Safe-Stage2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatTorqueSafeStage2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatTorqueSafeStage2PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Torque-Safe-Stage3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatTorqueSafeStage3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatTorqueSafeStage3PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Torque-Safe-Stage3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatTorqueSafeStage3EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatTorqueSafeStage3PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Torque-Safe-Stage3-OneKg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatTorqueSafeStage3OneKgEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatTorqueSafeStage3PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Torque-Safe-Stage3-OneKg-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatTorqueSafeStage3OneKgEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatTorqueSafeStage3PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Hardware-Limit-OneKg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatHardwareLimitOneKgEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatHardwareLimitOneKgPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Hardware-Limit-OneKg-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatHardwareLimitOneKgEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatHardwareLimitOneKgPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Hardware-Limit-OneKg-StageA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatHardwareLimitOneKgStageAEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatHardwareLimitOneKgStageAPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-Hardware-Limit-OneKg-StageA-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatHardwareLimitOneKgStageAEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatHardwareLimitOneKgStageAPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-BackYawLift-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatBackYawLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatBackYawLiftPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-Dog-Servo-Flat-BackYawLift-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoDogServoFlatBackYawLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoDogServoFlatBackYawLiftPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Flat-Fast-Elegant-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1FlatFastElegantEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1FlatFastElegantPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Flat-Fast-Elegant-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1FlatFastElegantEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1FlatFastElegantPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Flat-Fast-Robust-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1FlatFastRobustEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1FlatFastRobustPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Flat-Fast-Robust-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1FlatFastRobustEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1FlatFastRobustPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Approach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairApproachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairApproachPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Approach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairApproachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairApproachPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Approach-Goal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairApproachGoalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairApproachGoalPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Approach-Goal-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairApproachGoalEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairApproachGoalPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgePPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgePPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-X058-To-X070-Real-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeX058ToX070RealEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgeX058ToX070RealPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-X058-To-X070-Real-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeX058ToX070RealEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgeX058ToX070RealPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-X058-To-X068-Centered-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeX058ToX068CenteredEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgeX058ToX068CenteredPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-X058-To-X068-Centered-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeX058ToX068CenteredEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgeX058ToX068CenteredPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-X058-To-High-X080-Centered-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeX058ToHighX080CenteredEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgeX058ToHighX080CenteredPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-X058-To-High-X080-Centered-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeX058ToHighX080CenteredEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgeX058ToHighX080CenteredPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-X058-To-Mid-X068-High-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeX058ToMidX068HighEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgeX058ToMidX068HighPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-X058-To-Mid-X068-High-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeX058ToMidX068HighEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgeX058ToMidX068HighPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-X058-To-Mid-X072-Hold-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeX058ToMidX072HoldEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgeX058ToMidX072HoldPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-X058-To-Mid-X072-Hold-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeX058ToMidX072HoldEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgeX058ToMidX072HoldPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-X058-To-Mid-X070-Hold-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeX058ToMidX070HoldEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgeX058ToMidX070HoldPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-X058-To-Mid-X070-Hold-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeX058ToMidX070HoldEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgeX058ToMidX070HoldPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-X058-To-Mid-X070-Hold-ResetMix-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeX058ToMidX070HoldResetMixEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgeX058ToMidX070HoldResetMixPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-X058-To-Mid-X070-Hold-ResetMix-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeX058ToMidX070HoldResetMixEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgeX058ToMidX070HoldResetMixPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-X058-To-Mid-X066-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeX058ToMidX066LiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgeX058ToMidX066LiftPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Bridge-X058-To-Mid-X066-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairBridgeX058ToMidX066LiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairBridgeX058ToMidX066LiftPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Finisher-X070-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairFinisherX070EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairFinisherX070PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Finisher-X070-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairFinisherX070EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairFinisherX070PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Finisher-X070-Medium-ResetAction-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairFinisherX070MediumResetActionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairFinisherX070MediumResetActionPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Finisher-X070-Medium-ResetAction-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairFinisherX070MediumResetActionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairFinisherX070MediumResetActionPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Finisher-X068-Centered-ResetAction-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairFinisherX068CenteredResetActionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairFinisherX068CenteredResetActionPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Finisher-X068-Centered-ResetAction-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairFinisherX068CenteredResetActionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairFinisherX068CenteredResetActionPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Mixed-Approach-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbMixedApproachX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbMixedApproachX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Mixed-Approach-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbMixedApproachX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbMixedApproachX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Mixed-Approach-X068-Yaw-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbMixedApproachX068YawEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbMixedApproachX068YawPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Mixed-Approach-X068-Yaw-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbMixedApproachX068YawEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbMixedApproachX068YawPPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Curriculum-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbCurriculumX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbCurriculumX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Curriculum-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbCurriculumX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbCurriculumX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Random-LowMid-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbRandomLowMidX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbRandomLowMidX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Random-LowMid-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbRandomLowMidX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbRandomLowMidX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-LowStep-Stable-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbLowStepStableX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbLowStepStableX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-LowStep-Stable-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbLowStepStableX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbLowStepStableX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Switch-Near-Low-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbSwitchNearLowX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbSwitchNearLowX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Switch-Near-Low-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbSwitchNearLowX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbSwitchNearLowX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Near-Finisher-Low-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbNearFinisherLowX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbNearFinisherLowX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Near-Finisher-Low-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbNearFinisherLowX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbNearFinisherLowX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Near-Finisher-Relaxed-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbNearFinisherRelaxedX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbNearFinisherRelaxedX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Near-Finisher-Relaxed-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbNearFinisherRelaxedX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbNearFinisherRelaxedX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Near-Finisher-Edge-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbNearFinisherEdgeX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbNearFinisherEdgeX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Near-Finisher-Edge-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbNearFinisherEdgeX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbNearFinisherEdgeX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Switch-Near-Edge-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbSwitchNearEdgeX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbSwitchNearEdgeX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Switch-Near-Edge-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbSwitchNearEdgeX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbSwitchNearEdgeX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Switch-X058-Edge-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbSwitchX058EdgeX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbSwitchX058EdgeX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Switch-X058-Edge-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbSwitchX058EdgeX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbSwitchX058EdgeX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Switch-X070-Edge-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbSwitchX070EdgeX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbSwitchX070EdgeX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Switch-X070-Edge-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbSwitchX070EdgeX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbSwitchX070EdgeX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Bridge-Handoff-Edge-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbBridgeHandoffEdgeX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbBridgeHandoffEdgeX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Bridge-Handoff-Edge-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbBridgeHandoffEdgeX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbBridgeHandoffEdgeX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Bridge-Handoff-Centered-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbBridgeHandoffCenteredX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbBridgeHandoffCenteredX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Bridge-Handoff-Centered-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbBridgeHandoffCenteredX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbBridgeHandoffCenteredX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Switch-X058-Edge-BridgeAssist-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbSwitchX058EdgeBridgeAssistX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbSwitchX058EdgeBridgeAssistX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Switch-X058-Edge-BridgeAssist-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbSwitchX058EdgeBridgeAssistX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbSwitchX058EdgeBridgeAssistX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Switch-X058-Edge-BridgeAssist-Tight-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbSwitchX058EdgeBridgeAssistTightX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbSwitchX058EdgeBridgeAssistTightX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Switch-X058-Edge-BridgeAssist-Tight-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbSwitchX058EdgeBridgeAssistTightX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbSwitchX058EdgeBridgeAssistTightX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Switch-X058-RearCatch-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbSwitchX058RearCatchX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbSwitchX058RearCatchX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Switch-X058-RearCatch-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbSwitchX058RearCatchX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbSwitchX058RearCatchX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Switch-X058-RearCatch-Stage-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbSwitchX058RearCatchStageX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbSwitchX058RearCatchStageX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-Switch-X058-RearCatch-Stage-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbSwitchX058RearCatchStageX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbSwitchX058RearCatchStageX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-FlatStart-X058-RearStrict-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbFlatStartX058RearStrictX068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbFlatStartX058RearStrictX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-FlatStart-X058-RearStrict-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbFlatStartX058RearStrictX068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbFlatStartX058RearStrictX068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-FlatEdge-Commit-X058-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbFlatEdgeCommitX058X068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbFlatEdgeCommitX058X068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-FlatEdge-Commit-X058-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbFlatEdgeCommitX058X068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbFlatEdgeCommitX058X068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-FlatEdge-FrontContact-X058-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbFlatEdgeFrontContactX058X068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbFlatEdgeFrontContactX058X068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-FlatEdge-FrontContact-X058-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbFlatEdgeFrontContactX058X068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbFlatEdgeFrontContactX058X068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-FlatEdge-FrontFootOnly-X058-X068-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbFlatEdgeFrontFootOnlyX058X068EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbFlatEdgeFrontFootOnlyX058X068PPORunnerCfg",
    },
)

gym.register(
    id="Gogo-A1-Stair-Climb-FlatEdge-FrontFootOnly-X058-X068-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:GogoA1StairClimbFlatEdgeFrontFootOnlyX058X068EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GogoA1StairClimbFlatEdgeFrontFootOnlyX058X068PPORunnerCfg",
    },
)
