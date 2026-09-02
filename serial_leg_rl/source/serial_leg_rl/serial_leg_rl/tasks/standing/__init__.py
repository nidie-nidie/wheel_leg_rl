from __future__ import annotations

import gymnasium as gym


gym.register(
    id="SerialLeg-Standing-Direct-v0",
    entry_point="serial_leg_rl.tasks.standing.standing_env:StandingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "serial_leg_rl.tasks.standing.standing_env_cfg:StandingEnvCfg",
        "rsl_rl_cfg_entry_point": "serial_leg_rl.tasks.standing.agents.rsl_rl_ppo_cfg:StandingPPORunnerCfg",
    },
)

gym.register(
    id="SerialLeg-Standing-Rough-Direct-v0",
    entry_point="serial_leg_rl.tasks.standing.standing_env:StandingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "serial_leg_rl.tasks.standing.standing_env_cfg:StandingRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "serial_leg_rl.tasks.standing.agents.rsl_rl_ppo_cfg:StandingPPORunnerCfg",
    },
)
