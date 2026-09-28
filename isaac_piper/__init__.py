"""Isaac Sim / Isaac Lab Piper tasks.

Importing this package registers the Gym task.  Isaac modules are resolved
lazily through entry-point strings, so the package imports without a running
SimulationApp (the asset builder and plain-torch tests rely on that).
"""

import gymnasium as gym

TASK_ID = "Piper-PickPlace-Direct-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaac_piper.pick_place_env:PiperPickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaac_piper.pick_place_env_cfg:PiperPickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": "isaac_piper.agents.rsl_rl_ppo_cfg:PiperPickPlacePPORunnerCfg",
    },
)
