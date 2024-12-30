from my_gym.go2_flat.config import Go2EnvCfg, Go2PlayEnvCfg
from my_gym.base import BaseSymEnv, BaseAsymEnv

import gymnasium as gym
gym.register(
    id='Go2-Flat-Sym-Train',
    entry_point='my_gym.go2_flat:BaseSymEnv',
    disable_env_checker=True,
    kwargs={
        'cfg': Go2EnvCfg()
    }
)
gym.register(
    id='Go2-Flat-Sym-Play',
    entry_point='my_gym.go2_flat:BaseSymEnv',
    disable_env_checker=True,
    kwargs={
        'cfg': Go2PlayEnvCfg()
    }
)
gym.register(
    id='Go2-Flat-Asym-Train',
    entry_point='my_gym.go2_flat:BaseAsymEnv',
    disable_env_checker=True,
    kwargs={
        'cfg': Go2EnvCfg()
    }
)
gym.register(
    id='Go2-Flat-Asym-Play',
    entry_point='my_gym.go2_flat:BaseAsymEnv',
    disable_env_checker=True,
    kwargs={
        'cfg': Go2PlayEnvCfg()
    }
)
