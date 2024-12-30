from my_gym.go2_rough.config import Go2EnvCfg, Go2PlayEnvCfg
from my_gym.base import BaseSymEnv, BaseAsymEnv

import gymnasium as gym
gym.register(
    id='Go2-Rough-Sym-Train',
    entry_point='my_gym.go2_rough:BaseSymEnv',
    disable_env_checker=True,
    kwargs={
        'cfg': Go2EnvCfg()
    }
)
gym.register(
    id='Go2-Rough-Sym-Play',
    entry_point='my_gym.go2_rough:BaseSymEnv',
    disable_env_checker=True,
    kwargs={
        'cfg': Go2PlayEnvCfg()
    }
)
gym.register(
    id='Go2-Rough-Asym-Train',
    entry_point='my_gym.go2_rough:BaseAsymEnv',
    disable_env_checker=True,
    kwargs={
        'cfg': Go2EnvCfg()
    }
)
gym.register(
    id='Go2-Rough-Asym-Play',
    entry_point='my_gym.go2_rough:BaseAsymEnv',
    disable_env_checker=True,
    kwargs={
        'cfg': Go2PlayEnvCfg()
    }
)
