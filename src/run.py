from argparse import ArgumentParser
parser = ArgumentParser(description='run the ppo agent based of configuration file')
parser.add_argument('cfg_path', type=str)
parser.add_argument('--deterministic', type=bool, default=False)

from omni.isaac.lab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import IsaacLabWrapper

from src.utils import load_cfg, apply_cfg_mapping
import my_gym
import omni.isaac.lab_tasks

import gymnasium as gym
import torch as th


def main() :
    cfg = load_cfg(args_cli.cfg_path)
    env = gym.make(**cfg['run']['gym_kwargs'])
    env = IsaacLabWrapper(env)

    obs_space = env.observation_space
    act_space = env.action_space
    device = env.device
    n_envs = env.num_envs

    _mapping = {
        'obs_space' : obs_space,
        'act_space' : act_space,
        'device' : device,
        'n_envs' : n_envs
    }
    apply_cfg_mapping(cfg, _mapping)


    policy = cfg['policy_cls'](**cfg['policy_kwargs'])
    value = cfg['value_cls'](**cfg['value_kwargs']) if cfg['value_cls'] is not None else policy

    memory = RandomMemory(memory_size=cfg['ppo']['rollouts'], num_envs=n_envs, device=device)

    agent = PPO(
        models={'policy':policy, 'value':value},
        memory=memory,
        observation_space=obs_space,
        action_space=act_space,
        cfg=cfg['ppo']
    )
    agent.load(cfg['run']['load_path'])


    obs, info = env.reset()

    while simulation_app.is_running() :

        with th.no_grad() :
            p_obs = agent._state_preprocessor(obs)
            if not args_cli.deterministic :
                act, _, _ = agent.policy.act({'states': p_obs}, role='policy') # stochastic
            else :
                act, _, _ = agent.policy.compute({'states': p_obs}, role='policy') # deterministic
            val, _, _ = agent.value.act({'states': p_obs}, role='value')

        obs, rwd, ter, tru, info = env.step(act)

        #terrain_levels = env._env.env.env.scene.terrain.terrain_levels
        #print(terrain_levels.to(th.float32).mean(), th.bincount(terrain_levels))


    env.close()


if __name__ == '__main__' :
    main()
    simulation_app.close()
