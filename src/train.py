from argparse import ArgumentParser
parser = ArgumentParser(description='train the ppo agent based of configuration file')
parser.add_argument('cfg_path', type=str)

from omni.isaac.lab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from skrl.trainers.torch import SequentialTrainer
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import IsaacLabWrapper

from src.utils import load_cfg, apply_cfg_mapping
import my_gym
import omni.isaac.lab_tasks

import gymnasium as gym


def main() :
    cfg = load_cfg(args_cli.cfg_path)
    env = gym.make(**cfg['train']['gym_kwargs'])
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
    if cfg['train']['load_path'] is not None :
        agent.load(cfg['train']['load_path'])


    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg=cfg['train']['trainer_kwargs']
    )
    trainer.train()


    save_path = cfg['train']['save_path']
    if save_path is not None :
        agent.save(save_path)
    
    env.close()


if __name__ == '__main__' :
    main()
    simulation_app.close()
