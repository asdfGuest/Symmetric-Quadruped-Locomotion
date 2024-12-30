
from my_gym.base.isaac_env import BaseEnv

import torch as th
import numpy as np
import gymnasium as gym

from typing import Any, Type


# joint order in merged case
# 'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
# 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
# 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'

# joint order in decomposed case
# F*_hip_joint, R*_hip_joint, F*_thigh_joint, R*_thigh_joint, F*_calf_joint, R*_calf_joint

# sign of left hip joint is opposite in two case


def _interleave(x0:th.Tensor, x1:th.Tensor, dim=-1) :
    new_shape = list(x0.shape)
    new_shape[dim] *= 2
    new_tensor = th.empty(size=new_shape,dtype=x0.dtype,device=x0.device)
    
    slice0 = [slice(None) for _ in range(len(new_shape))]
    slice0[dim] = slice(0,None,2)
    slice1 = [slice(None) for _ in range(len(new_shape))]
    slice1[dim] = slice(1,None,2)

    new_tensor[slice0] = x0
    new_tensor[slice1] = x1
    return new_tensor

def _decompose_joint_tensor(joint_tensor:th.Tensor) :
    '''
    Args:
        joint_tensor: (n_batch,12k)
    Returns:
        joint left : (n_batch,6k)
        joint right : (n_batch,6k)
    '''
    joint_tensor = joint_tensor.clone()
    return joint_tensor[:,0::2], joint_tensor[:,1::2]


class BaseSymEnv(gym.Env):
    env_cls:Type[BaseEnv] = BaseEnv

    def __init__(self, *args, seperate_foot_reward:bool=False, **kwargs):
        self.env = self.env_cls(*args, **kwargs)
        self.n_envs = self.env.n_envs * 2
        self.num_envs = self.n_envs
        self.device = self.env.device
        self.seperate_foot_reward = seperate_foot_reward

        self.n_obs = self.env.single_observation_space['policy'].shape[0]
        self.n_act = 6

        self.single_observation_space = gym.spaces.Dict(
            policy=gym.spaces.Box(-np.inf, np.inf, (self.n_obs,), np.float32)
        )
        self.single_action_space = gym.spaces.Box(-np.inf, np.inf, (self.n_act,), np.float32)
        self.observation_space = gym.spaces.Dict(
            policy=gym.spaces.Box(-np.inf, np.inf, (self.n_envs,self.n_obs), np.float32)
        )
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (self.n_envs,self.n_act), np.float32)

    def _merge_action(self, action:th.Tensor) :
        '''
        Args:
            action: (2*n_envs,6)
        Returns:
            (n_envs,12)
        '''
        action = _interleave(action[1::2,:], action[0::2,:])
        action[:,0] = -action[:,0] #FL
        action[:,2] = -action[:,2] #RL
        return action

    def _decompose_observation(self, obs_pack) :
        linvel, angvel, projgrav, cmd, qpos, qvel, action = obs_pack
        linvel = linvel.clone()
        angvel = angvel.clone()
        projgrav = projgrav.clone()
        cmd = cmd.clone()
        qpos = qpos.clone()
        qvel = qvel.clone()
        action = action.clone()

        qpos_l, qpos_r = _decompose_joint_tensor(qpos)
        qvel_l, qvel_r = _decompose_joint_tensor(qvel)
        action_l, action_r = _decompose_joint_tensor(action)

        qpos_l[:,0::6] = -qpos_l[:,0::6]
        qpos_l[:,1::6] = -qpos_l[:,1::6]
        qvel_l[:,0::6] = -qvel_l[:,0::6]
        qvel_l[:,1::6] = -qvel_l[:,1::6]
        action_l[:,0::6] = -action_l[:,0::6]
        action_l[:,1::6] = -action_l[:,1::6]

        obs_r = th.cat([
            linvel,
            angvel,
            projgrav,
            cmd,
            qpos_l,
            qpos_r,
            qvel_l,
            qvel_r,
            action_l,
            action_r
        ],dim=-1)

        linvel[:,1::3] = -linvel[:,1::3] #linvel y
        angvel[:,0::3] = -angvel[:,0::3] #angvel x
        angvel[:,2::3] = -angvel[:,2::3] #angvel z
        projgrav[:,1::3] = -projgrav[:,1::3] #y
        cmd[:,1::3] = -cmd[:,1::3] #y
        cmd[:,2::3] = -cmd[:,2::3] #z

        obs_l = th.cat([
            linvel,
            angvel,
            projgrav,
            cmd,
            qpos_r,
            qpos_l,
            qvel_r,
            qvel_l,
            action_r,
            action_l
        ],dim=-1)

        obs = _interleave(obs_r, obs_l, dim=0)
        return obs

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) :
        obs_pack, info = self.env.reset(seed, options)
        obs = self._decompose_observation(obs_pack)
        return {'policy':obs}, info

    def step(self, action: th.Tensor) :
        action = self._merge_action(action)
        obs_pack, rwd_pack, termin, trunc, info = self.env.step(action)
        obs = self._decompose_observation(obs_pack)
        rwd_l = rwd_pack[0] + (rwd_pack[2] if self.seperate_foot_reward else rwd_pack[1])
        rwd_r = rwd_pack[0] + (rwd_pack[3] if self.seperate_foot_reward else rwd_pack[1])
        rwd = _interleave(rwd_r, rwd_l)
        termin = _interleave(termin, termin)
        trunc = _interleave(trunc, trunc)
        return {'policy':obs}, rwd, termin, trunc, info

    @classmethod
    def seed(cls, seed: int=-1) :
        return cls.env_cls.seed(seed)

    def render(self, recompute: bool = False) :
        return self.env.render(recompute)

    def close(self) :
        return self.env.close()

    def set_debug_vis(self, debug_vis: bool) :
        return self.env.set_debug_vis(debug_vis)


class BaseAsymEnv(gym.Env):
    env_cls:Type[BaseEnv] = BaseEnv

    def __init__(self, *args, **kwargs):
        self.env = self.env_cls(*args, **kwargs)
        self.n_envs = self.env.n_envs
        self.num_envs = self.env.num_envs
        self.device = self.env.device

        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) :
        obs_pack, info = self.env.reset(seed, options)
        obs = th.cat(obs_pack,dim=-1)
        return {'policy':obs}, info

    def step(self, action: th.Tensor) :
        obs_pack, rwd_pack, termin, trunc, info = self.env.step(action)
        obs = th.cat(obs_pack,dim=-1)
        rwd = rwd_pack[0] + rwd_pack[1]
        return {'policy':obs}, rwd, termin, trunc, info

    @classmethod
    def seed(cls, seed: int=-1) :
        return cls.env_cls.seed(seed)

    def render(self, recompute: bool = False) :
        return self.env.render(recompute)

    def close(self) :
        return self.env.close()

    def set_debug_vis(self, debug_vis: bool) :
        return self.env.set_debug_vis(debug_vis)
