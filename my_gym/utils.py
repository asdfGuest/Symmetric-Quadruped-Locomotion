
from omni.isaac.lab.envs import DirectRLEnv, VecEnvStepReturn

import torch as th

from collections import deque
from _collections_abc import Sequence
from typing import Callable


class _POST_INIT :
    pass
POST_INIT = _POST_INIT


class HistoryBuffer :
    '''
    - only shape of (n_envs, vec_dim) tensor can be storaged
    - you can reset buffer for specific indices of envs
    '''

    def __init__(self, n_envs:int, buffer_length:int, default_value:th.Tensor) :
        '''
        default_value: tensor shape of (vec_dim,)
        '''

        self.n_envs = n_envs
        self.device = default_value.device

        self.vec_dim = default_value.shape[0]
        self.buffer_length = buffer_length

        self._buffer = deque(maxlen=buffer_length)
        self._default_value = default_value.clone()
        self._ALL_INDICES = th.arange(start=0, end=self.n_envs, dtype=th.int64, )

        for _ in range(self.buffer_length) :
            self.push(default_value.expand(n_envs, *self._default_value.shape))

    def push(self, x:th.Tensor) :
        '''
        x: tensor shape of (n_envs, vec_dim)
        '''
        self._buffer.append(x.clone())

    def reset(self, env_ids: Sequence[int]|None) :
        env_ids = self._ALL_INDICES if env_ids is None else env_ids

        for k in range(self.buffer_length) :
            self._buffer[k][env_ids] = self._default_value.expand(len(env_ids),self.vec_dim)

    def __getitem__(self, index:int) -> th.Tensor :
        return self._buffer[index]

    def get_data(self) :
        return th.cat(list(self._buffer), dim=-1)


def direct_rl_env_wrong_step(self:DirectRLEnv, action: th.Tensor) -> VecEnvStepReturn:
    """Execute one time-step of the environment's dynamics.

    The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
    lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
    independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
    and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
    time-step is computed as the product of the two.

    This function performs the following steps:

    1. Pre-process the actions before stepping through the physics.
    2. Apply the actions to the simulator and step through the physics in a decimated manner.
    3. Compute the reward and done signals.
    4. Reset environments that have terminated or reached the maximum episode length.
    5. Apply interval events if they are enabled.
    6. Compute observations.

    Args:
        action: The actions to apply on the environment. Shape is (num_envs, action_dim).

    Returns:
        A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
    """
    action = action.to(self.device)
    # add action noise
    if self.cfg.action_noise_model:
        action = self._action_noise_model.apply(action)

    # process actions
    self._pre_physics_step(action)

    # check if we need to do rendering within the physics loop
    # note: checked here once to avoid multiple checks within the loop
    is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

    # perform physics stepping
    for _ in range(self.cfg.decimation):
        self._sim_step_counter += 1
        # set actions into buffers
        self._apply_action()
        # set actions into simulator
        self.scene.write_data_to_sim()
        # simulate
        self.sim.step(render=False)
        # render between steps only if the GUI or an RTX sensor needs it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
            self.sim.render()
        # update buffers at sim dt
        self.scene.update(dt=self.physics_dt)

    # post-step:
    # -- update env counters (used for curriculum generation)
    self.episode_length_buf += 1  # step in current episode (per env)
    self.common_step_counter += 1  # total step (common for all envs)

    self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
    self.reset_buf = self.reset_terminated | self.reset_time_outs

    # -- reset envs that terminated/timed-out and log the episode information
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
        self._reset_idx(reset_env_ids)
        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

    # post-step: step interval event
    if self.cfg.events:
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

    # update observations
    self.obs_buf = self._get_observations()
    self.reward_buf = self._get_rewards()

    # add observation noise
    # note: we apply no noise to the state space (since it is used for critic networks)
    if self.cfg.observation_noise_model:
        self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

    # return observations, rewards, resets and extras
    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


def my_direct_rl_env_step(self:DirectRLEnv, action: th.Tensor) -> VecEnvStepReturn:
    """Execute one time-step of the environment's dynamics.

    The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
    lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
    independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
    and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
    time-step is computed as the product of the two.

    This function performs the following steps:

    1. Pre-process the actions before stepping through the physics.
    2. Apply the actions to the simulator and step through the physics in a decimated manner.
    3. Compute the reward and done signals.
    4. Reset environments that have terminated or reached the maximum episode length.
    5. Apply interval events if they are enabled.
    6. Compute observations.

    Args:
        action: The actions to apply on the environment. Shape is (num_envs, action_dim).

    Returns:
        A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
    """
    action = action.to(self.device)
    # add action noise
    if self.cfg.action_noise_model:
        action = self._action_noise_model.apply(action)

    # process actions
    self._pre_physics_step(action)

    # check if we need to do rendering within the physics loop
    # note: checked here once to avoid multiple checks within the loop
    is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

    # perform physics stepping
    for _ in range(self.cfg.decimation):
        self._sim_step_counter += 1
        # set actions into buffers
        self._apply_action()
        # set actions into simulator
        self.scene.write_data_to_sim()
        # simulate
        self.sim.step(render=False)
        # render between steps only if the GUI or an RTX sensor needs it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
            self.sim.render()
        # update buffers at sim dt
        self.scene.update(dt=self.physics_dt)

    # post-step:
    # -- update env counters (used for curriculum generation)
    self.episode_length_buf += 1  # step in current episode (per env)
    self.common_step_counter += 1  # total step (common for all envs)

    # -- reset envs that terminated/timed-out and log the episode information
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
        self._reset_idx(reset_env_ids)
        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

    self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
    self.reset_buf = self.reset_terminated | self.reset_time_outs

    # post-step: step interval event
    if self.cfg.events:
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

    # update observations and rewards
    self.obs_buf = self._get_observations()
    self.reward_buf = self._get_rewards()

    # add observation noise
    # note: we apply no noise to the state space (since it is used for critic networks)
    if self.cfg.observation_noise_model:
        self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

    # return observations, rewards, resets and extras
    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


def get_linear_fn(x1:float, y1:float, x2:float, y2:float) -> Callable[[float],float]:

    def f(x) :
        x = max(x1, min(x2, x))
        return (x - x1) / (x2 - x1) * (y2 - y1) + y1

    return f

def get_linear_fn2(x1:float, y1:float, x2:float, y2:float, x3:float|None=None, y3:float|None=None) -> Callable[[float],float]:

    if x3 == None and y3 == None :
        return get_linear_fn(x1, y1, x2, y2)

    def f(x) :
        if x < x2 :
            x = max(x,x1)
            return (x - x1) / (x2 - x1) * (y2 - y1) + y1
        else :
            x = min(x,x3)
            return (x - x2) / (x3 - x2) * (y3 - y2) + y2

    return f