
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sensors import ContactSensor
import omni.isaac.lab.utils.math as math_utils

import my_gym.torch_utils as th_utils
from my_gym.utils import my_direct_rl_env_step, get_linear_fn2, HistoryBuffer

from my_gym.base.config import BaseEnvCfg
from my_gym.base.visualizator import VelocityVisualizator

import torch as th
import time

from _collections_abc import Sequence


class BaseEnv(DirectRLEnv) :

    def __init__(self, cfg:BaseEnvCfg) :
        super().__init__(cfg)
        self.cfg:BaseEnvCfg = self.cfg
        self.n_envs = self.cfg.env.n_envs
        self.extras['info'] = {}
        self.first_step = True

        self.dt_buff = []
        self.dt_buff_ptr = 0
        self.current_time = time.time()
        self.last_time = None

        self.termin_ids = self.contact_sensor.find_bodies(self.cfg.termination.body_names)[0]
        self.contact_pnt_ids = self.contact_sensor.find_bodies(self.cfg.reward.contact_pnt_names)[0]
        self.joint_pos_pnt_ids = self.robot.find_bodies(self.cfg.reward.joint_pos_pnt_names)[0]

        sample_freq = self.cfg.command.sample_freq
        self.command_sample_env_ids = [[] for _ in range(sample_freq)]
        for idx, val in enumerate(th.randperm(self.n_envs).tolist()) :
            self.command_sample_env_ids[idx%sample_freq].append(val)
        for idx in range(sample_freq) :
            self.command_sample_env_ids[idx].sort()
            self.command_sample_env_ids[idx] = th.tensor(self.command_sample_env_ids[idx], dtype=th.int32, device=self.device)

        self.cmd = th_utils.th_zeros_f((self.n_envs,4))
        self.curriculum = get_linear_fn2(**self.cfg.command.curriculum_kwargs)
        self.soft_start = get_linear_fn2(**self.cfg.reward.soft_start_kwargs)
        self.soft_exploit = get_linear_fn2(**self.cfg.reward.soft_exploit_kwargs)
        self.visualizator = VelocityVisualizator(self.n_envs,self.device) if self.cfg.extra.visualize else None

        self.worldgrav = -th_utils.worldvec_u.repeat(self.n_envs,1)
        th_zeros_f_3 = th_utils.th_zeros_f((3,))
        th_zeros_f_12 = th_utils.th_zeros_f((12,))

        self.action = th_utils.th_zeros_f((self.n_envs,12))
        self.last_action = th_utils.th_zeros_f((self.n_envs,12))

        n_history = self.cfg.observation.n_history
        self.obs_linvel = HistoryBuffer(self.n_envs, n_history, th_zeros_f_3)
        self.obs_angvel = HistoryBuffer(self.n_envs, n_history, th_zeros_f_3)
        self.obs_projgrav = HistoryBuffer(self.n_envs, n_history, th_zeros_f_3)
        self.obs_qpos = HistoryBuffer(self.n_envs, n_history, th_zeros_f_12)
        self.obs_qvel = HistoryBuffer(self.n_envs, n_history, th_zeros_f_12)
        self.obs_action = HistoryBuffer(self.n_envs, n_history, th_zeros_f_12)


    def _setup_scene(self) :
        self.robot:Articulation = self.scene['robot']
        self.contact_sensor:ContactSensor = self.scene['contact_sensor']


    def _pre_physics_step(self, actions: th.Tensor):
        self.last_action = self.action
        self.action = actions.clone()
        self.target_joint_pos = th.clip(
            input=self.robot.data.default_joint_pos + self.action * self.cfg.action.scale,
            min=self.robot.data.default_joint_limits[:,:,0],
            max=self.robot.data.default_joint_limits[:,:,1]
        )


    def _apply_action(self):
        self.robot.set_joint_position_target(self.target_joint_pos)


    def _update_command(self) :
        x_max = self.cfg.command.x.w * self.curriculum_rate + self.cfg.command.x.b
        y_max = self.cfg.command.y.w * self.curriculum_rate + self.cfg.command.y.b
        z_max = self.cfg.command.z.w * self.curriculum_rate + self.cfg.command.z.b

        if self.first_step :
            env_ids = list(range(self.n_envs))
        else :
            env_ids = self.command_sample_env_ids[self.common_step_counter % self.cfg.command.sample_freq]

        cmd = self.cmd[env_ids]
        cmd = (th_utils.th_rand_f(cmd.shape) - .5) * 2.0
        cmd[:,0] *= x_max
        cmd[:,1] *= y_max
        cmd[:,2] *= z_max
        cmd[:,3] *= th.pi
        self.cmd[env_ids] = cmd

        if self.cfg.command.heading :
            forward = math_utils.quat_apply(self.robot.data.root_quat_w, th_utils.worldvec_f.unsqueeze(0).expand(self.n_envs,3))
            heading = th.atan2(forward[:,1],forward[:,0])
            diff = (self.cmd[:,3]-heading) % (th.pi*2)
            diff -= (th.pi*2) * (diff>th.pi)
            self.cmd[:,2] = th.clip(diff,-2.,2.) * .5 * z_max


    def _pre_observations(self) :
        self.obs_linvel.push(self.robot.data.root_lin_vel_b)
        self.obs_angvel.push(self.robot.data.root_ang_vel_b)
        self.obs_projgrav.push(self.robot.data.projected_gravity_b)
        self.obs_qpos.push(self.robot.data.joint_pos)
        self.obs_qvel.push(self.robot.data.joint_vel)
        self.obs_action.push(self.action)

        self.projgrav = self.robot.data.projected_gravity_b
        self.contact = (self.contact_sensor.data.net_forces_w_history.norm(dim=-1) >
                        self.contact_sensor.cfg.force_threshold).max(dim=1)[0]

        self.curriculum_rate = self.curriculum(self.common_step_counter)
        self.soft_start_rate = self.soft_start(self.common_step_counter)
        self.soft_exploit_rate = self.soft_exploit(self.common_step_counter)

        self.extras['info']['curriculum'] = th.tensor(self.curriculum_rate)
        self.extras['info']['soft start'] = th.tensor(self.soft_start_rate)
        self.extras['info']['soft exploit'] = th.tensor(self.soft_exploit_rate)
        if self.scene.terrain.terrain_origins is not None :
            self.extras['info']['terrain level'] = self.scene.terrain.terrain_levels.to(th.float32).mean()
        if len(self.dt_buff) == self.cfg.extra.fps_track_buff_len :
            self.extras['info']['fps'] = th.tensor(len(self.dt_buff)/sum(self.dt_buff))

        self._update_command()
        if self.visualizator is not None :
            self.visualizator.update(
                xy_cmd=self.cmd[:,:2],
                root_pos_w=self.robot.data.root_pos_w,
                root_quat_w=self.robot.data.root_quat_w,
                root_linvel_b=self.robot.data.root_lin_vel_b
            )


    def _get_observations(self) :
        self._pre_observations()
        return (
            self.obs_linvel.get_data(),
            self.obs_angvel.get_data(),
            self.obs_projgrav.get_data(),
            self.cmd[:,:3],
            self.obs_qpos.get_data(),
            self.obs_qvel.get_data(),
            self.obs_action.get_data()
        )


    def _get_rewards(self) :
        lin_sq_error = th_utils.sq_norm(self.robot.data.root_lin_vel_b[:,:2]-self.cmd[:,0:2])
        ang_sq_error = th_utils.sq_norm(self.robot.data.root_ang_vel_b[:,2:]-self.cmd[:,2:3])
        n_contact = self.contact[:,self.contact_pnt_ids].sum(dim=-1)
        delta_action = self.action - self.last_action
        joint_pos = self.robot.data.joint_pos - self.robot.data.default_joint_pos

        r_track_lin = th.exp(-self.cfg.reward.k_track_lin_err_scale * lin_sq_error) * self.cfg.reward.k_track_lin
        r_track_ang = th.exp(-self.cfg.reward.k_track_ang_err_scale * ang_sq_error) * self.cfg.reward.k_track_ang
        r_pnt_lin = th_utils.sq_norm(self.robot.data.root_lin_vel_b[:,2:]) * self.cfg.reward.k_pnt_lin
        r_pnt_ang = th_utils.sq_norm(self.robot.data.root_ang_vel_b[:,:2]) * self.cfg.reward.k_pnt_ang
        r_contact = n_contact.clip(max=self.cfg.reward.contact_cnt_max) * self.cfg.reward.k_contact
        r_termin = self.terminated * self.cfg.reward.k_termin

        reward_common = {
            'r_track_lin' : r_track_lin,
            'r_track_ang' : r_track_ang,
            'r_pnt_lin' : r_pnt_lin * self.soft_start_rate,
            'r_pnt_ang' : r_pnt_ang * self.soft_start_rate,
            'r_contact' : r_contact * self.soft_start_rate,
            'r_termin' : r_termin * self.soft_start_rate,
        }

        r_pos = th_utils.sq_norm(joint_pos[:,self.joint_pos_pnt_ids]) * self.cfg.reward.k_pos
        r_trq = th_utils.sq_norm(self.robot.data.applied_torque) * self.cfg.reward.k_trq
        r_acc = th_utils.sq_norm(self.robot.data.joint_acc) * self.cfg.reward.k_acc
        r_action_diff_1 = th_utils.sq_norm(delta_action) * self.cfg.reward.k_action_diff_1

        reward_foot = {
            'r_pos' : r_pos,
            'r_trq' : r_trq,
            'r_acc' : r_acc,
            'r_action_diff_1' : r_action_diff_1,
        }
        for k in reward_foot :
            reward_foot[k] *= self.soft_start_rate * self.soft_exploit_rate

        self.extras['info'].update({k:th.mean(v) for k, v in reward_common.items()})
        self.extras['info'].update({k:th.mean(v) for k, v in reward_foot.items()})

        reward_common = th_utils.sum_tensor_list(list(reward_common.values()))
        reward_foot = th_utils.sum_tensor_list(list(reward_foot.values()))
        return reward_common, reward_foot, None, None


    def _get_dones(self) :
        self.termin_dict = {
            'angle' : th_utils.vec_ang(self.projgrav,self.worldgrav) > self.cfg.termination.angle_threshold,
            'contact' : th.any(self.contact[:,self.termin_ids],dim=-1)
        }
        terminated = th_utils.apply_tensor_list(list(self.termin_dict.values()),'or')
        truncated = self.episode_length_buf >= self.max_episode_length

        self.terminated = terminated.clone()
        self.extras['info']['terminated'] = th.sum(terminated)
        self.extras['info']['truncated'] = th.sum(truncated)

        if self.cfg.termination.trunc_with_termin :
            terminated |= truncated
        return terminated, truncated


    def _update_terrain(self, env_ids:Sequence[int]) :
        if self.scene.terrain.terrain_origins is None :
            return
        dist = (self.robot.data.root_pos_w[env_ids,:2]-self.scene.terrain.env_origins[env_ids,:2]).norm(dim=-1)
        move_up = dist > 4.0
        move_down = dist < th.norm(self.cmd[env_ids,:2],dim=-1)*self.max_episode_length_s*0.5
        self.scene.terrain.update_env_origins(
            env_ids,
            move_up=move_up,
            move_down=move_down*(~move_up)
        )


    def _reset_idx(self, env_ids:Sequence[int]) :
        super()._reset_idx(env_ids)

        self.action[env_ids,:] = 0.0
        self.last_action[env_ids,:] = 0.0
        self.obs_linvel.reset(env_ids)
        self.obs_angvel.reset(env_ids)
        self.obs_projgrav.reset(env_ids)
        self.obs_qpos.reset(env_ids)
        self.obs_qvel.reset(env_ids)
        self.obs_action.reset(env_ids)

        if len(env_ids) == self.n_envs and self.cfg.reset.distribute :
            self.episode_length_buf[:] = th.randint(low=0, high=self.max_episode_length, size=(self.n_envs,), device=self.device)

        self._update_terrain(env_ids)
        # root state
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:,:3] += self.scene.terrain.env_origins[env_ids]
        self.robot.write_root_state_to_sim(root_state, env_ids)
        # joint state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def step(self, action: th.Tensor) :
        self.last_time = self.current_time
        self.current_time = time.time()
        if len(self.dt_buff) < self.cfg.extra.fps_track_buff_len :
            self.dt_buff.append(0.)
        self.dt_buff[self.dt_buff_ptr] = self.current_time - self.last_time
        self.dt_buff_ptr = (self.dt_buff_ptr + 1) % len(self.dt_buff)

        if self.cfg.env.use_correct_step :
            returns = my_direct_rl_env_step(self, action)
        else :
            returns = super().step(action)

        self.first_step = False
        return returns
