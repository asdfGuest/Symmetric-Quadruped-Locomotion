import torch as th

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils

from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


_GOAL_VEL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path='/Visuals/Command/velocity_goal',
    markers={
        'arrow': sim_utils.UsdFileCfg(
            usd_path=f'{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd',
            scale=(0.5, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        )
    }
)
_BODY_VEL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path='/Visuals/Command/velocity_body',
    markers={
        'arrow': sim_utils.UsdFileCfg(
            usd_path=f'{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd',
            scale=(0.5, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        )
    }
)


class VelocityVisualizator() :
    def __init__(
            self,
            n_envs:int,
            device:th.device,
        ) :
        self.n_envs = n_envs
        self.device = device
        self.goal_vel_marker = VisualizationMarkers(_GOAL_VEL_MARKER_CFG)
        self.body_vel_marker = VisualizationMarkers(_BODY_VEL_MARKER_CFG)
        self.vel_marker_default_scale = th.tensor([[1.0, 0.7, 0.7]], dtype=th.float32, device=self.device).expand(self.n_envs,3)

    def update(
            self,
            xy_cmd:th.Tensor,
            root_pos_w:th.Tensor,
            root_quat_w:th.Tensor,
            root_linvel_b:th.Tensor
        ) :
        marker_pos = root_pos_w.clone()
        marker_pos[:,2] += .5

        # goal velocity marker
        cmd_yaw = th.atan2(xy_cmd[:,1],xy_cmd[:,0])
        yaw_like_zero = th.zeros_like(cmd_yaw)
        marker_rot = math_utils.quat_mul(
            root_quat_w,
            math_utils.quat_from_euler_xyz(yaw_like_zero, yaw_like_zero, cmd_yaw)
        )
        marker_scale = self.vel_marker_default_scale.clone()
        marker_scale[:,0] *= xy_cmd.norm(dim=-1)

        self.goal_vel_marker.visualize(
            translations=marker_pos,
            orientations=marker_rot,
            scales=marker_scale
        )

        # body velocity marker
        linve_b_yaw = th.atan2(root_linvel_b[:,1],root_linvel_b[:,0])
        marker_rot = math_utils.quat_mul(
            root_quat_w,
            math_utils.quat_from_euler_xyz(yaw_like_zero, yaw_like_zero, linve_b_yaw)
        )
        marker_scale = self.vel_marker_default_scale.clone()
        marker_scale[:,0] *= root_linvel_b[:,:2].norm(dim=-1)

        self.body_vel_marker.visualize(
            translations=marker_pos,
            orientations=marker_rot,
            scales=marker_scale
        )
