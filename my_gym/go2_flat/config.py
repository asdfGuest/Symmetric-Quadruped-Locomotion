
from omni.isaac.lab_assets import UNITREE_GO2_CFG
from omni.isaac.lab.terrains import TerrainImporterCfg
import omni.isaac.lab.sim as sim_utils

from omni.isaac.lab.utils import configclass

from my_gym.base import BaseSceneCfg, BaseEnvCfg


@configclass
class SceneCfg(BaseSceneCfg) :
    terrain = TerrainImporterCfg(
        prim_path='/World/ground',
        terrain_type='plane',
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode='multiply',
            restitution_combine_mode='multiply',
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot = UNITREE_GO2_CFG

    def __post_init__(self) :
        self.robot.prim_path = '{ENV_REGEX_NS}/Robot'
        self.robot.spawn.articulation_props.enabled_self_collisions = True
        super().__post_init__()


@configclass
class Go2EnvCfg(BaseEnvCfg) :
    scene = SceneCfg()

    class command(BaseEnvCfg.command) :
        curriculum_kwargs = {
            'x1':0000.0, 'y1':1.0,
            'x2':4000.0, 'y2':2.4,
            'x3':8000.0, 'y3':3.0,
        }
        class x :
            w = 1.0
            b = 0.0
        class y :
            w = 0.5
            b = 0.0
        class z :
            w = 0.5
            b = 0.0
        heading = False
        sample_freq = 200

    class reward(BaseEnvCfg.reward) :
        contact_pnt_names = (
            ['base'] + ['Head_upper', 'Head_lower'] +
            ['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip'] +
            ['FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh'] +
            ['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
        )
        soft_start_kwargs = {
            'x1':0000.0, 'y1':0.1,
            'x2':2000.0, 'y2':1.0,
        }
        soft_exploit_kwargs = {
            'x1': 5000.0, 'y1':0.5,
            'x2':10000.0, 'y2':1.0,
        }


@configclass
class Go2PlayEnvCfg(Go2EnvCfg) :
    class env(Go2EnvCfg.env) :
        n_envs = 32
    class command(Go2EnvCfg.command) :
        curriculum_kwargs = {
            'x1':0., 'y1':3.,
            'x2':1., 'y2':3.,
        }
    class reward(Go2EnvCfg.reward) :
        soft_start_kwargs = {
            'x1':0., 'y1':1.,
            'x2':1., 'y2':1.,
        }
        soft_exploit_kwargs = {
            'x1':0., 'y1':1.,
            'x2':1., 'y2':1.,
        }
    class reset(Go2EnvCfg.reset) :
        distribute = False
    class extra(Go2EnvCfg.extra) :
        visualize = True
