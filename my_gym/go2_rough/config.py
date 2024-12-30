
from omni.isaac.lab_assets import UNITREE_GO2_CFG
import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
import omni.isaac.lab.sim as sim_utils

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

from my_gym.base import BaseSceneCfg, BaseEnvCfg


TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        'pyramid_stairs': terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        'pyramid_stairs_inv': terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        'boxes': terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        'random_rough': terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        'hf_pyramid_slope': terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        'hf_pyramid_slope_inv': terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)


@configclass
class SceneCfg(BaseSceneCfg) :
    terrain = TerrainImporterCfg(
        prim_path='/World/ground',
        terrain_type='generator',
        terrain_generator=TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode='multiply',
            restitution_combine_mode='multiply',
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f'{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl',
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot = UNITREE_GO2_CFG

    def __post_init__(self) :
        self.robot.prim_path = '{ENV_REGEX_NS}/Robot'
        self.robot.spawn.articulation_props.enabled_self_collisions = True

        super().__post_init__()

        self.terrain.terrain_generator.sub_terrains['boxes'].grid_height_range = (0.025, 0.1)
        self.terrain.terrain_generator.sub_terrains['random_rough'].noise_range = (0.01, 0.06)
        self.terrain.terrain_generator.sub_terrains['random_rough'].noise_step = 0.01


@configclass
class Go2EnvCfg(BaseEnvCfg) :
    scene = SceneCfg()
    class reward(BaseEnvCfg.reward) :
        contact_pnt_names = (
            ['base'] + ['Head_upper', 'Head_lower'] +
            ['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip'] +
            ['FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh'] +
            ['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
        )
        joint_pos_pnt_names = ['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip']


@configclass
class Go2PlayEnvCfg(Go2EnvCfg) :
    class env(Go2EnvCfg.env) :
        n_envs = 128
    class command(Go2EnvCfg.command) :
        curriculum_kwargs = {
            'x1':0., 'y1':2.,
            'x2':1., 'y2':2.,
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
