
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.sensors import ContactSensorCfg
import omni.isaac.lab.sim as sim_utils

from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import RigidBodyMaterialCfg

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from my_gym.utils import POST_INIT

from gymnasium.spaces import Box
import numpy as np


@configclass
class BaseSceneCfg(InteractiveSceneCfg) :
    num_envs=POST_INIT
    env_spacing = 2.5

    sky_light = AssetBaseCfg(
        prim_path='/World/SkyLight',
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f'{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr',
        ),
    )
    terrain:TerrainImporterCfg = None
    robot:ArticulationCfg = None
    contact_sensor = ContactSensorCfg(
        history_length=POST_INIT,
        track_air_time=True
    )

    def __post_init__(self) :
        self.contact_sensor.prim_path = self.robot.prim_path+'/.*'


@configclass
class BaseEnvCfg(DirectRLEnvCfg) :
    sim = SimulationCfg(
        dt=POST_INIT,
        render_interval=POST_INIT,
        disable_contact_processing=True,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode='multiply',
            restitution_combine_mode='multiply'
        )
    )
    scene:BaseSceneCfg = None


    class env :
        physics_dt = 1/200
        decimation = 4
        episode_length_s = 20.0 # 1000 [step]

        n_envs = 4096
        n_obs = POST_INIT
        n_act = 12

        use_correct_step = True

    class observation :
        n_history = 6

    class action :
        scale = 0.25

    class termination :
        angle_threshold = 1.4 # 80degree
        body_names = []
        trunc_with_termin = True

    class reset :
        distribute = True

    class command :
        curriculum_kwargs = {
            'x1':0000.0, 'y1':1.0,
            'x2':4000.0, 'y2':2.0,
        }
        class x :
            w = 1.0
            b = 0.0
        class y :
            w = 0.5
            b = 0.0
        class z :
            w = 1.0
            b = 0.0
        heading = True
        sample_freq = 200

    class reward :
        k_track_lin = 1.2 #1.0
        k_track_lin_err_scale = 4.5 #4.0
        k_track_ang = 0.6 #0.5
        k_track_ang_err_scale = 3.0 #4.0
        k_pnt_lin = -2.0
        k_pnt_ang = -0.05
        k_contact = -0.3
        k_termin = -20.0
        k_pos = -0.03 #-0.1  OK
        k_trq = -1e-4 #-2e-4 OK
        k_acc = -5e-8 #-1e-7 OK
        k_action_diff_1 = -0.001

        joint_pos_pnt_names = []
        contact_pnt_names = []
        contact_cnt_max = 2

        soft_start_kwargs = {
            'x1':1000.0, 'y1':0.1,
            'x2':6000.0, 'y2':1.0,
        }
        soft_exploit_kwargs = {
            'x1':00000.0, 'y1':0.5,
            'x2':15000.0, 'y2':1.0,
        }

    class extra :
        visualize = False
        fps_track_buff_len = 120


    def __post_init__(self):

        self.decimation = self.env.decimation
        self.episode_length_s = self.env.episode_length_s

        self.sim.dt = self.env.physics_dt
        self.sim.render_interval = self.env.decimation

        self.scene.num_envs = self.env.n_envs
        self.scene.contact_sensor.history_length = self.env.decimation

        self.env.n_obs = (
            +3  #linvel
            +3  #angvel
            +3  #projgrav
            +12 #qpos
            +12 #qvel
            +12 #action
        ) * self.observation.n_history + 3 #cmd

        self.observation_space = Box(-np.inf, np.inf, shape=(self.env.n_obs,), dtype=np.float32)
        self.action_space = Box(-np.inf, np.inf, shape=(self.env.n_act,), dtype=np.float32)
