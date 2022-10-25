import glfw
import gym
import mujoco
import numpy as np
from gym import spaces
from scipy.spatial.transform import Rotation


class HoverEnv(gym.Env):
    """
    Gym environment to learn hovering with a Crazyflie.
    Observations: Quadcopter state
    Rewards: Weighted quadcopter state error
    Actions: Collective thrust + x,y,z torques
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 gui=False
                 ):
        self.gui = gui

        self.init_pos_mean = np.array([0, 0, 8])
        self.init_pos_max_dev = 0.1

        # Initialize MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("../assets/cf2.xml")
        self.data = mujoco.MjData(self.model)
        self.data.qpos = np.array([0, 0, 8, 1, 0, 0, 0])
        self.data.qpos[0:3] = self._init_pos_random(self.init_pos_mean, self.init_pos_max_dev)

        # Set environment properties
        self._set_action_space()
        self._set_observation_space()
        self.episode_length = 4  # Episode length in seconds
        self.timestep = 0.01  # Timestep in seconds, timestep in XML is 0.005

        if self.gui:
            self._init_gui()

        self.last_action = np.zeros((4,))

    def reset(self, **kwargs):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos = np.array([0, 0, 8, 1, 0, 0, 0])
        self.data.qpos[0:3] = self._init_pos_random(self.init_pos_mean, self.init_pos_max_dev)
        obs = self.compute_obs()
        return obs

    def render(self, **kwargs):
        if self.gui:
            viewport = mujoco.MjrRect(0, 0, 0, 0)
            viewport.width, viewport.height = glfw.get_framebuffer_size(self.window)
            mujoco.mjv_updateScene(self.model, self.data, self.option, pert=None, cam=self.camera,
                                   catmask=mujoco.mjtCatBit.mjCAT_ALL, scn=self.scene)
            mujoco.mjr_render(viewport, self.scene, self.context)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

    def close(self):
        if self.gui:
            glfw.terminate()

    def step(self, action):
        self.last_action = action
        self.data.ctrl[0] = action[0] * 0.5
        self.data.ctrl[1:4] = action[1:4] * 0.0005
        mujoco.mj_step(self.model, self.data, 2)
        obs = self.compute_obs()
        rew = self.compute_reward()
        done = self.compute_done()
        return obs, rew, done, {}  # No info yet

    def compute_obs(self):
        quat = self.data.qpos[3:7]
        rot = Rotation.from_quat(np.roll(quat, -1)).as_matrix().reshape((9,))
        # subtract mean position
        pos = self.data.qpos[0:3] - np.array([0, 0, 8])
        return np.hstack((pos, self.data.qvel[0:3], rot, self.data.qvel[3:6]))

    def compute_reward(self):
        pos_e = np.linalg.norm(self.state_vector()[0:3] - np.array([0, 0, 8]))
        ang_vel_e = np.linalg.norm(self.data.qvel[3:6])
        act_temp = self.last_action - np.array([0.55, 0, 0, 0])
        act_norm = np.linalg.norm(act_temp)
        r_pos = 0.4
        r_ang_vel = 0.1
        r_action = 0.1
        reward = (-r_pos * pos_e - r_ang_vel * ang_vel_e - r_action * act_norm) * self.data.time  # reward from sim-to-(multi)real
        if pos_e > 3:
            reward = -200*(self.episode_length - self.data.time)
        return reward

    def compute_done(self):
        pos_e = np.linalg.norm(self.state_vector()[0:3] - np.array([0, 0, 8]))
        if self.data.time > self.episode_length or pos_e > 3:
            return True
        else:
            return False

    def _set_action_space(self):
        # bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        # low, high = bounds.T
        low = np.array([0, -1, -1, -1])
        high = np.array([1, 1, 1, 1])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def state_vector(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat], dtype=np.float32)

    def _set_observation_space(self):
        low = np.full((18,), -float("inf"), dtype=np.float32)
        high = np.full((18,), float("inf"), dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        return self.observation_space

    def _init_gui(self):
        # Initialize the library
        if not glfw.init():
            return

        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(1280, 720, "Crazyflie in MuJoCo", None, None)
        if not self.window:
            glfw.terminate()
            return

        # Make the window's context current
        glfw.make_context_current(self.window)

        # initialize visualization data structures
        self.camera = mujoco.MjvCamera()
        self.camera.azimuth, self.camera.elevation = 170, -30
        self.camera.lookat, self.camera.distance = [0, 0, 8], 2
        self.option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=30)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

    @staticmethod
    def _init_pos_random(mean, max_dev):
        init_pos = mean + max_dev * (np.random.rand(3) * 2 - 1)
        return init_pos
