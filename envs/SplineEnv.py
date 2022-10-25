import glfw
import gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import scipy.interpolate as interp
from gym import spaces
from scipy.spatial.transform import Rotation
from ctrl.GeomControl import GeomControl


class SplineEnv(gym.Env):
    """
    Gym environment to learn spline trajectory planning with a Crazyflie.
    Observations: Quadcopter state
    Rewards: Weighted quadcopter tracking error
    Actions: Spline trajectory parameters
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
        init_pos = self._init_pos_random(self.init_pos_mean, self.init_pos_max_dev)
        self.data.qpos[0:3] = init_pos

        # Set environment properties
        self._set_action_space()
        self._set_observation_space()
        self.episode_length = 4  # Episode length in seconds
        self.timestep = 0.01  # Timestep in seconds, timestep in XML is 0.005
        self.traj_timestep = 0.5  # Timestep of trajectory generation (new spline section)
        self.num_step = 0

        # initialize spline trajectories
        self.spline = self._init_spline(init_pos)
        self.spline_vel = [spl.derivative() for spl in self.spline]

        # initialize controller
        self.controller = GeomControl(self.model, self.data)

        if self.gui:
            self._init_gui()

    def reset(self, **kwargs):
        # initialize mujoco simulation
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos = np.array([0, 0, 8, 1, 0, 0, 0])
        init_pos = self._init_pos_random(self.init_pos_mean, self.init_pos_max_dev)
        self.data.qpos[0:3] = init_pos
        # initialize spline trajectories
        self.spline = self._init_spline(init_pos)
        self.spline_vel = [spl.derivative() for spl in self.spline]

        self.num_step = 0
        obs = np.tile(self.compute_obs(), 5)
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
        spline_idx = self.num_step + 3
        # Construct new spline from action
        act = 0.1 * action
        self._update_spline(act, spline_idx)

        obs = np.zeros((int(self.traj_timestep/self.timestep/10), 18))

        for i in range(int(self.traj_timestep/self.timestep)):
            self._compute_control()
            mujoco.mj_step(self.model, self.data, 2)
            if i % 10 == 0:
                obs[int(i/10)] = self.compute_obs()
        # obs = self.compute_obs()
        rew = self.compute_reward(obs, act)
        obs = np.reshape(obs, (90,))
        done = self.compute_done()
        self.num_step += 1
        return obs, rew, done, {}  # No info yet

    def compute_obs(self):
        quat = self.data.qpos[3:7]
        rot = Rotation.from_quat(np.roll(quat, -1)).as_matrix().reshape((9,))
        # subtract mean position
        pos = self.data.qpos[0:3] - np.array([0, 0, 8])
        return np.hstack((pos, self.data.qvel[0:3], rot, self.data.qvel[3:6]))

    def compute_reward(self, obs, act):
        reward = 0
        r_pos = 0.4
        r_ang_vel = 0.1
        num_obs = obs.shape[0]
        for i in range(num_obs):
            cur_time = self.data.time - (num_obs-i-1)/num_obs*self.traj_timestep
            pos_e = np.linalg.norm(obs[i, 0:3])
            ang_vel_e = np.linalg.norm(obs[i, 15:18])
            reward += (-r_pos * pos_e - r_ang_vel * ang_vel_e) * cur_time
            if pos_e > 3:
                reward = -200*(self.episode_length - cur_time)
                break
        return reward

    def compute_done(self):
        pos_e = np.linalg.norm(self.state_vector()[0:3] - np.array([0, 0, 8]))
        if self.data.time > self.episode_length-1e-6 or pos_e > 3:
            return True
        else:
            return False

    def _set_action_space(self):
        # bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        # low, high = bounds.T
        low = np.array([-1, -1, -1])
        high = np.array([1, 1, 1])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def state_vector(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat], dtype=np.float32)

    def _set_observation_space(self):
        low = np.full((90,), -float("inf"), dtype=np.float32)
        high = np.full((90,), float("inf"), dtype=np.float32)
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

    def _init_spline(self, init_pos):
        n = int(self.episode_length / self.traj_timestep) + 1  # number of "sections"
        k = 3  # degree of spline
        t_max = self.episode_length
        t = [np.hstack((np.zeros(3), np.linspace(0, t_max, n-2), t_max * np.ones(3))) for _ in range(3)]  # knots
        c = [np.hstack((init_pos[i] * np.ones(3), np.zeros(n-1))) for i in range(3)]  # coefs
        spl = [interp.BSpline(t_, c_, k) for t_, c_ in zip(t, c)]
        # [plt.plot(np.linspace(0, 4, 101), spl_(np.linspace(0, 4, 101))) for spl_ in spl]
        # plt.show()
        return spl

    def _update_spline(self, act, spline_idx):
        for dim in range(3):
            c = self.spline[dim].c
            c[spline_idx] = c[spline_idx - 1] + act[dim]
            self.spline[dim] = interp.BSpline.construct_fast(self.spline[dim].t, c, self.spline[dim].k)
        self.spline_vel = [spl.derivative() for spl in self.spline]
        # [plt.plot(np.linspace(0, 5, 101), spl_(np.linspace(0, 5, 101))) for spl_ in self.spline]
        # plt.show()
        # print(self.data.time)
        # print('-')

    def _compute_control(self):
        t = self.data.time
        target_pos = np.array([spl(t) for spl in self.spline])
        target_vel = np.array([spl(t) for spl in self.spline_vel])
        pos = self.data.qpos[0:3]
        quat = self.data.qpos[3:7]
        vel = self.data.qvel[0:3]
        ang_vel = self.data.qvel[3:6]
        self.data.ctrl = self.controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos, target_vel=target_vel)
