"""Script for the evaluation of a reinforcement learning policy.
Example
-------
In a terminal, run as:
    $ python evaluate.py --logdir ./train_log/<date-and-time-of-policy-training>
"""

import argparse
import os
import time

import numpy as np
from gym.envs.registration import register
from envs.HoverEnv import HoverEnv
from envs.SplineEnv import SplineEnv
from assets.util import sync
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from assets.logger import Logger
from scipy.spatial.transform import Rotation
register(
    id="HoverEnv-v0",
    entry_point="envs:HoverEnv",
)


if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='./train_log/05.12.2022_15.43.23')
    ARGS = parser.parse_args()

    env = DummyVecEnv([lambda: SplineEnv(gui=True)])

    #### Save directory ########################################
    log_dir = os.path.dirname(os.path.abspath(__file__)) + ARGS.logdir

    stats_path = log_dir + "/vec/vec_normalize.pkl"
    env = VecNormalize.load(stats_path, env)
    env.training = True
    env.norm_reward = True

    model = PPO.load(log_dir + '/ppo_hover', env=env)

    obs = env.reset()
    simtime = 0
    start = time.time()
    i = 0
    num_runs = 5
    logger = Logger(env.envs[0].episode_length, env.envs[0].timestep, num_runs)
    for num_run in range(num_runs):
        env.reset()
        logger.reset_counter()
        while simtime < 10:
            simtime = env.envs[0].data.time
            action, _states = model.predict(obs,
                                            deterministic=True  # OPTIONAL 'deterministic=False'
                                            )
            # action = [simtime * np.array([0.1, -0.1, 0.05])]
            obs, reward, done, info = env.step(action)
            if 'terminal_observation' in info[0]:
                obs = info[0]['terminal_observation']
            # print(action)
            if env.envs[0].gui:
            # if env.envs[0].gui:
                env.render()
                # sync(i, start, env.envs[0].traj_timestep)
                # time.sleep(0.1)

            # Log simulation
            obs = np.reshape(obs, (5, 18))
            for i_ in range(obs.shape[0]):
                obs_ = obs[i_, :]
                time = simtime + i_/obs.shape[0]*env.envs[0].traj_timestep
                quat = np.roll(Rotation.from_matrix(np.reshape(obs_[6:15], (3, 3))).as_quat(), 1)
                state = np.hstack([obs_[0:6], quat, obs_[15:18]])
                logger.log(time, state, env.envs[0].data.ctrl, current_run=num_run)
            i = i + 1
            if done:
                break

    env.close()
    logger.plot()
