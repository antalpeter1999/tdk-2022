"""Script for the training of a reinforcement learning policy.
Example
-------
In a terminal, run as:
    $ python train.py
"""

import argparse
import os
import torch
from gym.envs.registration import register
from envs.HoverEnv import HoverEnv
from envs.SplineEnv import SplineEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from datetime import datetime
from assets.util import linear_schedule


register(
    id="HoverEnv-v0",
    entry_point="envs:HoverEnv",
)
register(
    id="SplineEnv-v0",
    entry_point="envs:SplineEnv",
)

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_path', type=str, default='----./train_log/05.02.2022_17.51.29')
    ARGS = parser.parse_args()

    #### Save directory ########################################
    log_dir = os.path.dirname(os.path.abspath(__file__)) + '/train_log/' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir + '/')

    model_kwargs = dict(activation_fn=torch.nn.ReLU,
                        net_arch=[128, 128, dict(vf=[128, 128], pi=[128, 128])]
                        )
    env = DummyVecEnv([lambda: SplineEnv(gui=False) for _ in range(20)])
    eval_env = DummyVecEnv([lambda: SplineEnv(gui=False)])

    if os.path.isfile(ARGS.init_path + '/ppo_spline.zip'):  # continue training
        path = ARGS.init_path + '/ppo_spline.zip'
        stats_path = ARGS.init_path + '/vec/vec_normalize.pkl'
        env = VecNormalize.load(stats_path, env)
        env.training = True
        env.norm_reward = True
        eval_env = VecNormalize.load(stats_path, eval_env)
        eval_env.norm_reward = True
        model = PPO.load(path,
                         env,
                         policy_kwargs=model_kwargs,
                         learning_rate=3e-4,
                         tensorboard_log=log_dir + '/tb/',
                         verbose=1
                         )
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
        model = PPO('MlpPolicy',
                    env,
                    policy_kwargs=model_kwargs,
                    learning_rate=3e-4,
                    tensorboard_log=log_dir + '/tb/',
                    verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 verbose=1,
                                 best_model_save_path=log_dir,
                                 log_path=log_dir,
                                 eval_freq=int(2000),
                                 deterministic=True,
                                 render=False
                                 )
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=log_dir + '/', name_prefix='ppo_spline')

    model.learn(total_timesteps=int(1e7),
                callback=[checkpoint_callback, eval_callback],
                log_interval=100)

    model.save(log_dir + "/ppo_spline")
    os.makedirs(log_dir + "/vec/")
    stats_path = log_dir + "/vec/vec_normalize.pkl"
    env.save(stats_path)
