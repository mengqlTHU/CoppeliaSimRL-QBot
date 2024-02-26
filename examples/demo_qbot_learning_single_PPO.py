from stable_baselines3 import A2C
from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

import sys
sys.path.append("../utils")
from callbackFunctions import VisdomCallback

sys.path.append("../QBot")
from QBotEnv import QBotEnv
from QBotEnv_T import QBotEnv_T
import gymnasium as gym

import os

if __name__ == "__main__":
    # ---------------- Create environment
    env = QBotEnv_T(port=30000) # action_type can be set as discrete or continuous


    # env = make_vec_env(QBotEnv, n_envs=2)

    # check_env(env)

    # ---------------- Callback functions
    log_dir = "../QBot/saved_models/tmp/PPO-Tcontrol"
    os.makedirs(log_dir, exist_ok=True)

    env = Monitor(env, log_dir)

    callback_visdom = VisdomCallback(name='visdom_qbot_ppo', check_freq=100, log_dir=log_dir)
    callback_save_best_model = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=500, deterministic=True, render=False)
    callback_list = CallbackList([callback_visdom, callback_save_best_model])


    # ---------------- Model
    # Option 1: create a new model
    # print("create a new model")
    # model = PPO(policy='MlpPolicy', env=env, learning_rate=3e-4, n_steps=512, n_epochs=10, verbose=True)

    # Option 2: load the model from files (note that the loaded model can be learned again)
    # print("load the model from files")
    # model = A2C.load("../QBot/saved_models/tmp/PPO/best_model", env=env)
    # model.learning_rate = 3e-4

    # Option 3: load the pre-trained model from files
    # print("load the pre-trained model from files")
    # if env.action_type == 'discrete':
    #     model = A2C.load("../CartPole/saved_models/best_model_discrete", env=env)
    # else:
    #     model = A2C.load("../CartPole/saved_models/best_model_continuous", env=env)


    # ---------------- Learning
    # print('Learning the model')
    # model.learn(total_timesteps=10000000, callback=callback_list) # 'MlpPolicy' = Actor Critic Policy
    # print('Finished')
    # del model # delete the model and load the best model to predict
    model = PPO.load("../QBot/saved_models/tmp/PPO-Tcontrol/best_model", env=env)


    # ---------------- Prediction
    print('Prediction')

    for _ in range(10):
        observation, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _state = model.predict(observation, deterministic=True)
            observation, reward, done, terminated, info = env.step(action)
            episode_reward += reward

        print([episode_reward, env.counts])

    env.close()
