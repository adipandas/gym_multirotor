import gym
import gym_multirotor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

ENV_NAME = "QuadrotorXHoverEnv-v0"

# Parallel environments
vec_env = make_vec_env(ENV_NAME, n_envs=4)
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=f'log_{ENV_NAME}', learning_rate=0.00005, clip_range=0.1)
model.learn(total_timesteps=10000000)
model.save(f"ppo_{ENV_NAME}")
del model
vec_env.close()

env = gym.make(ENV_NAME)
model = PPO.load(f"PPO_{ENV_NAME}")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
