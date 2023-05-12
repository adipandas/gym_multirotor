import gym
import gym_multirotor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch


SEED = 123
ENV_NAMES = ["QuadrotorXHoverEnv-v0", "TiltrotorPlus8DofHoverEnv-v0", "QuadrotorPlusHoverEnv-v0"]

for ENV_NAME in ENV_NAMES:
    vec_env = make_vec_env(ENV_NAME, n_envs=8, seed=SEED)     # Parallel environments
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=f'log_{ENV_NAME}',
        policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256])),
        learning_rate=0.00005,
        clip_range=0.05,
        seed=SEED,
        batch_size=256,
        max_grad_norm=0.2
    )
    model.learn(total_timesteps=20000000)
    model.save(f"./policy/PPO_{ENV_NAME}")
    del model
    vec_env.close()
