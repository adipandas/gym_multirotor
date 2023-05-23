"""
* QuadrotorPlusHoverEnv-v0
python enjoy.py --env-name "QuadrotorPlusHoverEnv-v0" --load-dir ./experiments_data/trained_models/QuadrotorPlusHoverEnv/QuadrotorPlusHoverEnv-0/ppo/

* TiltrotorPlus8DofHoverEnv-v0
python enjoy.py --env-name "TiltrotorPlus8DofHoverEnv-v0" \
--load-dir ./experiments_data/trained_models/TiltrotorPlus8DofHoverEnv/ConventionalTiltrotorPlus8DofHoverEnv-0/ppo/

python enjoy.py --env-name "TiltrotorPlus8DofHoverEnv-v0" \
--load-dir ./experiments_data/trained_models/TiltrotorPlus8DofHoverEnv/DevelopmentalTiltrotorPlus8DofHoverEnv-0/ppo/

"""

import argparse
import os
import sys
import torch
from ppo.envs import make_vec_envs
from ppo.utils import get_render_func, get_vec_normalize
import gym_multirotor


sys.path.append('ppo')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='QuadrotorPlusHoverEnv-v0',
    help='environment to train on (default: QuadrotorPlusHoverEnv-v0)')
parser.add_argument(
    '--load-dir',
    default='./experiments_data/trained_models/ppo/',
    help='directory to save agent logs (default: ./experiments_data/trained_models/ppo/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
args = parser.parse_args()

args.det = not args.non_det

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device='cpu',
    allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

actor_critic, obs_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"), map_location=lambda storage, loc: storage)

# We need to use the same statistics for normalization as used in training
vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.obs_rms = obs_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()

if render_func is not None:
    render_func('human')

t_max = 10
t_ = 0
while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    obs, reward, done, _ = env.step(action)
    masks.fill_(0.0 if done else 1.0)

    if done:
        t_ += 1
        env.reset()

    if render_func is not None:
        render_func('human')

    if t_ > t_max:
        break
