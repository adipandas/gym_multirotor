import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='On policy Deep-RL.')
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: ppo')
    parser.add_argument(
        '--lr', type=float, default=5e-5, help='Learning rate (default: 5e-5)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,
        help='how many training CPU processes to use (default: 1)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C/PPO (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.1,
        help='ppo clip parameter (default: 0.1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,
        help='log interval, one log per n updates (default: 1)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=20e6,
        help='number of environment steps to train (default: 2e6)')
    parser.add_argument(
        '--env-name',
        default='QuadrotorPlusHoverEnv-v0',
        help='environment to train on (default: QuadrotorPlusHoverEnv-v0)')
    parser.add_argument(
        '--log-dir',
        default='./experiments_data/logs',
        help='directory to save agent logs (default: ./experiments_data/logs)')
    parser.add_argument(
        '--save-dir',
        default='./experiments_data/trained_models/',
        help='directory to save agent logs (default: ./experiments_data/trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=True,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--exp-dir',
        action='store_true',
        default='exp_',
        help='experiment dir name for logging data in tensorboard(default: generated using env-name and datetime)')
    parser.add_argument(
        '--quadrotor-model-path',
        default=None,
        help='rl-agent trained on quadrotor used for transfering weights to tiltrotor'
    )
    parser.add_argument(
        '--freeze-quadrotor-weights',
        default=False,
        action='store_true',
        help='If True, transfer weights from quadrotor policy to tiltrotor policy and freeze the transfered weights. Default is `False`.'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=64,
        help='number of neurons in hidden layer of actor (default: 64)')
    parser.add_argument(
        '--robot-type',
        default="default",
        help='type of robot under consideration from the list ["quadrotor", "tiltrotor", "default"]')
    parser.add_argument(
        '--tiltrotor-branch-hidden-size',
        type=int,
        default=64,
        help='number of neurons in hidden layer of tiltrotor actor additional branch (default: 64)'
    )
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.exp_dir == "exp_":
        args.exp_dir = "./runs/" + args.exp_dir + args.env_name + "_" + str(args.seed)

    assert args.algo in ['ppo'], "Only ppo algo can be used. Other not configured."
    if args.recurrent_policy:
        assert args.algo in ['ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
