"""
Run a single experiment using following command in terminal:

python main_ppo.py --algo ppo --use-gae --log-interval 1 --num-steps 2000 --num-processes 1 \
--lr 5e-5 --entropy-coef 0 --clip-param 0.1 --value-loss-coef 0.5 --ppo-epoch 4 \
--num-mini-batch 32 --hidden-size 150 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 2000000 \
--no-cuda --use-proper-time-limits --seed 1000 --env-name QuadrotorPlusHoverEnv-v0 \
--log-dir ./experiments_data/logs/QuadrotorPlusHoverEnv/QuadrotorPlusHoverEnv-0 \
--save-dir ./experiments_data/trained_models/QuadrotorPlusHoverEnv/QuadrotorPlusHoverEnv-0/


python main_ppo.py --algo ppo --use-gae --log-interval 1 --num-steps 2000 --num-processes 1 \
--lr 5e-5 --entropy-coef 0 --clip-param 0.1 --value-loss-coef 0.5 --ppo-epoch 4 \
--num-mini-batch 32 --hidden-size 64 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 \
--no-cuda --use-proper-time-limits --seed 1000 --env-name TiltrotorPlus8DofHoverEnv-v0 \
--quadrotor-model-path ./experiments_data/trained_models/Quadrotor/QuadrotorPlusHoverEnv-1/ppo/QuadrotorPlusHoverEnv-v0.pt \
--log-dir ./experiments_data/logs/Tiltrotor/TiltrotorPlus8DofHoverEnv/Developmental_NN-0 \
--save-dir ./experiments_data/trained_models/TiltrotorPlus8DofHoverEnv/DevelopmentalTiltrotorPlus8DofHoverEnv-0/ \
--tiltrotor-branch-hidden-size 64

"""

import os
import time
from collections import deque
import numpy as np
import torch
from ppo import algo, utils
from ppo.arguments import get_args
from ppo.envs import make_vec_envs
from ppo.model import Policy
from ppo.storage import RolloutStorage
from ppo.evaluation import evaluate
import gym
import gym_multirotor


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    if args.env_name == "QuadrotorPlusHoverEnv-v0":
        base_kwargs = dict(recurrent=args.recurrent_policy,
                           hidden_size=args.hidden_size,
                           model="quadrotor")
    elif args.env_name == "TiltrotorPlus8DofHoverEnv-v0":
        base_kwargs = dict(recurrent=args.recurrent_policy,
                           hidden_size=args.hidden_size,
                           model="tiltrotor",
                           quadrotor_model_path=args.quadrotor_model_path,
                           quadrotor_weights_require_grad=(
                               not args.freeze_quadrotor_weights),
                           bhidden_size=args.tiltrotor_branch_hidden_size)
    else:
        base_kwargs = dict(recurrent=args.recurrent_policy,
                           hidden_size=args.hidden_size)

    _str_dashed_line_sepratr = "-"*46
    print(_str_dashed_line_sepratr)
    print("Model-base param:")
    print(base_kwargs)
    print(print(_str_dashed_line_sepratr))

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs=base_kwargs)
    actor_critic.to(device)

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        for step in range(args.num_steps):
            with torch.no_grad():
                # Sample actions
                (value, action, action_log_prob, recurrent_hidden_states) = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            filename_ = os.path.join(save_path, args.env_name + ".pt")
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], filename_)

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            str_len = 25
            print("-"*36)
            print("Steps"+" "*(str_len-len("Steps")) + "|" + f"{j:10d}")
            print("FPS"+" "*(str_len-len("FPS")) + "|" + f"{int(total_num_steps / (end - start)):10d}")
            print("#Episodes in batch"+" "*(str_len-len("#Episodes in batch")) + "|" + f"{len(episode_rewards):10d}")
            print("Mean Reward"+" "*(str_len-len("Mean Reward")) + "|" + f"{np.mean(episode_rewards):10.1f}")
            print("Median Reward"+" "*(str_len-len("Median Reward")) + "|" + f"{np.median(episode_rewards):10.1f}")
            print("Minimum Reward"+" "*(str_len-len("Minimum Reward")) + "|" + f"{np.min(episode_rewards):10.1f}")
            print("Maximum Reward"+" "*(str_len-len("Maximum Reward")) + "|" + f"{np.max(episode_rewards):10.1f}")
            print("Value Loss"+" "*(str_len-len("Value Loss")) + "|" + f"{value_loss:10.3f}")
            print("Action Loss"+" "*(str_len-len("Action Loss")) + "|" + f"{action_loss:10.3f}")
            print("Distribution Entropy"+" "*(str_len-len("Distribution Entropy")) + "|" + f"{dist_entropy:10.3f}")
            print("-"*36)

        if args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0:
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
