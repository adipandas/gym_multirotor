"""
Run this file to generate `yaml` file for running with Tmux.
You can run the generated YAML file with the command: ``tmuxp load <filename.yaml>``

Notes:
    Before training the developmental policy for tiltrotor, you will have to change the model path of quadrotor depending on the file structure. Provide the absolute path here.
        * For example: --quadrotor-model-path /home/my_pc/my_files/experiments_data/trained_models/Quadrotor/QuadrotorPlusHoverEnv-{experiment_number}/ppo/QuadrotorPlusHoverEnv-v0.pt


# Training Quadrotor with plus-configuration

```
python generate_tmux_yaml.py \
--num-seeds 5 \
--env-names "QuadrotorPlusHoverEnv-v0" \
--yaml-file run_all_QuadrotorPlusHoverEnv_v0.yaml \
--python-interpreter-path $(which python)
```

# Training Tiltrotor from scratch

```
python generate_tmux_yaml.py \
--num-seeds 5 \
--env-names "TiltrotorPlus8DofHoverEnv-v0" \
--yaml-file run_all_TiltrotorPlus8DofHoverEnv_v0.yaml \
--python-interpreter-path $(which python)
```

# Developmental learning in Tiltrotor Quadrotor

```
python generate_tmux_yaml.py \
--num-seeds 5 \
--env-names "TiltrotorPlus8DofHoverEnv-v0" \
--training-type "developmental" \
--yaml-file run_all_TiltrotorPlus8DofHoverEnv_v0_developmental.yaml \
--python-interpreter-path $(which python)
```

"""

import os
import argparse
import yaml

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--num-seeds',
                    type=int,
                    default=5,
                    help='Number of random seeds to generate')
parser.add_argument('--env-names',
                    default="QuadrotorPlusHoverEnv-v0",
                    help='Environment name separated by semicolons')
parser.add_argument('--yaml-file',
                    default="run_all",
                    help='Name of the yaml file to run using tmuxp (eg. `tmuxp load run_all.yaml`)')
parser.add_argument('--training-type',
                    default="default",
                    help='type of training (eg. ["default", "developmental"])')
parser.add_argument('--developmental-freeze-weights',
                    default=False,
                    action='store_true',
                    help='Freeze weights in developmental policy if True. Default is False.')
parser.add_argument('--python-interpreter-path',
                    default="python",
                    help='Python interpreter path. Example `/home/this_pc/miniconda3/envs/workenv/bin/python`. Default is python.')

args = parser.parse_args()

_this_file_dir = os.path.dirname(os.path.abspath(__file__))


ppo_mujoco_template = args.python_interpreter_path + " " + _this_file_dir + "/main_ppo.py --algo ppo --use-gae --log-interval 1 " \
                      "--num-steps 2000 --num-processes 1 --lr 5e-5 --entropy-coef 0 " \
                      "--clip-param 0.1 --value-loss-coef 0.5 --ppo-epoch 4 --num-mini-batch 32 " \
                      "--hidden-size 150 " \
                      "--gamma 0.99 --gae-lambda 0.95 --num-env-steps 20000000 --no-cuda --use-proper-time-limits " \
                      "--seed {3} " \
                      "--env-name {0} "


if args.env_names == "TiltrotorPlus8DofHoverEnv-v0":
    if args.training_type == "developmental":
        quadrotor_model_path = _this_file_dir + "/experiments_data/trained_models/QuadrotorPlusHoverEnv/QuadrotorPlusHoverEnv-{2}/ppo/QuadrotorPlusHoverEnv-v0.pt "
        ppo_mujoco_template += " --quadrotor-model-path " + quadrotor_model_path + \
                               "--log-dir " + _this_file_dir + "/experiments_data/logs/{1}/Developmental_NN-{2} " \
                               "--save-dir " + _this_file_dir + "/experiments_data/trained_models/{1}/Developmental{1}-{2}/ " \
                               "--tiltrotor-branch-hidden-size 64 "

        if args.developmental_freeze_weights:
            ppo_mujoco_template += "--freeze-quadrotor-weights "
    else:
        ppo_mujoco_template += "--log-dir " + _this_file_dir + "/experiments_data/logs/{1}/Conventional_NN-{2} " \
                              "--save-dir " + _this_file_dir + "/experiments_data/trained_models/{1}/Conventional{1}-{2}/ "
else:
    ppo_mujoco_template += "--log-dir " + _this_file_dir + "/experiments_data/logs/{1}/{1}-{2} " \
                           "--save-dir " + _this_file_dir + "/experiments_data/trained_models/{1}/{1}-{2}/ "


template = ppo_mujoco_template

config = {"session_name": "run-all-"+args.env_names, "windows": []}

for i in range(args.num_seeds):
    panes_list = []
    for env_name in args.env_names.split(';'):
        env_name_prefix_for_dirname = env_name.split('-')[0]
        experiment_count = i
        random_seed = i+1000
        panes_list.append(template.format(env_name,
                                          env_name_prefix_for_dirname,
                                          experiment_count,
                                          random_seed))

    config["windows"].append({"window_name": "seed-{}".format(i),
                              # "shell_command_before": "conda_activate && cd " + _this_file_dir,
                              "panes": panes_list})

file_name = args.yaml_file.split(".")[0] + ".yaml"
yaml.dump(config, open(file_name, "w"), default_flow_style=False)
