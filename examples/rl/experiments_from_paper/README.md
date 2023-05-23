# Developmental reinforcement learning of control policy of a quadcopter UAV with thrust vectoring rotors

### **Paper**: [https://arxiv.org/abs/2007.07793](https://arxiv.org/abs/2007.07793)



# Install
```
sudo apt install tmux
pip install --user tmuxp

pip install numpy
pip install matplotlib
pip install pandas
pip install jupyterlab
pip install torch
pip install stable-baselines3[extra]
pip install gym==0.21.0
pip install ipympl
```

# To generate results like shown in the paper

1. Run the following commands:

    ```
    cd gym_multirotor/examples/rl/experiments_from_paper

    python generate_tmux_yaml.py \
    --num-seeds 5 \
    --env-names "QuadrotorPlusHoverEnv-v0" \
    --yaml-file run_all_QuadrotorPlusHoverEnv_v0.yaml \
    --python-interpreter-path $(which python)

    python generate_tmux_yaml.py \
    --num-seeds 5 \
    --env-names "TiltrotorPlus8DofHoverEnv-v0" \
    --yaml-file run_all_TiltrotorPlus8DofHoverEnv_v0.yaml \
    --python-interpreter-path $(which python)

    python generate_tmux_yaml.py \
    --num-seeds 5 \
    --env-names "TiltrotorPlus8DofHoverEnv-v0" \
    --training-type "developmental" \
    --yaml-file run_all_TiltrotorPlus8DofHoverEnv_v0_developmental.yaml \
    --python-interpreter-path $(which python)

    ```

2. Run each generated yaml file one after another as per given sequence below. This will train the policies for the multirotor environments:

    ```
    tmuxp load run_all_QuadrotorPlusHoverEnv_v0.yaml
    ```

    ```
    tmuxp load run_all_TiltrotorPlus8DofHoverEnv_v0.yaml
    ```

    ```
    tmuxp load run_all_TiltrotorPlus8DofHoverEnv_v0_developmental.yaml
    ```

3. Plot results using [``visualize.ipynb``](visualize.ipynb)

4. Run the trained policy.

    * QuadrotorPlusHoverEnv-v0
        ```
        python enjoy.py --env-name "QuadrotorPlusHoverEnv-v0" --load-dir ./experiments_data/trained_models/QuadrotorPlusHoverEnv/QuadrotorPlusHoverEnv-0/ppo/
        ```

    * TiltrotorPlus8DofHoverEnv-v0
        ```
        python enjoy.py --env-name "TiltrotorPlus8DofHoverEnv-v0" \
        --load-dir ./experiments_data/trained_models/TiltrotorPlus8DofHoverEnv/ConventionalTiltrotorPlus8DofHoverEnv-0/ppo/
        ```

    * TiltrotorPlus8DofHoverEnv-v0 Developmental policy
        ```
        python enjoy.py --env-name "TiltrotorPlus8DofHoverEnv-v0" \
        --load-dir ./experiments_data/trained_models/TiltrotorPlus8DofHoverEnv/DevelopmentalTiltrotorPlus8DofHoverEnv-0/ppo/
        ```

# References:
1. https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
2. https://github.com/openai/baselines
3. https://stable-baselines3.readthedocs.io
