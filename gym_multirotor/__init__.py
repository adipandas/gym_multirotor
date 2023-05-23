from gym.envs.registration import register
from gym_multirotor import envs
from gym_multirotor.envs import mujoco
from gym_multirotor.envs.mujoco import *


# Quadrotors with Plus(+)-configuration
register(
    id='QuadrotorPlusHoverEnv-v0',
    entry_point='gym_multirotor.envs.mujoco.quadrotor_plus_hover:QuadrotorPlusHoverEnv',
    kwargs=dict(frame_skip=5, env_bounding_box=1.2, randomize_reset=True)
)

register(
    id='TiltrotorPlus8DofHoverEnv-v0',
    entry_point='gym_multirotor.envs.mujoco.tiltrotor_plus_hover:TiltrotorPlus8DofHoverEnv',
    kwargs=dict(frame_skip=5, env_bounding_box=1.2, randomize_reset=True)
)

# Quadrotor with X-configuration
register(
    id='QuadrotorXHoverEnv-v0',
    entry_point='gym_multirotor.envs.mujoco.quadrotor_x_hover:QuadrotorXHoverEnv',
    kwargs=dict(frame_skip=5, env_bounding_box=1.2, randomize_reset=True)
)
