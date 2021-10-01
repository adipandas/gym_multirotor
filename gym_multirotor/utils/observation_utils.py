import numpy as np
from gym_multirotor.utils.rotation_transformations import quat2rot


def get_ob_tiltrotor(mujoco_pos, mujoco_vel):
    """
    Convert mujoco observation with quaternion orientation to observation with orientation as
    a direction cosine matrix for tilt-rotor environments.
    This function returns the fully observed state of the tilt-rotor environment

    :param mujoco_pos: numpy array [shape (11,)] of position vector
    :param mujoco_vel: numpy array [shape (10,)] of velocity vector
    :return: numpy observation vector [shape (26,)]
    """

    pos = mujoco_pos[:3]
    quat = mujoco_pos[3:7]
    rot_mat = quat2rot(quat)
    rot_mat = rot_mat.flatten()
    tilt_angles = mujoco_pos[7:]
    return np.concatenate([pos, rot_mat, tilt_angles, mujoco_vel]).ravel()


def get_ob_quadrotor(mujoco_pos, mujoco_vel):
    """
    Function to convert mujoco observation with quaternion orientation to
    observation with orientation as direction cosine matrix for quad-copter environments

    :param mujoco_pos: numpy array [shape (7,)] of  position vector
    :param mujoco_vel: numpy array [shape (6,)] of velocity vector
    :return: numpy observation vector [shape (18,)]
    """

    pos = mujoco_pos[:3]                    # possibly this pos represents error

    quat = mujoco_pos[3:7]
    rot_mat = quat2rot(quat)
    rot_mat = rot_mat.flatten()

    return np.concatenate([pos, rot_mat, mujoco_vel]).ravel()


def get_partial_ob_tiltrotor(full_ob):
    """
    Function to evaluate partially observable state of the tilt-rotor uav environment

    :param full_ob: (numpy vector) complete observation vector of the environment
    :return observed_state: (numpy vector) partially observed state of the environment
    """

    e_pos = full_ob[:3]
    rot_mat = full_ob[3:12]
    vel = full_ob[16:19]
    angular_vel = full_ob[19:22]
    observed_state = np.concatenate([e_pos, rot_mat, vel, angular_vel]).ravel()
    return observed_state


def get_partial_observation(full_ob):
    """
    Function to evaluate partially observable state of the tilt-rotor uav environment
    Convenience function for `get_partial_ob_tiltrotor`

    :param full_ob: (numpy vector) complete observation vector of the environment
    :return observed_state: (numpy vector) partially observed state of the environment
    """
    return get_partial_ob_tiltrotor(full_ob)
