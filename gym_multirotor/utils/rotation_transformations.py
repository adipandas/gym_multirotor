import math
import numpy as np
from scipy.spatial.transform import Rotation


"""
The rotations can of two types:
1. In a global frame of reference (also known as rotation w.r.t. fixed or extrinsic frame) 
2. In a body-centred frame of reference (also known as rotation with respect to current frame of reference.
It is also referred as rotation w.r.t. intrinsic frame).

For more details on intrinsic and extrinsic frames refer: https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations

Euler angles as ROLL-PITCH-YAW refer the following links: 
* [Tait–Bryan angles](https://en.wikipedia.org/wiki/Euler_angles#Tait–Bryan_angles#Conventions)
* [Euler angls as YAW-PITCH-ROLL](https://en.wikipedia.org/wiki/Euler_angles#Conventions_2)
* [Rotation using Euler Angles](https://adipandas.github.io/posts/2020/02/euler-rotation/)
* [scipy: ``from_euler``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html#scipy.spatial.transform.Rotation.from_euler)
* [scipy: ``as_euler``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html#scipy.spatial.transform.Rotation.as_euler)

To get the angles as yaw-pitch-roll we calculate rotation with intrinsic frame of reference.
1. In intrinsic frame we start with `yaw` to go from inertial frame `0` to frame `1`. 
2. Than do `pitch` in frame `1` to go from frame `1` to frame `2`.
3. Than do `roll` in frame `2` to go from frame `2` to body frame `3`.
"""

INTRINSIC_ROTATION = "ZYX"
EXTRINSIC_ROTATION = "xyz"


def add_gaussian_noise(vector, noise_mag):
    """
    Add gaussian noise to the input vector.

    :param vector: vector of n-dimensions
    :type vector: numpy.ndarray
    :param noise_mag: magnitude of gaussian noise to add to input vector
    :type noise_mag: float
    :return: vector of same dimensions as input vector
    :rtype: numpy.ndarray
    """

    vector = vector + np.random.randn(*vector.shape) * float(noise_mag)
    return vector


def euler2quat_raw(rpy):
    """
    Euler angles of roll, pitch, yaw in radians. Returns quaternion in scalar first format.

    :param rpy: vector of (roll, pitch, yaw) with shape (3,)
    :type rpy: numpy.ndarray
    :return: quaternion as (w, x, y, z) with shape (4,)
    :rtype: numpy.ndarray
    """

    roll, pitch, yaw = rpy

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def quat2euler_raw(quat):
    """
    Convert quaternion orientation to euler angles.

    :param quat: quaternion as (w, x, y, z) with shape (4,)
    :type quat: numpy.ndarray
    :return: vector of (roll, pitch, yaw) with shape (3,)
    :rtype: numpy.ndarray
    """

    w, x, y, z = quat

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(math.pi*0.5, sinp)     # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2. * (w * z + x * y)
    cosy_cosp = 1. - 2. * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def quat2euler(quat, noise_mag=0):
    """
    Convert quaternion to euler.

    :param quat: quaternion in scalar first format
    :type quat: numpy.ndarray
    :param noise_mag: magnitude of gaussian noise added to orientation along each axis in radians
    :type noise_mag: float
    :return: numpy array of euler angles as roll, pitch, yaw (x, y, z) in radians
    :rtype: numpy.ndarray
    """

    quat = np.roll(quat, -1)                        # convert to scalar last
    rot = Rotation.from_quat(quat)                  # rotation object

    euler_angles = rot.as_euler(INTRINSIC_ROTATION)

    if noise_mag:
        euler_angles = add_gaussian_noise(euler_angles, noise_mag)

    rpy = euler_angles[::-1]

    return rpy


def euler2quat(euler, noise_mag=0):
    """
    Euler angles are transformed to corresponding quaternion.

    :param euler: vector of euler angles with shape (3,) in the order of roll-pitch-yaw (XYZ) in radians
    :type euler: numpy.ndarray
    :param noise_mag: magnitude of gaussian noise added to orientation along each axis in radians
    :type noise_mag: float
    :return: quaternion vector in scalar first format with shape (4,)
    :rtype: numpy.ndarray
    """

    euler = np.array([euler[2], euler[1], euler[0]])                # convert to YAW-PITCH-ROLL

    if noise_mag:
        euler = add_gaussian_noise(euler, noise_mag)

    rot = Rotation.from_euler(INTRINSIC_ROTATION, euler)
    quat_scalar_last = rot.as_quat()
    quat = np.roll(quat_scalar_last, 1)
    return quat


def quat2rot(quat, noise_mag=0):
    """
    Method to convert quaternion vector to 3x3 direction cosine matrix.

    :param quat:  quaternion (in scalar first format)
    :type quat: numpy.ndarray
    :param noise_mag: (float) magnitude of gaussian noise added to orientation along each axis in radians
    :type noise_mag: float
    :return: rotation matrix SO(3)
    :rtype: numpy.ndarray
    """

    quat = np.roll(quat, -1)                                        # quaternion in scalar last format
    rot = Rotation.from_quat(quat)                                  # rotation object

    euler_angles = rot.as_euler(INTRINSIC_ROTATION)                 # yaw-pitch-roll
    if noise_mag:
        euler_angles = add_gaussian_noise(euler_angles, noise_mag)

    rot_ = Rotation.from_euler(INTRINSIC_ROTATION, euler_angles)    # yaw-pitch-roll
    rot_mat = rot_.as_matrix()                                         # direction cosine matrix 3x3

    return rot_mat


def rot2quat(rot_mat, noise_mag=0):
    """
    Method to convert rotation matrix (SO3) to quaternion

    :param rot_mat: direction cosine matrix of 3x3 dimensions
    :type rot_mat: numpy.ndarray
    :param noise_mag: magnitude of gaussian noise added to orientation along each axis in radians.
    :type noise_mag: float
    :return quat: quaternion (in scalar first format) with a shape (4,).
    :rtype: numpy.ndarray
    """

    rot = Rotation.from_matrix(rot_mat)

    euler_angles = rot.as_euler(INTRINSIC_ROTATION)                                 # yaw-pitch-roll
    if noise_mag:
        euler_angles = add_gaussian_noise(euler_angles, noise_mag)

    rot_ = Rotation.from_euler(INTRINSIC_ROTATION, euler_angles)                     # yaw-pitch-roll
    quat_scalar_last = rot_.as_quat()
    quat = np.roll(quat_scalar_last, 1)

    return quat


def euler2rot(euler, noise_mag=0):
    """
    Convert euler angles to rotation (direction cosine) matrix

    :param euler: vector with shape (3,) including euler angles as (roll, pitch, yaw) in radians
    :type euler: numpy.ndarray
    :param noise_mag: magnitude of gaussian noise included in euler angle
    :type noise_mag: float
    :return: rotation matrix of shape (3, 3)
    :rtype: numpy.ndarray
    """

    euler = np.array([euler[2], euler[1], euler[0]])   # convert roll-pitch-yaw to yaw-pitch-roll

    if noise_mag:
        euler = add_gaussian_noise(euler, noise_mag)

    rot_ = Rotation.from_euler(INTRINSIC_ROTATION, euler)
    rot_mat = rot_.as_matrix()
    return rot_mat


def rot2euler(rot_mat, noise_mag=0):
    """
    Convert rotation matrix (SO3) to euler angles

    :param rot_mat: rotation matrix of shape (3, 3)
    :type rot_mat: numpy.ndarray
    :param noise_mag: magnitude of gaussian noise included in euler angle
    :type noise_mag: float
    :return: euler angles as (roll, pitch, yaw) with shape (3,)
    :rtype: numpy.ndarray
    """
    rot_ = Rotation.from_matrix(rot_mat)
    euler_angles = rot_.as_euler(INTRINSIC_ROTATION)  # yaw-pitch-roll
    if noise_mag:
        euler_angles = add_gaussian_noise(euler_angles, noise_mag)

    rpy = np.array([euler_angles[2], euler_angles[1], euler_angles[0]])
    return rpy


def quat2euler_scipy(quat):
    quat = np.roll(quat, shift=-1)  # scalar last
    rpy = Rotation.from_quat(quat).as_euler('xyz')
    return rpy


def euler2quat_scipy(rpy):
    quat = Rotation.from_euler('xyz', rpy).as_quat()
    quat = np.roll(quat, shift=1)  # scalar first
    return quat


def rotmat_world2body_scipy(rpy):
    rotmat = Rotation.from_euler('xyz', rpy).as_matrix()
    return rotmat


def rotmat_pqr2euler_rate(rpy):
    rotmat = np.array([
        [1, np.sin(rpy[0])*np.tan(rpy[1]), np.cos(rpy[0])*np.tan(rpy[1])],
        [0, np.cos(rpy[0]), -np.sin(rpy[1])],
        [0, np.sin(rpy[0])/np.cos(rpy[1]), np.cos(rpy[0])/np.cos(rpy[1])]
    ])
    return rotmat


def cross(a, b):
    a_skew = np.array(
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    )
    return np.dot(a_skew, b)
