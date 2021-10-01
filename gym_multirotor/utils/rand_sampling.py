import math
import numpy as np
from scipy.stats import special_ortho_group
from scipy.spatial.transform import Rotation
from gym_multirotor.utils.rotation_transformations import rot2quat


def sampleSO3():
    """
    Sampling SO(3) from uniform distribution.
    This function will sample Rotation matrix from a Uniform distribution

    :return: rotation matrix sampled from uniform distribution as numpy array of shape (3,3).
    """

    rand_rot = special_ortho_group.rvs(3)       # uniform sample from SO(3)
    return rand_rot


def sample_unit3d():
    """
    Sample 3d unit vector from unifrom distribution.

    Reference:
    https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume

    :return: numpy 3D unit vector sampled from uniform distribution
    """

    phi = np.random.uniform(0, 2*np.pi)
    costheta = np.random.uniform(-1, 1)
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

# TODO: Fix the random distribution of unit vector
# def sample_unit3d(np_random):
#     """
#     Uniformly sample points on a unit sphere
#
#     Parameters
#     ----------
#     np_random : random number generator object
#
#     References
#     ----------
#     .. [1] Generating uniformly distributed numbers on a sphere.
#        [blog](http://corysimon.github.io/articles/uniformdistn-on-sphere/)
#
#     Returns
#     -------
#     sample : numpy.ndarray
#         Random 3d unit vector.
#     """
#     theta = 2 * np_random.pi * np_random.uniform(0, 1)
#
#     rd = np_random.uniform(0, 1)
#     phi = np.arccos(1 - 2 * rd)
#
#     x = np.sin(phi) * np.cos(theta)
#     y = np.sin(phi) * np.sin(theta)
#     z = np.cos(phi)
#
#     sample = np.array([x, y, z])
#     return sample
#
#
# def sample_quaternion(np_random):
#     """
#
#     Parameters
#     ----------
#     np_random :
#
#     Returns
#     -------
#
#     """
#     x, y, z = sample_unit3d(np_random)
#     roll, pitch, yaw = np.arccos(x), np.arccos(y), np.arccos(z)
#     quat = np.array(p.getQuaternionFromEuler([roll, pitch, yaw]))
#     return quat



def sample_quat():
    """
    This function samples quaternion from uniform random distribution

    :return: numpy unit vector of shape (4,) as quaternion in scalar first format
    """
    rot_mat = sampleSO3()
    quat = rot2quat(rot_mat)
    return quat


class UniformUnitVector:
    def __init__(self, random_state):
        self.np_random = random_state

    def unit_vector(self):
        """
        Uniformly sample points on a unit sphere

        Returns
        -------
        sample : numpy.ndarray
            Random 3d unit vector.
        """

        theta = 2 * math.pi * self.np_random.uniform(0, 1)

        rd = self.np_random.uniform(0, 1)
        phi = np.arccos(1 - 2 * rd)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        sample = np.array([x, y, z])
        return sample


class UniformRotation:
    """
    Sample rotation from uniform distribution.

    Parameters
    ----------
    random_state : numpy.random.RandomState
        Random number generator RandomState object.
    """

    def __init__(self, random_state):
        self.np_random = random_state

    def rotmat(self):
        rotmat = special_ortho_group.rvs(dim=3, random_state=self.np_random)
        return rotmat

    def quaternion(self):
        rmat = self.rotmat()
        quat = Rotation.from_matrix(rmat).as_quat()
        return quat

    def rpy(self, degrees=False):
        rmat = self.rotmat()
        euler = Rotation.from_matrix(rmat).as_euler(seq='xyz', degrees=degrees)
        return euler


if __name__ == '__main__':
    random_state = np.random.RandomState(seed=10)
    ur = UniformRotation(random_state=random_state)
    a1 = []
    for i in range(100):
        a1.append(ur.rpy(degrees=True))
    a1 = np.array(a1)

    random_state = np.random.RandomState(seed=10)
    ur = UniformRotation(random_state=random_state)
    a2 = []
    for i in range(100):
        a2.append(ur.rpy(degrees=True))
    a2 = np.array(a2)

    np.testing.assert_array_almost_equal(a1, a2)
    print('Test Successfull...')

    print(random_state.standard_normal(size=4))
