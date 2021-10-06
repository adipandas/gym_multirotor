# *************************************************************************
# This file is a heavily modified version of the following code:
# https://github.com/ethz-asl/reinmav-gym/blob/master/gym_reinmav/envs/mujoco/mujoco_quad.py
# The corresponding copyright notice is provided below.
#
# Copyright (c) 2019, Autonomous Systems Lab
# Author: Dongho Kang <eastsky.kang@gmail.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# *************************************************************************


import os
import math
from abc import ABC
import numpy as np
from gym import spaces
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym_multirotor import utils as multirotor_utils


class UAVBaseEnv(mujoco_env.MujocoEnv, utils.EzPickle, ABC):
    """
    Base environment for UAV. Abstract class. Need to implement ``reset_model`` in every subclass along with few other methods..

    Args:
        xml_name (str): Name of the robot description xml file.
        frame_skip (int): Number of steps to skip after application of control command.
        error_tolerance (float): Error tolerance. Default is `0.05`.
        max_time_steps (int): Maximum number of timesteps in each episode. Default is `2500`.
        randomize_reset (bool): If `True`, initailize the environment with random state at the start of each episode. Default is `True`.
        disorient (bool): If True, random initialization and random orientation of the system at start of each episode. Default is ``True``.
        sample_SO3 (bool): If True, sample orientation uniformly from SO3 for random initialization of episode. Default is `False`.
        observation_noise_std (float): Standard deviation of noise added to observation vector. If zero, no noise is added to observation. Default is `0.`. If non-zero, a vector is sampled from normal distribution and added to observation vector.
        reduce_heading_error (bool): If `True`, reward function tries to reduce the heading error in orientation of the system. Default is `True`.
        env_bounding_box (float): Max. initial position error. Use to initialize the system in bounded space. Default is `1.2`.
        init_max_vel (float): Max. initial velocity error. Use to initialize the system in bounded space. Default is ``0.5``.
        init_max_angular_vel (float): Max. initial angular velocity error. Use to initialize the system in bounded space. Default is `pi/10`.
        init_max_attitude (float): Max. initial attitude error. Use to initialize the system in bounded space. Default is `pi/3.`.
        bonus_to_reach_goal (float): Bonus value or reward when RL agent takes the system to the goal state. Default is ``15.0``.
        max_reward_for_velocity_towards_goal (float): Max. reward possible when the agent is heading towards goal, i.e., velocity vector points towards goal direction. Default is ``2.0``.
        position_reward_constant (float): Position reward constant coefficient. Default is ``5.0``.
        orientation_reward_constant (float): Orientation reward constant coefficient. Default is ``0.02``.
        linear_velocity_reward_constant (float): Linear velocity reward constant. Default is ``0.01``.
        angular_velocity_reward_constant (float): Angular velocity reward constant. Default is ``0.001``.
        action_reward_constant (float): Action reward coefficient. Default is ``0.0025``.
        reward_for_staying_alive (float): Reward for staying alive in the environment. Default is ``5.0``.
        reward_scaling_coefficient (float): Reward multiplication factor which can be used to scale the value of reward to be greater or smaller. Default is ``1.0``.
    """

    obs_xyz_index = np.arange(0, 3)
    obs_rot_mat_index = np.arange(3, 12)
    obs_vel_index = np.arange(12, 15)
    obs_avel_index = np.arange(15, 18)

    action_index_thrust = np.arange(0, 4)

    def __init__(self,
                 xml_name="quadrotor_plus.xml",
                 frame_skip=5,
                 error_tolerance=0.05,
                 max_time_steps=1000,
                 randomize_reset=True,
                 disorient=True,
                 sample_SO3=False,
                 observation_noise_std=0,
                 reduce_heading_error=True,
                 env_bounding_box=1.2,
                 init_max_vel=0.5,
                 init_max_angular_vel=0.1*math.pi,
                 init_max_attitude=math.pi/3.0,
                 bonus_to_reach_goal=15.0,
                 max_reward_for_velocity_towards_goal=2.0,
                 position_reward_constant=5.0,
                 orientation_reward_constant=0.02,
                 linear_velocity_reward_constant=0.01,
                 angular_velocity_reward_constant=0.001,
                 action_reward_constant=0.0025,
                 reward_for_staying_alive=5.0,
                 reward_scaling_coefficient=1.0
                 ):
        xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets", xml_name))

        self.error_tolerance = error_tolerance
        """float: Error tolerance. Default is `0.05`."""

        self.max_time_steps = max_time_steps
        """
        int: Maximum number of timesteps in each episode. Default is `2500`.
        """

        # episode initialization parameters
        self.randomize_reset = randomize_reset
        """bool: If `True`, initailize the environment with random state at the start of each episode. Default is `True`.
        """

        self.disorient = disorient
        """bool: If True, random initialization and random orientation of the system at start of each episode. Default is ``True``.
        
        Notes:
            * If `self.disorient` is true and `self.sample_SO3` is true, randomly initialize orientation of robot for every episode and sample this orientation from uniform distribution of SO3 matrices.
            * If `self.disorient` is true, then randomly initialize the robot orientation at episode start.
            * If `self.disorient` is false, do not randomly initialize initial orientation of robot and do a deterministic
            initialization of the robot orientation as quaternion [1.0, 0., 0., 0.] for every episode.
        """

        self.sample_SO3 = sample_SO3
        """bool: If True, sample orientation uniformly from SO3 for random initialization of episode. Default is `False`.
        """

        self.observation_noise_std = observation_noise_std
        """float: Standard deviation of noise added to observation vector. If zero, no noise is added to observation. Default is `0.`. If non-zero, a vector is sampled from normal distribution and added to observation vector.
        """

        self.reduce_heading_error = reduce_heading_error
        """bool: If `True`, reward function tries to reduce the heading error in orientation of the system. Default is `True`."""

        # initial state randomizer bounds
        self.env_bounding_box = env_bounding_box
        """float: Max. initial position error. Use to initialize the system in bounded space. Default is ``1.2``."""

        self.init_max_vel = init_max_vel
        """float: Max. initial velocity error. Use to initialize the system in bounded space. Default is ``0.5``."""

        self.init_max_angular_vel = init_max_angular_vel
        """float: Max. initial angular velocity error. Use to initialize the system in bounded space. Default is `pi/10`."""

        self.init_max_attitude = init_max_attitude
        """float: Max. initial attitude error. Use to initialize the system in bounded space. Default is `pi/6.`."""

        self.bonus_to_reach_goal = bonus_to_reach_goal
        """float: Bonus value or reward when RL agent takes the system to the goal state. Default is ``15.0``.
        """

        self.max_reward_for_velocity_towards_goal = max_reward_for_velocity_towards_goal
        """float: Max. reward possible when the agent is heading towards goal, i.e., velocity vector points towards goal direction. Default is ``2.0``."""

        self.position_reward_constant = position_reward_constant
        """float: Position reward constant coefficient. Default is ``5.0``.
        """

        self.orientation_reward_constant = orientation_reward_constant
        """float: Orientation reward constant coefficient. Default is ``0.02``.
        """

        self.linear_velocity_reward_constant = linear_velocity_reward_constant
        """float: Linear velocity reward constant. Default is ``0.01``.
        """

        self.angular_velocity_reward_constant = angular_velocity_reward_constant
        """float: Angular velocity reward constant. Default is ``0.001``.
        """

        self.action_reward_constant = action_reward_constant
        """float: Action reward coefficient. Default is ``0.0025``.
        """

        self.reward_for_staying_alive = reward_for_staying_alive
        """float: Reward for staying alive in the environment. Default is ``5.0``.
        """

        self.reward_scaling_coefficient = reward_scaling_coefficient
        """float: Reward multiplication factor which can be used to scale the value of reward to be greater or smaller. Default is ``1.0``.
        """

        self.policy_range = [-1.0, 1.0]
        """tuple: Policy-output (a.k.a. action) range. Default is ``[-1.0, 1.0]``
        """

        self.policy_range_safe = [-0.8, 0.8]
        """tuple: Safe policy output range. Default is ``[-0.8, 0.8]``
        """

        # to record true states of robot in the simulator
        self.mujoco_qpos = None
        """numpy.ndarray: Mujoco pose vector of the system.
        """

        self.mujoco_qvel = None
        """numpy.ndarray: Mujoco velocity vector of the system.
        """

        self.previous_robot_observation = None
        """numpy.ndarray: Observation buffer to store previous robot observation
        """

        self.current_robot_observation = None
        """numpy.ndarray: Observation buffer to store current robot observation
        """

        self.previous_quat = None
        """numpy.ndarray: Buffer to keep record of orientation from mujoco. Stores previous quaternion.
        """

        self.current_quat = None
        """numpy.ndarray: Buffer to keep record of orientation from mujoco. Stores current orientation quaternion.
        """

        self.current_policy_action = np.array([-1, -1, -1, -1.])
        """numpy.ndarray: Buffer to hold current action input from policy to the environment. Stores the action vector scaled between (-1., 1.)
        """

        self.desired_position = np.array([0, 0, 3.0])
        """numpy.ndarray: Desired position of the system. Goal of the RL agent."""

        self._time = 0              # initialize time counter.
        self.gravity_mag = 9.81     # default value of acceleration due to gravity

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip)

        self.gravity_mag = float(abs(self.model.opt.gravity[2]))

    @property
    def env_bounding_box_norm(self):
        """
        Max. distance of the drone from the bounding box limits. Or maximum allowed distance of the drone from the desired position. It is the radius of the sphere within which the robot can observe the goal.

        Returns
        -------
        env_bounding_box_sphere_radius: float
        """
        return self.norm(np.array([self.env_bounding_box, self.env_bounding_box, self.env_bounding_box]))

    @property
    def error_tolerance_norm(self):
        """
        Returns the radius of the sphere within which the robot can be considered accurate.

        Returns
        -------
        error_tolerance_norm_sphere_radius: float
        """
        return self.norm(np.array([self.error_tolerance, self.error_tolerance, self.error_tolerance]))

    def step(self, a):
        """
        Take a step in the environment given an action

        :param a: action vector
        :type a: numpy.ndarray
        :return: tuple of agent-next_state, agent-reward, episode completetion flag and additional-info
        :rtype: tuple[numpy.ndarray, float, bool, dict]
        """
        reward = 0
        self.do_simulation(self.clip_action(a), self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all()
        done = not notdone
        info = {"reward_info": reward, "mujoco_qpos": self.mujoco_qpos, "mujoco_qvel": self.mujoco_qvel}
        return ob, reward, done, info

    def clip_action(self, action, a_min=-1.0, a_max=1.0):
        """
        Clip policy action vector to be within given minimum and maximum limits
        """
        action = np.clip(action, a_min=a_min, a_max=a_max)
        return action

    def viewer_setup(self):
        """This method is called when the viewer is initialized. Optionally implement this method, if you need to tinker with camera position and so forth.
        """
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 2.5

    @property
    def mass(self):
        """Return mass of the environment-body or robot.
        :return: mass of the robot
        :rtype: float
        """
        return self.model.body_mass[1]

    @property
    def inertia(self):
        """
        Returns
        -------
        numpy.ndarray: Inertia matrix of the system.
        """
        return np.array(self.sim.data.cinert)

    def get_motor_input(self, action):
        raise NotImplementedError

    def _get_obs(self):
        """
        Mujoco observations.

        :return: tuple of qpos, qvel and dict of copy of vectors qpos and qvel
        :rtype: tuple[numpy.ndarray, numpy.ndarray, dict]
        """
        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()

        self.mujoco_qpos = np.array(qpos)
        self.mujoco_qvel = np.array(qvel)

        obs = np.concatenate([qpos, qvel]).flatten()
        return obs

    def get_joint_qpos_qvel(self, joint_name):
        """

        Args:
            joint_name (str): Name of the joint.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Tuple containing the following elements in given order:
                - qpos (numpy.ndarray): True pose vector from the simulator.
                - qvel (numpy.ndarray): True velocity vector from the simulator.

        """
        qpos = self.data.get_joint_qpos(joint_name).copy()
        qvel = self.data.get_joint_qvel(joint_name).copy()
        return qpos, qvel

    def print_info(self):
        """
        Print information about the environment in the console. This method can be used for Debugging purposes.
        """
        print("Mass of robot :", self.mass)
        if isinstance(self.observation_space, spaces.Box):
            print("Observation dimensions :", self.observation_space.shape[0])
        else:
            print("Observation type: ", type(self.observation_space.sample()))
            print("Observation sample: ", self.observation_space.sample())

        print("Action dimensions :", self.action_space.shape[0])
        print("Min. action :", self.action_space.low)
        print("Max. action :", self.action_space.high)

        print("Actuator_control:", type(self.model.actuator_ctrlrange))
        print("actuator_forcerange:", self.model.actuator_forcerange)
        print("actuator_forcelimited:", self.model.actuator_forcelimited)
        print("actuator_ctrllimited:", self.model.actuator_ctrllimited)

    def orientation_error(self, quat):
        """
        Orientation error assuming desired orientation is (roll, pitch, yaw) = (0, 0, 0).

        Parameters
        ----------
        quat: numpy.ndarray
            Orientation quaternion of the robot with shape (4,). Quaternion vector is in scalar first format (q0, qx, qy, qz)

        Returns
        -------
        orientation_error: float
            Magnitude of error in orientation.

        """
        error = 0.
        rpy = multirotor_utils.quat2euler(quat)
        if self.reduce_heading_error:
            error += self.norm(rpy)
        else:
            error += self.norm(rpy[:2])     # exclude error in yaw

        return error

    def goal_reached(self, error_xyz):
        """
        This method checks if the given position error vector is close to zero or not.

        Parameters
        ----------
        error_xyz: numpy.ndarray
            Vector of error along xyz axes.

        Returns
        -------
        has_reached_goal: bool
            ``True`` if the system reached the goal position else ``False``. If the system reaches the goal position, error_xyz vector will be close to zero in magnitude.
        """
        return self.norm(error_xyz) < self.error_tolerance_norm

    def is_within_env_bounds(self, error_xyz):
        """
        This method checks if the robot is with the environment bounds or not. Environment bounds signify the range withing which the goal is observable. If this function returns ``True``, it automatically means that the goal is within the observable range of robot.

        Parameters
        ----------
        error_xyz: numpy.ndarray
            Vector of position error of the robot w.r.t. the target locations, i.e., vector = (target - robot_xyz).

        Returns
        -------
        within_bounds: bool
            ``True`` if the drone is within the environment limits else ``False``.
        """
        return self.norm(error_xyz) < self.env_bounding_box_norm

    def norm(self, vector):
        """
        Calculate the euclidean norm of a vector.

        Args:
            vector (numpy.ndarray): Vector of shape (n,)

        Returns:
            float: Norm or magnitude of the vector.
        """
        return np.linalg.norm(vector)

    def bound_violation_penalty(self, error_xyz):
        """

        Parameters
        ----------
        error_xyz: numpy.ndarray
            Error vector of robot position and desired position along x-y-z axes.

        Returns
        -------
        penalty: float
            If the robot is within the goal range, than the penalty is zero, else the penalty is high.

        Notes:
            You will need to subtract penalty from reward or when using this value in reward function multiply it with ``-1``.
        """
        penalty = 0.0
        if not self.is_within_env_bounds(error_xyz):
            penalty += self.bonus_to_reach_goal
        return penalty

    def bonus_reward_to_achieve_goal(self, error_xyz):
        """
        Return bonus reward value if the goal is achieved by the robot.

        Parameters
        ----------
        error_xyz: numpy.ndarray
            Error vector of robot position and desired position along x-y-z axes.

        Returns
        -------
        bonus_reward: float
            Bonus reward value if the goal is achieved using the robot and the control agent.
        """

        bonus = 0.0
        if self.goal_reached(error_xyz):
            bonus += self.bonus_to_reach_goal
        return bonus

    def reward_velocity_towards_goal(self, error_xyz, velocity):
        """
        Reward for velocity of the system towards goal.

        Parameters
        ----------
        error_xyz: numpy.ndarray
            Position error of the system along xyz-coordinates.

        velocity: numpy.ndarray
            Velocity vector (vx, vy, vz) of the system in body reference frame

        Returns
        -------
        reward: float
            Reward based on the system velocity inline with the desired position.
        """
        if self.goal_reached(error_xyz):
            return self.max_reward_for_velocity_towards_goal
        unit_xyz = error_xyz/(self.norm(error_xyz) + 1e-6)
        velocity_direction = velocity/(self.norm(velocity) + 1e-6)
        reward = np.dot(unit_xyz, velocity_direction)
        return reward

    def is_done(self, ob):
        """
        Returns ``True`` if the current episode is over.

        Parameters
        ----------
        ob: numpy.ndarray
            Observation vector

        Returns
        -------
        terminate: bool
            ``True`` if current episode is over else ``False``.
        """
        notdone = np.isfinite(ob).all() and self.is_within_env_bounds(ob[:3]) and (self._time < self.max_time_steps)
        done = not notdone
        return done

    def get_body_state(self, body_name):
        """
        Returns the state of the body with respect to world frame of reference (inertial frame).

        Parameters
        ----------
        body_name: str
            Name of the body used in the XML file.

        Returns
        -------
        state: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            State of the body as xyz ``(3,)``, rotation_matrix_flat ``(9,)``, linear_vel ``(3,)``, angular_vel ``(3,)`` as a tuple.
        """
        xyz = self.data.get_body_xpos(body_name).copy()
        mat = self.data.get_body_xmat(body_name).flatten().copy()
        vel = self.data.get_body_xvelp(body_name).copy()
        avel = self.data.get_body_xvelr(body_name).copy()
        return xyz, mat, vel, avel

    def get_body_states_for_plots(self, body_name):
        xyz, mat, vel, avel = self.get_body_state(body_name)
        quat = multirotor_utils.rot2quat(mat.reshape((3, 3)))
        rpy = multirotor_utils.quat2euler(quat)
        return xyz, rpy, vel, avel

    def get_body_state_in_body_frame(self, body_name, xyz_ref=None):
        """

        Args:
            body_name (str): Name of the body in XML File
            xyz_ref (numpy.ndarray): Reference XYZ of body frame

        Returns:
            xyz, mat, vel (in body frame of reference), avel

        """
        xyz, mat, vel, avel = self.get_body_state(body_name)        # in world frame

        if xyz_ref is not None:
            xyz = np.array(xyz) - np.array(xyz_ref)

        mat_ = mat.reshape((3, 3))
        vel = np.dot(mat_, vel)  # vel in body frame

        return xyz, mat, vel, avel
