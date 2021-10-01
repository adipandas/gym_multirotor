import numpy as np
from gym_multirotor import utils
from gym_multirotor.envs.mujoco.quadrotor_plus_hover import QuadrotorPlusHoverEnv


class TiltrotorPlus8DofHoverEnv(QuadrotorPlusHoverEnv):
    """
    Tiltrotor quadcopter with 8 dofs.

    * env_name: TiltrotorPlus8DofHoverEnv-v0

    Args:
        xml_name (str): Name of the xml file containing the model of the robot.
        frame_skip (int): Number of frames to skip after application of one action command.
    """

    obs_tilt_index = np.arange(18, 22)
    action_index_tilt = np.arange(4, 8)

    def __init__(self, xml_name="tiltrotor_plus_hover.xml", frame_skip=5):

        self.tilt_position_reward_constant = 1.0
        """float: Reward constant for position of the tilt servo.
        """

        super(TiltrotorPlus8DofHoverEnv, self).__init__(xml_name=xml_name, frame_skip=frame_skip)

    def _get_obs(self):
        """
        Returns the fully observed environment state.

        Returns:
             numpy.ndarray: Observation vector of shape (22,). The elements of the observation tuple include (xyz, rotation_matrix, linear_velocity, angular_velocity, tilt_angles).
        """

        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()

        self.mujoco_qvel = np.array(qvel)
        self.mujoco_qpos = np.array(qpos)

        e_pos = qpos[:3] - self.desired_position
        rot_mat = utils.quat2rot(qpos[3:7])
        tilt = qpos[7:11]

        lvel = qvel[0:3]        # linear velocity
        avel = qvel[3:6]        # angular velocity
        obs = np.concatenate([e_pos, rot_mat.flatten(), lvel, avel, tilt]).ravel()
        return obs

    def get_motor_input(self, a):
        """
        Transform policy actions to motor inputs.

        Args:
            a (numpy.ndarray): Action vector of shape (8,) from policy.

        Returns:
            numpy.ndarray: Action vector of shape (8,) to send as input to mujoco.
        """

        action_range = self.action_space.high - self.action_space.low
        avg_actuation = np.array([self.mass * 9.81 * 0.25,
                                  self.mass * 9.81 * 0.25,
                                  self.mass * 9.81 * 0.25,
                                  self.mass * 9.81 * 0.25,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0])     # 4 propellers and 4 tilt-servos
        motor_inputs = avg_actuation + a * action_range / (self.policy_range[1]-self.policy_range[0])
        return motor_inputs

    def get_reward(self, ob, a):
        """
        Method to evaluate reward of the agent based on the observation after taking the action.

        Args:
            ob (numpy.ndarray): Observation vector of the agent with shape (22,).
            a (numpy.ndarray): Action vector of the agent with shape (8,)

        Returns:
            tuple[float, dict]: Tuple containing follwing elements in the given order:
                - reward (float): Scalar reward based on observation and action.
                - reward_info (dict): Dictionary of reward for specific state values. This dictionary contains the reward values corresponding to the following keys - (position, orientation, linear_velocity, angular_velocity, action_thrust, action_tilt, alive_bonus, extra_bonus, extra_penalty).
        """
        ob_quad, ob_tilt = ob[:18].copy(), ob[18:22]
        action_quad, action_tilt = a[0:4], a[4:8]
        reward_quad, reward_info = super(TiltrotorPlus8DofHoverEnv, self).get_reward(ob_quad, action_quad)

        reward_tilt = self.norm(ob_tilt) * (-self.tilt_position_reward_constant)
        reward_action_tilt = self.norm(action_tilt) * (-self.action_reward_constant)

        reward_info['reward_tilt'] = reward_tilt
        reward_info['action_tilt'] = reward_action_tilt

        rewards = (reward_quad, reward_tilt, reward_action_tilt)
        reward = sum(rewards) * self.reward_scaling_coefficient
        return reward, reward_info

    def initialize_robot(self, randomize=True):
        """
        Method to randomize initial state of the robot.

        Args:
            randomize (bool): If ``True``, initialize the robot randomly.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Tuple containing the following vectors in given order:
                - qpose_init (numpy.ndarray): Robot's initial pose in mujoco with shape (11,).
                - qvel_init (numpy.ndarray): Robot's initial velocity in mujoco with shape (10,).
        """
        if not randomize:
            qpos_init = np.array([self.desired_position[0], self.desired_position[1], self.desired_position[2], 1., 0., 0., 0., 0., 0., 0., 0])
            qvel_init = np.zeros((10,))
            return qpos_init, qvel_init

        # attitude (roll pitch yaw)
        quat_init = np.array([1., 0., 0., 0.])
        if self.disorient and self.sample_SO3:
            rot_mat = utils.sampleSO3()
            quat_init = utils.rot2quat(rot_mat)
        elif self.disorient:
            attitude_euler_rand = np.random.uniform(low=-self.init_max_attitude, high=self.init_max_attitude, size=(3,))
            quat_init = utils.euler2quat(attitude_euler_rand)

        c = 0.2
        ep = np.random.uniform(low=-(self.env_bounding_box-c), high=(self.env_bounding_box-c), size=(3,))
        pos_init = ep + self.desired_position
        tilt_init = np.random.uniform(low=-self.init_max_attitude, high=self.init_max_attitude, size=(4,))
        vel_init = utils.sample_unit3d() * self.init_max_vel
        angular_vel_init = utils.sample_unit3d() * self.init_max_angular_vel
        tilt_vel = np.zeros((4,))
        qpos_init = np.concatenate([pos_init, quat_init, tilt_init]).ravel()
        qvel_init = np.concatenate([vel_init, angular_vel_init, tilt_vel]).ravel()
        return qpos_init, qvel_init
