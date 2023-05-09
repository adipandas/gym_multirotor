from gym_multirotor.envs.mujoco.quadrotor_plus_hover import QuadrotorPlusHoverEnv


class QuadrotorXHoverEnv(QuadrotorPlusHoverEnv):
    """
    Quadrotor with X-configuration.
    Environment designed to make the UAV hover at the desired position.

    Environment Name: QuadrotorXHoverEnv-v0

    Args:
        xml_name (str): Name of the xml file containing the model of the robot.
        frame_skip (int): Number of frames to skip after application of one action command.
    """

    def __init__(self, xml_name="quadrotor_x.xml", frame_skip=5, env_bounding_box=1.2, randomize_reset=False):
        super().__init__(xml_name, frame_skip=frame_skip, env_bounding_box=env_bounding_box, randomize_reset=randomize_reset)
