import numpy as np
from gym_multirotor.utils.low_pass_filter import FirstOrderLowPassFilter
from gym_multirotor.utils.noise import OUNoise


class ThrustMotor:
    """
    Thrust Motor Object to model the actuators.

    Parameters
    ----------
    kf : float
        Motor force coefficient.
    km : float
        Motor movement coefficient.
    control_timestep : float
        Time between two consecutive motor inputs.
    n_motor : int
        Number of motors.
    time_constant : float
        Time-constant of low pass filter used in this model.
    random_state : numpy.random.RandomState
        Random number generator RandomState object.
    mu : float
        OU Process mean.
    theta : float
        Decay rate of the OU process after a spike.
    sigma : float
        Scale factor for the OU process.
    sigma_min : float
        Scale factor lower limit for OU process.
    sigma_decay : float
        Scale factor decay rate for OU process.
    add_noise : bool
        If `True`, add noise to motor output.

    Attributes
    ----------
    kf : float
        Motor force coefficient.
    km : float
        Motor movement coefficient.
    lpf_motor_speed : float
        Low pass filter for motor speed.
    motor_noise : OUNoise
        Noise generator for thrust motor.
    add_noise : bool
        Add noise to motor outputs.
    """

    def __init__(
            self, kf, km, control_timestep, n_motor, random_state, time_constant=None,
            mu=0.0, theta=0.15, sigma=0.15, sigma_min=0.05, sigma_decay=1, add_noise=False
    ):
        self.kf = kf
        self.km = km

        self.add_noise = add_noise

        self.motor_noise = OUNoise(
            size=n_motor, random_state=random_state,
            mu=mu, theta=theta, sigma=sigma, sigma_decay=sigma_decay, sigma_min=sigma_min
        )

        self.lpf_motor_speed = FirstOrderLowPassFilter(
            control_timestep, input_shape=n_motor, time_constant=time_constant
        )

    def speed2thrust(self, speed):
        """
        Convert speed to motor thrust.

        Parameters
        ----------
        speed : float or numpy.ndarray
            Motor speed. (rad/s)

        Returns
        -------
        thrust : float or numpy.ndarray
            Thrust exerted by motor (in Newtons).
        """
        thrust = self.kf * speed * speed
        return thrust

    def thrust2speed(self, thrust):
        """
        Convert motor thrust to motor speed.

        Parameters
        ----------
        thrust : float or numpy.ndarray
            Thrust exerted by motor (in Newtons).

        Returns
        -------
        speed : float or numpy.ndarray
            Motor speed. (rad/s)

        """
        speed2 = np.abs(thrust) / self.kf
        speed = np.sqrt(speed2)
        return speed

    def speed_command2actual(self, speed_command):
        """
        Convert the commanded motor signal to actual motor signal.

        Note
        ----
        There is some delay for the motor in real world to reach the commanded value from its current value.
        This method applies that delay to the motor through a discrete time first order low pass filter.

        Parameters
        ----------
        speed_command : float or numpy.ndarray
            Commanded speed for the motor(s).

        Returns
        -------
        speed_actual : float or numpy.ndarray
            Actual speed for the motor(s).

        """
        speed_actual = self.lpf_motor_speed.step(speed_command)

        if self.add_noise:
            speed_actual += self.motor_noise.sample()

        speed_actual = np.clip(speed_actual, a_min=0., a_max=np.inf)
        return speed_actual

    def thrust_command2actual(self, thrust_command):
        """
        Convert the commanded motor thrust signal to actual motor signal.

        Note
        ----
        There is some delay for the motor in real world to reach the commanded value from its current value.
        This method applies that delay to the motor through a discrete time first order low pass filter.

        Parameters
        ----------
        thrust_command : float or numpy.ndarray
            Commanded thrust for the motor(s).

        Returns
        -------
        thrust_actual : float or numpy.ndarray
            Actual thrust for the motor(s).

        """
        thrust_sign = np.sign(thrust_command)

        speed_command = self.thrust2speed(thrust_command)

        speed_actual = self.speed_command2actual(speed_command)

        thrust_actual = self.speed2thrust(speed_actual) * thrust_sign

        return thrust_actual

    def reset(self):
        self.motor_noise.reset()
        self.lpf_motor_speed.reset()


if __name__ == '__main__':
    RPM_2_RAD_PER_SEC = 2 * np.pi / 60.

    kf = 6.11 * 10**-8 / (RPM_2_RAD_PER_SEC**2)
    km = 1.5 * 10**-9 / (RPM_2_RAD_PER_SEC**2)

    random_state = np.random.RandomState(seed=10)

    motor = ThrustMotor(
        kf=kf,
        km=km,
        control_timestep=0.01,
        n_motor=4,
        random_state=random_state,
        time_constant=None,
        mu=0.0,
        theta=0.15,
        sigma=0.15,
        sigma_min=0.05,
        sigma_decay=1,
        add_noise=True
    )

    for i in range(100):
        policy_input = random_state.uniform(low=0, high=1, size=(4,))
        a = motor.thrust_command2actual(policy_input)

        if i == 50:
            motor.reset()
