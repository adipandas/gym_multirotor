import numpy as np


class FirstOrderLowPassFilter:
    """
    Discrete first order low pass filter.

    Parameters
    ----------
    control_timestep : float
        Time between two consecutive inputs. Default is `0.01`.
    input_shape : int
        Shape of the input vector. Default is `4`.
    time_constant : float
        Time constant of this low pass filter.

    Note
    ----
    Smaller the time constant, faster will be the response of the motor.
    In other words, the motor will reach the commanded value faster.

    Attributes
    ----------
    input_shape : int
        Shape of the input vector.
    control_timestep :
        Time between two consecutive actuator inputs.
    time_constant : float
        Time constant of this low pass filter.
    alpha : float

    """
    def __init__(self, control_timestep=0.01, input_shape=4, time_constant=None):

        self.input_shape = input_shape
        self.control_timestep = control_timestep

        if time_constant is None:
            self.time_constant = 20. * control_timestep
        else:
            self.time_constant = time_constant

        self.alpha = self.control_timestep/self.time_constant

        self.u_buffer = np.zeros((input_shape,))

    def reset(self):
        self.u_buffer = np.zeros((self.input_shape,))

    def step(self, u):
        """
        Execute one time-step.

        Parameters
        ----------
        u : float or numpy.ndarray
            Input to the filter.

        Returns
        -------
        u_o : float or numpy.ndarray
            Filter output.
        """
        u_o = (u - self.u_buffer) * self.alpha + self.u_buffer

        self.u_buffer = np.array([u_o]).flatten()
        return u_o


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    lpf = FirstOrderLowPassFilter(control_timestep=0.01, input_shape=2, time_constant=0.04)
    inp = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

    out = []
    for c in range(inp.shape[1]):
        u = inp[:, c]
        out.append(lpf.step(u))
        # if c == 10:
        #     lpf.reset()

    out = np.array(out)
    plt.scatter(list(range(inp.shape[1])), out[:, 1])
    # plt.plot(out[:, 1])
    plt.plot(list(range(inp.shape[1])), inp[0, :], linewidth=2.0, color='red')
    plt.grid(True)
    plt.xticks(ticks=list(range(inp.shape[1])))
    plt.show()
    # print(out)
