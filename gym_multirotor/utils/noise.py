import numpy as np


class OUNoise:
    """

    Parameters
    ----------
    size : int
        Size of the process vector.
    random_state : numpy.random.RandomState
        Random number generator RandomState object.
    mu : float
        process mean.
    theta : float
        Decay rate of the process after a spike.
    sigma : float
        Scale factor.
    sigma_min : float
        Scale factor lower limit.
    sigma_decay : float
        Scale factor decay rate.

    Attributes
    ----------
    size : int
        Size of the process vector.
    np_random : numpy.random.RandomState
        Random number generator RandomState object.
    mu : numpy.ndarray
        process mean of shape (size,).
    theta : float
        Decay rate of the process after a spike.
    sigma : float
        Scale factor.
    sigma_min : float
        Scale factor lower limit.
    sigma_decay : float
        Scale factor decay rate.


    References
    ----------
    Wikipedia, https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

    """
    def __init__(self, size, random_state, mu=0.0, theta=0.15, sigma=0.15, sigma_min=0.05, sigma_decay=1):

        self.mu = np.ones((size,)) * mu
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.np_random = random_state
        self.size = size

        self.reset()

    def reset(self):
        self.state = self.mu.copy()
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * self.np_random.standard_normal(self.size)
        self.state = x + dx
        return self.state


if __name__ == '__main__':
    random_state = np.random.RandomState(seed=10)
    oun = OUNoise(size=4, random_state=random_state, mu=0.0, theta=0.15, sigma=0.15, sigma_min=0.05, sigma_decay=1)

    oun.reset()

    print(oun.sample())
