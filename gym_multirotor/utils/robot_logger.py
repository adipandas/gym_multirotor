

class Logger:
    def __init__(self):
        self.time = []
        self.state = []
        self.desired_state = []
        self.omega = []
        self.tilt = []

    def add(self, state, desired_state, omega, tilt):
        self.state.append(state)
        self.desired_state.append(desired_state)
        self.omega.append(omega)
        self.tilt.append(tilt)
