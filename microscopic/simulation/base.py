

class SUMOBaseSimulation:
    def __init__(self, config):
        self.config = config
        self.simulation_time = 0

    def step(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def setup(self):
        pass

    def check(self):
        pass

    def run(self):
        pass

    def post_run(self):
        pass