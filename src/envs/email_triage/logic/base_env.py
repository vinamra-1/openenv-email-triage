class Environment:
    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    @property
    def state(self):
        raise NotImplementedError