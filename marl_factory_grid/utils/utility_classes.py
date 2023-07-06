import gymnasium as gym


class MarlFrameStack(gym.ObservationWrapper):
    """todo @romue404"""
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        if isinstance(self.env, gym.wrappers.FrameStack) and self.env.unwrapped.n_agents > 1:
            return observation[0:].swapaxes(0, 1)
        return observation
