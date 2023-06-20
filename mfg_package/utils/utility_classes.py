import gymnasium as gym


class EnvCombiner(object):

    def __init__(self, *envs_cls):
        self._env_dict = {env_cls.__name__: env_cls for env_cls in envs_cls}

    @staticmethod
    def combine_cls(name, *envs_cls):
        return type(name, envs_cls, {})

    def build(self):
        name = f'{"".join([x.lower().replace("factory").capitalize() for x in self._env_dict.keys()])}Factory'

        return self.combine_cls(name, tuple(self._env_dict.values()))


class MarlFrameStack(gym.ObservationWrapper):
    """todo @romue404"""
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        if isinstance(self.env, gym.wrappers.FrameStack) and self.env.unwrapped.n_agents > 1:
            return observation[0:].swapaxes(0, 1)
        return observation
