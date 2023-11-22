from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


class MarlFrameStack(gym.ObservationWrapper):
    """todo @romue404"""
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        if isinstance(self.env, gym.wrappers.FrameStack) and self.env.unwrapped.n_agents > 1:
            return observation[0:].swapaxes(0, 1)
        return observation


@dataclass
class RenderEntity:
    """
    This class defines the interface to communicate with the Renderer. Name and pos are used to load an asset file
    named name.png and place it at the given pos.
    """
    name: str
    pos: np.array
    value: float = 1
    value_operation: str = 'none'
    state: str = None
    id: int = 0
    aux: Any = None
    real_name: str = 'none'


@dataclass
class Floor:
    """
    This class defines Entity like Floor-Objects, which do not come with the overhead.
    Solely used for field-of-view calculation.
    """

    @property
    def encoding(self):
        return 1

    @property
    def name(self):
        return f"Floor({self.pos})"

    @property
    def pos(self):
        return self.x, self.y

    x: int
    y: int
    var_is_blocking_light: bool = False

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"Floor{self.pos}"
