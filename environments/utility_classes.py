from typing import NamedTuple, Union
import gym
from gym.wrappers.frame_stack import FrameStack


class AgentRenderOptions(object):
    SEPERATE = 'seperate'
    COMBINED = 'combined'
    LEVEL = 'lvl'
    NOT = 'not'


class MovementProperties(NamedTuple):
    allow_square_movement: bool = True
    allow_diagonal_movement: bool = False
    allow_no_op: bool = False


class ObservationProperties(NamedTuple):
    render_agents: AgentRenderOptions = AgentRenderOptions.SEPERATE
    omit_agent_self: bool = True
    additional_agent_placeholder: Union[None, str, int] = None
    cast_shadows: bool = True
    frames_to_stack: int = 0
    pomdp_r: int = 0
    show_global_position_info: bool = True


class MarlFrameStack(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        if isinstance(self.env, FrameStack) and self.env.unwrapped.n_agents > 1:
            return observation[0:].swapaxes(0, 1)
        return observation

