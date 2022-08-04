from typing import NamedTuple, Union
import gym
from gym.wrappers.frame_stack import FrameStack


class EnvCombiner(object):

    def __init__(self, *envs_cls):
        self._env_dict = {env_cls.__name__: env_cls for env_cls in envs_cls}

    @staticmethod
    def combine_cls(name, *envs_cls):
        return type(name,envs_cls,{})

    def build(self):
        name = f'{"".join([x.lower().replace("factory").capitalize() for x in self._env_dict.keys()])}Factory'

        return self.combine_cls(name, tuple(self._env_dict.values()))


class AgentRenderOptions(object):
    """
    Class that specifies the available options for the way agents are represented in the env observation.

    SEPERATE:
    Each agent is represented in a seperate slice as Constant.OCCUPIED_CELL value (one hot)

    COMBINED:
    For all agent, value of Constant.OCCUPIED_CELL is added to a zero-value slice at the agents position (sum(SEPERATE))

    LEVEL:
    The combined slice is added to the LEVEL-slice. (Agents appear as obstacle / wall)

    NOT:
    The position of individual agents can not be read from the observation.
    """

    SEPERATE = 'seperate'
    COMBINED = 'combined'
    LEVEL = 'lvl'
    NOT = 'not'


class MovementProperties(NamedTuple):
    """
    Property holder; for setting multiple related parameters through a single parameter. Comes with default values.
    """

    """Allow the manhattan style movement on a grid (move to cells that are connected by square edges)."""
    allow_square_movement: bool = True

    """Allow diagonal movement on the grid (move to cells that are connected by square corners)."""
    allow_diagonal_movement: bool = False

    """Allow the agent to just do nothing; not move (NO-OP)."""
    allow_no_op: bool = False


class ObservationProperties(NamedTuple):
    """
    Property holder; for setting multiple related parameters through a single parameter. Comes with default values.
    """

    """How to represent agents in the observation space. This may also alter the obs-shape."""
    render_agents: AgentRenderOptions = AgentRenderOptions.SEPERATE

    """Obserations are build per agent; whether the current agent should be represented in its own observation."""
    omit_agent_self: bool = True

    """Their might be the case you want to modify the agents obs-space, so that it can be used with additional obs.
       The additional slice can be filled with any number"""
    additional_agent_placeholder: Union[None, str, int] = None

    """Whether to cast shadows (make floortiles and items hidden).; """
    cast_shadows: bool = True

    """Frame Stacking is a methode do give some temporal information to the agents. 
    This paramters controls how many "old-frames" """
    frames_to_stack: int = 0

    """Specifies the radius (_r) of the agents field of view. Please note, that the agents grid cellis not taken 
        accountance for. This means, that the resulting field of view diameter = `pomdp_r * 2 + 1`.
        A 'pomdp_r' of 0 always returns the full env == no partial observability."""
    pomdp_r: int = 2

    """Whether to place a visual encoding on walkable tiles around the doors. This is helpfull when the doors can be 
    operated from their surrounding area. So the agent can more easily get a notion of where to choose the door option.
    However, this is not necesarry at all. 
        """
    indicate_door_area: bool = False

    """Whether to add the agents normalized global position as float values (2,1) to a seperate information slice.
        More optional informations are to come.
        """
    show_global_position_info: bool = False


class MarlFrameStack(gym.ObservationWrapper):
    """todo @romue404"""
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        if isinstance(self.env, FrameStack) and self.env.unwrapped.n_agents > 1:
            return observation[0:].swapaxes(0, 1)
        return observation
