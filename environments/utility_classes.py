from enum import Enum
from typing import NamedTuple, Union


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
    cast_shadows = True
    frames_to_stack: int = 0
    pomdp_r: int = 0
