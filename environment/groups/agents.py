from environment.groups.env_objects import EnvObjects
from environment.groups.mixins import PositionMixin
from environment.entity.agent import Agent
import environment.constants as c


class Agents(PositionMixin, EnvObjects):
    _entity = Agent
    is_blocking_light = False
    can_move = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def obs_pairs(self):
        return [(a.name, a) for a in self]

    @property
    def action_space(self):
        from gymnasium import spaces
        space = spaces.Tuple([spaces.Discrete(len(x.actions)) for x in self])
        return space

    @property
    def named_action_space(self):
        named_space = dict()
        for agent in self:
            named_space[agent.name] = {action.name: idx for idx, action in enumerate(agent.actions)}
        return named_space
