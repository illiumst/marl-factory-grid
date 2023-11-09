from typing import List, Union

from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.entity.util import GlobalPosition
from marl_factory_grid.environment.groups.collection import Collection
from marl_factory_grid.utils.results import Result
from marl_factory_grid.utils.states import Gamestate


class Combined(Collection):

    @property
    def var_has_position(self):
        return True

    @property
    def name(self):
        return f'{super().name}({self._ident or self._names})'

    @property
    def names(self):
        return self._names

    def __init__(self, names: List[str], *args, identifier: Union[None, str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ident = identifier
        self._names = names or list()

    @property
    def obs_tag(self):
        return self.name

    @property
    def obs_pairs(self):
        return [(name, None) for name in self.names]


class GlobalPositions(Collection):

    _entity = GlobalPosition

    var_is_blocking_light = False
    var_can_be_bound = True
    var_can_collide = False
    var_has_position = False

    def __init__(self, *args, **kwargs):
        super(GlobalPositions, self).__init__(*args, **kwargs)

    def spawn(self, agents, level_shape, *args, **kwargs):
        self.add_items([self._entity(agent, level_shape, *args, **kwargs) for agent in agents])
        return [Result(identifier=f'{self.name}_spawn', validity=c.VALID, value=len(self))]

    def trigger_spawn(self, state: Gamestate, *args, **kwargs) -> [Result]:
        return self.spawn(state[c.AGENT], state.lvl_shape, *args, **kwargs)
