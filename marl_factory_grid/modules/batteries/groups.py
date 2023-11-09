from typing import Union, List, Tuple

from marl_factory_grid.environment import constants as c
from marl_factory_grid.environment.groups.collection import Collection
from marl_factory_grid.modules.batteries.entitites import ChargePod, Battery
from marl_factory_grid.utils.results import Result


class Batteries(Collection):
    _entity = Battery

    var_has_position = False
    var_can_be_bound = True

    @property
    def obs_tag(self):
        return self.__class__.__name__

    def __init__(self, size, initial_charge_level: float=1.0, *args, **kwargs):
        super(Batteries, self).__init__(size, *args, **kwargs)
        self.initial_charge_level = initial_charge_level

    def spawn(self, coords_or_quantity: Union[int, List[Tuple[(int, int)]]], agents, *entity_args, **entity_kwargs):
        batteries = [self._entity(self.initial_charge_level, agent) for _, agent in enumerate(agents)]
        self.add_items(batteries)

    def trigger_spawn(self, state, *entity_args, coords_or_quantity=None,  **entity_kwargs):
        self.spawn(0, state[c.AGENT])
        return Result(identifier=f'{self.name}_spawn', validity=c.VALID, value=len(self))


class ChargePods(Collection):
    _entity = ChargePod

    def __init__(self, *args, **kwargs):
        super(ChargePods, self).__init__(*args, **kwargs)

    def __repr__(self):
        return super(ChargePods, self).__repr__()
