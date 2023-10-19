from marl_factory_grid.environment.groups.env_objects import Collection
from marl_factory_grid.environment.groups.mixins import PositionMixin
from marl_factory_grid.modules.destinations.entitites import Destination
from marl_factory_grid.environment import constants as c
from marl_factory_grid.modules.destinations import constants as d


class Destinations(PositionMixin, Collection):
    _entity = Destination
    is_blocking_light: bool = False
    can_collide: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return super(Destinations, self).__repr__()

    @staticmethod
    def trigger_destination_spawn(n_dests, state):
        coordinates = state.entities.floorlist[:n_dests]
        if destinations := [Destination(pos) for pos in coordinates]:
            state[d.DESTINATION].add_items(destinations)
            state.print(f'{n_dests} new destinations have been spawned')
            return c.VALID
        else:
            state.print('No Destiantions are spawning, limit is reached.')
            return c.NOT_VALID


