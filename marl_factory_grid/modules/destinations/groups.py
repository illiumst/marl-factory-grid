from marl_factory_grid.environment.groups.collection import Collection
from marl_factory_grid.modules.destinations.entitites import Destination
from marl_factory_grid.environment import constants as c
from marl_factory_grid.modules.destinations import constants as d


class Destinations(Collection):
    _entity = Destination

    @property
    def var_is_blocking_light(self):
        return False

    @property
    def var_can_collide(self):
        return False

    @property
    def var_can_move(self):
        return False

    @property
    def var_has_position(self):
        return True

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


