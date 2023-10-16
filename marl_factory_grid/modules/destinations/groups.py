from marl_factory_grid.environment.groups.env_objects import EnvObjects
from marl_factory_grid.environment.groups.mixins import PositionMixin, HasBoundMixin
from marl_factory_grid.modules.destinations.entitites import Destination, BoundDestination
from marl_factory_grid.environment import constants as c
from marl_factory_grid.modules.destinations import constants as d


class Destinations(PositionMixin, EnvObjects):
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


class BoundDestinations(HasBoundMixin, Destinations):
    _entity = BoundDestination

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ReachedDestinations(Destinations):
    _entity = Destination
    is_blocking_light = False
    can_collide = False

    def __init__(self, *args, **kwargs):
        super(ReachedDestinations, self).__init__(*args, **kwargs)

    def __repr__(self):
        return super(ReachedDestinations, self).__repr__()
