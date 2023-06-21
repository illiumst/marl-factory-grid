from marl_factory_grid.environment.groups.env_objects import EnvObjects
from marl_factory_grid.environment.groups.mixins import PositionMixin
from marl_factory_grid.modules.destinations.entitites import Destination


class Destinations(PositionMixin, EnvObjects):

    _entity = Destination
    is_blocking_light: bool = False
    can_collide: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return super(Destinations, self).__repr__()


class ReachedDestinations(Destinations):
    _entity = Destination
    is_blocking_light = False
    can_collide = False

    def __init__(self, *args, **kwargs):
        super(ReachedDestinations, self).__init__(*args, **kwargs)

    def __repr__(self):
        return super(ReachedDestinations, self).__repr__()
