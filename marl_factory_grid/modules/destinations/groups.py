from marl_factory_grid.environment.groups.collection import Collection
from marl_factory_grid.modules.destinations.entitites import Destination


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

    @property
    def var_can_be_bound(self):
        return True

    def __init__(self, *args, **kwargs):
        """
        A collection of destinations.
        """
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return super(Destinations, self).__repr__()
