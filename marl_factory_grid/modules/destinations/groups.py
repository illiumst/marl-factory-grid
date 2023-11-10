from marl_factory_grid.environment.groups.collection import Collection
from marl_factory_grid.modules.destinations.entitites import Destination


class Destinations(Collection):
    _entity = Destination

    var_is_blocking_light = False
    var_can_collide = False
    var_can_move = False
    var_has_position = True
    var_can_be_bound = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return super(Destinations, self).__repr__()
