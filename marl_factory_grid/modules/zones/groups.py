from marl_factory_grid.environment.groups.objects import Objects
from marl_factory_grid.modules.zones import Zone


class Zones(Objects):

    symbol = None
    _entity = Zone
    var_can_move = False

    def __init__(self, *args, **kwargs):
        super(Zones, self).__init__(*args, can_collide=True, **kwargs)
